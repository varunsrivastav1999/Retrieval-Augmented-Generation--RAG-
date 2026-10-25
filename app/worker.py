"""Ingestion worker process: runs the job queue and watches the media folder.

Started as its own container (`python -m app.worker`) so parsing/OCR/embedding
never competes with query latency inside the API process.

Media layout (bind-mount a host folder to /media to drop files in):
  /media/<collection>/...            -> tenant "default", collection from folder
  /media/uploads/<tenant>/<coll>/... -> files uploaded through the API
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import time

from sqlalchemy import select

from app.config import get_settings
from app.db import Document, SessionLocal, init_db
from app.logging_setup import setup_logging
from app.rag import cache, jobs, models, vector_store
from app.rag.ingestion import index_file, remove_document
from app.rag.parsers import is_supported_file

log = logging.getLogger("worker")
stop = threading.Event()
HEARTBEAT = "/tmp/worker.heartbeat"


def heartbeat() -> None:
    try:
        with open(HEARTBEAT, "w") as f:
            f.write(str(time.time()))
    except OSError:
        pass


def _tenant_for(path: str) -> str:
    root = os.path.realpath(get_settings().media_path)
    rel = os.path.relpath(path, root).split(os.sep)
    return rel[1] if len(rel) > 2 and rel[0] == "uploads" else "default"


def scan_media() -> dict:
    """Queue new/changed files and drop documents whose file disappeared."""
    s = get_settings()
    root = os.path.realpath(s.media_path)
    if not os.path.isdir(root):
        return {"queued": 0, "removed": 0}
    on_disk = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if name.startswith(".") or not is_supported_file(name):
                continue
            full = os.path.realpath(os.path.join(dirpath, name))
            try:
                st = os.stat(full)
            except OSError:
                continue
            on_disk[full] = (st.st_size, st.st_mtime)

    with SessionLocal() as db:
        known = {d.path: d for d in db.execute(select(
            Document.id, Document.tenant_id, Document.path, Document.size_bytes, Document.mtime, Document.status))}
    queued = 0
    for path, (size, mtime) in on_disk.items():
        doc = known.get(path)
        if doc is None or doc.size_bytes != size or doc.mtime != mtime or doc.status in {"pending"}:
            jobs.enqueue(_tenant_for(path), path)
            queued += 1

    missing = [d for p, d in known.items() if p.startswith(root + os.sep) and p not in on_disk]
    removed = 0
    if missing and len(missing) > max(10, 0.5 * len(known)):
        # Protect against an unmounted/empty media volume wiping the index.
        log.warning("refusing to remove %d of %d documents in one scan (media volume missing?)", len(missing), len(known))
    else:
        for d in missing:
            remove_document(d.id)
            cache.bump_index_version(d.tenant_id)
            removed += 1
    if queued or removed:
        log.info("media scan", extra={"files": len(on_disk), "queued": queued, "removed": removed})
    return {"queued": queued, "removed": removed, "files": len(on_disk)}


def process(job: dict) -> None:
    s = get_settings()
    job_id = job["id"]
    path = job["source_path"]
    if not os.path.isfile(path):
        jobs.finish(job_id, "failed", "file no longer exists")
        return

    progressed = {"batches": 0}

    def progress(chunks: int, _rows: int) -> None:
        heartbeat()
        progressed["batches"] += 1
        jobs.update(job_id, chunks_total=chunks)

    # Keep the job visibly alive during long parse/OCR phases (no batches yet).
    alive = threading.Event()

    def keepalive() -> None:
        while not alive.wait(30):
            heartbeat()
            try:
                jobs.update(job_id)
            except Exception:
                pass

    threading.Thread(target=keepalive, name=f"job-{job_id[:8]}", daemon=True).start()
    try:
        result = index_file(path, job["tenant_id"], force=bool(job["force_reindex"]), progress=progress)
        jobs.finish(job_id, "skipped" if result["status"] == "skipped" else "succeeded",
                    document_id=result.get("document_id"), chunks_total=result.get("chunks", 0), progress_pct=100.0)
        for extracted in result.get("extracted_files", []):
            jobs.enqueue(job["tenant_id"], extracted)
    except Exception as exc:
        permanent = isinstance(exc, (ValueError, FileNotFoundError))
        # A long load that made progress before a transient outage (DB/Qdrant restart)
        # does not use up its attempts: it resumes where it stopped.
        attempts = 0 if (progressed["batches"] and not permanent) else job["attempts"]
        if not permanent and attempts < s.job_max_attempts:
            jobs.update(job_id, status="queued", attempts=attempts, error=f"{type(exc).__name__}: {exc}"[:4000])
        else:
            jobs.finish(job_id, "failed", f"{type(exc).__name__}: {exc}")
    finally:
        alive.set()


def main() -> None:
    s = get_settings()
    setup_logging(s.log_level, s.log_json)
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    init_db()
    for _ in range(60):
        try:
            vector_store.ensure_collection()
            break
        except Exception as exc:
            log.warning("qdrant not ready (%s), retrying", exc)
            time.sleep(2)
    models.load_models()
    os.makedirs(os.path.join(s.media_path, "uploads"), exist_ok=True)
    # Jobs left "running" by a crashed/restarted worker resume quickly (progress
    # updates refresh updated_at every batch, so live jobs of other workers are safe).
    recovered = jobs.requeue_stale(120, s.job_max_attempts)
    if recovered:
        log.info("re-queued %d interrupted job(s)", recovered)
    log.info("worker ready", extra={"media": s.media_path})

    last_scan = 0.0
    last_stale = 0.0
    while not stop.is_set():
        heartbeat()
        now = time.time()
        try:
            if s.auto_scan_seconds > 0 and now - last_scan >= s.auto_scan_seconds:
                last_scan = now
                scan_media()
            if now - last_stale >= 60:
                last_stale = now
                jobs.requeue_stale(s.job_stale_seconds, s.job_max_attempts)
            job = jobs.claim()
        except Exception as exc:  # database/Qdrant outage: back off and retry
            log.warning("worker loop error: %s", exc)
            stop.wait(5)
            continue
        if job is None:
            stop.wait(s.worker_poll_seconds)
            continue
        process(job)
    log.info("worker stopped")


if __name__ == "__main__":
    main()
