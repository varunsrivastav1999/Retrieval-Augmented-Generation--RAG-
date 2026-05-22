"""
=============================================================================
 i-Tips RAG: Background Ingestion Worker & Auto-Scanner
=============================================================================
 - Universal file worker (all formats, not just PDF)
 - Auto-scan /media for ALL supported file types
 - Background chunking starts automatically when files appear
 - Progress tracking per job
 - Stale job recovery
=============================================================================
"""

import os
import threading
import time
import uuid
import shutil
import zipfile
import tarfile
from glob import glob
from typing import Dict, List, Optional

from sqlalchemy import text

from app.database import IngestionJob, SessionLocal, utcnow
from app.rag.parsers import SUPPORTED_EXTENSIONS, is_supported_file


# ---------------------------------------------------------------------------
# Job CRUD
# ---------------------------------------------------------------------------
def create_ingestion_job(
    tenant_id: str,
    source_path: str,
    force_reindex: bool = False,
    file_type: Optional[str] = None,
) -> IngestionJob:
    """Create a new ingestion job for any file type."""
    db = SessionLocal()
    try:
        if file_type is None:
            from app.rag.parsers import get_file_type
            file_type = get_file_type(source_path)

        job = IngestionJob(
            id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            source_path=source_path,
            source_name=os.path.basename(source_path),
            status="queued",
            force_reindex=force_reindex,
            file_type=file_type,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return job
    finally:
        db.close()


def create_ingestion_jobs(tenant_id: str, source_paths: List[str], force_reindex: bool = False) -> List[IngestionJob]:
    return [create_ingestion_job(tenant_id, source_path, force_reindex) for source_path in source_paths]


def get_ingestion_job(job_id: str) -> Optional[IngestionJob]:
    db = SessionLocal()
    try:
        return db.get(IngestionJob, job_id)
    finally:
        db.close()


def mark_stale_running_jobs_queued(timeout_seconds: int) -> int:
    db = SessionLocal()
    try:
        result = db.execute(
            text(
                "UPDATE ingestion_jobs "
                "SET status = 'queued', updated_at = now() "
                "WHERE status = 'running' "
                "AND updated_at < now() - (:timeout_seconds * interval '1 second')"
            ),
            {"timeout_seconds": timeout_seconds},
        )
        db.commit()
        return result.rowcount or 0
    finally:
        db.close()


def claim_next_ingestion_job() -> Optional[Dict[str, str]]:
    db = SessionLocal()
    try:
        result = db.execute(
            text(
                "UPDATE ingestion_jobs "
                "SET status = 'running', attempts = attempts + 1, "
                "updated_at = now(), error = NULL "
                "WHERE id = ("
                "  SELECT id FROM ingestion_jobs "
                "  WHERE status IN ('queued', 'retry') "
                "  ORDER BY created_at ASC "
                "  LIMIT 1 "
                "  FOR UPDATE SKIP LOCKED"
                ") "
                "RETURNING id, tenant_id, source_path, force_reindex, file_type"
            ),
        )
        job = result.mappings().first()
        db.commit()
        return dict(job) if job else None
    finally:
        db.close()


def update_ingestion_job(job_id: str, **fields) -> None:
    db = SessionLocal()
    try:
        job = db.get(IngestionJob, job_id)
        if not job:
            return
        for key, value in fields.items():
            setattr(job, key, value)
        job.updated_at = utcnow()
        db.commit()
    finally:
        db.close()


def complete_ingestion_job(job_id: str, chunks_total: int, chunks_inserted: int) -> None:
    update_ingestion_job(
        job_id,
        status="succeeded",
        chunks_total=chunks_total,
        chunks_inserted=chunks_inserted,
        completed_at=utcnow(),
        progress_pct=100.0,
        error=None,
    )


def fail_ingestion_job(job_id: str, error: str) -> None:
    update_ingestion_job(
        job_id,
        status="failed",
        error=error[:4000],
        completed_at=utcnow(),
    )


# ---------------------------------------------------------------------------
# File Discovery — scan /media for ALL supported file types
# ---------------------------------------------------------------------------
def find_all_supported_files(media_path: str, include_scan: bool = True) -> List[str]:
    """
    Recursively scan media_path for ALL supported file types.
    Returns sorted list of absolute file paths.
    """
    if not include_scan or not os.path.isdir(media_path):
        return []

    found_files = set()
    for ext in SUPPORTED_EXTENSIONS:
        # Glob for each extension (case-insensitive by trying both)
        pattern_lower = os.path.join(media_path, f"**/*{ext}")
        pattern_upper = os.path.join(media_path, f"**/*{ext.upper()}")
        found_files.update(glob(pattern_lower, recursive=True))
        found_files.update(glob(pattern_upper, recursive=True))

    # Filter out hidden files and directories
    filtered = [
        f for f in found_files
        if os.path.isfile(f)
        and not os.path.basename(f).startswith(".")
        and is_supported_file(f)
    ]

    return sorted(filtered)


# ---------------------------------------------------------------------------
# Universal Ingestion Worker
# ---------------------------------------------------------------------------
def ingestion_worker_loop(stop_event: threading.Event, poll_seconds: float = 5.0) -> None:
    """Background worker that processes queued ingestion jobs for any file type."""
    from app.rag.ingestion import ingest_file

    while not stop_event.is_set():
        job = claim_next_ingestion_job()
        if not job:
            stop_event.wait(poll_seconds)
            continue

        try:
            # Check if this is an archive
            if job.get("file_type") == "archive":
                source_path = job["source_path"]
                tenant_id = job["tenant_id"]
                job_id = job["id"]
                
                extract_dir = os.path.join(os.path.dirname(source_path), f"extracted_{job_id}")
                os.makedirs(extract_dir, exist_ok=True)
                
                # Unpack archive
                if source_path.endswith(".zip"):
                    with zipfile.ZipFile(source_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                elif source_path.endswith((".tar", ".tar.gz", ".tgz")):
                    with tarfile.open(source_path, 'r:*') as tar_ref:
                        tar_ref.extractall(extract_dir)
                
                # Find all supported files inside
                extracted_files = find_all_supported_files(extract_dir, include_scan=True)
                
                # Create jobs for them
                if extracted_files:
                    create_ingestion_jobs(tenant_id, extracted_files, force_reindex=bool(job.get("force_reindex", False)))
                
                # Complete the archive job itself
                complete_ingestion_job(job_id, chunks_total=len(extracted_files), chunks_inserted=len(extracted_files))
                continue

            ingest_file(
                job["source_path"],
                tenant_id=job["tenant_id"],
                job_id=job["id"],
                force_reindex=bool(job.get("force_reindex", False)),
            )
        except Exception as exc:
            fail_ingestion_job(job["id"], str(exc))


def start_ingestion_worker(
    stop_event: threading.Event,
    poll_seconds: float = 5.0,
    stale_timeout_seconds: int = 1800,
) -> threading.Thread:
    """Start the background ingestion worker thread."""
    recovered = mark_stale_running_jobs_queued(stale_timeout_seconds)
    if recovered:
        print(f"[IngestWorker] Re-queued {recovered} stale ingestion jobs")

    thread = threading.Thread(
        target=ingestion_worker_loop,
        args=(stop_event, poll_seconds),
        name="rag-ingestion-worker",
        daemon=True,
    )
    thread.start()
    return thread
