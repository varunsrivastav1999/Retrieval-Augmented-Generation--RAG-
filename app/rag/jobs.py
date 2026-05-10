import os
import threading
import time
import uuid
from typing import Dict, List, Optional

from sqlalchemy import text

from app.database import IngestionJob, SessionLocal, utcnow


def create_ingestion_job(tenant_id: str, source_path: str, force_reindex: bool = False) -> IngestionJob:
    db = SessionLocal()
    try:
        job = IngestionJob(
            id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            source_path=source_path,
            source_name=os.path.basename(source_path),
            status="queued",
            force_reindex=force_reindex,
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
                "RETURNING id, tenant_id, source_path, force_reindex"
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
        error=None,
    )


def fail_ingestion_job(job_id: str, error: str) -> None:
    update_ingestion_job(
        job_id,
        status="failed",
        error=error[:4000],
        completed_at=utcnow(),
    )


def ingestion_worker_loop(stop_event: threading.Event, poll_seconds: float = 5.0) -> None:
    from app.rag.ingestion import ingest_pdf

    while not stop_event.is_set():
        job = claim_next_ingestion_job()
        if not job:
            stop_event.wait(poll_seconds)
            continue

        try:
            ingest_pdf(
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
