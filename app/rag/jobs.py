"""Postgres-backed ingestion job queue (FOR UPDATE SKIP LOCKED; safe with many workers)."""

from __future__ import annotations

import os
import uuid
from typing import List, Optional

from sqlalchemy import select, text as sql

from app.db import IngestJob, SessionLocal, utcnow
from app.rag.parsers import get_file_type


def enqueue(tenant_id: str, path: str, force: bool = False) -> IngestJob:
    """Queue a file unless an identical job is already pending."""
    path = os.path.realpath(path)
    with SessionLocal() as db:
        existing = db.execute(select(IngestJob).where(
            IngestJob.tenant_id == tenant_id, IngestJob.source_path == path,
            IngestJob.status.in_(["queued", "running"]))).scalars().first()
        if existing:
            return existing
        job = IngestJob(id=str(uuid.uuid4()), tenant_id=tenant_id, source_path=path,
                        source_name=os.path.basename(path), file_type=get_file_type(path),
                        status="queued", force_reindex=force)
        db.add(job)
        db.commit()
        return job


def claim() -> Optional[dict]:
    with SessionLocal() as db:
        row = db.execute(sql("""
            UPDATE ingest_jobs SET status='running', attempts=attempts+1, updated_at=now(), error=NULL
            WHERE id = (SELECT id FROM ingest_jobs WHERE status='queued'
                        ORDER BY created_at LIMIT 1 FOR UPDATE SKIP LOCKED)
            RETURNING id, tenant_id, source_path, force_reindex, attempts""")).mappings().first()
        db.commit()
        return dict(row) if row else None


def update(job_id: str, **fields) -> None:
    with SessionLocal() as db:
        job = db.get(IngestJob, job_id)
        if job is None:
            return
        for k, v in fields.items():
            setattr(job, k, v)
        job.updated_at = utcnow()
        db.commit()


def finish(job_id: str, status: str, error: Optional[str] = None, **fields) -> None:
    update(job_id, status=status, error=(error or None) and error[:4000], completed_at=utcnow(), **fields)


def requeue_stale(timeout_seconds: int, max_attempts: int) -> int:
    with SessionLocal() as db:
        res = db.execute(sql("""
            UPDATE ingest_jobs SET status = CASE WHEN attempts >= :max THEN 'failed' ELSE 'queued' END,
                   error = CASE WHEN attempts >= :max THEN 'worker stopped while processing (max attempts reached)' ELSE error END,
                   updated_at = now()
            WHERE status='running' AND updated_at < now() - (:t * interval '1 second')"""),
            {"t": timeout_seconds, "max": max_attempts})
        db.commit()
        return res.rowcount or 0


def recent(tenant_id: str, limit: int = 50) -> List[IngestJob]:
    with SessionLocal() as db:
        return list(db.execute(select(IngestJob).where(IngestJob.tenant_id == tenant_id)
                               .order_by(IngestJob.created_at.desc()).limit(limit)).scalars())


def queue_depth() -> dict:
    with SessionLocal() as db:
        rows = db.execute(sql("SELECT status, count(*) FROM ingest_jobs WHERE status IN ('queued','running') GROUP BY status")).all()
    return {r[0]: r[1] for r in rows}
