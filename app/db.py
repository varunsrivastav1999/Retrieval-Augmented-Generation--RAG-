"""PostgreSQL schema and session handling (system of record for documents/chunks).

Vectors live in Qdrant; Postgres holds the text, document registry, job queue,
conversation history and the query audit log. Tables are new (documents,
chunks, ingest_jobs, ...) so a legacy database is never altered destructively.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from sqlalchemy import (
    BigInteger, Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, Sequence,
    String, Text, UniqueConstraint, create_engine, text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from app.config import get_settings

log = logging.getLogger(__name__)
Base = declarative_base()


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("tenant_id", "path", name="uq_documents_tenant_path"),
        Index("ix_documents_tenant_status", "tenant_id", "status"),
        Index("ix_documents_tenant_collection", "tenant_id", "collection"),
    )
    id = Column(String(36), primary_key=True)
    tenant_id = Column(String(80), nullable=False, default="default")
    path = Column(Text, nullable=False)
    name = Column(String(512), nullable=False)
    collection = Column(String(128), nullable=False, default="general")
    file_type = Column(String(32))
    sha256 = Column(String(64))
    size_bytes = Column(BigInteger, default=0)
    mtime = Column(Float)
    pages = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    status = Column(String(16), nullable=False, default="pending")  # pending|indexing|indexed|failed
    error = Column(Text)
    created_at = Column(DateTime, nullable=False, default=utcnow)
    updated_at = Column(DateTime, nullable=False, default=utcnow, onupdate=utcnow)
    indexed_at = Column(DateTime)


chunk_id_seq = Sequence("chunks_id_seq", metadata=Base.metadata)


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        Index("ix_chunks_doc_ord", "document_id", "ord"),
        Index("ix_chunks_section_ord", "section_id", "ord", postgresql_where=text("section_id IS NOT NULL")),
        Index("ix_chunks_identifiers", "identifiers", postgresql_using="gin"),
    )
    id = Column(BigInteger, chunk_id_seq, primary_key=True)  # == Qdrant point id
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    tenant_id = Column(String(80), nullable=False)
    ord = Column(Integer, nullable=False)
    page = Column(Integer)
    section_id = Column(String(64))
    section_title = Column(Text)
    content_type = Column(String(24), nullable=False, default="text")
    text = Column(Text, nullable=False)
    identifiers = Column(ARRAY(Text), nullable=False, default=list)


class IngestJob(Base):
    __tablename__ = "ingest_jobs"
    __table_args__ = (Index("ix_ingest_jobs_status_created", "status", "created_at"),
                      Index("ix_ingest_jobs_tenant_created", "tenant_id", "created_at"))
    id = Column(String(36), primary_key=True)
    tenant_id = Column(String(80), nullable=False, default="default")
    document_id = Column(String(36))
    source_path = Column(Text, nullable=False)
    source_name = Column(String(512), nullable=False)
    file_type = Column(String(32))
    status = Column(String(16), nullable=False, default="queued")  # queued|running|succeeded|failed|skipped
    attempts = Column(Integer, nullable=False, default=0)
    force_reindex = Column(Boolean, nullable=False, default=False)
    chunks_total = Column(Integer, nullable=False, default=0)
    progress_pct = Column(Float, nullable=False, default=0.0)
    error = Column(Text)
    created_at = Column(DateTime, nullable=False, default=utcnow)
    updated_at = Column(DateTime, nullable=False, default=utcnow, onupdate=utcnow)
    completed_at = Column(DateTime)


class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = (Index("ix_conversations_tenant_updated", "tenant_id", "updated_at"),)
    id = Column(String(36), primary_key=True)
    tenant_id = Column(String(80), nullable=False, default="default")
    title = Column(String(200), nullable=False, default="New conversation")
    created_at = Column(DateTime, nullable=False, default=utcnow)
    updated_at = Column(DateTime, nullable=False, default=utcnow, onupdate=utcnow)


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (Index("ix_messages_conversation", "conversation_id", "id"),)
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id = Column(String(36), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(16), nullable=False)  # user|assistant
    content = Column(Text, nullable=False)
    payload = Column(JSONB)  # sources, verification, timings
    created_at = Column(DateTime, nullable=False, default=utcnow)


class QueryLog(Base):
    __tablename__ = "query_log"
    __table_args__ = (Index("ix_query_log_ts", "ts"),)
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    ts = Column(DateTime, nullable=False, default=utcnow)
    tenant_id = Column(String(80), nullable=False)
    query = Column(Text, nullable=False)
    intent = Column(String(24))
    mode = Column(String(16))
    status = Column(String(16))  # answered|not_found|error
    cache_hit = Column(Boolean, default=False)
    latency_ms = Column(Integer)
    retrieval_ms = Column(Integer)
    llm_ms = Column(Integer)
    top_score = Column(Float)
    n_sources = Column(Integer)


_settings = get_settings()
engine = create_engine(
    _settings.database_url,
    pool_size=_settings.db_pool_size,
    max_overflow=10,
    pool_pre_ping=True,      # transparently replaces connections after a DB restart
    pool_recycle=1800,
    connect_args={"connect_timeout": 5, "application_name": "industrial-rag"},
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db(retries: int = 30, delay: float = 2.0) -> None:
    """Create tables/indexes idempotently, waiting for the database if needed."""
    for attempt in range(1, retries + 1):
        try:
            with engine.begin() as conn:
                conn.execute(text("SELECT pg_advisory_xact_lock(724001)"))  # one migrator at a time
                Base.metadata.create_all(bind=conn)
            log.info("database schema ready")
            return
        except Exception as exc:  # pragma: no cover - depends on infra timing
            if attempt == retries:
                raise
            log.warning("database not ready (attempt %d/%d): %s", attempt, retries, exc)
            time.sleep(delay)


def db_ping() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
