import os
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    Boolean,
    DateTime,
    Float,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rag_user:rag_password@postgres:5432/rag_db")

EMBEDDING_DIM = int(os.getenv("RAG_EMBEDDING_DIM", "1024"))
LEGACY_EMBEDDING_MODEL = os.getenv(
    "RAG_LEGACY_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "doc_id",
            "chunk_hash",
            "embedding_model",
            name="uq_document_chunk_scope",
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(String, default="default", nullable=False, index=True)
    doc_id = Column(String, index=True)
    chunk_hash = Column(String, index=True)
    text_content = Column(Text)
    section = Column(Integer)
    doc_metadata = Column(JSON)
    embedding_model = Column(String, default=LEGACY_EMBEDDING_MODEL, nullable=False, index=True)
    created_at = Column(DateTime, default=utcnow, nullable=False)
    # --- NEW columns for 17-layer architecture ---
    file_type = Column(String, default="pdf", nullable=True, index=True)
    parent_chunk_id = Column(Integer, nullable=True, index=True)
    confidence_score = Column(Float, nullable=True)
    # Note: Vectors are now stored in Qdrant. These columns only hold textual metadata.
    # --- NEW columns for Multi-modal / Vision ---
    quantized_embedding = Column(Text, nullable=True)
    # --- NEW columns for RAPTOR ---
    raptor_level = Column(Integer, default=0, nullable=False, index=True)


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id = Column(String, primary_key=True)
    tenant_id = Column(String, default="default", nullable=False, index=True)
    source_path = Column(Text, nullable=False)
    source_name = Column(String, nullable=False)
    status = Column(String, default="queued", nullable=False, index=True)
    attempts = Column(Integer, default=0, nullable=False)
    chunks_total = Column(Integer, default=0, nullable=False)
    chunks_inserted = Column(Integer, default=0, nullable=False)
    error = Column(Text)
    force_reindex = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=utcnow, nullable=False)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)
    completed_at = Column(DateTime)
    # --- NEW columns ---
    file_type = Column(String, default="pdf", nullable=True)
    progress_pct = Column(Float, default=0.0, nullable=True)

import sqlalchemy.exc

def init_db():
    # Skip pgvector extensions because we use Qdrant for vectors
    pass
            
    try:
        Base.metadata.create_all(bind=engine)
    except sqlalchemy.exc.IntegrityError as e:
        print(f"[DB] IntegrityError during create_all (ignored, likely multi-worker race condition): {e}")
    except sqlalchemy.exc.ProgrammingError as e:
        print(f"[DB] ProgrammingError during create_all (ignored, likely multi-worker race condition): {e}")

    _run_schema_migrations()


def _execute_best_effort(conn, statement: str):
    try:
        conn.execute(text(statement))
        conn.commit()
    except Exception as exc:
        conn.rollback()
        print(f"[DB] Skipped migration/index statement: {exc}")


def _run_schema_migrations():
    # SQLite schema updates should be done via alembic or fresh db
    # We rely on Base.metadata.create_all for initial setup
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
