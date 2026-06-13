import os
from datetime import datetime

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
from pgvector.sqlalchemy import Vector

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rag_user:rag_password@postgres:5432/rag_db")
EMBEDDING_DIM = int(os.getenv("RAG_EMBEDDING_DIM", "1024"))
RAG_USE_HALFVEC = os.getenv("RAG_USE_HALFVEC", "false").lower() in {"1", "true", "yes", "on"}
try:
    from pgvector.sqlalchemy import Halfvec as _Halfvec
    VECTOR_TYPE = _Halfvec if RAG_USE_HALFVEC else Vector
except ImportError:
    VECTOR_TYPE = Vector
LEGACY_EMBEDDING_MODEL = os.getenv(
    "RAG_LEGACY_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

engine = create_engine(DATABASE_URL, pool_size=50, max_overflow=100, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def utcnow():
    return datetime.utcnow()

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
    embedding = Column(VECTOR_TYPE(EMBEDDING_DIM))
    created_at = Column(DateTime, default=utcnow, nullable=False)
    # --- NEW columns for 12-layer architecture ---
    file_type = Column(String, default="pdf", nullable=True, index=True)
    parent_chunk_id = Column(Integer, nullable=True, index=True)
    confidence_score = Column(Float, nullable=True)
    # --- NEW columns for Multi-modal / Vision ---
    image_embedding = Column(Vector(768), nullable=True)
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
    with engine.connect() as conn:
        try:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"[DB] Skipped extension creation: {e}")
            
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
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS tenant_id varchar"))
        conn.execute(text("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding_model varchar"))
        conn.execute(text("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS created_at timestamp"))
        conn.execute(text("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS file_type varchar"))
        conn.execute(text("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS raptor_level integer DEFAULT 0"))
        _execute_best_effort(
            conn,
            "ALTER TABLE document_chunks ADD COLUMN parent_chunk_id INTEGER"
        )
        _execute_best_effort(
            conn,
            "ALTER TABLE document_chunks ADD COLUMN confidence_score FLOAT"
        )
        _execute_best_effort(
            conn,
            "ALTER TABLE document_chunks ADD COLUMN image_embedding vector(512)"
        )
        _execute_best_effort(
            conn,
            "ALTER TABLE document_chunks ADD COLUMN quantized_embedding text"
        )
        _execute_best_effort(
            conn,
            "CREATE INDEX ON document_chunks USING hnsw (image_embedding vector_cosine_ops)"
        )
        conn.execute(text("ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS force_reindex boolean"))
        conn.execute(text("ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS file_type varchar"))
        conn.execute(text("ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS progress_pct float"))
        conn.execute(text("UPDATE ingestion_jobs SET force_reindex = false WHERE force_reindex IS NULL"))
        conn.execute(text("ALTER TABLE ingestion_jobs ALTER COLUMN force_reindex SET NOT NULL"))
        conn.execute(text("UPDATE document_chunks SET tenant_id = 'default' WHERE tenant_id IS NULL"))
        conn.execute(
            text(
                "UPDATE document_chunks "
                "SET embedding_model = :model "
                "WHERE embedding_model IS NULL"
            ),
            {"model": LEGACY_EMBEDDING_MODEL},
        )
        conn.execute(text("UPDATE document_chunks SET created_at = now() WHERE created_at IS NULL"))
        conn.execute(text("UPDATE document_chunks SET file_type = 'pdf' WHERE file_type IS NULL"))
        conn.execute(text("ALTER TABLE document_chunks ALTER COLUMN tenant_id SET NOT NULL"))
        conn.execute(text("ALTER TABLE document_chunks ALTER COLUMN embedding_model SET NOT NULL"))
        conn.execute(text("ALTER TABLE document_chunks ALTER COLUMN created_at SET NOT NULL"))
        conn.execute(text("ALTER TABLE document_chunks DROP CONSTRAINT IF EXISTS document_chunks_chunk_hash_key"))
        conn.execute(text("DROP INDEX IF EXISTS ix_document_chunks_chunk_hash"))
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_document_chunks_chunk_hash "
                "ON document_chunks (chunk_hash)"
            )
        )
        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_document_chunk_scope "
                "ON document_chunks (tenant_id, doc_id, chunk_hash, embedding_model)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_document_chunks_tenant_model "
                "ON document_chunks (tenant_id, embedding_model)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_document_chunks_file_type "
                "ON document_chunks (file_type)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_document_chunks_parent_chunk "
                "ON document_chunks (parent_chunk_id)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_ingestion_jobs_status_created_at "
                "ON ingestion_jobs (status, created_at)"
            )
        )
        conn.commit()

        # Drop the old 'simple' dictionary index if it exists, then recreate
        # with 'english' to match the retrieval queries and enable stemming.
        _execute_best_effort(
            conn,
            "DROP INDEX IF EXISTS ix_document_chunks_text_search",
        )
        _execute_best_effort(
            conn,
            "CREATE INDEX IF NOT EXISTS ix_document_chunks_text_search "
            "ON document_chunks USING gin "
            "(to_tsvector('english', coalesce(text_content, '')))",
        )
        _execute_best_effort(
            conn,
            "CREATE INDEX IF NOT EXISTS ix_document_chunks_text_trgm "
            "ON document_chunks USING gin (text_content gin_trgm_ops)",
        )
        _execute_best_effort(
            conn,
            "CREATE INDEX IF NOT EXISTS ix_document_chunks_embedding_hnsw "
            "ON document_chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)",
        )

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
