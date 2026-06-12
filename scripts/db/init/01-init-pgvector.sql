-- =============================================================================
-- i-Tips RAG — Database Initialization
-- =============================================================================
-- Runs once on first container start via docker-entrypoint-initdb.d
-- Creates extensions, indexes, and any required schema migrations.
-- =============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS uuid-ossp;

-- Create the rag schema (clean namespace)
CREATE SCHEMA IF NOT EXISTS rag;
SET search_path TO rag, public;

-- =============================================================================
-- Core tables are created by SQLAlchemy at app startup.
-- This script handles DB-level setup that SQLAlchemy cannot:
--   1. HNSW indexes (only raw SQL can set index parameters)
--   2. Full-text search indexes
--   3. Trigram indexes
--   4. Composite indexes for common query patterns
-- =============================================================================

-- HNSW index for vector similarity search (cosine distance)
-- m=16 (connections per node), ef_construction=64 (build quality)
-- Only created if the document_chunks table already has data
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.tables
    WHERE table_name = 'document_chunks' AND table_schema = 'public'
  ) THEN
    -- HNSW index for text embeddings
    IF NOT EXISTS (
      SELECT 1 FROM pg_indexes WHERE indexname = 'ix_document_chunks_embedding_hnsw'
    ) THEN
      CREATE INDEX ix_document_chunks_embedding_hnsw
        ON document_chunks USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    END IF;

    -- HNSW index for image embeddings
    IF NOT EXISTS (
      SELECT 1 FROM pg_indexes WHERE indexname = 'ix_document_chunks_image_embedding_hnsw'
    ) THEN
      CREATE INDEX ix_document_chunks_image_embedding_hnsw
        ON document_chunks USING hnsw (image_embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    END IF;

    -- Full-text search GIN index
    IF NOT EXISTS (
      SELECT 1 FROM pg_indexes WHERE indexname = 'ix_document_chunks_text_search'
    ) THEN
      CREATE INDEX ix_document_chunks_text_search
        ON document_chunks USING gin
        (to_tsvector('english', coalesce(text_content, '')));
    END IF;

    -- Trigram GIN index for fuzzy text matching
    IF NOT EXISTS (
      SELECT 1 FROM pg_indexes WHERE indexname = 'ix_document_chunks_text_trgm'
    ) THEN
      CREATE INDEX ix_document_chunks_text_trgm
        ON document_chunks USING gin (text_content gin_trgm_ops);
    END IF;

    -- Composite indexes for common query patterns
    IF NOT EXISTS (
      SELECT 1 FROM pg_indexes WHERE indexname = 'ix_document_chunks_tenant_model'
    ) THEN
      CREATE INDEX ix_document_chunks_tenant_model
        ON document_chunks (tenant_id, embedding_model);
    END IF;

    IF NOT EXISTS (
      SELECT 1 FROM pg_indexes WHERE indexname = 'ix_document_chunks_file_type'
    ) THEN
      CREATE INDEX ix_document_chunks_file_type
        ON document_chunks (file_type);
    END IF;

    IF NOT EXISTS (
      SELECT 1 FROM pg_indexes WHERE indexname = 'ix_document_chunks_parent_chunk'
    ) THEN
      CREATE INDEX ix_document_chunks_parent_chunk
        ON document_chunks (parent_chunk_id);
    END IF;

    -- Unique constraint for deduplication
    IF NOT EXISTS (
      SELECT 1 FROM pg_indexes WHERE indexname = 'uq_document_chunk_scope'
    ) THEN
      CREATE UNIQUE INDEX uq_document_chunk_scope
        ON document_chunks (tenant_id, doc_id, chunk_hash, embedding_model);
    END IF;
  END IF;
END $$;
