-- =============================================================================
-- Enterprise Level RAG — PostgreSQL Performance Tuning
-- =============================================================================
-- These settings optimize pgvector for low-latency HNSW queries.
-- Applied at the session/connection level so existing clusters are unaffected.
-- =============================================================================

-- Effective cache size (assumes 4GB+ RAM for PostgreSQL container)
ALTER SYSTEM SET effective_cache_size = '2GB';

-- Parallel query workers for vector search
ALTER SYSTEM SET max_parallel_workers_per_gather = 2;
ALTER SYSTEM SET parallel_tuple_cost = 0.1;
ALTER SYSTEM SET parallel_setup_cost = 100;

-- Work memory for sorting/indexing operations
ALTER SYSTEM SET work_mem = '64MB';
ALTER SYSTEM SET maintenance_work_mem = '256MB';

-- Shared buffers (25% of expected RAM)
ALTER SYSTEM SET shared_buffers = '1GB';

-- WAL for reliability without killing performance
ALTER SYSTEM SET wal_buffers = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;

-- Autovacuum tuning for vector tables
ALTER SYSTEM SET autovacuum_max_workers = 3;
ALTER SYSTEM SET autovacuum_naptime = '60s';
ALTER SYSTEM SET autovacuum_vacuum_threshold = 1000;
ALTER SYSTEM SET autovacuum_analyze_threshold = 500;

-- Load configuration
SELECT pg_reload_conf();
