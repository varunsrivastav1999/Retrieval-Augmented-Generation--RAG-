"""Prometheus metrics (scraped at /metrics)."""

from prometheus_client import Counter, Gauge, Histogram

QUERY_SECONDS = Histogram(
    "rag_query_seconds", "End-to-end query latency", ["mode", "status"],
    buckets=(0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2.5, 5, 10, 20, 40, 80))
STAGE_SECONDS = Histogram(
    "rag_stage_seconds", "Latency per pipeline stage", ["stage"],
    buckets=(0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 1, 2, 5, 10, 30))
QUERIES = Counter("rag_queries_total", "Queries by outcome", ["status", "intent"])
CACHE_HITS = Counter("rag_cache_hits_total", "Answer cache hits")
LLM_ERRORS = Counter("rag_llm_errors_total", "LLM failures (fallback to evidence answer)", ["kind"])
LOW_CONFIDENCE = Counter("rag_low_confidence_answers_total", "Answers whose verification confidence was low")
INDEXED_CHUNKS = Gauge("rag_indexed_chunks", "Chunks in the vector index (approximate)")
