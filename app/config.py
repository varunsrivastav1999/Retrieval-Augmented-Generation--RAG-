"""Central configuration. Every tunable is read once from the environment here."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _str(name: str, default: str, *aliases: str) -> str:
    for key in (name, *aliases):
        raw = os.getenv(key)
        if raw not in (None, ""):
            return raw
    return default


def _database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    user = _str("POSTGRES_USER", "rag")
    password = _str("POSTGRES_PASSWORD", "rag")
    host = _str("POSTGRES_HOST", "postgres")
    port = _str("POSTGRES_PORT", "5432")
    db = _str("POSTGRES_DB", "rag")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


def _redis_url() -> str:
    url = os.getenv("REDIS_URL")
    if url:
        return url
    host = _str("REDIS_HOST", "redis")
    port = _str("REDIS_PORT", "6379")
    password = os.getenv("REDIS_PASSWORD", "")
    auth = f":{password}@" if password else ""
    return f"redis://{auth}{host}:{port}/0"


def _ollama_base() -> str:
    # Accept the legacy form http://ollama:11434/api/generate as well.
    url = _str("OLLAMA_BASE_URL", "http://ollama:11434", "OLLAMA_URL")
    return url.split("/api/")[0].rstrip("/")


@dataclass(frozen=True)
class Settings:
    env: str = field(default_factory=lambda: _str("RAG_ENV", "production").lower())
    log_level: str = field(default_factory=lambda: _str("RAG_LOG_LEVEL", "INFO").upper())
    log_json: bool = field(default_factory=lambda: _bool("RAG_LOG_JSON", True))

    # --- storage -----------------------------------------------------------
    database_url: str = field(default_factory=_database_url)
    db_pool_size: int = field(default_factory=lambda: _int("RAG_DB_POOL_SIZE", 10))
    redis_url: str = field(default_factory=_redis_url)
    qdrant_url: str = field(default_factory=lambda: _str("QDRANT_URL", "http://qdrant:6333"))
    qdrant_collection: str = field(default_factory=lambda: _str("RAG_QDRANT_COLLECTION", "rag_chunks"))
    qdrant_timeout: int = field(default_factory=lambda: _int("RAG_QDRANT_TIMEOUT", 10))
    media_path: str = field(default_factory=lambda: _str("MEDIA_PATH", "/media"))
    max_upload_mb: int = field(default_factory=lambda: _int("RAG_MAX_UPLOAD_MB", 512))

    # --- models (all served from the local cache baked into the image) -----
    model_cache: str = field(default_factory=lambda: _str("RAG_MODEL_CACHE", "/models"))
    embedding_model: str = field(default_factory=lambda: _str("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
    embedding_dim: int = field(default_factory=lambda: _int("RAG_EMBEDDING_DIM", 384))
    query_instruction: str = field(default_factory=lambda: _str(
        "RAG_QUERY_INSTRUCTION", "Represent this sentence for searching relevant passages: "))
    reranker_model: str = field(default_factory=lambda: _str("RAG_RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2"))
    # Full cross-encoder reranking (off: measured +0.3 pt for ~290 ms on CPU). The reranker
    # still scores the top RAG_RERANK_GATE_N hits for the "not in documents" decision.
    rerank_enabled: bool = field(default_factory=lambda: _bool("RAG_RERANK_ENABLED", False))
    rerank_gate_n: int = field(default_factory=lambda: _int("RAG_RERANK_GATE_N", 3))
    sparse_model: str = field(default_factory=lambda: _str("RAG_SPARSE_MODEL", "Qdrant/bm25"))
    model_threads: int = field(default_factory=lambda: _int("RAG_MODEL_THREADS", 0))  # 0 = onnxruntime default

    # --- retrieval ---------------------------------------------------------
    top_k: int = field(default_factory=lambda: _int("RAG_TOP_K", 6))
    max_top_k: int = field(default_factory=lambda: _int("RAG_MAX_TOP_K", 50))
    prefetch_limit: int = field(default_factory=lambda: _int("RAG_PREFETCH_LIMIT", 40))
    rerank_candidates: int = field(default_factory=lambda: _int("RAG_RERANK_CANDIDATES", 24))
    hnsw_ef: int = field(default_factory=lambda: _int("RAG_HNSW_EF", 96))
    # Below this reranker score (and with no exact identifier match) the
    # question is treated as "not covered by the documents".
    min_relevance: float = field(default_factory=lambda: _float("RAG_MIN_RELEVANCE", -4.0))
    # Above this score an answer is trusted even if some question words are not in the evidence.
    strong_relevance: float = field(default_factory=lambda: _float("RAG_STRONG_RELEVANCE", 3.0))
    section_expand_chars: int = field(default_factory=lambda: _int("RAG_SECTION_EXPAND_CHARS", 2000))
    # Evidence characters sent to the LLM. Measured on CPU (qwen2.5:3b): ~6 s to first
    # token at 3k chars, ~14 s at 6k. Raise on GPU hardware.
    context_max_chars: int = field(default_factory=lambda: _int("RAG_CONTEXT_MAX_CHARS", 3500))

    # --- LLM ---------------------------------------------------------------
    llm_backend: str = field(default_factory=lambda: _str("RAG_LLM_BACKEND", "ollama").lower())
    ollama_url: str = field(default_factory=_ollama_base)
    openai_base_url: str = field(default_factory=lambda: _str("OPENAI_BASE_URL", "", "VLLM_BASE_URL").rstrip("/"))
    openai_api_key: str = field(default_factory=lambda: _str("OPENAI_API_KEY", "EMPTY", "VLLM_API_KEY"))
    llm_model: str = field(default_factory=lambda: _str("RAG_LLM_MODEL", "qwen2.5:3b", "OLLAMA_MODEL"))
    llm_num_ctx: int = field(default_factory=lambda: _int("RAG_LLM_NUM_CTX", 4096))
    llm_max_tokens: int = field(default_factory=lambda: _int("RAG_LLM_MAX_TOKENS", 300))
    llm_timeout: int = field(default_factory=lambda: _int("RAG_LLM_TIMEOUT", 180))
    llm_keep_alive: str = field(default_factory=lambda: _str("RAG_LLM_KEEP_ALIVE", "24h"))
    llm_concurrency: int = field(default_factory=lambda: _int("RAG_LLM_CONCURRENCY", 1))
    # How long a question waits for a busy LLM before answering with evidence only.
    llm_queue_wait: float = field(default_factory=lambda: _float("RAG_LLM_QUEUE_WAIT_SECONDS", 20.0))
    llm_seed: int = field(default_factory=lambda: _int("RAG_LLM_SEED", 42))

    # --- cache -------------------------------------------------------------
    cache_enabled: bool = field(default_factory=lambda: _bool("RAG_CACHE_ENABLED", True))
    cache_ttl_seconds: int = field(default_factory=lambda: _int("RAG_CACHE_TTL_SECONDS", 86400))

    # --- ingestion worker --------------------------------------------------
    ingest_batch_size: int = field(default_factory=lambda: _int("RAG_INGEST_BATCH_SIZE", 256))
    chunk_chars: int = field(default_factory=lambda: _int("RAG_CHUNK_CHARS", 900))
    worker_poll_seconds: float = field(default_factory=lambda: _float("RAG_WORKER_POLL_SECONDS", 2.0))
    auto_scan_seconds: int = field(default_factory=lambda: _int("RAG_AUTO_SCAN_SECONDS", 60))
    job_stale_seconds: int = field(default_factory=lambda: _int("RAG_JOB_STALE_SECONDS", 300))
    job_max_attempts: int = field(default_factory=lambda: _int("RAG_JOB_MAX_ATTEMPTS", 3))
    ocr_enabled: bool = field(default_factory=lambda: _bool("RAG_OCR_ENABLED", True))
    use_docling: bool = field(default_factory=lambda: _bool("RAG_USE_DOCLING", False))

    # --- API ---------------------------------------------------------------
    api_key: str = field(default_factory=lambda: _str("RAG_API_KEY", ""))
    cors_origins: List[str] = field(default_factory=lambda: [
        o.strip() for o in _str("RAG_CORS_ORIGINS", "").split(",") if o.strip()])


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
