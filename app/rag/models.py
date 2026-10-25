"""Local ONNX models (fastembed): dense embedder, BM25 sparse encoder, reranker.

All models are baked into the Docker image at build time and loaded from the
local cache only (HF_HUB_OFFLINE=1), so nothing is downloaded at runtime.
ONNX Runtime keeps PyTorch out of the API image and is faster on CPU.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import List, Optional, Sequence

import numpy as np

from app.config import get_settings

log = logging.getLogger(__name__)
_lock = threading.Lock()
_dense = None
_sparse = None
_reranker = None
_load_error: Optional[str] = None


def _kwargs() -> dict:
    s = get_settings()
    kw = {"cache_dir": os.path.join(s.model_cache, "fastembed")}
    if s.model_threads > 0:
        kw["threads"] = s.model_threads
    if os.getenv("HF_HUB_OFFLINE", "0") == "1":
        kw["local_files_only"] = True
    return kw


def load_models() -> None:
    """Load all models (idempotent, thread-safe)."""
    global _dense, _sparse, _reranker, _load_error
    with _lock:
        if _dense is not None:
            return
        s = get_settings()
        started = time.perf_counter()
        try:
            import onnxruntime
            onnxruntime.disable_telemetry_events()  # offline sites: never collect/send telemetry
            from fastembed import SparseTextEmbedding, TextEmbedding
            dense = TextEmbedding(s.embedding_model, **_kwargs())
            sparse = SparseTextEmbedding(s.sparse_model, **_kwargs())
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            reranker = TextCrossEncoder(s.reranker_model, **_kwargs())
            dim = len(next(iter(dense.embed(["warmup"]))))
            if dim != s.embedding_dim:
                raise RuntimeError(f"embedding dim {dim} != RAG_EMBEDDING_DIM {s.embedding_dim}")
            _dense, _sparse, _reranker, _load_error = dense, sparse, reranker, None
            log.info("models loaded", extra={"seconds": round(time.perf_counter() - started, 2),
                                             "embedding": s.embedding_model, "reranker": s.reranker_model if reranker else None})
        except Exception as exc:
            _load_error = f"{type(exc).__name__}: {exc}"
            log.exception("model loading failed")
            raise


def models_ready() -> bool:
    return _dense is not None


def load_error() -> Optional[str]:
    return _load_error


def _require():
    if _dense is None:
        load_models()


def embed_query(text: str) -> List[float]:
    _require()
    s = get_settings()
    vec = next(iter(_dense.embed([s.query_instruction + text])))
    return vec.tolist()


def embed_documents(texts: Sequence[str], batch_size: int = 64) -> List[List[float]]:
    _require()
    return [v.tolist() for v in _dense.embed(list(texts), batch_size=batch_size)]


def sparse_query(text: str):
    _require()
    return next(iter(_sparse.query_embed(text)))


def sparse_documents(texts: Sequence[str], batch_size: int = 256):
    _require()
    return list(_sparse.embed(list(texts), batch_size=batch_size))


def rerank(query: str, docs: Sequence[str]) -> Optional[List[float]]:
    """Cross-encoder relevance scores (logits), or None when disabled."""
    _require()
    if _reranker is None or not docs:
        return None
    return [float(x) for x in _reranker.rerank(query, list(docs), batch_size=32)]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    va, vb = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(va @ vb / denom) if denom else 0.0


def model_info() -> dict:
    s = get_settings()
    return {
        "embedding_model": s.embedding_model,
        "embedding_dim": s.embedding_dim,
        "sparse_model": s.sparse_model,
        "reranker_model": s.reranker_model,
        "full_rerank": s.rerank_enabled,
        "loaded": models_ready(),
        "error": _load_error,
    }
