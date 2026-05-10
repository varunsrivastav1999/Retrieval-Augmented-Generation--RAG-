import hashlib
import math
import os
import re
from glob import glob
from functools import lru_cache
from typing import Any, Iterable, List, Sequence


EMBEDDING_DIM = int(os.getenv("RAG_EMBEDDING_DIM", "384"))
RAG_ENV = os.getenv("RAG_ENV", "local").lower()
EMBEDDING_MODEL = os.getenv(
    "RAG_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
RERANKER_MODEL = os.getenv(
    "RAG_RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)
MODEL_CACHE_DIR = os.getenv("SENTENCE_TRANSFORMERS_HOME") or os.getenv("HF_HOME")
HF_OFFLINE = os.getenv("RAG_HF_OFFLINE", "false").lower() in {"1", "true", "yes", "on"}
ALLOW_HASH_FALLBACK = os.getenv("RAG_ALLOW_HASH_FALLBACK", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
ALLOW_RERANKER_FALLBACK = os.getenv(
    "RAG_ALLOW_RERANKER_FALLBACK",
    str(ALLOW_HASH_FALLBACK).lower(),
).lower() in {"1", "true", "yes", "on"}
REQUIRE_REAL_MODELS = os.getenv(
    "RAG_REQUIRE_REAL_MODELS",
    "true" if RAG_ENV in {"prod", "production"} else "false",
).lower() in {"1", "true", "yes", "on"}

if HF_OFFLINE:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


class HashingEmbedder:
    """Deterministic local fallback that preserves the 384-d pgvector shape."""

    def __init__(self, dimension: int = EMBEDDING_DIM):
        self.dimension = dimension

    def encode(self, sentences: Any, **_: Any) -> Any:
        is_single = isinstance(sentences, str)
        texts = [sentences] if is_single else list(sentences)
        vectors = [self._encode_one(text) for text in texts]
        return vectors[0] if is_single else vectors

    def _encode_one(self, text: str) -> List[float]:
        tokens = _TOKEN_RE.findall((text or "").lower())
        if not tokens:
            tokens = ["_empty_"]

        features = list(tokens)
        features.extend(f"{left} {right}" for left, right in zip(tokens, tokens[1:]))

        vector = [0.0] * self.dimension
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            weight = 2.0 if " " not in feature else 1.0
            vector[index] += sign * weight

        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector


class LexicalReranker:
    """Cheap local reranker for offline local development."""

    def predict(self, pairs: Sequence[Sequence[str]], **_: Any) -> List[float]:
        return [self._score(query, text) for query, text in pairs]

    def _score(self, query: str, text: str) -> float:
        query_terms = _TOKEN_RE.findall((query or "").lower())
        text_terms = _TOKEN_RE.findall((text or "").lower())
        if not query_terms or not text_terms:
            return 0.0

        text_set = set(text_terms)
        overlap = sum(1 for term in query_terms if term in text_set)
        coverage = overlap / len(query_terms)
        density = overlap / max(len(text_terms), 1)
        phrase_bonus = 0.2 if query.lower() in text.lower() else 0.0
        return coverage + density + phrase_bonus


@lru_cache(maxsize=1)
def get_optimal_device() -> str:
    """
    Returns the best available hardware accelerator in priority order:
    1. MPS (Apple Silicon) - Best for Mac local
    2. CUDA (NVIDIA) - Best for Production/Linux
    3. CPU (Fallback) - Default
    """
    device_override = os.getenv("RAG_MODEL_DEVICE")
    if device_override:
        return device_override

    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except (ImportError, Exception) as e:
        print(f"[RAG Hardware] Torch check failed: {e}. Defaulting to CPU.")

    return "cpu"


def _model_kwargs() -> dict:
    kwargs = {}
    if MODEL_CACHE_DIR:
        kwargs["cache_folder"] = MODEL_CACHE_DIR
    
    device = get_optimal_device()
    kwargs["device"] = device
    
    # Prominent status banner for Docker logs
    print("\n" + "="*50)
    print(f"  RAG HARDWARE STATUS: {device.upper()}")
    print(f"  MODE: {'🚀 GPU ACCELERATED' if device in ['mps', 'cuda'] else '🐢 CPU ONLY'}")
    print("="*50 + "\n")
    
    return kwargs


def _fallback_embedding_id() -> str:
    return f"fallback:hashing:{EMBEDDING_DIM}:v1"


def _fallback_reranker_id() -> str:
    return "fallback:lexical:v1"


def _model_appears_cached(model_name: str) -> bool:
    if not MODEL_CACHE_DIR:
        return False

    org, _, name = model_name.partition("/")
    hf_safe_name = f"models--{org}--{name}" if name else f"models--{model_name}"
    candidates = [
        os.path.join(MODEL_CACHE_DIR, model_name),
        os.path.join(MODEL_CACHE_DIR, hf_safe_name),
        os.path.join(MODEL_CACHE_DIR, "hub", hf_safe_name),
        os.path.join(os.path.dirname(MODEL_CACHE_DIR), "hub", hf_safe_name),
    ]
    for candidate in candidates:
        if glob(os.path.join(candidate, "snapshots", "*")):
            return True
        if os.path.exists(os.path.join(candidate, "config.json")):
            return True
        if os.path.exists(os.path.join(candidate, "modules.json")):
            return True
    return False


@lru_cache(maxsize=1)
def get_embedding_model() -> Any:
    if HF_OFFLINE and ALLOW_HASH_FALLBACK and not _model_appears_cached(EMBEDDING_MODEL):
        print(
            "[RAG Embeddings] Using deterministic local hashing embeddings "
            f"because {EMBEDDING_MODEL!r} is not in the offline cache."
        )
        return HashingEmbedder()

    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(EMBEDDING_MODEL, **_model_kwargs())
    except Exception as exc:
        if not ALLOW_HASH_FALLBACK:
            raise RuntimeError(
                "Embedding model is unavailable. Check network/DNS or pre-populate "
                "the Hugging Face cache, or enable RAG_ALLOW_HASH_FALLBACK=true "
                "for local development."
            ) from exc

        print(
            "[RAG Embeddings] Falling back to deterministic local hashing "
            f"embeddings because {EMBEDDING_MODEL!r} could not be loaded: {exc}"
        )
        return HashingEmbedder()


def get_embedding_model_id() -> str:
    model = get_embedding_model()
    if isinstance(model, HashingEmbedder):
        return _fallback_embedding_id()
    return EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_reranker_model() -> Any:
    if HF_OFFLINE and ALLOW_RERANKER_FALLBACK and not _model_appears_cached(RERANKER_MODEL):
        print(
            "[RAG Reranker] Using lexical reranking because "
            f"{RERANKER_MODEL!r} is not in the offline cache."
        )
        return LexicalReranker()

    try:
        from sentence_transformers import CrossEncoder

        kwargs = {}
        tokenizer_args = {}
        automodel_args = {}
        if MODEL_CACHE_DIR:
            tokenizer_args["cache_dir"] = MODEL_CACHE_DIR
            automodel_args["cache_dir"] = MODEL_CACHE_DIR
        kwargs["device"] = get_optimal_device()
        if HF_OFFLINE:
            tokenizer_args["local_files_only"] = True
            automodel_args["local_files_only"] = True
        if tokenizer_args:
            kwargs["tokenizer_args"] = tokenizer_args
        if automodel_args:
            kwargs["automodel_args"] = automodel_args
        return CrossEncoder(RERANKER_MODEL, **kwargs)
    except Exception as exc:
        if not ALLOW_RERANKER_FALLBACK:
            raise RuntimeError(
                "Reranker model is unavailable. Check network/DNS or pre-populate "
                "the Hugging Face cache, or enable RAG_ALLOW_RERANKER_FALLBACK=true "
                "for local development."
            ) from exc

        print(
            "[RAG Reranker] Falling back to lexical reranking because "
            f"{RERANKER_MODEL!r} could not be loaded: {exc}"
        )
        return LexicalReranker()


def get_reranker_model_id() -> str:
    model = get_reranker_model()
    if isinstance(model, LexicalReranker):
        return _fallback_reranker_id()
    return RERANKER_MODEL


def validate_runtime_models() -> None:
    embedding_id = get_embedding_model_id()
    reranker_id = get_reranker_model_id()
    if REQUIRE_REAL_MODELS and (
        embedding_id.startswith("fallback:") or reranker_id.startswith("fallback:")
    ):
        raise RuntimeError(
            "Production requires real embedding and reranker models. Disable fallback "
            "or pre-populate the model cache before starting the service."
        )


def runtime_model_info() -> dict:
    embedding_id = get_embedding_model_id()
    reranker_id = get_reranker_model_id()
    return {
        "environment": RAG_ENV,
        "device": get_optimal_device(),
        "embedding_model": embedding_id,
        "reranker_model": reranker_id,
        "embedding_dimension": EMBEDDING_DIM,
        "hf_offline": HF_OFFLINE,
        "using_embedding_fallback": embedding_id.startswith("fallback:"),
        "using_reranker_fallback": reranker_id.startswith("fallback:"),
        "requires_real_models": REQUIRE_REAL_MODELS,
    }


def _as_vector(value: Any) -> List[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]


def encode_text(text: str) -> List[float]:
    return _as_vector(get_embedding_model().encode(text))


def encode_texts(texts: Iterable[str]) -> List[List[float]]:
    text_list = list(texts)
    if not text_list:
        return []

    encoded = get_embedding_model().encode(text_list)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    return [_as_vector(vector) for vector in encoded]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)
