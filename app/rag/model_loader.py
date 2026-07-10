import hashlib
import json
import math
import os
import re
import threading
from glob import glob
from functools import lru_cache
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Optional

import requests


EMBEDDING_DIM = int(os.getenv("RAG_EMBEDDING_DIM", "1024"))
RAG_ENV = os.getenv("RAG_ENV", "local").lower()
EMBEDDING_MODEL = os.getenv(
    "RAG_EMBEDDING_MODEL",
    "BAAI/bge-small-en-v1.5",
)
RERANKER_MODEL = os.getenv(
    "RAG_RERANKER_MODEL",
    "BAAI/bge-reranker-base",
)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
LLM_BACKEND = os.getenv("RAG_LLM_BACKEND", os.getenv("LLM_BACKEND", "ollama")).lower()
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", os.getenv("OPENAI_API_KEY", "EMPTY"))
MODEL_CACHE_DIR = os.getenv("HF_HOME")
ST_CACHE_DIR = os.getenv("SENTENCE_TRANSFORMERS_HOME")
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

RAG_EMBEDDING_QUANTIZE = os.getenv("RAG_EMBEDDING_QUANTIZE", "none").lower()
RAG_USE_HALFVEC = os.getenv("RAG_USE_HALFVEC", "false").lower() in {"1", "true", "yes", "on"}

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
        
    def rerank(self, query: str, documents: List[str], k: int=10) -> List[dict]:
        scores = self.predict([(query, d) for d in documents])
        results = [{"content": doc, "score": score, "rank": i+1} for i, (doc, score) in enumerate(zip(documents, scores))]
        return sorted(results, key=lambda x: x["score"], reverse=True)[:k]

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

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            print(f"[RAG Hardware] PyTorch CUDA is False. PyTorch built for CUDA: {torch.version.cuda}")
            import subprocess
            try:
                subprocess.check_output(["nvidia-smi"])
                print("[RAG Hardware] nvidia-smi works inside container, but PyTorch cannot use it. (Likely an NVIDIA Driver version mismatch or missing CUDA libs).")
            except Exception as e:
                print(f"[RAG Hardware] nvidia-smi missing inside container. The GPU is NOT passed through to Docker! Ensure you run start.sh and not docker compose directly.")
    except Exception as e:
        print(f"[RAG Hardware] Torch check failed: {e}. Defaulting to CPU.")

    return "cpu"


def _model_kwargs() -> dict:
    kwargs = {}
    # Do not manually pass cache_folder, let sentence-transformers use SENTENCE_TRANSFORMERS_HOME
    
    device = get_optimal_device()
    kwargs["device"] = device
    
    # Enhanced diagnostics for Mac/Docker users
    is_docker = os.path.exists("/.dockerenv")
    
    print("\n" + "="*60)
    print(f"  RAG HARDWARE STATUS: {device.upper()}")
    
    if device == "cpu" and is_docker:
        print("  MODE: 🐢 CPU ONLY")
        print("  NOTE: If on Mac, Docker cannot access your GPU (MPS). Run natively to use it.")
        print("        If on Linux, ensure NVIDIA drivers are passed properly and match PyTorch CUDA version.")
    elif device in ["mps", "cuda"]:
        print(f"  MODE: 🚀 GPU ACCELERATED ({device.upper()})")
    else:
        print("  MODE: 🐢 CPU ONLY")
        
    print("="*60 + "\n")
    
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
    st_safe_name = model_name.replace("/", "_")
    candidates = [
        os.path.join(MODEL_CACHE_DIR, "hub", hf_safe_name) if MODEL_CACHE_DIR else "",
        os.path.join(ST_CACHE_DIR, st_safe_name) if ST_CACHE_DIR else "",
        os.path.join(MODEL_CACHE_DIR, model_name) if MODEL_CACHE_DIR else "",
    ]
    candidates = [c for c in candidates if c]
    for candidate in candidates:
        if glob(os.path.join(candidate, "snapshots", "*")):
            return True
        if os.path.exists(os.path.join(candidate, "config.json")):
            return True
        if os.path.exists(os.path.join(candidate, "modules.json")):
            return True
    return False


def get_ollama_generate_url() -> str:
    base_url = os.getenv("OLLAMA_URL", "http://ollama:11434/api/generate")
    return base_url


def use_openai_compatible_llm() -> bool:
    configured_url = VLLM_BASE_URL or os.getenv("OLLAMA_URL", "")
    return (
        LLM_BACKEND in {"vllm", "openai", "openai-compatible", "openai_compatible"}
        or bool(VLLM_BASE_URL)
        or "/v1/" in configured_url
        or configured_url.rstrip("/").endswith("/v1")
    )


def get_openai_completion_url() -> str:
    base_url = VLLM_BASE_URL or os.getenv("OLLAMA_URL", "http://vllm:8000/v1")
    base_url = base_url.rstrip("/")
    if base_url.endswith("/completions") or base_url.endswith("/chat/completions"):
        return base_url
    if base_url.endswith("/v1"):
        return f"{base_url}/completions"
    return f"{base_url}/v1/completions"


def get_openai_models_url() -> str:
    completion_url = get_openai_completion_url()
    if completion_url.endswith("/chat/completions"):
        return completion_url[: -len("/chat/completions")] + "/models"
    if completion_url.endswith("/completions"):
        return completion_url[: -len("/completions")] + "/models"
    return completion_url.rstrip("/") + "/models"


def use_chat_completions() -> bool:
    return get_openai_completion_url().endswith("/chat/completions")


def build_llm_payload(
    prompt: str,
    *,
    system: str = "",
    stream: bool = False,
    num_predict: Optional[int] = None,
    temperature: float = 0.0,
    num_ctx: Optional[int] = None,
    extra_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    max_tokens = num_predict or int(os.getenv("OLLAMA_NUM_PREDICT", "1024"))
    if use_openai_compatible_llm():
        if use_chat_completions():
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            payload = {
                "model": OLLAMA_MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": stream,
            }
        else:
            full_prompt = f"{system}\n\n{prompt}" if system else prompt
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": full_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": stream,
            }
        if extra_options:
            payload.update({k: v for k, v in extra_options.items() if k not in {"num_ctx", "num_gpu"}})
        return payload

    options = {
        "num_predict": max_tokens,
        "temperature": temperature,
    }
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    if extra_options:
        options.update(extra_options)
    return {
        "model": OLLAMA_MODEL,
        "system": system,
        "prompt": prompt,
        "stream": stream,
        "options": options,
    }


def post_llm_generate(
    prompt: str,
    *,
    system: str = "",
    stream: bool = False,
    timeout: Optional[float] = None,
    num_predict: Optional[int] = None,
    temperature: float = 0.0,
    num_ctx: Optional[int] = None,
    extra_options: Optional[Dict[str, Any]] = None,
):
    payload = build_llm_payload(
        prompt,
        system=system,
        stream=stream,
        num_predict=num_predict,
        temperature=temperature,
        num_ctx=num_ctx,
        extra_options=extra_options,
    )
    if use_openai_compatible_llm():
        headers = {"Content-Type": "application/json"}
        if VLLM_API_KEY:
            headers["Authorization"] = f"Bearer {VLLM_API_KEY}"
        return requests.post(
            get_openai_completion_url(),
            json=payload,
            headers=headers,
            stream=stream,
            timeout=timeout,
        )
    return requests.post(
        get_ollama_generate_url(),
        json=payload,
        stream=stream,
        timeout=timeout,
    )


def extract_llm_text(response) -> str:
    data = response.json()
    if "response" in data:
        return str(data.get("response") or "").strip()
    choices = data.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    if "text" in choice:
        return str(choice.get("text") or "").strip()
    message = choice.get("message") or {}
    return str(message.get("content") or "").strip()


def iter_llm_tokens(response) -> Iterator[str]:
    for line in response.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8").strip()
        if not decoded:
            continue
        if decoded.startswith("data:"):
            decoded = decoded[len("data:"):].strip()
        if decoded == "[DONE]":
            break
        try:
            data = json.loads(decoded)
        except json.JSONDecodeError:
            continue

        if "response" in data:
            token = data.get("response", "")
            if token:
                yield token
            continue

        choices = data.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        token = choice.get("text")
        if token is None:
            delta = choice.get("delta") or choice.get("message") or {}
            token = delta.get("content")
        if token:
            yield token


def generate_llm_text(
    prompt: str,
    *,
    system: str = "",
    timeout: Optional[float] = None,
    num_predict: Optional[int] = None,
    temperature: float = 0.0,
    num_ctx: Optional[int] = None,
    extra_options: Optional[Dict[str, Any]] = None,
) -> str:
    response = post_llm_generate(
        prompt,
        system=system,
        stream=False,
        timeout=timeout,
        num_predict=num_predict,
        temperature=temperature,
        num_ctx=num_ctx,
        extra_options=extra_options,
    )
    if response.status_code == 200:
        return extract_llm_text(response)
    print(f"[LLM] Generation failed status={response.status_code}: {response.text[:500]}")
    return ""

_embedding_model = None
_embedding_model_lock = threading.Lock()

def clear_embedding_model_cache():
    global _embedding_model
    with _embedding_model_lock:
        _embedding_model = None

def get_embedding_model() -> Any:
    global _embedding_model
    with _embedding_model_lock:
        if _embedding_model is not None:
            return _embedding_model

        if HF_OFFLINE and ALLOW_HASH_FALLBACK and not _model_appears_cached(EMBEDDING_MODEL):
            print(
                "[RAG Embeddings] Using deterministic local hashing embeddings "
                f"because {EMBEDDING_MODEL!r} is not in the offline cache."
            )
            _embedding_model = HashingEmbedder()
            return _embedding_model

        try:
            from sentence_transformers import SentenceTransformer

            _embedding_model = SentenceTransformer(EMBEDDING_MODEL, **_model_kwargs())
            return _embedding_model
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
            _embedding_model = HashingEmbedder()
            return _embedding_model


def get_embedding_model_id() -> str:
    """Return the model ID matching the currently active embedding model.
    If using HashingEmbedder fallback, return the fallback ID so vectors
    are correctly labeled and never mixed with real embeddings."""
    model = get_embedding_model()
    if isinstance(model, HashingEmbedder):
        return _fallback_embedding_id()
    return EMBEDDING_MODEL


_reranker_model = None
_reranker_model_lock = threading.Lock()

def clear_reranker_model_cache():
    global _reranker_model
    with _reranker_model_lock:
        _reranker_model = None

def get_reranker_model() -> Any:
    global _reranker_model
    with _reranker_model_lock:
        if _reranker_model is not None:
            return _reranker_model

        if HF_OFFLINE and ALLOW_RERANKER_FALLBACK and not _model_appears_cached(RERANKER_MODEL):
            print(
                "[RAG Reranker] Using lexical reranking because "
                f"{RERANKER_MODEL!r} is not in the offline cache."
            )
            _reranker_model = LexicalReranker()
            return _reranker_model

        try:
            if "colbert" in RERANKER_MODEL.lower():
                from ragatouille import RAGPretrainedModel
                print(f"[RAG Reranker] Loading ColBERT Late-Interaction model: {RERANKER_MODEL}")
                _reranker_model = RAGPretrainedModel.from_pretrained(RERANKER_MODEL)
                return _reranker_model
            else:
                from sentence_transformers import CrossEncoder

                kwargs = {}
                tokenizer_args = {}
                automodel_args = {}
                kwargs["device"] = get_optimal_device()
                if HF_OFFLINE:
                    tokenizer_args["local_files_only"] = True
                    automodel_args["local_files_only"] = True
                if tokenizer_args:
                    kwargs["tokenizer_args"] = tokenizer_args
                if automodel_args:
                    kwargs["automodel_args"] = automodel_args
                _reranker_model = CrossEncoder(RERANKER_MODEL, **kwargs)
                return _reranker_model
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
            _reranker_model = LexicalReranker()
            return _reranker_model


def get_reranker_model_id() -> str:
    """Return the model ID matching the currently active reranker model."""
    model = get_reranker_model()
    if isinstance(model, LexicalReranker):
        return _fallback_reranker_id()
    return RERANKER_MODEL


def validate_runtime_models() -> None:
    if not REQUIRE_REAL_MODELS:
        return
    model = get_embedding_model()
    reranker = get_reranker_model()
    if isinstance(model, HashingEmbedder) or isinstance(reranker, LexicalReranker):
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
        "quantization": RAG_EMBEDDING_QUANTIZE,
        "use_halfvec": RAG_USE_HALFVEC,
    }


def _as_vector(value: Any) -> List[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]


def encode_text(text: str, quantize: bool = False) -> List[float]:
    vec = _as_vector(get_embedding_model().encode(text))
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
        
    if quantize and RAG_EMBEDDING_QUANTIZE == "int8":
        from app.rag.quantization import quantize_embedding, dequantize_embedding
        q_bytes, scale, zp = quantize_embedding(vec)
        vec = dequantize_embedding(q_bytes, scale, zp, len(vec))
    return vec


def encode_texts(texts: Iterable[str], quantize: bool = False) -> List[List[float]]:
    text_list = list(texts)
    if not text_list:
        return []

    # Hard limit the attention batch matrices to protect VRAM constraints
    encoded = get_embedding_model().encode(text_list, batch_size=16)
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
        
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    vecs = [_as_vector(vector) for vector in encoded]
    if quantize and RAG_EMBEDDING_QUANTIZE == "int8" and vecs:
        from app.rag.quantization import quantize_vector_batch, dequantize_embedding
        q_bytes, scale, zp = quantize_vector_batch(vecs)
        vecs = [dequantize_embedding(qb, scale, zp, len(v)) for qb, v in zip(q_bytes, vecs)]
    return vecs


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)





if __name__ == "__main__":
    import os
    print("\n" + "="*60)
    print("  RAG MODEL PRE-LOADER (FOR DOCKER & OFFLINE USE)")
    print("="*60)
    
    # Force online mode for the download phase
    os.environ["RAG_HF_OFFLINE"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    
    # We import inside main to avoid circular dependency issues if any
    import sys
    _mod = sys.modules.get("app.rag.model_loader") or sys.modules.get(__name__) or __import__(__name__)
    get_embedding_model = _mod.get_embedding_model
    get_reranker_model = _mod.get_reranker_model
    HashingEmbedder = _mod.HashingEmbedder
    LexicalReranker = _mod.LexicalReranker
    
    print("\n1. Downloading Embedding Model...")
    try:
        clear_embedding_model_cache()
        embedder = get_embedding_model()
        if isinstance(embedder, HashingEmbedder):
            print("❌ ERROR: Download failed, fell back to Hashing.")
        else:
            print("✅ SUCCESS: Embedding model is ready.")
    except Exception as e:
        print(f"❌ ERROR: {e}")

    print("\n2. Downloading Reranker Model...")
    try:
        clear_reranker_model_cache()
        reranker = get_reranker_model()
        if isinstance(reranker, LexicalReranker):
            print("❌ ERROR: Download failed, fell back to Lexical.")
        else:
            print("✅ SUCCESS: Reranker model is ready.")
    except Exception as e:
        print(f"❌ ERROR: {e}")

    print("\n" + "="*60)
    print("  PRE-LOADING COMPLETE. YOU CAN NOW START IN OFFLINE MODE.")
    print("="*60 + "\n")
