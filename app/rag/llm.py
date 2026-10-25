"""Local LLM client (Ollama by default, any OpenAI-compatible server optional).

Generation is deterministic (temperature 0, fixed seed) and concurrency is
bounded so a CPU-only box is never asked to run more generations than it can.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Iterator, Optional

import httpx

from app.config import get_settings

log = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    pass


class LLMBusy(RuntimeError):
    pass


_settings = get_settings()
_sem = threading.BoundedSemaphore(max(1, _settings.llm_concurrency))
_http: Optional[httpx.Client] = None
_health_cache = {"ts": 0.0, "value": None}


def _client() -> httpx.Client:
    global _http
    if _http is None:
        s = get_settings()
        _http = httpx.Client(timeout=httpx.Timeout(s.llm_timeout, connect=3.0))
    return _http


def _is_openai() -> bool:
    s = get_settings()
    return s.llm_backend in {"openai", "vllm", "openai-compatible"} and bool(s.openai_base_url)


def stream_chat(system: str, prompt: str, wait_seconds: Optional[float] = None,
                max_tokens: Optional[int] = None) -> Iterator[str]:
    """Yield answer tokens. Raises LLMBusy if the generation slot is not freed in time."""
    if not _sem.acquire(timeout=get_settings().llm_queue_wait if wait_seconds is None else wait_seconds):
        raise LLMBusy("LLM is busy with other requests")
    try:
        yield from (_stream_openai(system, prompt, max_tokens) if _is_openai() else _stream_ollama(system, prompt, max_tokens))
    finally:
        _sem.release()


def _stream_ollama(system: str, prompt: str, max_tokens: Optional[int] = None) -> Iterator[str]:
    s = get_settings()
    body = {
        "model": s.llm_model,
        "stream": True,
        "keep_alive": s.llm_keep_alive,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "options": {"temperature": 0, "top_k": 1, "top_p": 1, "seed": s.llm_seed,
                    "num_ctx": s.llm_num_ctx, "num_predict": max_tokens or s.llm_max_tokens},
    }
    try:
        with _client().stream("POST", f"{s.ollama_url}/api/chat", json=body) as resp:
            if resp.status_code != 200:
                detail = resp.read().decode("utf-8", "replace")[:300]
                raise LLMUnavailable(f"Ollama returned {resp.status_code}: {detail}")
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if data.get("error"):
                    raise LLMUnavailable(data["error"])
                token = (data.get("message") or {}).get("content", "")
                if token:
                    yield token
                if data.get("done"):
                    break
    except httpx.HTTPError as exc:
        raise LLMUnavailable(f"Ollama unreachable: {exc}") from exc


def _stream_openai(system: str, prompt: str, max_tokens: Optional[int] = None) -> Iterator[str]:
    s = get_settings()
    body = {
        "model": s.llm_model, "stream": True, "temperature": 0, "seed": s.llm_seed,
        "max_tokens": max_tokens or s.llm_max_tokens,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
    }
    headers = {"Authorization": f"Bearer {s.openai_api_key}"}
    try:
        with _client().stream("POST", f"{s.openai_base_url}/chat/completions", json=body, headers=headers) as resp:
            if resp.status_code != 200:
                raise LLMUnavailable(f"LLM server returned {resp.status_code}")
            for line in resp.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                choice = (json.loads(payload).get("choices") or [{}])[0]
                token = (choice.get("delta") or {}).get("content")
                if token:
                    yield token
    except httpx.HTTPError as exc:
        raise LLMUnavailable(f"LLM server unreachable: {exc}") from exc


def health(max_age: float = 5.0) -> dict:
    """{'status': 'ok'|'model_missing'|'down', 'model': ...} (cached briefly)."""
    now = time.time()
    if _health_cache["value"] and now - _health_cache["ts"] < max_age:
        return _health_cache["value"]
    s = get_settings()
    value = {"status": "down", "model": s.llm_model, "backend": "openai" if _is_openai() else "ollama"}
    try:
        if _is_openai():
            r = _client().get(f"{s.openai_base_url}/models", timeout=3.0)
            value["status"] = "ok" if r.status_code == 200 else "down"
        else:
            r = _client().get(f"{s.ollama_url}/api/tags", timeout=3.0)
            r.raise_for_status()
            names = {m.get("name") for m in r.json().get("models", [])}
            wanted = {s.llm_model, f"{s.llm_model}:latest"}
            value["status"] = "ok" if names & wanted else "model_missing"
    except Exception as exc:
        value["error"] = str(exc)[:200]
    _health_cache.update(ts=now, value=value)
    return value


def warmup() -> bool:
    """Load the model into memory so the first user question does not pay for it."""
    if _is_openai():
        return health()["status"] == "ok"
    s = get_settings()
    try:
        r = _client().post(f"{s.ollama_url}/api/generate", json={
            "model": s.llm_model, "prompt": "", "keep_alive": s.llm_keep_alive,
            "options": {"num_ctx": s.llm_num_ctx}}, timeout=600.0)
        return r.status_code == 200
    except Exception as exc:
        log.warning("llm warmup failed: %s", exc)
        return False
