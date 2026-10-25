"""Redis answer cache keyed on the exact normalised question + filters + index version.

Exact (not "semantic") matching on purpose: for maintenance, "SRVO-050" and
"SRVO-056" must never share an answer. The index version is bumped by the
ingestion worker, so any re-index invalidates cached answers instantly.
Redis being down only disables caching; it never fails a request.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Optional

import redis

from app.config import get_settings

log = logging.getLogger(__name__)
_client: Optional[redis.Redis] = None


def client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.Redis.from_url(get_settings().redis_url, decode_responses=True,
                                       socket_timeout=0.5, socket_connect_timeout=0.5, health_check_interval=30)
    return _client


def ping() -> bool:
    try:
        return bool(client().ping())
    except Exception:
        return False


def index_version(tenant_id: str) -> str:
    try:
        return client().get(f"rag:index_version:{tenant_id}") or "0"
    except Exception:
        return "nocache"


def bump_index_version(tenant_id: str) -> None:
    try:
        client().incr(f"rag:index_version:{tenant_id}")
    except Exception as exc:
        log.warning("could not bump index version: %s", exc)


def make_key(tenant_id: str, query: str, **scope) -> Optional[str]:
    version = index_version(tenant_id)
    if version == "nocache":
        return None
    s = get_settings()
    norm = re.sub(r"\s+", " ", query.strip().lower())
    payload = json.dumps({"q": norm, "t": tenant_id, "v": version, "m": [s.embedding_model, s.reranker_model, s.llm_model],
                          **scope}, sort_keys=True)
    return "rag:answer:" + hashlib.sha256(payload.encode()).hexdigest()


def get(key: Optional[str]) -> Optional[dict]:
    if not key or not get_settings().cache_enabled:
        return None
    try:
        raw = client().get(key)
        return json.loads(raw) if raw else None
    except Exception:
        return None


def put(key: Optional[str], value: dict) -> None:
    if not key or not get_settings().cache_enabled:
        return
    try:
        client().setex(key, get_settings().cache_ttl_seconds, json.dumps(value, default=str))
    except Exception:
        pass
