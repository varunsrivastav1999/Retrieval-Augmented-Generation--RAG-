"""Hybrid retrieval: dense + BM25 + exact identifiers -> weighted RRF -> rerank.

Deterministic for a given index state: no LLM calls, no randomness.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

from sqlalchemy import text as sql

from app.config import get_settings
from app.db import SessionLocal
from app.rag import models, vector_store
from app.rag.identifiers import extract_identifiers, normalize_identifier

log = logging.getLogger(__name__)

RRF_K = 60
WEIGHTS = {"dense": 1.0, "sparse": 1.0, "ids": 2.0}
# Identifiers shorter than this (e.g. robot "R07") are too generic to be used
# as a hard filter, but still count towards the exact-match bonus.
STRICT_ID_MIN_LEN = 4
ID_BONUS = 4.0  # added to the reranker logit when all strict query identifiers are present


@dataclass
class Hit:
    id: int
    document_id: str
    document: str
    collection: str
    file_type: str
    page: Optional[int]
    section_id: Optional[str]
    section_title: Optional[str]
    content_type: str
    ord: int
    text: str
    identifiers: List[str]
    fused: float = 0.0
    rerank: Optional[float] = None
    score: float = 0.0
    matched_ids: List[str] = field(default_factory=list)
    signals: List[str] = field(default_factory=list)

    def public(self) -> dict:
        d = asdict(self)
        d.pop("identifiers", None)
        return d


@dataclass
class Retrieval:
    hits: List[Hit]
    query_ids: List[str]
    missing_ids: List[str]           # strict identifiers that exist nowhere in the index
    timings: Dict[str, float]
    top_score: Optional[float]       # best cross-encoder relevance among the top hits (None = no reranker)


def _fetch_rows(ids: Sequence[int]) -> Dict[int, dict]:
    if not ids:
        return {}
    q = sql("""
        SELECT c.id, c.document_id, c.page, c.section_id, c.section_title, c.content_type, c.ord,
               c.text, c.identifiers, d.name, d.collection, d.file_type
        FROM chunks c JOIN documents d ON d.id = c.document_id
        WHERE c.id = ANY(:ids)""")
    with SessionLocal() as db:
        return {r.id: r._asdict() for r in db.execute(q, {"ids": list(ids)})}


def _ids_present(tenant_id: str, ids: Sequence[str]) -> set:
    if not ids:
        return set()
    q = sql("""SELECT DISTINCT unnest(identifiers) AS i FROM
               (SELECT identifiers FROM chunks WHERE tenant_id = :t AND identifiers && :ids LIMIT 200) s""")
    with SessionLocal() as db:
        return {r.i for r in db.execute(q, {"t": tenant_id, "ids": list(ids)})} & set(ids)


def retrieve(
    query: str,
    tenant_id: str = "default",
    *,
    top_k: Optional[int] = None,
    collections: Sequence[str] = (),
    file_types: Sequence[str] = (),
    document_ids: Sequence[str] = (),
    extra_ids: Sequence[str] = (),
    rerank: bool = True,
) -> Retrieval:
    s = get_settings()
    top_k = min(top_k or s.top_k, s.max_top_k)
    t0 = time.perf_counter()

    # Codes are often typed in lower case ("srvo-050", "p-500"): also scan the upper-cased question.
    found = [*extract_identifiers(query), *extract_identifiers(query.upper()), *extra_ids]
    query_ids = list(dict.fromkeys(normalize_identifier(t) for t in found))
    strict_ids = [i for i in query_ids if len(i) >= STRICT_ID_MIN_LEN]

    dense = models.embed_query(query)
    sparse = models.sparse_query(query)
    t_embed = time.perf_counter()

    lists = vector_store.hybrid_search(
        dense, sparse, tenant_id, identifiers=strict_ids, collections=collections,
        file_types=file_types, document_ids=document_ids, limit=s.prefetch_limit)
    t_search = time.perf_counter()

    fused: Dict[int, float] = defaultdict(float)
    signals: Dict[int, List[str]] = defaultdict(list)
    for name, ranked in lists.items():
        for rank, (pid, _score) in enumerate(ranked, 1):
            fused[pid] += WEIGHTS[name.split(":")[0]] / (RRF_K + rank)
            signals[pid].append(name)
    # Fetch more candidates than needed so exact-identifier bonuses can reorder them.
    pool_size = max(top_k, s.rerank_candidates)
    pool = sorted(fused, key=fused.get, reverse=True)[:pool_size]
    rows = _fetch_rows(pool)
    t_fetch = time.perf_counter()

    hits: List[Hit] = []
    qset = set(query_ids)
    for pid in pool:
        r = rows.get(pid)
        if r is None:  # vector without a row (deleted document) -> ignore
            continue
        ident = list(r["identifiers"] or [])
        hits.append(Hit(
            id=pid, document_id=r["document_id"], document=r["name"], collection=r["collection"],
            file_type=r["file_type"] or "", page=r["page"], section_id=r["section_id"],
            section_title=r["section_title"], content_type=r["content_type"], ord=r["ord"], text=r["text"],
            identifiers=ident, fused=fused[pid], matched_ids=sorted(qset.intersection(ident)), signals=signals[pid]))

    # Ranking: weighted RRF + exact-identifier bonus. The cross-encoder is used for
    # full reranking only when enabled (benchmarks: +0.3 pt accuracy for ~290 ms on
    # CPU), but always scores the top few hits to decide "not in the documents".
    fused_scale = 100.0
    strict = set(strict_ids)
    for h in hits:
        h.score = h.fused * fused_scale
        if strict:
            h.score += ID_BONUS * len(strict.intersection(h.matched_ids)) / len(strict)
    hits.sort(key=lambda h: (-h.score, h.id))
    full = rerank and s.rerank_enabled
    scored = hits if full else hits[: s.rerank_gate_n]
    rr_scores = models.rerank(query, [f"{h.section_title or ''}\n{h.text}".strip() for h in scored]) if scored else None
    t_rerank = time.perf_counter()
    if rr_scores is not None:
        for h, sc in zip(scored, rr_scores):
            h.rerank = round(sc, 4)
        if full:
            for h in hits:
                h.score = h.rerank + (ID_BONUS * len(strict.intersection(h.matched_ids)) / len(strict) if strict else 0.0)
            hits.sort(key=lambda h: (-h.score, h.id))
    for h in hits:
        h.score = round(h.score, 4)
    relevance = max((h.rerank for h in scored if h.rerank is not None), default=None) if rr_scores is not None else None
    hits = hits[:top_k]

    found = {i for h in hits for i in h.matched_ids}
    unresolved = [i for i in strict_ids if i not in found]
    missing = [i for i in unresolved if i not in _ids_present(tenant_id, unresolved)] if unresolved else []
    t_end = time.perf_counter()

    timings = {
        "embed_ms": (t_embed - t0) * 1000, "search_ms": (t_search - t_embed) * 1000,
        "fetch_ms": (t_fetch - t_search) * 1000, "rerank_ms": (t_rerank - t_fetch) * 1000,
        "total_ms": (t_end - t0) * 1000,
    }
    return Retrieval(hits=hits, query_ids=query_ids, missing_ids=missing,
                     timings={k: round(v, 1) for k, v in timings.items()},
                     top_score=relevance)


def expand_sections(hits: Sequence[Hit], max_chars: int, max_sections: int = 3) -> Dict[int, str]:
    """For the best hits, return the surrounding section text (ordered chunks).

    Gives the answer step complete procedures / alarm entries instead of a
    single fragment. Bounded by `max_chars` per section.
    """
    targets = []
    seen = set()
    for h in hits:
        if h.section_id and h.section_id not in seen and h.content_type == "text":
            seen.add(h.section_id)
            targets.append(h)
        if len(targets) >= max_sections:
            break
    if not targets:
        return {}
    q = sql("""SELECT section_id, ord, text FROM chunks
               WHERE section_id = ANY(:sids) ORDER BY section_id, ord LIMIT 400""")
    with SessionLocal() as db:
        rows = db.execute(q, {"sids": [h.section_id for h in targets]}).all()
    by_section: Dict[str, List] = defaultdict(list)
    for r in rows:
        by_section[r.section_id].append(r)
    out: Dict[int, str] = {}
    for h in targets:
        parts = by_section.get(h.section_id, [])
        # Grow outward from the hit so the matching chunk is always included.
        ords = [p.ord for p in parts]
        if h.ord not in ords:
            continue
        idx = ords.index(h.ord)
        chosen = {idx}
        total = len(parts[idx].text)
        lo, hi = idx - 1, idx + 1
        while (lo >= 0 or hi < len(parts)) and total < max_chars:
            for j in (hi, lo):
                if 0 <= j < len(parts) and j not in chosen and total + len(parts[j].text) <= max_chars:
                    chosen.add(j)
                    total += len(parts[j].text)
            lo, hi = lo - 1, hi + 1
        out[h.id] = "\n".join(parts[j].text for j in sorted(chosen))
    return out
