"""Query pipeline: intent -> cache -> hybrid retrieval -> evidence -> LLM -> verification.

`run()` yields events; the streaming endpoint forwards them as SSE and the JSON
endpoint folds them into one response, so both share a single code path.

Latency design: evidence (documented passages with citations) is produced
without the LLM and emitted first (typically well under one second on CPU);
LLM synthesis, when enabled, streams afterwards and is verified sentence by
sentence against the evidence.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

from sqlalchemy import select

from app.config import get_settings
from app.db import Conversation, Message, QueryLog, SessionLocal, utcnow
from app.metrics import CACHE_HITS, LLM_ERRORS, LOW_CONFIDENCE, QUERIES, QUERY_SECONDS, STAGE_SECONDS
from app.rag import cache, llm, relations, search
from app.rag.answer import (NOT_FOUND, NOT_FOUND_TEXT, SUMMARY_SYSTEM_PROMPT, build_sources, build_summary_prompt,
                            evidence_markdown, exact_answer, exact_markdown, uncovered_terms, verify_answer)
from app.rag.identifiers import extract_identifiers, normalize_identifier
from app.rag.intent import INTENTS, detect_intent, is_greeting

log = logging.getLogger(__name__)
MODES = {"auto", "evidence", "answer"}


@dataclass
class QueryOptions:
    query: str
    tenant_id: str = "default"
    mode: str = "auto"                 # auto: evidence + LLM if available; evidence: no LLM; answer: LLM required
    top_k: Optional[int] = None
    collections: List[str] = field(default_factory=list)
    file_types: List[str] = field(default_factory=list)
    document_ids: List[str] = field(default_factory=list)
    conversation_id: Optional[str] = None
    use_cache: bool = True


def _followup_ids(conversation_id: Optional[str], query: str) -> List[str]:
    """Carry identifiers from the previous question into short follow-ups ("how do I reset it?")."""
    if not conversation_id or extract_identifiers(query) or len(query.split()) > 14:
        return []
    with SessionLocal() as db:
        prev = db.execute(select(Message.content).where(
            Message.conversation_id == conversation_id, Message.role == "user"
        ).order_by(Message.id.desc()).limit(1)).scalar()
    return [normalize_identifier(i) for i in extract_identifiers(prev or "")][:3]


def _persist(opts: QueryOptions, result: dict) -> None:
    try:
        with SessionLocal() as db:
            t = result.get("timings", {})
            db.add(QueryLog(
                tenant_id=opts.tenant_id, query=opts.query[:4000], intent=result.get("intent"), mode=opts.mode,
                status=result.get("status"), cache_hit=bool(result.get("cache_hit")),
                latency_ms=int(t.get("total_ms", 0)), retrieval_ms=int(t.get("retrieval_ms", 0)),
                llm_ms=int(t.get("llm_ms", 0)), top_score=result.get("top_score"),
                n_sources=len(result.get("sources", []))))
            if opts.conversation_id:
                conv = db.get(Conversation, opts.conversation_id)
                if conv is None:
                    conv = Conversation(id=opts.conversation_id, tenant_id=opts.tenant_id, title=opts.query[:120])
                    db.add(conv)
                elif conv.title == "New conversation":
                    conv.title = opts.query[:120]
                conv.updated_at = utcnow()
                db.add(Message(conversation_id=conv.id, role="user", content=opts.query))
                db.add(Message(conversation_id=conv.id, role="assistant", content=result.get("answer", ""), payload={
                    k: result.get(k) for k in ("status", "intent", "sources", "verification", "timings", "related",
                                               "evidence", "exact", "exact_markdown", "missing_identifiers", "mode_used")}))
            db.commit()
    except Exception as exc:  # audit must never break answering
        log.warning("could not persist query: %s", exc)


def run(opts: QueryOptions) -> Iterator[dict]:
    s = get_settings()
    t0 = time.perf_counter()
    mode = opts.mode if opts.mode in MODES else "auto"
    query = opts.query.strip()

    if is_greeting(query):
        result = {"status": "answered", "intent": "greeting", "answer": "Hello. Ask about an alarm code, tag, "
                  "fault, procedure or past maintenance case and I will answer from the indexed documents.",
                  "sources": [], "timings": {"total_ms": 0.0}, "verification": None, "mode_used": "evidence"}
        _persist(opts, result)
        yield {"type": "done", **result}
        return

    intent = detect_intent(query)
    scope = {"mode": mode, "k": opts.top_k, "c": sorted(opts.collections), "f": sorted(opts.file_types),
             "d": sorted(opts.document_ids)}
    extra_ids = _followup_ids(opts.conversation_id, query)
    key = cache.make_key(opts.tenant_id, query + " " + " ".join(extra_ids), **scope) if opts.use_cache else None
    cached = cache.get(key)
    if cached:
        CACHE_HITS.inc()
        cached["cache_hit"] = True
        cached["timings"] = {**cached.get("timings", {}), "total_ms": round((time.perf_counter() - t0) * 1000, 1)}
        yield {"type": "retrieval", **{k: cached.get(k) for k in ("intent", "sources", "evidence", "exact", "exact_markdown",
                                                                    "related", "missing_identifiers", "status")}}
        _persist(opts, cached)
        yield {"type": "done", **cached}
        return

    # ---------------- retrieval ----------------
    try:
        retrieval = search.retrieve(
            query, opts.tenant_id, top_k=opts.top_k or intent.top_k, collections=opts.collections,
            file_types=opts.file_types, document_ids=opts.document_ids, extra_ids=extra_ids)
    except Exception as exc:
        log.exception("retrieval failed")
        QUERIES.labels("error", intent.name).inc()
        message = f"Search service unavailable: {type(exc).__name__}"
        yield {"type": "error", "status": "error", "message": message}
        _persist(opts, {"status": "error", "intent": intent.name, "answer": message,
                        "timings": {"total_ms": round((time.perf_counter() - t0) * 1000, 1)}})
        return
    for stage in ("embed_ms", "search_ms", "fetch_ms", "rerank_ms"):
        STAGE_SECONDS.labels(stage[:-3]).observe(retrieval.timings[stage] / 1000)

    hits = retrieval.hits
    expansions = search.expand_sections(hits, s.section_expand_chars) if intent.expand_section and hits else {}
    sources = build_sources(hits, expansions, s.context_max_chars)
    strict_ids = {i for i in retrieval.query_ids if len(i) >= search.STRICT_ID_MIN_LEN}
    strict_match = any(strict_ids.intersection(h.matched_ids) for h in hits)
    any_id_match = any(h.matched_ids for h in hits)
    relevant = bool(hits) and (any_id_match or (retrieval.top_score is not None and retrieval.top_score >= s.min_relevance))

    status = "answered"
    missing_note = ""
    if retrieval.missing_ids:
        missing_note = f"No indexed document mentions {', '.join(retrieval.missing_ids)}."
    uncovered = uncovered_terms(query, sources) if hits else []
    if retrieval.missing_ids and not strict_match:
        status = "not_found"  # the question is about a code/tag we have no documentation for
    elif not relevant:
        status = "not_found"
    elif (len(uncovered) >= 2 and not strict_match and (retrieval.top_score is None or retrieval.top_score < s.strong_relevance)):
        # e.g. "replace the forklift hydraulic pump": the evidence is about pumps, never about forklifts.
        # Reworded questions look the same, so show the closest documented text with a clear warning
        # instead of refusing; the LLM is not asked to write a confident answer from it.
        status = "partial"
        missing_note = (f"Closest match only — the indexed documents do not mention: {', '.join(uncovered[:4])}. "
                        "Check that this is the equipment/topic you asked about.")

    shown = status in {"answered", "partial"}
    related = None
    if retrieval.query_ids and shown:
        try:
            related = relations.related_identifiers(
                opts.tenant_id, retrieval.query_ids, exclude_chunks=[h.id for h in hits[:1]])
        except Exception as exc:
            log.warning("relations lookup failed: %s", exc)
    evidence = evidence_markdown(sources) if shown else ""
    quotes = exact_answer(query, intent, sources, retrieval.query_ids) if shown else []
    exact = exact_markdown(quotes)
    if missing_note and shown:
        evidence = f"**Note:** {missing_note}\n\n{evidence}"
        exact = f"**Note:** {missing_note}\n\n{exact}"
    public_sources = [x.public() for x in sources] if shown else []
    t_ctx = time.perf_counter()

    yield {"type": "retrieval", "intent": intent.name, "status": status,
           "sources": public_sources, "evidence": evidence, "exact": quotes, "exact_markdown": exact, "related": related,
           "missing_identifiers": retrieval.missing_ids,
           "timings": {**retrieval.timings, "retrieval_ms": round((t_ctx - t0) * 1000, 1)}}

    # ---------------- answer ----------------
    answer, verification, mode_used, llm_ms, notice = "", None, "evidence", 0.0, None
    if status == "not_found":
        answer = (missing_note + " " if missing_note else "") + NOT_FOUND_TEXT
    elif mode == "evidence" or status == "partial":
        answer = exact or evidence
    else:
        health = llm.health()
        if health["status"] != "ok":
            LLM_ERRORS.labels("unavailable").inc()
            notice = f"Local LLM unavailable ({health['status']}); showing the exact document text only."
            answer = exact or evidence
        else:
            t_llm = time.perf_counter()
            tokens: List[str] = []
            passthrough = False
            try:
                prompt = build_summary_prompt(query, quotes, sources)
                if missing_note:
                    prompt += f"\nNOTE: {missing_note} State this explicitly in the answer."
                for tok in llm.stream_chat(SUMMARY_SYSTEM_PROMPT, prompt, max_tokens=s.llm_summary_tokens):
                    tokens.append(tok)
                    if passthrough:
                        yield {"type": "token", "text": tok}
                        continue
                    # Hold tokens back while the reply could still be the NOT_FOUND marker.
                    head = "".join(tokens).lstrip()
                    if head and not (NOT_FOUND.startswith(head) or head.startswith(NOT_FOUND)):
                        passthrough = True
                        yield {"type": "token", "text": "".join(tokens)}
                generated = "".join(tokens).strip()
                if generated.startswith(NOT_FOUND) and quotes:
                    # The exact document text stands; the AI simply had nothing to add.
                    answer = exact or evidence
                    notice = "The AI did not add a summary; the exact document text is shown."
                elif generated.startswith(NOT_FOUND):
                    status, answer = "not_found", NOT_FOUND_TEXT
                else:
                    answer, mode_used = generated, "llm"
                    verification = verify_answer(answer, sources, known_ids=retrieval.missing_ids)
                    if verification["confidence"] == "low":
                        LOW_CONFIDENCE.inc()
            except (llm.LLMBusy, llm.LLMUnavailable) as exc:
                LLM_ERRORS.labels(type(exc).__name__).inc()
                notice = f"{exc}. Showing the exact document text only."
                answer = exact or evidence
            llm_ms = (time.perf_counter() - t_llm) * 1000
            STAGE_SECONDS.labels("llm").observe(llm_ms / 1000)

    total_ms = (time.perf_counter() - t0) * 1000
    result = {
        "status": status, "intent": intent.name, "answer": answer, "mode_used": mode_used, "notice": notice,
        "sources": public_sources,
        "evidence": evidence, "exact": quotes, "exact_markdown": exact, "related": related, "verification": verification,
        "missing_identifiers": retrieval.missing_ids, "top_score": retrieval.top_score, "cache_hit": False,
        "timings": {**retrieval.timings, "retrieval_ms": round((t_ctx - t0) * 1000, 1),
                    "llm_ms": round(llm_ms, 1), "total_ms": round(total_ms, 1)},
    }
    QUERIES.labels(status, intent.name).inc()
    QUERY_SECONDS.labels(mode_used, status).observe(total_ms / 1000)
    if status != "error" and not notice:
        cache.put(key, result)
    _persist(opts, result)
    yield {"type": "done", **result}


def run_to_completion(opts: QueryOptions) -> dict:
    final = {}
    for event in run(opts):
        if event["type"] in {"done", "error"}:
            final = event
    final.pop("type", None)
    return final


def new_conversation_id() -> str:
    return str(uuid.uuid4())


__all__ = ["QueryOptions", "run", "run_to_completion", "new_conversation_id", "INTENTS"]
