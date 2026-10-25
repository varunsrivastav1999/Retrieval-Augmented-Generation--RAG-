"""Evidence assembly, prompt construction and deterministic answer verification.

Accuracy rules for maintenance use:
  * the model sees only numbered sources and must cite them as [n];
  * documented facts, inferred suggestions and "not in documents" are kept apart;
  * after generation every sentence, identifier and number is checked against
    the sources without another LLM call (fast and reproducible).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence

from app.rag.identifiers import extract_identifiers, normalize_identifier
from app.rag.intent import Intent
from app.rag.search import Hit

NOT_FOUND = "NOT_FOUND"
NOT_FOUND_TEXT = ("This information is not available in the indexed documents. "
                  "Check the spelling of codes/tags, the document filters, or upload the relevant manual.")


@dataclass
class Source:
    n: int
    chunk_id: int
    document_id: str
    document: str
    collection: str
    page: int | None
    section: str | None
    content_type: str
    text: str
    snippet: str
    score: float
    matched_ids: List[str]

    def public(self) -> dict:
        return {k: getattr(self, k) for k in (
            "n", "chunk_id", "document_id", "document", "collection", "page", "section",
            "content_type", "text", "snippet", "score", "matched_ids")}


def build_sources(hits: Sequence[Hit], expansions: Dict[int, str], max_chars: int) -> List[Source]:
    sources: List[Source] = []
    used = 0
    seen_sections = set()
    for h in hits:
        if h.section_id and h.section_id in seen_sections and h.id not in expansions:
            continue  # already covered by an expanded section
        body = expansions.get(h.id, h.text)
        if used and used + len(body) > max_chars:
            body = body[: max(0, max_chars - used)]
            if len(body) < 200:
                break
        if h.section_id and h.id in expansions:
            seen_sections.add(h.section_id)
        used += len(body)
        sources.append(Source(
            n=len(sources) + 1, chunk_id=h.id, document_id=h.document_id, document=h.document,
            collection=h.collection, page=h.page, section=h.section_title, content_type=h.content_type,
            text=body, snippet=h.text[:400], score=h.score, matched_ids=h.matched_ids))
    return sources


def uncovered_terms(question: str, sources: Sequence[Source]) -> List[str]:
    """Content words of the question that appear nowhere in the evidence (e.g. "forklift")."""
    seen = set()
    for s in sources:
        seen.update(_tokens(f"{s.section or ''} {s.document} {s.text}"))
    out = []
    for t in dict.fromkeys(_tokens(question)):
        if len(t) > 3 and not any(ch.isdigit() for ch in t) and t not in seen:
            out.append(t)
    return out


def evidence_markdown(sources: Sequence[Source], limit: int = 3) -> str:
    """Deterministic, LLM-free answer: the best documented passages verbatim."""
    parts = []
    for s in sources[:limit]:
        where = f"{s.document}" + (f" — {s.section}" if s.section else "") + (f" (p. {s.page})" if s.page else "")
        quoted = "\n".join(f"> {line}" if line.strip() else ">" for line in s.text.strip().splitlines())
        parts.append(f"**[{s.n}] {where}**\n\n{quoted}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Exact (verbatim) answers — no LLM
# ---------------------------------------------------------------------------
_UNIT_START = re.compile(r"^\s*(\d{1,2}[.)]\s|[-•*]\s|[A-Z][A-Za-z /-]{1,30}:\s|WARNING|CAUTION|NOTE|DANGER)")
_STEP = re.compile(r"^\s*\d{1,2}[.)]\s")
# Query wording -> (label/content pattern, weight) of the lines that answer it.
_CUES = [
    (re.compile(r"\b(remed\w*|fix|solve|what (should|do) i do|corrective|action|reset|clear)\b", re.I),
     [(re.compile(r"(^|; )(remedy|corrective action|action taken|check|solution|measure): ", re.I), 1.0)]),
    (re.compile(r"\b(when (is|does|do|will|can)\b|condition|raised|trigger\w*|occurs?|means?|meaning)\b", re.I),
     [(re.compile(r"(^|; )(condition|description): ", re.I), 1.0), (re.compile(r"(^|; )message: ", re.I), 0.15)]),
    (re.compile(r"\b(cause|why|reason)\b", re.I),
     [(re.compile(r"(^|; )(probable cause|possible cause|cause|root cause): ", re.I), 1.0)]),
    (re.compile(r"\b(torque|tighten\w*)\b", re.I), [(re.compile(r"\bNm\b", re.I), 1.0)]),
    (re.compile(r"\b(part number|p/n|spare part|kit)\b", re.I), [(re.compile(r"\bP/N\b", re.I), 1.0)]),
    (re.compile(r"\b(interval|how often|frequency)\b", re.I), [(re.compile(r"^interval\b|every \d", re.I), 1.0)]),
    (re.compile(r"\b(default|range|minimum|maximum|limit|setpoint)\b", re.I),
     [(re.compile(r"(^|; )(default|minimum|maximum|range|limit|high limit|hihi limit)\b", re.I), 1.0)]),
]
_ROW = re.compile(r"^[^;:]{1,40}: .*; [^;:]{1,40}: ")


def _units(text: str) -> List[str]:
    """Re-join wrapped PDF lines into sentences/steps while keeping the original words.
    Table/record rows ("Header: value; Header: value") stay whole."""
    units: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if _ROW.match(line):
            units.append(line)
        elif units and not _UNIT_START.match(line) and not units[-1].endswith(":") and not _ROW.match(units[-1]):
            units[-1] = f"{units[-1]} {line}"  # wrapped continuation line
        else:
            units.append(line)
    return units


def _quote(src: Source, text: str) -> dict:
    return {"n": src.n, "document_id": src.document_id, "document": src.document, "page": src.page,
            "section": src.section, "content_type": src.content_type, "text": text}


def exact_answer(question: str, intent: Intent, sources: Sequence[Source], query_ids: Sequence[str] = (),
                 max_sources: int = 4) -> List[dict]:
    """Verbatim quotes [{n, document, page, section, text}] that answer the question (no LLM).

    Lines from the top passages are scored by word overlap with the question, by
    question cues ("remedy", "when raised", "torque", "part number", ...), by exact
    identifiers and by how well the passage's document/section matches; the best
    lines are returned exactly as written. Procedures return every step in order.
    """
    if not sources:
        return []
    q_tokens = set(_tokens(question))
    ids = {i.upper() for i in query_ids}
    cue_sets = [labels for qrx, labels in _CUES if qrx.search(question)]
    step_no = re.search(r"\bstep (\d{1,2})\b", question, re.I)
    wants_single = re.search(r"\b(torque|part number|p/n|step \d+|what is step|interval|default|range|value)\b", question, re.I)
    wants_procedure = (intent.name == "procedure" or re.search(
        r"\b(procedure|steps|how (do|to)|replac\w+|instructions?)\b", question, re.I)) and not wants_single

    def context_score(src: Source) -> float:
        ctx = set(_tokens(f"{src.section or ''} {re.sub(r'[_.]+', ' ', src.document)}"))
        score = 0.5 * len(q_tokens & ctx) / (len(q_tokens) or 1)
        if ids and any(i in (src.section or "").upper() for i in ids):
            score += 0.6  # a section dedicated to the asked code (e.g. "2.4 Alarm PSU-171 ...")
        elif ids and any(i in src.document.upper() for i in ids):
            score += 0.3
        return score

    pool = list(sources[:max_sources])
    if wants_procedure:
        best_src, best = None, -1.0
        for rank, src in enumerate(pool):
            units = _units(src.text)
            steps = [u for u in units if _STEP.match(u)]
            if len(steps) < 3:
                continue
            sc = context_score(src) + len(q_tokens & set(_tokens(src.text))) / (len(q_tokens) or 1) - 0.1 * rank
            if sc > best:
                best_src, best = (src, units, steps), sc
        if best_src:
            src, units, steps = best_src
            lead = [u for u in units if not _STEP.match(u) and re.match(r"^(interval|caution|warning|note|danger|required)", u, re.I)]
            return [_quote(src, "\n".join(lead + steps))]

    scored = []
    for rank, src in enumerate(pool):
        ctx = context_score(src)
        for idx, u in enumerate(_units(src.text)):
            score = len(q_tokens & set(_tokens(u))) / (len(q_tokens) or 1) + ctx - 0.15 * rank
            for labels in cue_sets:
                score += max((w for rx, w in labels if rx.search(u)), default=0.0)
            if ids and any(i in u.upper() for i in ids):
                score += 0.6
            if step_no and re.match(rf"^\s*{step_no.group(1)}[.)]\s", u):
                score += 2.0
            if len(u) < 12:
                score -= 0.5
            scored.append((score, rank, idx, u, src))
    if not scored:
        return []
    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    top_score, top_rank = scored[0][0], scored[0][1]
    # Quote up to 3 strong lines from the winning passage, in document order.
    same = [x for x in scored if x[1] == top_rank and x[0] >= max(0.35, top_score - 0.25)][:3]
    same.sort(key=lambda x: x[2])
    quotes = [_quote(scored[0][4], "\n".join(x[3] for x in same))]
    # Two-part questions ("when is it raised AND what is the remedy?"): if the first
    # quote does not answer one of the asked parts, add the best line that does.
    chosen_text = "\n".join(x[3] for x in same)
    for labels in cue_sets:
        if any(rx.search(line) for rx, w in labels if w >= 1.0 for line in chosen_text.splitlines()):
            continue
        extra = [x for x in scored if any(rx.search(x[3]) for rx, w in labels if w >= 1.0)
                 and (not ids or any(i in x[3].upper() or i in (x[4].section or "").upper() for i in ids))]
        if extra:
            quotes.append(_quote(extra[0][4], extra[0][3]))
            chosen_text += "\n" + extra[0][3]
    return quotes[:2]


def exact_markdown(quotes: Sequence[dict]) -> str:
    parts = []
    for q in quotes:
        where = q["document"] + (f" — {q['section']}" if q.get("section") else "") + (f", page {q['page']}" if q.get("page") else "")
        body = "\n".join(f"> {line}" for line in q["text"].splitlines())
        parts.append(f"{body}\n\n— *exact text from* **[{q['n']}] {where}**")
    return "\n\n".join(parts)


SYSTEM_PROMPT = (
    "You are a maintenance assistant for an industrial robot painting plant (robots, PLCs, SCADA, "
    "sensors, painting process). Answer ONLY with information from the numbered SOURCES.\n"
    "Rules:\n"
    "1. After every factual sentence cite its source like [1] or [2][3].\n"
    "2. Copy alarm codes, tag names, addresses, values, units and procedure steps exactly as written.\n"
    "3. Never invent codes, part numbers, values or steps. Never use outside knowledge for facts.\n"
    "4. If the SOURCES do not answer the question, reply with exactly: NOT_FOUND\n"
    "5. If you add a suggestion that is not stated in the sources, put it under the heading "
    "'Inferred (not documented)' and keep it short.\n"
    "6. Be concise. Use short paragraphs, numbered steps or bullet lists."
)


def build_prompt(question: str, intent: Intent, sources: Sequence[Source]) -> str:
    blocks = []
    for s in sources:
        where = s.document + (f" | {s.section}" if s.section else "") + (f" | page {s.page}" if s.page else "")
        blocks.append(f"[{s.n}] ({where})\n{s.text.strip()}")
    return (
        "SOURCES:\n" + "\n\n".join(blocks) +
        f"\n\nQUESTION: {question}\n"
        f"ANSWER GUIDANCE: {intent.guidance}\n"
        "Answer with citations:"
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
_STOP = set("""a an the is are was were be been being have has had do does did will would could should may might
shall can to of in for on with at by from as into through during before after above below between out off over under
again then once and but or nor not so yet both each every all any few more most other some such no only own same than
too very just because if when where how what which who this that these those it its they them their there here also
use used using must per via up down please check""".split())
_TOKEN = re.compile(r"[a-z0-9][a-z0-9._%/-]*")
_NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?(?![\w])")
_CITE = re.compile(r"\[(\d{1,2})\]")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9*\[(-])|\n+")


def _stem(token: str) -> str:
    """Tiny suffix stripper so 'detected'/'detect', 'checks'/'check' compare equal."""
    if len(token) > 4 and not any(ch.isdigit() for ch in token):
        for suffix in ("ing", "ed", "es", "s"):
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                return token[: -len(suffix)]
    return token


def _tokens(text: str) -> List[str]:
    out = []
    for t in _TOKEN.findall(text.lower()):
        t = t.strip("._/-")
        if len(t) > 1 and t not in _STOP:
            out.append(_stem(t))
    return out


def verify_answer(answer: str, sources: Sequence[Source], known_ids: Sequence[str] = ()) -> dict:
    """Check every sentence / identifier / number against the sources.

    `known_ids` are identifiers the answer may legitimately mention although no
    source contains them (e.g. "PNT1-999 is not in the documents").
    """
    if not answer.strip() or answer.strip() == NOT_FOUND:
        return {"confidence": "n/a", "supported_ratio": None, "unsupported": [], "unknown_identifiers": [],
                "unknown_numbers": [], "citations": []}
    by_n = {s.n: s.text.lower() for s in sources}
    all_text = "\n".join(by_n.values())
    all_tokens = set(_tokens(all_text))
    source_ids = {normalize_identifier(i) for s in sources for i in extract_identifiers(s.text, limit=512)}
    source_ids |= {normalize_identifier(i) for s in sources for i in extract_identifiers(s.section or "")}
    source_ids |= {normalize_identifier(i) for i in known_ids}
    source_numbers = set(_NUMBER.findall(all_text))

    inferred_section = False
    checked = supported = 0
    unsupported: List[str] = []
    cited = set()
    for raw in _SENT.split(answer):
        sent = raw.strip()
        if not sent:
            continue
        if re.search(r"inferred\s*\(not documented\)", sent, re.I):
            inferred_section = True
            continue
        refs = [int(n) for n in _CITE.findall(sent)]
        cited.update(refs)
        toks = _tokens(_CITE.sub(" ", sent))
        if len(toks) < 3 or inferred_section or sent.rstrip("*_ ").endswith(":"):  # list lead-ins are not claims
            continue
        pool = set(_tokens(" ".join(by_n[r] for r in refs if r in by_n))) if refs else all_tokens
        overlap = sum(1 for t in toks if t in pool) / len(toks)
        checked += 1
        if overlap >= 0.6:
            supported += 1
        else:
            unsupported.append(sent[:240])

    body = answer.split("Inferred (not documented)")[0]
    unknown_ids = sorted({normalize_identifier(i) for i in extract_identifiers(body)} - source_ids)
    nums = {n for n in _NUMBER.findall(_CITE.sub(" ", re.sub(r"(?m)^\s*\d+[.)]\s", " ", body)))}
    unknown_numbers = sorted(n for n in nums - source_numbers if n.replace(",", ".") not in source_numbers)
    ratio = (supported / checked) if checked else 1.0
    if ratio >= 0.9 and not unknown_ids and not unknown_numbers:
        confidence = "high"
    elif ratio >= 0.6 and not unknown_ids:
        confidence = "medium"
    else:
        confidence = "low"
    return {
        "confidence": confidence,
        "supported_ratio": round(ratio, 3),
        "unsupported": unsupported[:5],
        "unknown_identifiers": unknown_ids[:10],
        "unknown_numbers": unknown_numbers[:10],
        "citations": sorted(c for c in cited if c in by_n),
    }
