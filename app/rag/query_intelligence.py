"""
=============================================================================
 Enterprise Level RAG: Query Intelligence Engine
=============================================================================
 Techniques implemented:
   - FLARE: Forward-looking Active Retrieval Augmented Generation
   - Query Decomposition: Two-tier (regex + LLM) sub-question generation
   - Dynamic Metadata Filter Extraction: Two-tier (regex + LLM) filter parsing

 All functions are designed to be non-blocking and fail-safe.
 LLM calls use minimal token budgets (num_predict ≤ 80) for near-zero latency.
=============================================================================
"""

import re
import json
import os
import requests
from typing import Dict, List, Optional

from app.rag.model_loader import get_ollama_generate_url, OLLAMA_MODEL

# ---------------------------------------------------------------------------
# FLARE: Mid-Generation Retrieval Monitor
# ---------------------------------------------------------------------------

def flare_mid_generation_retrieval(
    partial_answer: str,
    original_query: str,
    existing_context: List[Dict],
    min_sentence_len: int = 30,
    is_full_answer: bool = False,
) -> Optional[str]:
    """
    FLARE mid-generation: extract claims from partial/generated answer and
    generate a targeted search query for ungrounded claims.

    In streaming mode (is_full_answer=False): checks only the last sentence.
    In full-answer mode (is_full_answer=True): checks ALL sentences for safety.

    Returns a search query string if re-retrieval is needed, or None if
    the partial answer is sufficiently grounded in existing context.
    """
    if not partial_answer or len(partial_answer) < min_sentence_len:
        return None

    sentences = re.split(r'(?<=[.!?])\s+', partial_answer)
    if not sentences:
        return None

    context_text = " ".join(
        c.get("text", "") for c in (existing_context or [])
    ).lower()

    # Determine which sentences to check
    if is_full_answer:
        candidates = sentences
    else:
        candidates = [sentences[-1].strip()]

    stopwords = {"this", "that", "with", "from", "have", "been", "will",
                 "would", "could", "should", "their", "there", "which",
                 "what", "when", "where", "how", "does", "based", "also",
                 "used", "using", "such", "each", "than", "then", "very",
                 "more", "most", "some", "just", "only"}

    for sentence in candidates:
        sentence = sentence.strip()
        if len(sentence) < min_sentence_len:
            continue

        terms = set(re.findall(r'[A-Za-z0-9][A-Za-z0-9_-]{2,}', sentence.lower()))
        terms -= stopwords
        if not terms:
            continue

        # Word-boundary matching to avoid false positives
        matched = sum(1 for t in terms if re.search(r'\b' + re.escape(t) + r'\b', context_text))
        overlap_ratio = matched / len(terms)

        if overlap_ratio < 0.4:
            missing = [t for t in terms if not re.search(r'\b' + re.escape(t) + r'\b', context_text)]
            if missing:
                flare_query = f"{original_query} {' '.join(missing[:5])}"
                print(f"[FLARE] Triggered. Overlap: {overlap_ratio:.0%} "
                      f"Missing terms: {missing[:5]}")
                return flare_query

    return None


# ---------------------------------------------------------------------------
# Query Decomposition — Tier 1: Rule-based (0ms)
# ---------------------------------------------------------------------------

# Conjunctive split: "What is X and how does Y work"
_CONJUNCTIVE_RE = re.compile(
    r'^(.+?)\s+and\s+((?:how|what|why|where|when|which|who|explain|describe|list|tell)\s+.+)$',
    re.IGNORECASE,
)
# Comparative split: "difference between X and Y" / "compare X and Y"
_COMPARATIVE_RE = re.compile(
    r'(?:difference(?:s)?\s+between|compare|contrast|distinguish)\s+(.+?)\s+and\s+(.+)',
    re.IGNORECASE,
)
# Multi-part: "list all X and explain Y"
_LIST_EXPLAIN_RE = re.compile(
    r'^(list\s+.+?)\s+and\s+(explain\s+.+)$',
    re.IGNORECASE,
)

# Complexity heuristic — only call LLM Tier 2 when query seems multi-faceted
_COMPLEX_TRIGGERS = frozenset([
    "and", "also", "additionally", "furthermore", "as well as",
    "both", "multiple", "various", "several", "what are the steps",
    "how does", "why does", "when should", "what happens when",
    "relationship between", "how does it work",
])


def _rule_based_decompose(query: str) -> List[str]:
    """
    Tier 1: Pure regex decomposition — 0ms, always runs first.
    Returns a list of sub-questions if the query can be split, else [query].
    """
    q = query.strip()

    # Comparative: "difference between X and Y"
    m = _COMPARATIVE_RE.search(q)
    if m:
        x, y = m.group(1).strip(), m.group(2).strip()
        return [
            q,
            f"What is {x}?",
            f"What is {y}?",
            f"What are the differences between {x} and {y}?",
        ]

    # List + Explain pattern
    m = _LIST_EXPLAIN_RE.match(q)
    if m:
        return [q, m.group(1).strip() + "?", m.group(2).strip() + "?"]

    # Conjunctive: "What is X and how does Y work"
    m = _CONJUNCTIVE_RE.match(q)
    if m:
        part1 = m.group(1).strip().rstrip("?") + "?"
        part2 = m.group(2).strip().rstrip("?") + "?"
        return [q, part1, part2]

    return [q]


def _llm_decompose(query: str) -> List[str]:
    """
    Tier 2: LLM-driven decomposition for complex queries Tier 1 cannot split.
    Uses num_predict=80, temperature=0.0 — fast and deterministic.
    Falls back to [query] on any failure.
    """
    prompt = (
        f"Break the following question into 2 or 3 specific, self-contained retrieval "
        f"sub-questions that together fully cover the original question. "
        f"Output a JSON array of strings ONLY, no explanation.\n\n"
        f"Question: {query}"
    )
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "num_predict": 80,
            "temperature": 0.0,
            "num_ctx": int(os.getenv("OLLAMA_CONTEXT_LENGTH", "32768")),
        },
    }
    try:
        resp = requests.post(get_ollama_generate_url(), json=payload, timeout=8.0)
        if resp.status_code == 200:
            raw = resp.json().get("response", "").strip()
            # Extract JSON array (may be wrapped in ```json ... ```)
            if "```" in raw:
                raw = re.search(r'\[.*?\]', raw, re.DOTALL)
                raw = raw.group(0) if raw else "[]"
            sub_qs = json.loads(raw)
            if isinstance(sub_qs, list) and all(isinstance(q, str) for q in sub_qs):
                result = [query] + [sq.strip() for sq in sub_qs if sq.strip()]
                print(f"[QueryDecomp/LLM] '{query}' → {len(result)} sub-questions")
                return result
    except Exception as e:
        print(f"[QueryDecomp/LLM] Failed, using original query: {e}")
    return [query]


def decompose_query(query: str) -> List[str]:
    """
    Two-tier query decomposition for enterprise RAG.

    Tier 1 (always, 0ms): Rule-based regex decomposition.
    Tier 2 (only if Tier 1 returns 1 result AND query is complex): LLM decomposition.

    Returns a deduplicated list: [original_query, sub_q1, sub_q2, ...]
    Always includes the original as the first element so RRF fusion covers it.

    Example:
        "What is a contactor and how does it differ from a relay?"
        → ["What is a contactor and...", "What is a contactor?",
           "What is a relay?", "What are the differences between a contactor and a relay?"]
    """
    if not query or len(query.strip()) < 10:
        return [query]

    # Tier 1: Rule-based
    variants = _rule_based_decompose(query)
    if len(variants) > 1:
        print(f"[QueryDecomp/Rule] '{query}' → {len(variants)} sub-questions")
        return list(dict.fromkeys(variants))  # deduplicate, preserve order

    # Tier 2: LLM — only for complex queries
    q_lower = query.lower()
    is_complex = (
        len(query.split()) > 8
        and any(trigger in q_lower for trigger in _COMPLEX_TRIGGERS)
    )
    if is_complex:
        return list(dict.fromkeys(_llm_decompose(query)))

    return [query]


# ---------------------------------------------------------------------------
# Dynamic Metadata Filter Extraction — Tier 1: Regex (0ms)
# ---------------------------------------------------------------------------

# Document/file reference patterns
_FILE_PATTERNS = [
    # "in the welding manual" / "from the installation guide"
    re.compile(
        r'(?:in|from|within|inside|of)\s+the\s+([a-z0-9\s\-_]+?)'
        r'\s+(?:manual|guide|document|specification|spec|datasheet|report|pdf|file|handbook|procedure|standard)',
        re.IGNORECASE,
    ),
    # "the welding manual" without preposition
    re.compile(
        r'the\s+([a-z0-9\s\-_]+?)\s+'
        r'(?:manual|guide|document|specification|spec|datasheet|report|pdf|file|handbook|procedure|standard)',
        re.IGNORECASE,
    ),
    # "welding manual" at start
    re.compile(
        r'^([a-z0-9\s\-_]+?)\s+'
        r'(?:manual|guide|document|specification|spec|datasheet)',
        re.IGNORECASE,
    ),
]

# Page reference patterns
_PAGE_PATTERNS = [
    re.compile(r'\bon\s+page\s+(\d+)\b', re.IGNORECASE),
    re.compile(r'\bpage\s+(\d+)\b', re.IGNORECASE),
    re.compile(r'\bp\.\s*(\d+)\b', re.IGNORECASE),
]

# Page range patterns: "pages 10 to 15" / "pages 10-15"
_PAGE_RANGE_RE = re.compile(
    r'\bpages?\s+(\d+)\s*(?:to|-|–|through)\s*(\d+)\b', re.IGNORECASE
)

# Section/chapter reference patterns
_SECTION_PATTERNS = [
    re.compile(r'\b(?:in\s+)?section\s+(\d+(?:\.\d+)*)\b', re.IGNORECASE),
    re.compile(r'\b(?:in\s+)?chapter\s+(\d+)\b', re.IGNORECASE),
    re.compile(r'\bsec\.\s*(\d+(?:\.\d+)*)\b', re.IGNORECASE),
]

# Trigger words that suggest a document-scoped query (needed for LLM Tier 2)
_DOC_TRIGGER_WORDS = frozenset([
    "manual", "guide", "document", "specification", "spec", "datasheet",
    "report", "pdf", "file", "handbook", "procedure", "standard",
    "from the", "in the", "chapter", "section", "page",
])


def _regex_extract_filters(query: str) -> Dict:
    """Tier 1: Pure regex filter extraction — 0ms."""
    filters: Dict = {}

    # Page range (check first — strip it from query before file matching
    # so "pages 10 to 15 of the electrical guide" doesn't greedily match
    # "pages 10 to 15 of electrical" as the target_file)
    query_for_file = query
    m = _PAGE_RANGE_RE.search(query)
    if m:
        filters["page_range"] = [int(m.group(1)), int(m.group(2))]
        # Remove the page-range clause from the string used for file matching
        query_for_file = query[:m.start()].strip() + " " + query[m.end():].strip()
    else:
        # Single page
        for pat in _PAGE_PATTERNS:
            m = pat.search(query)
            if m:
                filters["page"] = int(m.group(1))
                break

    # File / document filter (run against page-stripped query)
    for pat in _FILE_PATTERNS:
        m = pat.search(query_for_file)
        if m:
            candidate = m.group(1).strip().lower()
            # Clean up dangling prepositions left over from stripping page ranges
            for prep in ["of ", "in ", "from "]:
                if candidate.startswith(prep):
                    candidate = candidate[len(prep):].strip()
            # Reject generic stopwords or single-word prepositions captured as artifact
            _reject = {"the", "a", "an", "this", "that", "above", "following", "of", "by", "its"}
            if candidate not in _reject and len(candidate) > 2:
                filters["target_file"] = candidate
                break

    # Section / chapter (against original query)
    for pat in _SECTION_PATTERNS:
        m = pat.search(query)
        if m:
            filters["section"] = m.group(1)
            break

    return filters


def _llm_extract_filters(query: str) -> Dict:
    """
    Tier 2: LLM filter extraction — fires only when regex finds nothing
    AND doc-like trigger words are present.
    Uses num_predict=40, temperature=0.0 — extremely fast.
    """
    prompt = (
        f"Extract document metadata filters from this question. "
        f"Output ONLY a JSON object with these optional keys: "
        f"target_file (string: partial filename or topic), page (integer), section (string). "
        f"If nothing to extract, output {{}}.\n\n"
        f"Question: {query}"
    )
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "num_predict": 40,
            "temperature": 0.0,
            "num_ctx": int(os.getenv("OLLAMA_CONTEXT_LENGTH", "32768")),
        },
    }
    try:
        resp = requests.post(get_ollama_generate_url(), json=payload, timeout=6.0)
        if resp.status_code == 200:
            raw = resp.json().get("response", "").strip()
            # Extract JSON object from response
            m = re.search(r'\{[^}]*\}', raw, re.DOTALL)
            if m:
                result = json.loads(m.group(0))
                if isinstance(result, dict):
                    # Sanitize — keep only expected keys with valid types
                    clean: Dict = {}
                    if isinstance(result.get("target_file"), str) and result["target_file"]:
                        clean["target_file"] = result["target_file"].lower().strip()
                    if isinstance(result.get("page"), int):
                        clean["page"] = result["page"]
                    if isinstance(result.get("section"), (str, int)) and result["section"]:
                        clean["section"] = str(result["section"])
                    if clean:
                        print(f"[MetaFilter/LLM] Extracted: {clean}")
                    return clean
    except Exception as e:
        print(f"[MetaFilter/LLM] Failed: {e}")
    return {}


def extract_metadata_filters(query: str) -> Dict:
    """
    Two-tier dynamic metadata filter extraction.

    Tier 1 (always, 0ms): Regex patterns for file, page, section references.
    Tier 2 (only if Tier 1 finds nothing AND doc-like terms present): LLM extraction.

    Returns a dict compatible with perform_hybrid_search(metadata_filters=...).
    Returns {} (no filters) for generic queries.

    Examples:
        "torque spec in the welding manual"      → {"target_file": "welding"}
        "installation steps on page 5"           → {"page": 5}
        "wiring in section 3.2"                  → {"section": "3.2"}
        "pages 10 to 15 of the electrical guide" → {"page_range": [10,15], "target_file": "electrical"}
        "What is a DC sensor?"                   → {}
    """
    if not query:
        return {}

    # Tier 1: Regex
    filters = _regex_extract_filters(query)
    if filters:
        print(f"[MetaFilter/Regex] Extracted: {filters}")
        return filters

    # Tier 2: LLM — only when doc-like trigger words are present
    q_lower = query.lower()
    has_doc_trigger = any(t in q_lower for t in _DOC_TRIGGER_WORDS)
    if has_doc_trigger:
        return _llm_extract_filters(query)

    return {}
