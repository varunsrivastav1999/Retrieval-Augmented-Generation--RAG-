"""
=============================================================================
 Enterprise Level RAG v6.0: Query Classifier & Topic Detection
=============================================================================
 Two-tier query classification:
   Tier 1: Rule-based regex patterns (0ms, always runs)
   Tier 2: LLM fallback for ambiguous queries (~200ms, fires only when needed)

 Each query type maps to a RetrievalStrategy with different:
   - top_k, expand_section, multi_query, section_search, map_reduce settings

 Query Types:
   FACT        → "What is Register D100?" → top-k chunks only
   TOPIC       → "Explain USB Communication" → full section retrieval
   CHAPTER     → "Summarize Chapter 3" → all chapter chunks
   SUMMARY     → "Give overview of the manual" → map-reduce
   TROUBLESHOOT→ "USB not working" → troubleshooting sections
   PROCEDURE   → "How to install USB driver" → step-by-step
   COMPARISON  → "Difference between RS232 and USB" → multi-section
   PARAMETER   → "List all USB parameters" → table/parameter lookup
   TABLE       → "Show voltage ratings table" → table retrieval

 100% offline, fail-safe, no external dependencies.
=============================================================================
"""

import os
import re
import json
import requests
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Query Types
# ---------------------------------------------------------------------------

class QueryType(str, Enum):
    FACT = "fact"
    TOPIC = "topic"
    CHAPTER = "chapter"
    SUMMARY = "summary"
    TROUBLESHOOT = "troubleshoot"
    PROCEDURE = "procedure"
    COMPARISON = "comparison"
    PARAMETER = "parameter"
    TABLE = "table"


# ---------------------------------------------------------------------------
# Retrieval Strategies
# ---------------------------------------------------------------------------

@dataclass
class RetrievalStrategy:
    """Configuration for how retrieval should behave for a given query type."""
    query_type: QueryType
    top_k: int = 12
    expand_section: bool = False
    multi_query: bool = False
    section_search: bool = False
    chapter_retrieval: bool = False
    map_reduce: bool = False
    table_search: bool = False
    max_section_chunks: int = 40
    multi_query_count: int = 6
    context_ordering: str = "relevance"  # "relevance" | "document_order" | "section_order"
    prompt_style: str = "standard"       # "standard" | "comprehensive" | "procedural" | "diagnostic" | "comparative"


# Strategy definitions per query type
STRATEGIES: Dict[QueryType, RetrievalStrategy] = {
    QueryType.FACT: RetrievalStrategy(
        query_type=QueryType.FACT,
        top_k=12,
        expand_section=False,
        multi_query=False,
        context_ordering="relevance",
        prompt_style="standard",
    ),
    QueryType.TOPIC: RetrievalStrategy(
        query_type=QueryType.TOPIC,
        top_k=40,
        expand_section=True,
        multi_query=True,
        section_search=True,
        max_section_chunks=40,
        multi_query_count=6,
        context_ordering="section_order",
        prompt_style="comprehensive",
    ),
    QueryType.CHAPTER: RetrievalStrategy(
        query_type=QueryType.CHAPTER,
        top_k=80,
        expand_section=True,
        multi_query=False,
        chapter_retrieval=True,
        max_section_chunks=80,
        context_ordering="document_order",
        prompt_style="comprehensive",
    ),
    QueryType.SUMMARY: RetrievalStrategy(
        query_type=QueryType.SUMMARY,
        top_k=60,
        expand_section=True,
        multi_query=False,
        map_reduce=True,
        max_section_chunks=100,
        context_ordering="document_order",
        prompt_style="comprehensive",
    ),
    QueryType.TROUBLESHOOT: RetrievalStrategy(
        query_type=QueryType.TROUBLESHOOT,
        top_k=30,
        expand_section=True,
        multi_query=True,
        section_search=True,
        multi_query_count=4,
        context_ordering="section_order",
        prompt_style="diagnostic",
    ),
    QueryType.PROCEDURE: RetrievalStrategy(
        query_type=QueryType.PROCEDURE,
        top_k=20,
        expand_section=True,
        multi_query=False,
        section_search=True,
        context_ordering="document_order",
        prompt_style="procedural",
    ),
    QueryType.COMPARISON: RetrievalStrategy(
        query_type=QueryType.COMPARISON,
        top_k=40,
        expand_section=True,
        multi_query=True,
        section_search=True,
        multi_query_count=4,
        context_ordering="relevance",
        prompt_style="comparative",
    ),
    QueryType.PARAMETER: RetrievalStrategy(
        query_type=QueryType.PARAMETER,
        top_k=30,
        expand_section=False,
        multi_query=True,
        table_search=True,
        multi_query_count=4,
        context_ordering="section_order",
        prompt_style="standard",
    ),
    QueryType.TABLE: RetrievalStrategy(
        query_type=QueryType.TABLE,
        top_k=20,
        expand_section=False,
        multi_query=False,
        table_search=True,
        context_ordering="relevance",
        prompt_style="standard",
    ),
}


# ---------------------------------------------------------------------------
# Tier 1: Rule-Based Classification (0ms)
# ---------------------------------------------------------------------------

# Topic patterns — user wants complete information about a subject
_TOPIC_PATTERNS = [
    re.compile(r"\b(?:explain|describe)\s+(?:everything|all|complete|full|entire|detailed)\s+(?:about|details?\s+(?:of|about)?|info(?:rmation)?\s+(?:of|about|on)?)", re.IGNORECASE),
    re.compile(r"\b(?:tell|give)\s+(?:me\s+)?(?:all|everything|complete|full|entire|detailed)\s+(?:about|details?\s+(?:of|about)?|info(?:rmation)?\s+(?:of|about|on)?)", re.IGNORECASE),
    re.compile(r"\b(?:complete|full|detailed|entire)\s+(?:guide|explanation|description|info(?:rmation)?|details?)\s+(?:of|about|on|for)\b", re.IGNORECASE),
    re.compile(r"\b(?:explain|describe|discuss)\s+(?:the\s+)?(?:\w+\s+){0,2}(?:communication|protocol|interface|system|module|feature|function|operation|process|mechanism)\b", re.IGNORECASE),
    re.compile(r"\b(?:all|every)\s+(?:parameter|command|register|alarm|error|feature|setting|option|config(?:uration)?)\s+(?:of|for|about|related|in)\b", re.IGNORECASE),
    re.compile(r"\b(?:explain|describe)\s+(?:the\s+)?(?:full|complete|entire)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(?:is|are)\s+(?:all|the\s+different|the\s+various)\b", re.IGNORECASE),
    re.compile(r"\b(?:list|enumerate)\s+(?:all|every|each)\b", re.IGNORECASE),
]

# Chapter patterns — user wants an entire chapter
_CHAPTER_PATTERNS = [
    re.compile(r"\b(?:summarize|explain|describe|give)\s+(?:the\s+)?(?:full\s+|entire\s+|complete\s+)?chapter\s+(\d+)", re.IGNORECASE),
    re.compile(r"\bchapter\s+(\d+)\s+(?:summary|overview|content|details?)", re.IGNORECASE),
    re.compile(r"\b(?:entire|whole|full|complete)\s+chapter\b", re.IGNORECASE),
]

# Summary patterns — user wants a summary of document/section
_SUMMARY_PATTERNS = [
    re.compile(r"\b(?:summarize|summary)\s+(?:the\s+)?(?:entire\s+|whole\s+|full\s+|complete\s+)?(?:document|manual|file|pdf|guide|book|report)\b", re.IGNORECASE),
    re.compile(r"\b(?:overview|outline)\s+(?:of\s+)?(?:the\s+)?(?:entire\s+|whole\s+)?(?:document|manual|file|pdf)\b", re.IGNORECASE),
    re.compile(r"\b(?:give|provide)\s+(?:me\s+)?(?:a\s+|an\s+)?(?:brief\s+|short\s+)?(?:summary|overview)\b", re.IGNORECASE),
]

# Troubleshooting patterns
_TROUBLESHOOT_PATTERNS = [
    re.compile(r"\b(?:troubleshoot|diagnos(?:e|is|tic)|debug|fix|solve|resolv)\b", re.IGNORECASE),
    re.compile(r"\bnot\s+(?:working|responding|connecting|communicating|functioning)\b", re.IGNORECASE),
    re.compile(r"\b(?:error|fault|failure|problem|issue)\s+(?:with|in|when|during)\b", re.IGNORECASE),
    re.compile(r"\b(?:complete|full)\s+troubleshoot(?:ing)?\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(?:to\s+do|should\s+I\s+do)\s+(?:if|when)\b", re.IGNORECASE),
]

# Procedure patterns
_PROCEDURE_PATTERNS = [
    re.compile(r"\b(?:how\s+to|steps?\s+(?:to|for)|procedure\s+(?:to|for)|process\s+(?:to|for))\b", re.IGNORECASE),
    re.compile(r"\b(?:install(?:ation)?|setup|configur(?:e|ation)|calibrat(?:e|ion)|assembl(?:e|y))\s+(?:guide|steps?|procedure|instructions?|process)\b", re.IGNORECASE),
    re.compile(r"\b(?:step[\s-]by[\s-]step|instructions?\s+(?:for|to))\b", re.IGNORECASE),
]

# Comparison patterns
_COMPARISON_PATTERNS = [
    re.compile(r"\b(?:difference|differ|compare|comparison|contrast|distinguish|versus|vs\.?)\b", re.IGNORECASE),
    re.compile(r"\b(?:between)\s+.+\s+(?:and|&|vs\.?)\s+", re.IGNORECASE),
    re.compile(r"\b(?:which\s+(?:one|is\s+better)|pros?\s+and\s+cons?|advantages?\s+(?:and|vs)\s+disadvantages?)\b", re.IGNORECASE),
]

# Parameter patterns — looking for specific parameters, settings, values
_PARAMETER_PATTERNS = [
    re.compile(r"\b(?:list|show|display|what\s+are)\s+(?:the\s+)?(?:all\s+)?(?:parameter|setting|register|config(?:uration)?|option|value|specification|spec)\b", re.IGNORECASE),
    re.compile(r"\b(?:parameter|register|setting)\s+(?:list|table|value|range|default)\b", re.IGNORECASE),
    re.compile(r"\b(?:all|every|each)\s+(?:parameter|register|setting|config(?:uration)?)\b", re.IGNORECASE),
]

# Table patterns — looking for a specific table
_TABLE_PATTERNS = [
    re.compile(r"\b(?:show|display|find|where\s+is)\s+(?:the\s+)?(?:\w+\s+){0,3}table\b", re.IGNORECASE),
    re.compile(r"\btable\s+(?:of|for|with|showing|listing)\b", re.IGNORECASE),
    re.compile(r"\b(?:rating|specification|spec|wiring|pin(?:out)?|dimension)\s+table\b", re.IGNORECASE),
]


def _rule_based_classify(query: str) -> Optional[QueryType]:
    """
    Tier 1: Pure rule-based classification — 0ms.
    Returns QueryType if confidently classified, None if ambiguous.
    """
    q = query.strip()
    if not q:
        return QueryType.FACT

    # Order matters: more specific patterns checked first

    # Chapter (very specific pattern)
    for p in _CHAPTER_PATTERNS:
        if p.search(q):
            return QueryType.CHAPTER

    # Summary of document
    for p in _SUMMARY_PATTERNS:
        if p.search(q):
            return QueryType.SUMMARY

    # Comparison (before topic, since "difference between X" could overlap)
    for p in _COMPARISON_PATTERNS:
        if p.search(q):
            return QueryType.COMPARISON

    # Troubleshooting
    for p in _TROUBLESHOOT_PATTERNS:
        if p.search(q):
            return QueryType.TROUBLESHOOT

    # Procedure
    for p in _PROCEDURE_PATTERNS:
        if p.search(q):
            return QueryType.PROCEDURE

    # Table (before parameter to avoid overlap)
    for p in _TABLE_PATTERNS:
        if p.search(q):
            return QueryType.TABLE

    # Parameter
    for p in _PARAMETER_PATTERNS:
        if p.search(q):
            return QueryType.PARAMETER

    # Topic (broad topic request)
    for p in _TOPIC_PATTERNS:
        if p.search(q):
            return QueryType.TOPIC

    return None  # Ambiguous — needs LLM Tier 2


# ---------------------------------------------------------------------------
# Tier 2: LLM-Based Classification (~200ms)
# ---------------------------------------------------------------------------

def _llm_classify(query: str) -> QueryType:
    """
    Tier 2: LLM-based classification for ambiguous queries.
    Uses ultra-fast settings: num_predict=20, temperature=0.0.
    Falls back to FACT on any failure.
    """
    try:
        from app.rag.model_loader import get_ollama_generate_url, OLLAMA_MODEL
    except ImportError:
        return QueryType.FACT

    prompt = (
        "Classify this question into exactly ONE category. "
        "Output ONLY the category name, nothing else.\n\n"
        "Categories:\n"
        "- fact: Simple factual question (What is X? Who made Y?)\n"
        "- topic: Broad topic request (Explain USB Communication, Tell me everything about X)\n"
        "- chapter: Chapter-level request (Summarize Chapter 3)\n"
        "- summary: Document summary request (Summarize the manual)\n"
        "- troubleshoot: Problem/error solving (X not working, how to fix Y)\n"
        "- procedure: Step-by-step instructions (How to install X, Setup procedure)\n"
        "- comparison: Comparing two things (Difference between X and Y)\n"
        "- parameter: Parameter/setting lookup (List all parameters, What registers)\n"
        "- table: Table lookup (Show the ratings table, Wiring diagram)\n\n"
        f"Question: {query}\n"
        "Category:"
    )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "num_predict": 20,
            "temperature": 0.0,
            "num_ctx": int(os.getenv("OLLAMA_CONTEXT_LENGTH", "32768")),
        },
    }

    try:
        resp = requests.post(get_ollama_generate_url(), json=payload, timeout=6.0)
        if resp.status_code == 200:
            raw = resp.json().get("response", "").strip().lower()
            # Extract category name
            for qt in QueryType:
                if qt.value in raw:
                    print(f"[QueryClassifier/LLM] '{query}' → {qt.value}")
                    return qt
    except Exception as e:
        print(f"[QueryClassifier/LLM] Failed: {e}")

    return QueryType.FACT


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_query(query: str) -> Tuple[QueryType, RetrievalStrategy]:
    """
    Two-tier query classification for enterprise RAG.

    Tier 1 (always, 0ms): Rule-based regex classification.
    Tier 2 (only if Tier 1 is ambiguous): LLM classification (~200ms).

    Returns:
        (QueryType, RetrievalStrategy) tuple with retrieval configuration.

    Examples:
        "Explain USB Communication"       → (TOPIC, {top_k=40, expand_section=True, ...})
        "What is Register D100?"          → (FACT, {top_k=12, expand_section=False, ...})
        "How to install USB driver"       → (PROCEDURE, {top_k=20, expand_section=True, ...})
        "Summarize the entire manual"     → (SUMMARY, {top_k=60, map_reduce=True, ...})
        "USB not working"                → (TROUBLESHOOT, {top_k=30, expand_section=True, ...})
    """
    if not query or len(query.strip()) < 3:
        qt = QueryType.FACT
        return qt, STRATEGIES[qt]

    # Tier 1: Rule-based
    qt = _rule_based_classify(query)
    if qt is not None:
        strategy = STRATEGIES[qt]
        print(f"[QueryClassifier/Rule] '{query}' → {qt.value} "
              f"(top_k={strategy.top_k}, section={strategy.expand_section}, "
              f"multi_query={strategy.multi_query})")
        return qt, strategy

    # Tier 2: LLM fallback
    qt = _llm_classify(query)
    strategy = STRATEGIES[qt]
    return qt, strategy


def get_strategy(query_type: QueryType) -> RetrievalStrategy:
    """Get the retrieval strategy for a given query type."""
    return STRATEGIES.get(query_type, STRATEGIES[QueryType.FACT])
