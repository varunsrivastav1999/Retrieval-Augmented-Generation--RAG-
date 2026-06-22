"""
=============================================================================
 Enterprise Level RAG: Query Router v2.0
=============================================================================
 4-tier semantic query routing:
   1. EXACT_LOOKUP   → SQL ILIKE / GIN exact match   (~0ms, 0 LLM tokens)
   2. COMPARISON     → SQL row fetch + LLM diff       (~5ms, ~200 LLM tokens)
   3. AGGREGATION    → SQL GROUP BY / COUNT / SUM     (~5ms, 0-50 LLM tokens)
   4. NARRATIVE      → Hybrid RAG + ColBERT + LLM     (~2s, ~2000 LLM tokens)

 Plus the existing routes:
   5. GRAPH          → Neo4j Cypher
   6. RAPTOR         → Hierarchical summary index
   7. SQL_META       → DB metadata / file counts

 Query classification is rule-based (0ms) with LLM fallback for edge cases.
=============================================================================
"""

from __future__ import annotations

import re
import os
import requests
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

from app.rag.model_loader import get_ollama_generate_url, OLLAMA_MODEL

# ---------------------------------------------------------------------------
# Route constants
# ---------------------------------------------------------------------------

class QueryRoute:
    EXACT_LOOKUP  = "exact_lookup"    # SQL ILIKE / GIN exact cell match
    COMPARISON    = "comparison"      # Compare 2+ items (SQL data + LLM diff)
    AGGREGATION   = "aggregation"     # COUNT / LIST / SUM (SQL only)
    NARRATIVE     = "narrative"       # Explain / describe (RAG + LLM)
    GRAPH         = "graph"           # Entity relationships (Neo4j)
    RAPTOR        = "raptor"          # Global/overview (RAPTOR index)
    SQL_META      = "sql"             # File/document metadata counts
    VECTOR        = "vector"          # Generic vector search fallback

    ALL = {EXACT_LOOKUP, COMPARISON, AGGREGATION, NARRATIVE, GRAPH, RAPTOR, SQL_META, VECTOR}


@dataclass
class RouteResult:
    """Result of query classification."""
    route: str
    confidence: float = 1.0           # 0.0–1.0
    catalogue_patterns: list = field(default_factory=list)  # e.g. ["ECL2412SD"]
    numeric_filters: dict = field(default_factory=dict)     # e.g. {"amps": (">=", 200)}
    comparison_entities: list = field(default_factory=list) # e.g. ["ECL2412SD", "SNC3412"]
    needs_llm: bool = False
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Pattern banks
# ---------------------------------------------------------------------------

# Engineering catalogue / model numbers: ECL2412SD, SNC2448L1125, DK10-1A, EQL40200D
_CATALOGUE_RE = re.compile(r'\b([A-Z]{2,5}\d{2,6}[A-Z0-9\-]*)\b')

# Numeric filters: "200A", "24 circuits", "34 inches", "5kV"
_NUMERIC_RE = re.compile(
    r'\b(\d+(?:\.\d+)?)\s*'
    r'(A|amp|amps?|V|volt|volts?|W|watt|kW|kV|in(?:ch(?:es)?)?|mm|cm|m|kg|lb|circuit|slot|phase|Hz|MVA)\b',
    re.IGNORECASE
)

# Exact lookup signals
_EXACT_PATTERNS = [
    r'\b(catalogue\s*(?:no|number|#)?|model\s*(?:no|number|#)?|part\s*(?:no|number|#)?)\b',
    r'\b(door\s*kit|lug\s*data|mounting\s*type|slot\s*qty|main\s*(?:amps?|breaker|lug))\b',
    r'\b(dimension|width|height|depth|weight)\s*of\b',
    r'\b(what\s+is\s+the\s+(?:catalogue|model|part|door))\b',
    r'\b(give\s+me\s+the|find\s+the|show\s+me\s+the)\b.*\b(number|code|value|rating|spec)\b',
    r'\brating\s+of\b',
    r'\bhow\s+(?:much|tall|wide|deep|heavy)\s+is\b',
]

# Comparison signals
_COMPARISON_PATTERNS = [
    r'\b(vs\.?|versus|vs\s|compare[sd]?|comparison|difference|differ)\b',
    r'\b(between\s+.+\s+and|vs\.?\s)',
    r'\b(which\s+(?:has|is|are)\s+(?:more|less|larger|smaller|higher|lower|better|worse))\b',
    r'\b(more\s+circuits?|fewer\s+amps?|larger\s+than|smaller\s+than)\b',
]

# Aggregation signals
_AGGREGATION_PATTERNS = [
    r'\b(list\s+all|show\s+all|give\s+me\s+all|all\s+(?:models?|catalogues?|products?|options?))\b',
    r'\b(how\s+many|count\s+of|number\s+of|total\s+number)\b',
    r'\b(every\s+(?:model|option|type|variant))\b',
    r'\b(available\s+(?:models?|options?|variants?))\b',
    r'\b(which\s+(?:models?|products?)\s+(?:have|support|are\s+rated|come\s+in))\b',
    r'\b(all\s+\d+[-\s]?(?:amp|A|circuit|phase))\b',
]

# Narrative / explanation signals
_NARRATIVE_PATTERNS = [
    r'\b(what\s+is\s+(?:a|an|the)\s+(?!catalogue|part|model))',
    r'\b(how\s+does|how\s+do|how\s+to)\b',
    r'\b(explain|describe|elaborate|tell\s+me\s+about|overview\s+of)\b',
    r'\b(why\s+(?:is|are|do|does|would|should))\b',
    r'\b(difference\s+between\s+(?!ECL|SNC|EQL|SEQ))',  # conceptual, not model comparison
    r'\b(best\s+(?:practice|approach|way)\s+to)\b',
    r'\b(safety|installation|maintenance|troubleshoot)\b',
]

# Graph signals (entity relationships)
_GRAPH_PATTERNS = [
    r'\b(relationship|relation|connected?\s+to|linked?\s+to|associated?\s+with)\b',
    r'\b(who\s+(?:manufactures?|supplies?|distributes?))\b',
    r'\b(network|hierarchy|organisation|org\s+chart)\b',
]

# RAPTOR signals (global/thematic)
_RAPTOR_PATTERNS = [
    r'\b(overall|global|entire|whole|complete)\s+(summary|overview|theme|picture)\b',
    r'\b(what\s+(?:is|are)\s+(?:this|the)\s+(?:document|corpus|catalogue)\s+about)\b',
    r'\b(summarize\s+(?:the\s+)?(?:entire|whole|complete|all))\b',
]

# SQL metadata signals
_SQL_META_PATTERNS = [
    r'\b(how\s+many\s+(?:files?|documents?|pages?))\b',
    r'\b(file\s*(?:type|format|count)|document\s*count|total\s+files?)\b',
    r'\b(what\s+(?:files?|documents?)\s+(?:have\s+been\s+)?(?:uploaded|ingested|indexed))\b',
]


def _compile(patterns: list) -> re.Pattern:
    return re.compile("|".join(patterns), re.IGNORECASE)


_EXACT_RE       = _compile(_EXACT_PATTERNS)
_COMPARISON_RE  = _compile(_COMPARISON_PATTERNS)
_AGGREGATION_RE = _compile(_AGGREGATION_PATTERNS)
_NARRATIVE_RE   = _compile(_NARRATIVE_PATTERNS)
_GRAPH_RE       = _compile(_GRAPH_PATTERNS)
_RAPTOR_RE      = _compile(_RAPTOR_PATTERNS)
_SQL_META_RE    = _compile(_SQL_META_PATTERNS)


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

def classify_query_full(query: str) -> RouteResult:
    """
    Classify a query into its optimal execution route.
    Returns a RouteResult with route, confidence, extracted patterns, and LLM flag.

    Decision order (highest specificity first):
    1. Graph patterns         → GRAPH
    2. RAPTOR patterns        → RAPTOR
    3. SQL meta patterns      → SQL_META
    4. Catalogue numbers      → EXACT_LOOKUP (highest priority for catalogues)
    5. Comparison signals     → COMPARISON
    6. Aggregation signals    → AGGREGATION
    7. Exact lookup signals   → EXACT_LOOKUP
    8. Narrative signals      → NARRATIVE
    9. Default                → VECTOR (hybrid RAG)
    """
    q = query.strip()
    catalogues = _CATALOGUE_RE.findall(q)
    numerics   = _extract_numeric_filters(q)

    # 1. Infrastructure/meta routes (highest override)
    if _GRAPH_RE.search(q):
        return RouteResult(route=QueryRoute.GRAPH, confidence=0.9, reasoning="graph relationship patterns")

    if _RAPTOR_RE.search(q):
        return RouteResult(route=QueryRoute.RAPTOR, confidence=0.9, reasoning="global summary patterns")

    if _SQL_META_RE.search(q):
        return RouteResult(route=QueryRoute.SQL_META, confidence=0.9, reasoning="document metadata patterns")

    # 2. Catalogue numbers detected → most likely exact or comparison
    if catalogues:
        if _COMPARISON_RE.search(q) or len(catalogues) >= 2:
            return RouteResult(
                route=QueryRoute.COMPARISON,
                confidence=0.95,
                catalogue_patterns=catalogues,
                numeric_filters=numerics,
                needs_llm=True,
                comparison_entities=catalogues,
                reasoning=f"catalogue patterns {catalogues} + comparison signals",
            )
        # Single catalogue + no comparison → exact lookup
        return RouteResult(
            route=QueryRoute.EXACT_LOOKUP,
            confidence=0.98,
            catalogue_patterns=catalogues,
            numeric_filters=numerics,
            needs_llm=False,
            reasoning=f"catalogue pattern detected: {catalogues}",
        )

    # 3. Comparison without explicit catalogue numbers
    if _COMPARISON_RE.search(q):
        return RouteResult(
            route=QueryRoute.COMPARISON,
            confidence=0.85,
            numeric_filters=numerics,
            needs_llm=True,
            reasoning="comparison signals without explicit catalogue numbers",
        )

    # 4. Aggregation
    if _AGGREGATION_RE.search(q):
        return RouteResult(
            route=QueryRoute.AGGREGATION,
            confidence=0.85,
            numeric_filters=numerics,
            needs_llm=len(numerics) == 0,  # pure aggregation needs no LLM
            reasoning="aggregation/listing patterns",
        )

    # 5. Exact lookup without catalogue number (dimension, weight, rating of known product)
    if _EXACT_RE.search(q) and numerics:
        return RouteResult(
            route=QueryRoute.EXACT_LOOKUP,
            confidence=0.80,
            numeric_filters=numerics,
            needs_llm=False,
            reasoning="exact lookup signals with numeric filter",
        )

    # 6. Pure narrative/explanation
    if _NARRATIVE_RE.search(q):
        return RouteResult(
            route=QueryRoute.NARRATIVE,
            confidence=0.80,
            needs_llm=True,
            reasoning="narrative/explanation signals",
        )

    # 7. Default: hybrid vector search
    return RouteResult(
        route=QueryRoute.VECTOR,
        confidence=0.5,
        catalogue_patterns=catalogues,
        numeric_filters=numerics,
        needs_llm=True,
        reasoning="no strong signal detected, falling back to vector search",
    )


def _extract_numeric_filters(query: str) -> dict:
    """Extract numeric constraints from a query string.
    Returns a dict like {"amps": (">=", 200), "circuits": ("=", 24)}
    """
    filters = {}
    for match in _NUMERIC_RE.finditer(query):
        value_str, unit = match.group(1), match.group(2).lower()
        value = float(value_str)

        # Determine operator from context (word before the number)
        start = match.start()
        prefix = query[max(0, start - 20):start].lower()

        if any(w in prefix for w in ["at least", "minimum", "more than", ">="]):
            op = ">="
        elif any(w in prefix for w in ["at most", "maximum", "less than", "under", "<="]):
            op = "<="
        elif any(w in prefix for w in ["exactly", "equal to"]):
            op = "="
        else:
            op = "="  # default exact match

        # Normalize unit to canonical field name
        unit_map = {
            "a": "amps", "amp": "amps", "amps": "amps",
            "v": "volts", "volt": "volts", "volts": "volts",
            "circuit": "circuits", "circuits": "circuits",
            "slot": "slots", "slots": "slots",
            "phase": "phases", "phases": "phases",
            "in": "width_in", "inch": "width_in", "inches": "width_in",
            "w": "watts", "kw": "kw", "kv": "kv",
        }
        canonical = unit_map.get(unit, unit)
        filters[canonical] = (op, value)

    return filters


# ---------------------------------------------------------------------------
# Backward-compatible Router class (wraps classify_query_full)
# ---------------------------------------------------------------------------

class Router:
    """
    Enhanced two-tier query router v2.0.
    Backward-compatible with the original Router interface.
    Now routes to 8 destinations including exact_lookup, comparison, aggregation.
    """

    TIERS = QueryRoute.ALL

    def route_query(self, query: str) -> str:
        """Return the route string (backward-compatible)."""
        result = classify_query_full(query)
        print(f"[Router] {result.route.upper()} (confidence={result.confidence:.2f}) — {result.reasoning}")
        return result.route

    def route_query_full(self, query: str) -> RouteResult:
        """Return the full RouteResult with all extracted metadata."""
        result = classify_query_full(query)
        print(f"[Router] {result.route.upper()} (confidence={result.confidence:.2f}) — {result.reasoning}")
        return result


# Global singleton (backward-compatible)
query_router = Router()


# ---------------------------------------------------------------------------
# Execution strategy helper
# ---------------------------------------------------------------------------

def get_execution_strategy(route_result: RouteResult) -> dict:
    """
    Given a RouteResult, return the execution strategy dict used by main.py.
    Describes what sub-systems to call and in what order.
    """
    r = route_result.route

    if r == QueryRoute.EXACT_LOOKUP:
        return {
            "primary": "sql_exact",
            "fallback": "hybrid_search",
            "needs_rerank": False,
            "needs_llm": False,
            "max_llm_tokens": 0,
            "sql_params": {
                "catalogue_patterns": route_result.catalogue_patterns,
                "numeric_filters": route_result.numeric_filters,
            },
        }
    elif r == QueryRoute.COMPARISON:
        return {
            "primary": "sql_exact",
            "fallback": "hybrid_search",
            "needs_rerank": True,
            "needs_llm": True,
            "max_llm_tokens": 300,
            "sql_params": {
                "catalogue_patterns": route_result.catalogue_patterns,
                "numeric_filters": route_result.numeric_filters,
            },
        }
    elif r == QueryRoute.AGGREGATION:
        return {
            "primary": "sql_aggregate",
            "fallback": "hybrid_search",
            "needs_rerank": False,
            "needs_llm": False,
            "max_llm_tokens": 100,
            "sql_params": {
                "numeric_filters": route_result.numeric_filters,
            },
        }
    elif r == QueryRoute.NARRATIVE:
        return {
            "primary": "hybrid_search",
            "fallback": None,
            "needs_rerank": True,
            "needs_llm": True,
            "max_llm_tokens": 2000,
            "sql_params": {},
        }
    elif r == QueryRoute.GRAPH:
        return {
            "primary": "graph",
            "fallback": "hybrid_search",
            "needs_rerank": False,
            "needs_llm": True,
            "max_llm_tokens": 500,
            "sql_params": {},
        }
    elif r == QueryRoute.RAPTOR:
        return {
            "primary": "raptor",
            "fallback": "hybrid_search",
            "needs_rerank": False,
            "needs_llm": True,
            "max_llm_tokens": 2000,
            "sql_params": {},
        }
    elif r == QueryRoute.SQL_META:
        return {
            "primary": "sql_meta",
            "fallback": None,
            "needs_rerank": False,
            "needs_llm": False,
            "max_llm_tokens": 0,
            "sql_params": {},
        }
    else:  # VECTOR fallback
        return {
            "primary": "hybrid_search",
            "fallback": None,
            "needs_rerank": True,
            "needs_llm": True,
            "max_llm_tokens": 1500,
            "sql_params": {},
        }
