"""
=============================================================================
 Enterprise Level RAG v6.0: Multi-Query Topic Expander
=============================================================================
 For topic-level queries, generates multiple sub-queries covering all aspects
 of the topic. Two-tier expansion:

   Tier 1: TOC-based expansion (0ms) — use document section titles as sub-queries
   Tier 2: LLM-based expansion (~300ms) — generate sub-queries via LLM

 Example:
   "USB Communication" →
     - "USB driver installation"
     - "USB cable connection setup"  
     - "USB configuration parameters"
     - "USB commands and protocol"
     - "USB troubleshooting and error codes"
     - "USB limitations and compatibility"

 100% offline, fail-safe.
=============================================================================
"""

import os
import re
import json
import requests
from typing import Any, Dict, List, Optional

from app.rag.model_loader import get_ollama_generate_url, OLLAMA_MODEL


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MULTI_QUERY_COUNT = int(os.getenv("RAG_TOPIC_MULTI_QUERY_COUNT", "6"))


# ---------------------------------------------------------------------------
# Tier 1: TOC-Based Expansion (0ms)
# ---------------------------------------------------------------------------

def expand_from_sections(
    query: str,
    tenant_id: str,
    db=None,
    max_queries: int = MULTI_QUERY_COUNT,
) -> List[str]:
    """
    Tier 1: Use document section titles to generate sub-queries.
    Searches the document_sections table for titles matching the query topic.
    
    Returns list of sub-queries based on actual document structure.
    """
    if db is None:
        return [query]

    try:
        from app.rag.model_loader import get_embedding_model_id
        from sqlalchemy import text as sql_text

        embedding_model = get_embedding_model_id()

        # Extract the topic keywords from the query
        topic_keywords = _extract_topic_keywords(query)
        if not topic_keywords:
            return [query]

        # Search for section titles matching topic keywords
        keyword_pattern = "%".join(topic_keywords[:3])  # Top 3 keywords
        result = db.execute(sql_text("""
            SELECT DISTINCT title, heading_path, level
            FROM document_sections
            WHERE tenant_id = :tenant_id
              AND (
                title ILIKE :pattern
                OR heading_path::text ILIKE :pattern
              )
            ORDER BY level, title
            LIMIT :limit
        """), {
            "tenant_id": tenant_id,
            "pattern": f"%{keyword_pattern}%",
            "limit": max_queries * 2,
        }).fetchall()

        if not result:
            # Broader search: any keyword matches
            conditions = " OR ".join(
                [f"title ILIKE '%{kw}%'" for kw in topic_keywords[:3]]
            )
            result = db.execute(sql_text(f"""
                SELECT DISTINCT title, heading_path, level
                FROM document_sections
                WHERE tenant_id = :tenant_id
                  AND ({conditions})
                ORDER BY level, title
                LIMIT :limit
            """), {
                "tenant_id": tenant_id,
                "limit": max_queries * 2,
            }).fetchall()

        if result:
            sub_queries = [query]  # Always include original
            seen = {query.lower()}
            for row in result:
                title = row[0] or ""
                if title.lower() not in seen and len(title) > 3:
                    sub_queries.append(title)
                    seen.add(title.lower())

            if len(sub_queries) > 1:
                print(f"[MultiQuery/TOC] '{query}' → {len(sub_queries)} sub-queries from TOC")
                return sub_queries[:max_queries + 1]  # +1 for original

    except Exception as e:
        print(f"[MultiQuery/TOC] Section-based expansion failed: {e}")

    return [query]


# ---------------------------------------------------------------------------
# Tier 2: LLM-Based Expansion (~300ms)
# ---------------------------------------------------------------------------

def expand_with_llm(
    query: str,
    max_queries: int = MULTI_QUERY_COUNT,
) -> List[str]:
    """
    Tier 2: Use LLM to generate sub-queries covering all aspects of a topic.
    Uses num_predict=120, temperature=0.3 for creative but focused expansion.
    
    Returns list: [original_query, sub_query_1, sub_query_2, ...]
    """
    prompt = (
        f"You are a technical documentation search assistant. "
        f"The user wants COMPLETE information about a topic from a technical manual. "
        f"Generate {max_queries} specific search queries that together would retrieve "
        f"ALL relevant sections about this topic.\n\n"
        f"Cover different aspects like: setup, configuration, commands, parameters, "
        f"troubleshooting, limitations, error codes, examples, wiring, specifications.\n\n"
        f"Output a JSON array of strings ONLY, no explanation.\n\n"
        f"Topic: {query}"
    )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "num_predict": 120,
            "temperature": 0.3,
            "num_ctx": int(os.getenv("OLLAMA_CONTEXT_LENGTH", "32768")),
        },
    }

    try:
        resp = requests.post(get_ollama_generate_url(), json=payload, timeout=10.0)
        if resp.status_code == 200:
            raw = resp.json().get("response", "").strip()
            # Extract JSON array
            if "```" in raw:
                match = re.search(r'\[.*?\]', raw, re.DOTALL)
                raw = match.group(0) if match else "[]"
            else:
                match = re.search(r'\[.*?\]', raw, re.DOTALL)
                if match:
                    raw = match.group(0)

            sub_queries = json.loads(raw)
            if isinstance(sub_queries, list) and all(isinstance(q, str) for q in sub_queries):
                result = [query] + [sq.strip() for sq in sub_queries if sq.strip()]
                print(f"[MultiQuery/LLM] '{query}' → {len(result)} sub-queries")
                return result[:max_queries + 1]

    except Exception as e:
        print(f"[MultiQuery/LLM] LLM expansion failed: {e}")

    return [query]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def expand_topic_queries(
    query: str,
    tenant_id: str,
    db=None,
    max_queries: int = MULTI_QUERY_COUNT,
    use_llm: bool = True,
) -> List[str]:
    """
    Two-tier topic-aware multi-query expansion.
    
    Tier 1 (always, 0ms): TOC-based expansion from document_sections table.
    Tier 2 (only if Tier 1 returns ≤1 result): LLM-based expansion.
    
    Returns deduplicated list: [original_query, sub_q1, sub_q2, ...]
    Always includes the original query as first element.
    
    Args:
        query: The user's original query
        tenant_id: Tenant isolation
        db: SQLAlchemy session for TOC lookup
        max_queries: Maximum number of sub-queries to generate
        use_llm: Whether to use LLM Tier 2 if Tier 1 fails
        
    Returns:
        List of queries covering all aspects of the topic
    """
    if not query or len(query.strip()) < 3:
        return [query]

    # Tier 1: TOC-based expansion
    sub_queries = expand_from_sections(query, tenant_id, db, max_queries)
    if len(sub_queries) > 1:
        return _deduplicate_queries(sub_queries, max_queries + 1)

    # Tier 2: LLM-based expansion
    if use_llm:
        sub_queries = expand_with_llm(query, max_queries)
        return _deduplicate_queries(sub_queries, max_queries + 1)

    # Fallback: keyword-based expansion
    return _keyword_expand(query, max_queries)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_topic_keywords(query: str) -> List[str]:
    """Extract meaningful topic keywords from a query, filtering stopwords."""
    stopwords = {
        "explain", "describe", "tell", "give", "show", "list", "what", "how",
        "all", "every", "each", "complete", "full", "entire", "detailed",
        "about", "information", "info", "details", "the", "a", "an", "is",
        "are", "was", "were", "me", "please", "everything", "related", "to",
        "of", "in", "on", "for", "with", "from", "and", "or", "that", "this",
    }
    words = re.findall(r'[a-zA-Z0-9]+', query.lower())
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    return keywords


def _keyword_expand(query: str, max_queries: int) -> List[str]:
    """Simple keyword-based expansion as last resort."""
    keywords = _extract_topic_keywords(query)
    if not keywords:
        return [query]

    topic = " ".join(keywords[:3])
    aspects = [
        f"{topic} configuration",
        f"{topic} installation setup",
        f"{topic} commands",
        f"{topic} troubleshooting errors",
        f"{topic} parameters settings",
        f"{topic} specifications limitations",
    ]
    result = [query] + aspects[:max_queries]
    return result


def _deduplicate_queries(queries: List[str], max_count: int) -> List[str]:
    """Remove duplicate or near-duplicate queries, preserving order."""
    seen: set = set()
    unique: List[str] = []
    for q in queries:
        normalized = q.strip().lower()
        if normalized not in seen and normalized:
            seen.add(normalized)
            unique.append(q.strip())
    return unique[:max_count]
