"""
=============================================================================
 Enterprise Level RAG: Layer 13 — Query Intelligence Engine
=============================================================================
 World-Best RAG Techniques:
  - FLARE: Forward-looking Active Retrieval Augmented Generation
  
 This layer ensures ANY type of question returns the most accurate answer,
 even if the user's query is poorly worded or complex.
=============================================================================
"""

import re
from typing import Dict, List, Optional

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
