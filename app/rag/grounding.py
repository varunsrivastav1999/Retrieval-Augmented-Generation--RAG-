"""
=============================================================================
 i-Tips RAG: Layers 9 & 10 — Hallucination Guard & Answer Verification
=============================================================================
 Layer 9:  Pre-generation grounding check — refuse to answer if no relevant
           content exists in the documents.
 Layer 10: Post-generation verification — score answer confidence and ensure
           every claim is traceable to a source chunk.

 ZERO HALLUCINATION POLICY:
   If information is NOT in the uploaded documents, the system will say:
   "This information is not available in the uploaded documents."
   It will NEVER give a general/made-up answer.
=============================================================================
"""

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.rag.model_loader import cosine_similarity, encode_text, encode_texts


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Minimum grounding score to proceed with LLM generation
GROUNDING_THRESHOLD = 0.25

# Minimum confidence to return an answer
CONFIDENCE_THRESHOLD = 0.20

# Keywords that lower the threshold (very short/vague queries)
VAGUE_QUERY_WORDS = {"what", "how", "why", "when", "where", "which", "who", "tell", "explain", "describe", "show"}


NOT_FOUND_RESPONSE = (
    "This information is not available in the uploaded documents. "
    "Please upload relevant documents containing this information, or rephrase your question."
)


# ---------------------------------------------------------------------------
# Layer 9: Pre-Generation Grounding Score
# ---------------------------------------------------------------------------
def compute_grounding_score(
    query: str,
    chunks: List[Dict[str, Any]],
    query_embedding: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Compute how well the retrieved chunks actually answer the query.
    
    Returns:
        {
            "score": float (0.0 - 1.0),
            "is_grounded": bool,
            "keyword_overlap": float,
            "semantic_similarity": float,
            "detail": str,
        }
    """
    if not chunks:
        return {
            "score": 0.0,
            "is_grounded": False,
            "keyword_overlap": 0.0,
            "semantic_similarity": 0.0,
            "detail": "No chunks retrieved from knowledge base.",
        }

    # --- 1. Keyword overlap ---
    query_tokens = _extract_tokens(query)
    if not query_tokens:
        return {
            "score": 0.0,
            "is_grounded": False,
            "keyword_overlap": 0.0,
            "semantic_similarity": 0.0,
            "detail": "Empty query.",
        }

    # Check keyword overlap across ALL chunks combined
    all_chunk_text = " ".join(c.get("text", "") for c in chunks).lower()
    all_chunk_tokens = set(_extract_tokens(all_chunk_text))
    
    matched_tokens = [t for t in query_tokens if t in all_chunk_tokens]
    keyword_overlap = len(matched_tokens) / len(query_tokens) if query_tokens else 0.0

    # --- 2. Semantic similarity (top-3 chunks average) ---
    if query_embedding is None:
        query_embedding = encode_text(query)

    chunk_texts = [c.get("text", "") for c in chunks[:5]]  # Top 5 for speed
    if chunk_texts:
        chunk_embeddings = encode_texts(chunk_texts)
        similarities = [cosine_similarity(query_embedding, ce) for ce in chunk_embeddings]
        semantic_similarity = max(similarities) if similarities else 0.0
    else:
        semantic_similarity = 0.0

    # --- 3. Combined score ---
    # Weight: 40% keyword, 60% semantic
    combined_score = (0.4 * keyword_overlap) + (0.6 * semantic_similarity)

    # --- 4. Adjust threshold for vague queries ---
    effective_threshold = GROUNDING_THRESHOLD
    query_lower = query.lower().split()
    if len(query_lower) <= 3 or any(w in VAGUE_QUERY_WORDS for w in query_lower):
        effective_threshold = max(0.15, GROUNDING_THRESHOLD - 0.1)

    is_grounded = combined_score >= effective_threshold

    detail = (
        f"Grounding: {combined_score:.3f} "
        f"(keyword={keyword_overlap:.3f}, semantic={semantic_similarity:.3f}) "
        f"threshold={effective_threshold:.3f} → {'PASS' if is_grounded else 'BLOCKED'}"
    )

    return {
        "score": round(combined_score, 4),
        "is_grounded": is_grounded,
        "keyword_overlap": round(keyword_overlap, 4),
        "semantic_similarity": round(semantic_similarity, 4),
        "detail": detail,
    }


# ---------------------------------------------------------------------------
# Layer 10: Post-Generation Answer Verification
# ---------------------------------------------------------------------------
def verify_answer_grounding(
    answer: str,
    source_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    After LLM generates an answer, verify how well it's grounded in the sources.
    
    Returns:
        {
            "confidence": str ("high" | "medium" | "low"),
            "confidence_score": float (0.0 - 1.0),
            "grounded_sentences": int,
            "total_sentences": int,
            "evidence": List[dict],
        }
    """
    if not answer or not source_chunks:
        return {
            "confidence": "low",
            "confidence_score": 0.0,
            "grounded_sentences": 0,
            "total_sentences": 0,
            "evidence": [],
        }

    # Split answer into sentences
    sentences = _split_sentences(answer)
    if not sentences:
        return {
            "confidence": "low",
            "confidence_score": 0.0,
            "grounded_sentences": 0,
            "total_sentences": 0,
            "evidence": [],
        }

    # Check each sentence against source chunks
    all_source_text = " ".join(c.get("text", "") for c in source_chunks).lower()
    source_tokens = set(_extract_tokens(all_source_text))

    grounded_count = 0
    evidence = []

    for sentence in sentences:
        sentence_tokens = _extract_tokens(sentence)
        if len(sentence_tokens) < 3:
            grounded_count += 1  # Skip very short sentences (formatting, etc.)
            continue

        # Check token overlap with sources
        overlap = sum(1 for t in sentence_tokens if t in source_tokens)
        overlap_ratio = overlap / len(sentence_tokens) if sentence_tokens else 0.0

        if overlap_ratio >= 0.4:
            grounded_count += 1
            # Find the best matching source chunk
            best_match = _find_best_matching_chunk(sentence, source_chunks)
            if best_match:
                evidence.append({
                    "sentence": sentence[:200],
                    "source": best_match.get("citation", ""),
                    "overlap_ratio": round(overlap_ratio, 3),
                })

    # Calculate confidence
    confidence_score = grounded_count / len(sentences) if sentences else 0.0

    if confidence_score >= 0.7:
        confidence = "high"
    elif confidence_score >= 0.4:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "confidence": confidence,
        "confidence_score": round(confidence_score, 4),
        "grounded_sentences": grounded_count,
        "total_sentences": len(sentences),
        "evidence": evidence[:10],  # Limit evidence list size
    }


# ---------------------------------------------------------------------------
# Strict Grounding Prompt Builder
# ---------------------------------------------------------------------------
def build_strict_grounding_prompt(
    question: str,
    context_text: str,
    broad_query: bool = False,
    parent: Optional[str] = None,
    child: Optional[str] = None,
) -> str:
    """
    Build the LLM prompt with STRICT grounding instructions.
    The LLM is explicitly forbidden from using general knowledge.
    """
    topic_hint = " / ".join([v for v in [parent, child] if v])
    topic_line = f"The user is looking at topic area: {topic_hint}.\n" if topic_hint else ""

    broad_instruction = (
        "The user wants a comprehensive response. "
        "Cover every relevant topic present in the context, group by topic, "
        "and do not stop after the first matching paragraph.\n"
        if broad_query
        else ""
    )

    return (
        "═══════════════════════════════════════════════════════════\n"
        "  SYSTEM: i-Tips Technical Intelligence Agent\n"
        "  MODE: STRICT DOCUMENT GROUNDING — ZERO HALLUCINATION\n"
        "═══════════════════════════════════════════════════════════\n\n"
        "CRITICAL RULES:\n"
        "1. ONLY use information from the CONTEXT below. Do NOT use your general knowledge.\n"
        "2. If the answer is NOT in the context, respond EXACTLY with:\n"
        '   "This information is not available in the uploaded documents."\n'
        "3. Do NOT make up, infer, or guess any information not explicitly in the context.\n"
        "4. ABSOLUTELY PROHIBITED: NEVER start your answer with phrases like 'Based on the provided context', 'According to the documents', or 'The context states'. Start directly with the factual answer.\n"
        "5. Do NOT copy, echo, or quote the context block verbatim. Synthesize a concise answer.\n"
        "6. Do NOT include citations like 'From [Document]' in your text. Just output the answer itself.\n"
        "7. Render tables as Markdown Pipe Tables.\n"
        "8. Use bold text for key terms and bullet points for lists.\n"
        "9. Keep answers extremely short and precise — 1 to 2 sentences maximum. Do not over-explain.\n"
        "10. Provide ONLY the answer. Do NOT include conversational filler, greetings, or extra words.\n\n"
        "USER QUERY HANDLING:\n"
        "- The user might use broken English or non-standard terms. Focus on INTENT.\n"
        "- If the user's terms differ from document terms, match them intelligently.\n"
        f"{broad_instruction}"
        f"{topic_line}\n"
        f"CONTEXT FROM KNOWLEDGE BASE:\n{context_text}\n\n"
        f"USER QUESTION: {question}\n\n"
        "ANSWER (using ONLY the context above):"
    )


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------
_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more", "most", "other",
    "some", "such", "no", "only", "own", "same", "than", "too", "very",
    "just", "because", "if", "when", "where", "how", "what", "which", "who",
    "whom", "this", "that", "these", "those", "i", "me", "my", "we", "our",
    "you", "your", "he", "him", "his", "she", "her", "it", "its", "they",
    "them", "their", "about",
}

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _extract_tokens(text: str) -> List[str]:
    """Extract meaningful tokens from text, removing stop words."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Simple sentence splitter
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]


def _find_best_matching_chunk(
    sentence: str,
    chunks: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Find the chunk that best matches a given sentence."""
    sentence_tokens = set(_extract_tokens(sentence))
    if not sentence_tokens:
        return None

    best_chunk = None
    best_overlap = 0.0

    for chunk in chunks:
        chunk_tokens = set(_extract_tokens(chunk.get("text", "")))
        if not chunk_tokens:
            continue
        overlap = len(sentence_tokens & chunk_tokens) / len(sentence_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_chunk = chunk

    return best_chunk if best_overlap > 0.3 else None
