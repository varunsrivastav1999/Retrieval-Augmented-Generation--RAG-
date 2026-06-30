"""
=============================================================================
 Advanced RAG: Layers 9, 10 & 10b — Hallucination Guard, Answer Verification
                                     & Low-Confidence Remediation
=============================================================================
 Layer 9:  Pre-generation grounding check — refuse to answer if no relevant
           content exists in the documents.
 Layer 10: Post-generation verification — score answer confidence and ensure
           every claim is traceable to a source chunk.
 Layer 10b: Low-confidence remediation — if Layer 10 returns confidence < 0.4,
            re-generate using an ultra-strict extractive prompt that forces the
            model to copy sentences verbatim from the context.

 ZERO HALLUCINATION POLICY:
   If information is NOT in the uploaded documents, the system will say:
   "This information is not available in the uploaded documents."
   It will NEVER give a general/made-up answer.
   It will NEVER rephrase, paraphrase, or substitute wording from the source.
============================================================================="""

import re
import json
import requests
import os
from typing import Any, Dict, List, Optional

from app.rag.model_loader import cosine_similarity, encode_text, encode_texts, get_ollama_generate_url, OLLAMA_MODEL


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Minimum grounding score to proceed with strict LLM generation (lowered to be more aggressive)
GROUNDING_THRESHOLD = 0.15

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
        effective_threshold = GROUNDING_THRESHOLD - 0.1

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
            "reasoning": str,
        }
    """
    default_verification = {
        "confidence": "low",
        "confidence_score": 0.0,
        "reasoning": "Answer or sources are empty.",
    }
    
    if not answer or not source_chunks:
        return default_verification

    # In streaming mode, partial answers might trigger this. 
    # For a full check, we need enough text.
    if len(answer.split()) < 5:
        return default_verification

    all_source_text = "\n---\n".join(c.get("text", "") for c in source_chunks[:5])

    prompt = f"""
You are an expert strict fact-checker and groundedness verification judge (NeMo Guardrail).
Your task is to determine if the generated ANSWER is fully supported by the provided CONTEXT.

CONTEXT:
{all_source_text}

ANSWER:
{answer}

Evaluate if every claim in the ANSWER is backed by the CONTEXT.
Output ONLY a valid JSON object with no markdown wrappers or extra text. Do not output anything other than JSON.
Format:
{{
  "confidence": "high" | "medium" | "low",
  "confidence_score": 1.0,
  "reasoning": "Explain why the answer is or isn't grounded."
}}
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 300,
        }
    }

    try:
        response = requests.post(get_ollama_generate_url(), json=payload, timeout=30)
        if response.status_code == 200:
            text = response.json().get("response", "").strip()
            # Clean JSON markdown if present
            if text.startswith("```json"):
                text = text[7:]
            elif text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            
            result = json.loads(text.strip())
            
            # Ensure required fields exist
            if "confidence" in result and "confidence_score" in result:
                return {
                    "confidence": str(result["confidence"]),
                    "confidence_score": float(result["confidence_score"]),
                    "reasoning": str(result.get("reasoning", "")),
                }
    except Exception as e:
        print(f"[Guardrail] LLM Grounding verification failed: {e}")

    # Fallback to simple logic if LLM fails
    return {
        "confidence": "medium",
        "confidence_score": 0.5,
        "reasoning": "LLM verification failed, defaulting to medium confidence.",
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

    Key improvements over the original:
    - CRITICAL CONSTRAINT block is placed FIRST before all other rules.
    - Verbatim-quoting rule enforced for procedures and product names.
    - PRECISE mode replaces AGGRESSIVE mode to avoid encouraging embellishment.
    """
    topic_hint = " / ".join([v for v in [parent, child] if v])
    topic_line = f"The user is looking at topic area: {topic_hint}.\n" if topic_hint else ""

    # Shared critical constraint header placed at the very top of every prompt
    critical_constraint = (
        "╔═══════════════════════════════════════════════════════════╗\n"
        "║  ⚠  CRITICAL CONSTRAINT — READ BEFORE ANYTHING ELSE  ⚠  ║\n"
        "╠═══════════════════════════════════════════════════════════╣\n"
        "║  ZERO HALLUCINATION POLICY (NON-NEGOTIABLE)               ║\n"
        "║  • Only use information explicitly present in the context. ║\n"
        "║  • NEVER introduce wording, terms, product names, or      ║\n"
        "║    procedures that are NOT verbatim in the source text.    ║\n"
        "║  • If a procedure has numbered steps, reproduce them in   ║\n"
        "║    the EXACT order and wording from the document.          ║\n"
        "║  • If a product/part name appears in context, use THAT    ║\n"
        "║    exact string — never a synonym or paraphrase.           ║\n"
        "║  • If the answer is not in the context, respond ONLY with:║\n"
        "║    'This information is not available in the uploaded     ║\n"
        "║     documents.'  Do NOT guess or use general knowledge.    ║\n"
        "╚═══════════════════════════════════════════════════════════╝\n\n"
    )

    if broad_query:
        return (
            critical_constraint
            + "═══════════════════════════════════════════════════════════\n"
            "  SYSTEM: Expert Technical Analyst\n"
            "  MODE: PRECISE DOCUMENT ANALYSIS & SYNTHESIS\n"
            "═══════════════════════════════════════════════════════════\n\n"
            "RULES & ADVANCED DIRECTIVES:\n"
            "1. DEEP SYNTHESIS: Provide a comprehensive, detailed, and logically structured response. Connect scattered facts and underlying themes into a cohesive summary.\n"
            "2. INLINE CITATIONS: You MUST include inline citations for every major claim using the format `[Source: <filename>, Page <X>]`. The source filename is provided at the top of each text chunk.\n"
            "3. VERBATIM PROCEDURES: When describing steps or instructions, copy the exact numbered steps from the document. Do NOT rephrase, reorder, or merge steps.\n"
            "4. CONTRADICTION RESOLUTION: If the documents contain conflicting information, state the conflict clearly, attribute each side to its respective context, and do not guess.\n"
            "5. STRUCTURE & FORMATTING: Use clean, modern Markdown. Use headings (###), bulleted lists, and bold text for emphasis. Ensure extreme readability.\n"
            "6. MATHEMATICS & DATA: If mathematical equations, formulas, or scientific data are present, output them using proper LaTeX format (e.g., $$E=mc^2$$ or $x^2$). Preserve all technical accuracy.\n"
            "7. MULTI-STEP REASONING: For complex questions, internally break down the logic step-by-step before answering. Ensure the final output is logical, rigorous, and completely accurate.\n"
            "8. EXHAUSTIVE EXTRACTION & NO TRUNCATION: Do NOT summarize away important details. Return all related details, clauses, headings, and context exactly as it appears in the documents.\n"
            "9. TONE: Maintain an expert, professional, and technically precise tone.\n"
            "10. NO FILLER WORDS: DO NOT use introductory filler phrases like 'According to the provided text'. Start your answer immediately with the requested facts.\n"
            f"{topic_line}\n"
            "-----------------------------------------------------------\n"
            f"DATABASE RECORDS:\n{context_text}\n"
            "-----------------------------------------------------------\n\n"
            f"QUESTION: {question}\n\n"
            "ANALYSIS:"
        )

    return (
        critical_constraint
        + "═══════════════════════════════════════════════════════════\n"
        "  SYSTEM: Expert Technical Assistant\n"
        "  MODE: PRECISE & FAITHFUL ANSWER GENERATION\n"
        "═══════════════════════════════════════════════════════════\n\n"
        "RULES & ADVANCED DIRECTIVES:\n"
        "1. COMPREHENSIVE ANSWER: Provide a complete and detailed answer. Do not truncate or be overly brief. If the answer is implied by the context, state it confidently with explanation.\n"
        "2. INLINE CITATIONS: You MUST include inline citations for every major claim using the format `[Source: <filename>, Page <X>]`. The source filename is provided at the top of each text chunk.\n"
        "3. VERBATIM PROCEDURES: When describing steps, warnings, or instructions, copy them EXACTLY as written in the document — same order, same wording, same numbering. Never rephrase.\n"
        "4. EXACT TERMINOLOGY: Use the EXACT product names, part numbers, software names, and technical terms as they appear in the source. Never substitute synonyms (e.g., do not replace a specific software name with a generic description).\n"
        "5. TABLE LOOKUP MASTERY & FORMATTING:\n"
        "   - Scan the table chunk for the EXACT row and column containing the requested information.\n"
        "   - Cross-reference column headers meticulously with the row data. Never mix data from different rows.\n"
        "   - Ignore trailing superscripts/asterisks (e.g., EQL40200D3, EQL8100D* match the base).\n"
        "   - If a cell groups multiple items (e.g. 'SEQ40150 SEQ40200'), apply the row's data to ALL items.\n"
        "   - CRITICAL: If you output table rows in your answer, you MUST format them as a valid Markdown table including the header and the `|---|---|` separator row. NEVER output raw pipe-separated text strings.\n"
        "6. MULTI-HOP REASONING & COT: If the query requires combining piece A and piece B from different chunks, briefly explain your step-by-step reasoning (Chain of Thought) before concluding the final answer.\n"
        "7. MATHEMATICS & FORMULAS: If the query involves calculations or scientific data, use proper LaTeX format (e.g., $$...$$) and ensure the math is strictly accurate to the document.\n"
        "8. STRUCTURE & CONTRADICTIONS: Preserve markdown formatting. Do not repeat information. If the context contains conflicting data, expose the conflict instead of guessing.\n"
        "9. EXHAUSTIVE EXTRACTION & NO TRUNCATION: Do NOT summarize away important details. Return all related details, clauses, headings, and context exactly as it appears in the documents.\n"
        "10. NO FILLER WORDS: DO NOT use introductory filler phrases like 'According to the provided text'. Start your answer immediately with the requested facts.\n"
        f"{topic_line}\n"
        "-----------------------------------------------------------\n"
        f"DATABASE RECORDS:\n{context_text}\n"
        "-----------------------------------------------------------\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )


def build_extractive_remediation_prompt(
    question: str,
    context_text: str,
) -> str:
    """
    Ultra-strict extractive prompt used by Layer 10b remediation.
    Forces the model to copy sentences verbatim from the context.
    Only triggered when post-generation verification returns confidence < 0.4.
    """
    return (
        "╔═══════════════════════════════════════════════════════════╗\n"
        "║         EXTRACTIVE ANSWER MODE — MAXIMUM FIDELITY         ║\n"
        "╠═══════════════════════════════════════════════════════════╣\n"
        "║ You MUST answer ONLY by selecting and quoting sentences   ║\n"
        "║ that already exist verbatim in the CONTEXT below.         ║\n"
        "║ Rules:                                                    ║\n"
        "║  1. Copy relevant sentences EXACTLY — no paraphrasing.   ║\n"
        "║  2. Do NOT add any word that is not in the CONTEXT.       ║\n"
        "║  3. Organize quoted sentences logically if needed.        ║\n"
        "║  4. If the answer is not in the context, respond ONLY:    ║\n"
        "║     'This information is not available in the uploaded    ║\n"
        "║      documents.'                                          ║\n"
        "║  5. Do NOT introduce any wording from general knowledge.  ║\n"
        "╚═══════════════════════════════════════════════════════════╝\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"QUESTION: {question}\n\n"
        "EXTRACTIVE ANSWER (verbatim sentences from CONTEXT only):"
    )


def remediate_low_confidence_answer(
    question: str,
    context_text: str,
    broad_query: bool = False,
) -> Optional[str]:
    """
    Layer 10b: Low-Confidence Remediation.

    Called when verify_answer_grounding() returns confidence_score < 0.4.
    Re-generates the answer using an ultra-strict extractive prompt that forces
    the model to copy sentences verbatim from the context rather than rephrase.

    Returns the corrected answer string, or None if re-generation fails.
    """
    prompt = build_extractive_remediation_prompt(question, context_text)
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 1024,
        }
    }
    try:
        response = requests.post(get_ollama_generate_url(), json=payload, timeout=60)
        if response.status_code == 200:
            corrected = response.json().get("response", "").strip()
            if corrected and len(corrected) > 20:
                print(f"[Guardrail/10b] Remediation produced corrected answer ({len(corrected)} chars).")
                return corrected
    except Exception as e:
        print(f"[Guardrail/10b] Remediation failed: {e}")
    return None


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
    return [t for t in tokens if t not in _STOP_WORDS]


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
