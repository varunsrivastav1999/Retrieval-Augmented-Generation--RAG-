"""
=============================================================================
 Enterprise Level RAG: Layer 13 — Query Intelligence Engine
=============================================================================
 World-Best RAG Techniques:
  1. Query Expansion — automatically adds synonyms & related terms
  2. Query Decomposition — breaks complex questions into sub-queries
  3. Spelling Correction — fixes typos before searching
  4. Multi-Hop Reasoning — chains multiple chunks for complex answers
  5. Reciprocal Rank Fusion across sub-queries
  
 This layer ensures ANY type of question returns the most accurate answer,
 even if the user's query is poorly worded or complex.
=============================================================================
"""

import re
import json
from typing import Dict, List, Optional, Tuple
import requests
from app.rag.model_loader import get_ollama_generate_url, OLLAMA_MODEL


# ---------------------------------------------------------------------------
# Spelling / Typo Correction (offline, no API)
# ---------------------------------------------------------------------------
# Common industrial term corrections (expandable)
TERM_CORRECTIONS = {
    # Common typos → correct terms
    "senser": "sensor",
    "sensar": "sensor",
    "sesnor": "sensor",
    "moter": "motor",
    "mottor": "motor",
    "valv": "valve",
    "valev": "valve",
    "presure": "pressure",
    "pressur": "pressure",
    "temprature": "temperature",
    "temperture": "temperature",
    "tempreture": "temperature",
    "maintanance": "maintenance",
    "maintainence": "maintenance",
    "maintence": "maintenance",
    "calibartion": "calibration",
    "calibraton": "calibration",
    "hidraulic": "hydraulic",
    "hydralic": "hydraulic",
    "pneumtic": "pneumatic",
    "pnuematic": "pneumatic",
    "electic": "electric",
    "electircal": "electrical",
    "voltege": "voltage",
    "amphere": "ampere",
    "frequecy": "frequency",
    "torqe": "torque",
    "diagramm": "diagram",
    "specfication": "specification",
    "specifcation": "specification",
    "instlation": "installation",
    "instalation": "installation",
    "troble": "trouble",
    "troublshooting": "troubleshooting",
    "trobleshoot": "troubleshoot",
    "proceedure": "procedure",
    "procedur": "procedure",
    "safty": "safety",
    "saftey": "safety",
    "waranty": "warranty",
    "warenty": "warranty",
    "circit": "circuit",
    "circut": "circuit",
    "compresser": "compressor",
    "compresor": "compressor",
    "coolent": "coolant",
    "exhust": "exhaust",
    "filtter": "filter",
    "genrator": "generator",
    "pumpe": "pump",
    "conveyer": "conveyor",
    "robbot": "robot",
    "robet": "robot",
    "weldng": "welding",
    "paintng": "painting",
    "assemby": "assembly",
    "assmbly": "assembly",
}


def correct_query_spelling(query: str) -> str:
    """Fix common typos in user queries using offline dictionary."""
    words = query.split()
    corrected = []
    for word in words:
        lower = word.lower()
        if lower in TERM_CORRECTIONS:
            corrected.append(TERM_CORRECTIONS[lower])
        else:
            corrected.append(word)
    result = " ".join(corrected)
    if result != query:
        print(f"[QueryIntel] Spelling corrected: '{query}' → '{result}'")
    return result


# ---------------------------------------------------------------------------
# Query Expansion — add synonyms for better recall
# ---------------------------------------------------------------------------
SYNONYM_MAP = {
    "error": ["fault", "failure", "alarm", "warning", "issue", "problem"],
    "fix": ["repair", "resolve", "solution", "troubleshoot", "correct"],
    "start": ["begin", "initiate", "startup", "power on", "activate"],
    "stop": ["halt", "shutdown", "deactivate", "turn off", "cease"],
    "speed": ["rpm", "velocity", "rate", "frequency"],
    "hot": ["overheating", "temperature high", "thermal"],
    "cold": ["cooling", "temperature low", "freeze"],
    "noise": ["vibration", "sound", "rattle", "hum", "buzz"],
    "leak": ["leakage", "drip", "seepage", "overflow"],
    "broken": ["damaged", "failed", "defective", "faulty", "malfunction"],
    "replace": ["change", "swap", "substitute", "renewal"],
    "check": ["inspect", "verify", "examine", "test", "diagnose"],
    "install": ["mount", "setup", "fit", "attach", "connect"],
    "remove": ["detach", "disconnect", "uninstall", "disassemble"],
    "clean": ["wash", "purge", "flush", "decontaminate"],
    "adjust": ["calibrate", "tune", "set", "configure", "align"],
    "measure": ["gauge", "meter", "reading", "value", "level"],
    "manual": ["guide", "handbook", "documentation", "instruction"],
    "part": ["component", "spare", "piece", "element"],
    "power": ["supply", "voltage", "current", "watt", "energy"],
    "oil": ["lubricant", "lubrication", "grease"],
    "pipe": ["tube", "hose", "line", "duct", "conduit"],
    "switch": ["relay", "contactor", "breaker", "toggle"],
    "display": ["screen", "panel", "indicator", "monitor", "HMI"],
    "robot": ["manipulator", "arm", "axis", "servo"],
    "weld": ["welding", "spot weld", "arc weld", "seam weld"],
    "paint": ["painting", "coating", "spray", "booth"],
    "conveyor": ["belt", "chain", "roller", "transfer"],
}


def expand_query(query: str, max_expansions: int = 3) -> str:
    """
    Expand user query with synonyms to improve recall.
    Only adds terms NOT already in the query.
    """
    query_lower = query.lower()
    expansions = []

    for term, synonyms in SYNONYM_MAP.items():
        if term in query_lower.split():
            for syn in synonyms[:2]:  # Max 2 synonyms per term
                if syn.lower() not in query_lower:
                    expansions.append(syn)
                    if len(expansions) >= max_expansions:
                        break
        if len(expansions) >= max_expansions:
            break

    if expansions:
        expanded = f"{query} {' '.join(expansions)}"
        print(f"[QueryIntel] Expanded: '{query}' → '{expanded}'")
        return expanded

    return query


# ---------------------------------------------------------------------------
# Query Decomposition — break complex questions into sub-queries
# ---------------------------------------------------------------------------
MULTI_PART_PATTERNS = [
    # "X and Y" → two sub-queries
    re.compile(r"(.+?)\s+and\s+(.+)", re.IGNORECASE),
    # "X or Y" → two sub-queries
    re.compile(r"(.+?)\s+or\s+(.+)", re.IGNORECASE),
    # "X, also Y" → two sub-queries
    re.compile(r"(.+?),\s*also\s+(.+)", re.IGNORECASE),
    # "X. Also Y" → two sub-queries
    re.compile(r"(.+?)\.\s*Also\s+(.+)", re.IGNORECASE),
]

# Words that when followed by "and" indicate a list, not separate questions
CONJUNCTION_SAFE_WORDS = {
    "pros", "cons", "advantages", "disadvantages", "input", "output",
    "start", "stop", "left", "right", "up", "down", "on", "off",
    "high", "low", "minimum", "maximum", "min", "max",
}


def decompose_query(query: str) -> List[str]:
    """
    Break complex multi-part questions into sub-queries.
    Returns list of sub-queries (may be just the original if not decomposable).
    """
    # Don't decompose short queries
    if len(query.split()) <= 6:
        return [query]

    # Check for multi-part patterns
    for pattern in MULTI_PART_PATTERNS:
        match = pattern.match(query)
        if match:
            part1, part2 = match.group(1).strip(), match.group(2).strip()

            # Skip if it's a safe conjunction ("pros and cons", "start and stop")
            last_word = part1.split()[-1].lower() if part1 else ""
            first_word = part2.split()[0].lower() if part2 else ""
            if last_word in CONJUNCTION_SAFE_WORDS or first_word in CONJUNCTION_SAFE_WORDS:
                return [query]

            # Only decompose if both parts are substantial
            if len(part1.split()) >= 3 and len(part2.split()) >= 3:
                print(f"[QueryIntel] Decomposed: '{query}' → ['{part1}', '{part2}']")
                return [part1, part2]

    return [query]


# ---------------------------------------------------------------------------
# Multi-Hop Chunk Chaining — connect related chunks for complex answers
# ---------------------------------------------------------------------------
def chain_related_chunks(
    primary_chunks: List[Dict],
    all_chunks: List[Dict],
    max_hops: int = 2,
) -> List[Dict]:
    """
    Multi-hop reasoning: find chunks that are referenced by or related to
    the primary chunks. This handles questions like:
    "What happens when sensor X fails?" → needs the sensor chunk + the alarm chunk.
    """
    if not primary_chunks or not all_chunks:
        return primary_chunks

    # Extract key entities from primary chunks
    primary_entities = set()
    for chunk in primary_chunks:
        text = chunk.get("text", "").lower()
        # Extract potential entity references (capitalized words, model numbers, codes)
        words = re.findall(r'\b[A-Z][A-Za-z0-9_-]{2,}\b', chunk.get("text", ""))
        primary_entities.update(w.lower() for w in words)

    if not primary_entities:
        return primary_chunks

    # Find related chunks that mention the same entities
    chained_ids = {c.get("id") for c in primary_chunks}
    related = []

    for chunk in all_chunks:
        if chunk.get("id") in chained_ids:
            continue
        chunk_text = chunk.get("text", "").lower()
        # Count entity overlaps
        overlap = sum(1 for entity in primary_entities if entity in chunk_text)
        if overlap >= 2:  # At least 2 shared entities
            chunk["_chain_score"] = overlap
            related.append(chunk)

    # Sort by overlap and take top N
    related.sort(key=lambda x: x.get("_chain_score", 0), reverse=True)
    hop_chunks = related[:max_hops]

    if hop_chunks:
        print(f"[QueryIntel] Multi-hop: chained {len(hop_chunks)} related chunks")

    return primary_chunks + hop_chunks


# ---------------------------------------------------------------------------
# Confidence-Based Answer Ranking
# ---------------------------------------------------------------------------
def compute_answer_confidence(
    query: str,
    chunks: List[Dict],
) -> float:
    """
    Compute overall confidence that the retrieved chunks can answer the query.
    Uses multiple signals:
    - Rerank score distribution
    - Keyword coverage
    - Content density
    Returns 0.0 to 1.0
    """
    if not chunks:
        return 0.0

    # Signal 1: Best rerank score
    rerank_scores = [c.get("rerank_score", 0.0) for c in chunks]
    best_rerank = max(rerank_scores) if rerank_scores else 0.0
    # Normalize rerank score (typically -10 to +10)
    rerank_signal = min(1.0, max(0.0, (best_rerank + 5) / 10))

    # Signal 2: Keyword coverage
    query_words = set(re.findall(r'[a-z0-9]+', query.lower()))
    query_words -= {"what", "is", "the", "a", "an", "how", "to", "do", "can", "in", "of", "for"}
    if query_words:
        all_text = " ".join(c.get("text", "") for c in chunks[:3]).lower()
        matched = sum(1 for w in query_words if w in all_text)
        keyword_signal = matched / len(query_words)
    else:
        keyword_signal = 0.5

    # Signal 3: Content density (more text = more likely to contain answer)
    total_chars = sum(len(c.get("text", "")) for c in chunks[:3])
    density_signal = min(1.0, total_chars / 1000)

    # Weighted combination
    confidence = (0.5 * rerank_signal) + (0.3 * keyword_signal) + (0.2 * density_signal)
    return round(min(1.0, confidence), 4)


# ---------------------------------------------------------------------------
# Master Query Intelligence Pipeline
# ---------------------------------------------------------------------------
def intelligent_query_pipeline(query: str) -> Dict:
    """
    Layer 13: Full Query Intelligence Pipeline.
    1. Correct spelling
    2. Decompose into sub-queries
    3. Expand with synonyms
    
    Returns:
    {
        "original": str,
        "corrected": str,
        "sub_queries": List[str],
        "expanded_queries": List[str],
        "primary_search_query": str,
    }
    """
    # Step 1: Spelling correction
    corrected = correct_query_spelling(query)

    # Step 2: Decompose complex queries
    sub_queries = decompose_query(corrected)

    # Step 3: Expand each sub-query with synonyms
    expanded = [expand_query(sq) for sq in sub_queries]

    # Primary search query = first expanded sub-query
    primary = expanded[0] if expanded else corrected

    return {
        "original": query,
        "corrected": corrected,
        "sub_queries": sub_queries,
        "expanded_queries": expanded,
        "primary_search_query": primary,
    }

def reformulate_query(query: str) -> str:
    """
    CRAG: Reformulate query for a second retrieval attempt if grounding was low.
    Strips question words and extracts keywords.
    """
    query_lower = query.lower()
    
    # Strip common question prefixes
    prefixes = [
        "what is the ", "what is a ", "what is ", "what are ",
        "how to ", "how do i ", "how do you ", "how does ", "how can ",
        "why is ", "why does ", "when is ", "where is ", "where are ",
        "can you explain ", "explain ", "describe ", "tell me about "
    ]
    
    for prefix in prefixes:
        if query_lower.startswith(prefix):
            query_lower = query_lower[len(prefix):]
            break
            
    # Remove question marks and extra spaces
    query_lower = query_lower.replace("?", "").strip()
    
    # Expand what's left
    reformulated = expand_query(query_lower)
    print(f"[CRAG] Reformulated query: '{query}' -> '{reformulated}'")
    return reformulated

def text_to_sql_filters(query: str) -> dict:
    """
    Self-Query Retriever: Uses Ollama to extract metadata filters from the natural language query.
    Extracts 'file_type' (e.g., pdf, docx, txt) and 'page' (integer) if mentioned.
    Returns a dictionary of filters to be applied in the vector search.
    """
    prompt = f"""
    You are a metadata filter extractor.
    Extract the 'file_type' (extension) and 'page' number from this query if present.
    Return ONLY a raw JSON object with keys "file_type" (string or null) and "page" (integer or null).
    Do NOT include any markdown, explanation, or code blocks.
    
    Query: "{query}"
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0}
    }
    try:
        response = requests.post(get_ollama_generate_url(), json=payload, timeout=5)
        if response.status_code == 200:
            text = response.json().get("response", "").strip()
            if text.startswith("```json"): text = text[7:-3].strip()
            elif text.startswith("```"): text = text[3:-3].strip()
            filters = json.loads(text)
            return {k: v for k, v in filters.items() if v is not None}
    except Exception as e:
        print(f"[Self-Query] Filter extraction failed: {e}")
    return {}


# ---------------------------------------------------------------------------
# FLARE: Forward-looking Active Retrieval Augmented Generation
# ---------------------------------------------------------------------------
def flare_query_decomposition(original_query: str, retry_count: int, partial_answer: str = "") -> List[str]:
    """
    FLARE: Generate alternative search queries when initial retrieval fails.
    
    Retry 1: Extract keywords, drop question words (existing CRAG)
    Retry 2: Generate 2 semantically different search angles via LLM
    Retry 3: Decompose into sub-questions, each targeting a different aspect
    
    Returns list of queries to search.
    """
    if retry_count <= 1:
        return [reformulate_query(original_query)]

    if retry_count == 2:
        # LLM-based: generate alternative search angles
        prompt = f"""You are a search query rewriter for a technical document RAG system.
The original query: "{original_query}"
The search returned no relevant results.

Generate exactly 2 alternative search queries that might find the answer in technical documentation.
Each query should use different keywords, synonyms, and phrasing from the original.
Return ONLY a JSON array of strings. No markdown, no explanation.

Example: ["query one", "query two"]
"""
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 200}
        }
        try:
            response = requests.post(get_ollama_generate_url(), json=payload, timeout=10)
            if response.status_code == 200:
                text = response.json().get("response", "").strip()
                if text.startswith("```json"): text = text[7:-3].strip()
                elif text.startswith("```"): text = text[3:-3].strip()
                queries = json.loads(text)
                if isinstance(queries, list) and len(queries) >= 1:
                    print(f"[FLARE] Retry 2: Generated alternative queries: {queries}")
                    return queries[:2]
        except Exception as e:
            print(f"[FLARE] LLM query generation failed: {e}")

        return [reformulate_query(original_query)]

    # Retry 3+: decompose into sub-questions
    prompt = f"""You are a query decomposer for a technical RAG system.
The query: "{original_query}"

Decompose this query into 3 simple fact-seeking sub-questions.
Each sub-question should ask about ONE specific piece of information.
Return ONLY a JSON array of 3 strings. No markdown, no explanation.

Example: ["What is the part number for X?", "What is the torque specification?", "What material is used?"]
"""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.5, "num_predict": 300}
    }
    try:
        response = requests.post(get_ollama_generate_url(), json=payload, timeout=15)
        if response.status_code == 200:
            text = response.json().get("response", "").strip()
            if text.startswith("```json"): text = text[7:-3].strip()
            elif text.startswith("```"): text = text[3:-3].strip()
            queries = json.loads(text)
            if isinstance(queries, list) and len(queries) >= 1:
                print(f"[FLARE] Retry 3: Decomposed into sub-queries: {queries}")
                return queries[:3]
    except Exception as e:
        print(f"[FLARE] LLM decomposition failed: {e}")
    return [reformulate_query(original_query)]


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

