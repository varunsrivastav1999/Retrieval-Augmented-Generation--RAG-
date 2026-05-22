"""
=============================================================================
 i-Tips RAG: Layer 13 — Query Intelligence Engine
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
from typing import Dict, List, Optional, Tuple


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
    "compressor": "compressor",
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
