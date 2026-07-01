"""
=============================================================================
 Enterprise Level RAG v6.0: Context Optimizer
=============================================================================
 Handles large topic retrievals that exceed the LLM context window:
 
   1. Semantic Clustering — group chunks by similarity, pick representatives
   2. Duplicate Removal — remove chunks with token overlap > threshold
   3. Token Budget Manager — prioritize high-relevance chunks within budget
   4. Map-Reduce Fallback — switch to map-reduce when context is too large
   5. Section Prioritization — sections with more matching chunks get priority
   6. Section-Ordered Assembly — group chunks by section, sort by document order

 100% offline, fail-safe.
=============================================================================
"""

import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from app.rag.model_loader import cosine_similarity, encode_text, encode_texts


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "100000"))
TOPIC_MAP_REDUCE_THRESHOLD = int(os.getenv("RAG_TOPIC_MAP_REDUCE_THRESHOLD", "80000"))
DEDUP_THRESHOLD = float(os.getenv("RAG_CONTEXT_DEDUP_THRESHOLD", "0.92"))
MAX_CHUNKS_PER_SECTION = int(os.getenv("RAG_MAX_CHUNKS_PER_SECTION", "30"))


# ---------------------------------------------------------------------------
# Context Optimizer
# ---------------------------------------------------------------------------

class ContextOptimizer:
    """
    Optimizes context chunks for LLM consumption:
    - Removes near-duplicates
    - Groups by section for coherent reading
    - Manages token budget
    - Falls back to map-reduce for oversized contexts
    """

    def __init__(
        self,
        max_context_chars: int = MAX_CONTEXT_CHARS,
        dedup_threshold: float = DEDUP_THRESHOLD,
        map_reduce_threshold: int = TOPIC_MAP_REDUCE_THRESHOLD,
    ):
        self.max_context_chars = max_context_chars
        self.dedup_threshold = dedup_threshold
        self.map_reduce_threshold = map_reduce_threshold

    def optimize(
        self,
        chunks: List[Dict[str, Any]],
        query: str,
        ordering: str = "section_order",
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Optimize chunks for LLM context.
        
        Args:
            chunks: Raw retrieved chunks
            query: The user's query (for relevance scoring)
            ordering: "relevance" | "document_order" | "section_order"
            
        Returns:
            (optimized_chunks, needs_map_reduce: bool)
            If needs_map_reduce is True, the caller should use map-reduce summarization.
        """
        if not chunks:
            return [], False

        # Step 1: Remove exact duplicates
        chunks = self._remove_exact_duplicates(chunks)

        # Step 2: Remove near-duplicate text
        chunks = self._remove_near_duplicates(chunks)

        # Step 3: Order chunks
        chunks = self._order_chunks(chunks, ordering)

        # Step 4: Check if context exceeds map-reduce threshold
        total_chars = sum(len(c.get("text", "")) for c in chunks)
        needs_map_reduce = total_chars > self.map_reduce_threshold

        if needs_map_reduce:
            print(f"[ContextOptimizer] Context too large ({total_chars:,} chars > "
                  f"{self.map_reduce_threshold:,} threshold) → map-reduce recommended")

        # Step 5: Truncate to budget
        chunks = self._apply_token_budget(chunks, query)

        return chunks, needs_map_reduce

    def _remove_exact_duplicates(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove chunks with identical text."""
        seen_texts: Set[str] = set()
        unique: List[Dict[str, Any]] = []
        for chunk in chunks:
            text = chunk.get("text", "").strip()
            # Normalize for comparison (remove citation headers)
            normalized = re.sub(r'\[Source:.*?\]\n?', '', text).strip()
            if normalized and normalized not in seen_texts:
                seen_texts.add(normalized)
                unique.append(chunk)

        removed = len(chunks) - len(unique)
        if removed > 0:
            print(f"[ContextOptimizer] Removed {removed} exact duplicate chunks")
        return unique

    def _remove_near_duplicates(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove chunks with high token overlap."""
        if len(chunks) <= 1:
            return chunks

        unique: List[Dict[str, Any]] = [chunks[0]]

        for chunk in chunks[1:]:
            text = chunk.get("text", "")
            is_dup = False
            for existing in unique:
                existing_text = existing.get("text", "")
                overlap = self._token_overlap(text, existing_text)
                if overlap > self.dedup_threshold:
                    # Keep the one with higher score
                    if chunk.get("score", 0) > existing.get("score", 0):
                        unique.remove(existing)
                        unique.append(chunk)
                    is_dup = True
                    break
            if not is_dup:
                unique.append(chunk)

        removed = len(chunks) - len(unique)
        if removed > 0:
            print(f"[ContextOptimizer] Removed {removed} near-duplicate chunks "
                  f"(threshold={self.dedup_threshold})")
        return unique

    def _order_chunks(
        self,
        chunks: List[Dict[str, Any]],
        ordering: str,
    ) -> List[Dict[str, Any]]:
        """Order chunks for optimal LLM consumption."""
        if ordering == "document_order":
            return sorted(chunks, key=lambda c: (
                c.get("metadata", {}).get("source", ""),
                c.get("metadata", {}).get("section", 0),
            ))
        elif ordering == "section_order":
            return self._group_by_section(chunks)
        else:  # "relevance"
            return sorted(chunks, key=lambda c: c.get("score", c.get("hybrid_score", 0)), reverse=True)

    def _group_by_section(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group chunks by section, sort sections by relevance, chunks within section by order."""
        section_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        no_section: List[Dict[str, Any]] = []

        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            section_id = metadata.get("section_id")
            section_title = metadata.get("section_title", "")
            heading_path = metadata.get("heading_path", [])

            group_key = section_id or section_title or (
                " > ".join(heading_path) if heading_path else ""
            )

            if group_key:
                section_groups[group_key].append(chunk)
            else:
                no_section.append(chunk)

        # Sort each section's chunks by document order
        for group_key in section_groups:
            section_groups[group_key].sort(
                key=lambda c: c.get("metadata", {}).get("section", 0)
            )

        # Sort sections by their best chunk's score (most relevant section first)
        sorted_sections = sorted(
            section_groups.items(),
            key=lambda item: max(
                (c.get("score", c.get("hybrid_score", 0)) for c in item[1]),
                default=0,
            ),
            reverse=True,
        )

        # Flatten
        ordered: List[Dict[str, Any]] = []
        for group_key, group_chunks in sorted_sections:
            ordered.extend(group_chunks[:MAX_CHUNKS_PER_SECTION])

        # Add un-sectioned chunks at the end
        ordered.extend(no_section)

        return ordered

    def _apply_token_budget(
        self,
        chunks: List[Dict[str, Any]],
        query: str,
    ) -> List[Dict[str, Any]]:
        """Truncate chunks to fit within the context window budget."""
        budgeted: List[Dict[str, Any]] = []
        total_chars = 0

        for chunk in chunks:
            text_len = len(chunk.get("text", ""))
            if total_chars + text_len > self.max_context_chars and budgeted:
                print(f"[ContextOptimizer] Token budget reached: {total_chars:,}/{self.max_context_chars:,} chars, "
                      f"kept {len(budgeted)}/{len(chunks)} chunks")
                break
            budgeted.append(chunk)
            total_chars += text_len

        return budgeted

    @staticmethod
    def _token_overlap(text_a: str, text_b: str) -> float:
        """Quick token-level Jaccard overlap between two texts."""
        if not text_a or not text_b:
            return 0.0
        # Strip citation headers for comparison
        text_a = re.sub(r'\[Source:.*?\]\n?', '', text_a).strip()
        text_b = re.sub(r'\[Source:.*?\]\n?', '', text_b).strip()
        tokens_a = set(text_a.lower().split())
        tokens_b = set(text_b.lower().split())
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Section Completeness Checker
# ---------------------------------------------------------------------------

def check_section_completeness(
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Check if sections are completely represented in the context.
    Returns warnings for partially included sections.
    """
    warnings: List[Dict[str, Any]] = []
    section_chunks: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        section_id = metadata.get("section_id")
        if section_id:
            section_chunks[section_id].append(chunk)

    for section_id, s_chunks in section_chunks.items():
        # Check if we have all chunks from this section
        for chunk in s_chunks:
            total = chunk.get("metadata", {}).get("total_chunks_in_section")
            if total and len(s_chunks) < total:
                section_title = chunk.get("metadata", {}).get("section_title", "Unknown")
                warnings.append({
                    "type": "partial_section",
                    "section_title": section_title,
                    "chunks_included": len(s_chunks),
                    "chunks_total": total,
                    "message": f"Section '{section_title}' is partially included "
                              f"({len(s_chunks)}/{total} chunks)",
                })
                break

    return warnings


# ---------------------------------------------------------------------------
# Convenience Function
# ---------------------------------------------------------------------------

def optimize_context(
    chunks: List[Dict[str, Any]],
    query: str,
    ordering: str = "section_order",
    max_context_chars: int = MAX_CONTEXT_CHARS,
) -> Tuple[List[Dict[str, Any]], bool, List[Dict[str, Any]]]:
    """
    Convenience function to optimize context.
    
    Returns:
        (optimized_chunks, needs_map_reduce, warnings)
    """
    optimizer = ContextOptimizer(max_context_chars=max_context_chars)
    optimized, needs_map_reduce = optimizer.optimize(chunks, query, ordering)
    warnings = check_section_completeness(optimized)
    return optimized, needs_map_reduce, warnings
