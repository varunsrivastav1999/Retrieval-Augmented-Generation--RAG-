"""
=============================================================================
 Enterprise Level RAG v6.0: Section-Aware Retrieval
=============================================================================
 Solves the core "incomplete topic retrieval" problem:
   
   1. Section Search: Search section_embeddings collection to find matching sections
   2. Parent Section Expansion: Given a matched chunk, retrieve ALL sibling chunks
   3. Section Tree Retrieval: Get all chunks from a section + child subsections
   4. Section-ordered deduplication and merging

 This is the primary retrieval path for TOPIC, CHAPTER, TROUBLESHOOT,
 and PROCEDURE query types.

 100% offline, async-safe, fail-safe.
=============================================================================
"""

import os
from typing import Any, Dict, List, Optional, Set, Tuple

from sqlalchemy.orm import Session
from sqlalchemy import text

from app.database import DocumentChunk, SessionLocal
from app.rag.model_loader import encode_text, get_embedding_model_id, cosine_similarity

try:
    from app.rag.qdrant_client import get_qdrant_client
    from qdrant_client.http import models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENABLE_SECTION_RETRIEVAL = os.getenv("RAG_ENABLE_SECTION_RETRIEVAL", "true").lower() in {"1", "true", "yes", "on"}
MAX_SECTION_CHUNKS = int(os.getenv("RAG_MAX_SECTION_CHUNKS", "40"))
SECTION_SEARCH_TOP_K = int(os.getenv("RAG_SECTION_SEARCH_TOP_K", "5"))
CONTEXT_DEDUP_THRESHOLD = float(os.getenv("RAG_CONTEXT_DEDUP_THRESHOLD", "0.92"))


# ---------------------------------------------------------------------------
# Section-Level Search
# ---------------------------------------------------------------------------

def search_sections(
    query: str,
    tenant_id: str,
    top_k: int = SECTION_SEARCH_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Search the section_embeddings Qdrant collection to find matching sections.
    
    Returns list of section matches with:
      - section_id, title, heading_path, page_start, page_end, score
    """
    if not QDRANT_AVAILABLE or not ENABLE_SECTION_RETRIEVAL:
        return []

    try:
        qdrant = get_qdrant_client()
        query_vector = encode_text(query)
        embedding_model = get_embedding_model_id()

        # Check if section_embeddings collection exists
        try:
            collections = [c.name for c in qdrant.get_collections().collections]
            if "section_embeddings" not in collections:
                return []
        except Exception:
            return []

        qdrant_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(value=tenant_id),
                ),
                models.FieldCondition(
                    key="embedding_model",
                    match=models.MatchValue(value=embedding_model),
                ),
            ]
        )

        results = qdrant.search(
            collection_name="section_embeddings",
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=top_k,
        )

        sections = []
        for point in results:
            payload = point.payload or {}
            sections.append({
                "section_id": payload.get("section_id", str(point.id)),
                "title": payload.get("title", ""),
                "heading_path": payload.get("heading_path", []),
                "page_start": payload.get("page_start"),
                "page_end": payload.get("page_end"),
                "document_id": payload.get("document_id"),
                "document_name": payload.get("document_name"),
                "level": payload.get("level", "section"),
                "score": point.score,
            })

        if sections:
            print(f"[SectionRetriever] Found {len(sections)} matching sections: "
                  f"{[s['title'] for s in sections[:3]]}")

        return sections

    except Exception as e:
        print(f"[SectionRetriever] Section search failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Section Chunk Retrieval (expand section → all its chunks)
# ---------------------------------------------------------------------------

def retrieve_section_chunks(
    section_id: str,
    tenant_id: str,
    db: Optional[Session] = None,
    max_chunks: int = MAX_SECTION_CHUNKS,
    include_children: bool = True,
) -> List[Dict[str, Any]]:
    """
    Retrieve ALL chunks belonging to a section (and optionally its child sections).
    
    This is the core function that solves "incomplete topic retrieval":
    instead of returning top-k chunks, it returns EVERY chunk in the matched section.
    
    Args:
        section_id: The section UUID to retrieve chunks for
        tenant_id: Tenant isolation
        db: SQLAlchemy session (created if not provided)
        max_chunks: Maximum chunks to return from this section
        include_children: Also retrieve chunks from child subsections
        
    Returns:
        List of chunk dicts ordered by section (document order)
    """
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True

    try:
        embedding_model = get_embedding_model_id()

        # Build section_id filter — include child sections if requested
        section_ids = [section_id]
        if include_children:
            child_ids = _get_child_section_ids(section_id, db)
            section_ids.extend(child_ids)

        # Query all chunks with matching section_id
        # Uses the doc_metadata JSON field for section_id lookup
        candidates = []

        for sid in section_ids:
            # Use the dedicated section_id column (v6.0)
            rows = (
                db.query(DocumentChunk)
                .filter(
                    DocumentChunk.tenant_id == tenant_id,
                    DocumentChunk.embedding_model == embedding_model,
                    DocumentChunk.section_id == sid,
                )
                .order_by(DocumentChunk.section)
                .limit(max_chunks)
                .all()
            )

            if not rows:
                # Fallback: search by section_id inside doc_metadata JSON using cast
                try:
                    rows = (
                        db.query(DocumentChunk)
                        .filter(
                            DocumentChunk.tenant_id == tenant_id,
                            DocumentChunk.embedding_model == embedding_model,
                        )
                        .filter(
                            text("doc_metadata->>'section_id' = :sid")
                        ).params(sid=sid)
                        .order_by(DocumentChunk.section)
                        .limit(max_chunks)
                        .all()
                    )
                except Exception:
                    rows = []

            for row in rows:
                candidates.append(_chunk_from_db_row(row))

        # Deduplicate by chunk ID
        seen_ids: Set = set()
        unique_chunks = []
        for chunk in candidates:
            cid = chunk.get("id")
            if cid not in seen_ids:
                seen_ids.add(cid)
                unique_chunks.append(chunk)

        if unique_chunks:
            print(f"[SectionRetriever] Retrieved {len(unique_chunks)} chunks for "
                  f"section {section_id[:8]}... (include_children={include_children})")

        return unique_chunks[:max_chunks]

    except Exception as e:
        print(f"[SectionRetriever] Section chunk retrieval failed: {e}")
        return []
    finally:
        if close_db:
            db.close()


def retrieve_parent_section_chunks(
    chunk: Dict[str, Any],
    tenant_id: str,
    db: Optional[Session] = None,
    max_chunks: int = MAX_SECTION_CHUNKS,
) -> List[Dict[str, Any]]:
    """
    Given a matched chunk, find its parent section and return ALL sibling chunks.
    
    This expands a single matched chunk to include its entire section context,
    solving the problem of fragmented topic retrieval.
    """
    metadata = chunk.get("metadata", {})
    section_id = metadata.get("section_id")

    if section_id:
        return retrieve_section_chunks(section_id, tenant_id, db, max_chunks, include_children=True)

    # Fallback: use heading_path matching
    heading_path = metadata.get("heading_path", [])
    if not heading_path:
        # Last resort: expand by document + section range
        return _expand_by_section_range(chunk, tenant_id, db, max_chunks)

    return _retrieve_by_heading_path(heading_path, tenant_id, db, max_chunks)


def retrieve_by_topic(
    query: str,
    tenant_id: str,
    db: Optional[Session] = None,
    top_sections: int = SECTION_SEARCH_TOP_K,
    max_chunks_per_section: int = MAX_SECTION_CHUNKS,
) -> List[Dict[str, Any]]:
    """
    High-level topic retrieval: search for matching sections, then retrieve
    all chunks from those sections.
    
    This is the primary entry point for TOPIC query types.
    
    Pipeline:
      1. Search section_embeddings → find matching sections
      2. For each matched section → retrieve all chunks
      3. Merge, deduplicate, order by section
    """
    matching_sections = search_sections(query, tenant_id, top_k=top_sections)

    if not matching_sections:
        return []

    all_chunks: List[Dict[str, Any]] = []
    seen_ids: Set = set()

    for section in matching_sections:
        section_chunks = retrieve_section_chunks(
            section["section_id"],
            tenant_id,
            db,
            max_chunks=max_chunks_per_section,
            include_children=True,
        )
        for chunk in section_chunks:
            cid = chunk.get("id")
            if cid not in seen_ids:
                seen_ids.add(cid)
                # Boost score based on section match quality
                chunk["section_match_score"] = section["score"]
                chunk["matched_section_title"] = section["title"]
                all_chunks.append(chunk)

    # Sort by document order (section index)
    all_chunks.sort(key=lambda c: (
        c.get("metadata", {}).get("source", ""),
        c.get("metadata", {}).get("section", 0),
    ))

    print(f"[SectionRetriever] Topic retrieval: {len(all_chunks)} total chunks "
          f"from {len(matching_sections)} sections")

    return all_chunks


# ---------------------------------------------------------------------------
# Chunk Merging & Deduplication
# ---------------------------------------------------------------------------

def merge_and_deduplicate(
    section_chunks: List[Dict[str, Any]],
    direct_chunks: List[Dict[str, Any]],
    dedup_threshold: float = CONTEXT_DEDUP_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    Merge section-retrieved chunks with direct vector search chunks.
    Deduplicate by ID and by semantic similarity.
    
    Priority: section chunks first (complete sections), then direct matches.
    """
    # Phase 1: ID-based deduplication
    seen_ids: Set = set()
    merged: List[Dict[str, Any]] = []

    # Section chunks get priority
    for chunk in section_chunks:
        cid = chunk.get("id")
        if cid is not None and cid not in seen_ids:
            seen_ids.add(cid)
            merged.append(chunk)

    # Add direct chunks that aren't already included
    for chunk in direct_chunks:
        cid = chunk.get("id")
        if cid is not None and cid not in seen_ids:
            seen_ids.add(cid)
            merged.append(chunk)
        elif cid is None:
            # Chunk without ID — add it
            merged.append(chunk)

    # Phase 2: Semantic deduplication (remove near-duplicate text)
    if len(merged) > 1 and dedup_threshold < 1.0:
        merged = _semantic_deduplicate(merged, dedup_threshold)

    return merged


def _semantic_deduplicate(
    chunks: List[Dict[str, Any]],
    threshold: float,
) -> List[Dict[str, Any]]:
    """Remove semantically near-duplicate chunks using text comparison."""
    if len(chunks) <= 1:
        return chunks

    unique: List[Dict[str, Any]] = [chunks[0]]

    for chunk in chunks[1:]:
        text = chunk.get("text", "")
        is_duplicate = False

        for existing in unique:
            existing_text = existing.get("text", "")
            # Quick length-based filter
            if abs(len(text) - len(existing_text)) > max(len(text), len(existing_text)) * 0.3:
                continue

            # Token overlap ratio (faster than embedding comparison)
            overlap = _text_overlap_ratio(text, existing_text)
            if overlap > threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique.append(chunk)

    if len(unique) < len(chunks):
        print(f"[SectionRetriever] Dedup: {len(chunks)} → {len(unique)} chunks "
              f"(removed {len(chunks) - len(unique)} near-duplicates)")

    return unique


def _text_overlap_ratio(text_a: str, text_b: str) -> float:
    """Quick token-level overlap ratio between two texts."""
    if not text_a or not text_b:
        return 0.0
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _chunk_from_db_row(row: DocumentChunk) -> Dict[str, Any]:
    """Convert a DocumentChunk ORM object to a retrieval candidate dict."""
    metadata = row.doc_metadata or {}
    return {
        "id": str(row.id),
        "text": metadata.get("parent_text") or row.text_content or "",
        "score": 0.0,
        "hybrid_score": 0.0,
        "dense_score": 0.0,
        "file_type": row.file_type or "unknown",
        "table_group": metadata.get("table_group"),
        "metadata": {
            "tenant_id": row.tenant_id,
            "source": row.doc_id,
            "section": row.section,
            "type": metadata.get("type", "text"),
            "page_num": metadata.get("page_num"),
            "embedding_model": row.embedding_model,
            "file_type": row.file_type or "unknown",
            "entities": metadata.get("entities", []),
            "table_group": metadata.get("table_group"),
            "table_id": metadata.get("table_id"),
            "section_title": metadata.get("section_title", row.section_title or ""),
            "cell_values": metadata.get("cell_values"),
            "header_path": metadata.get("header_path", []),
            "row_index": metadata.get("row_index"),
            # v6.0 section-aware fields
            "section_id": metadata.get("section_id"),
            "heading_path": metadata.get("heading_path", []),
            "chunk_index_in_section": metadata.get("chunk_index_in_section"),
            "total_chunks_in_section": metadata.get("total_chunks_in_section"),
        },
    }


def _get_child_section_ids(section_id: str, db: Session) -> List[str]:
    """Get IDs of all child sections (recursive) from document_sections table."""
    try:
        # Check if document_sections table exists
        result = db.execute(text(
            "SELECT id FROM document_sections WHERE parent_id = :parent_id"
        ), {"parent_id": section_id}).fetchall()

        child_ids = [row[0] for row in result]
        # Recurse
        grandchildren = []
        for cid in child_ids:
            grandchildren.extend(_get_child_section_ids(cid, db))
        return child_ids + grandchildren
    except Exception:
        # Table doesn't exist yet or query failed
        return []


def _expand_by_section_range(
    chunk: Dict[str, Any],
    tenant_id: str,
    db: Optional[Session],
    max_chunks: int,
) -> List[Dict[str, Any]]:
    """
    Fallback expansion: retrieve chunks around the matched chunk by section index.
    Uses ±20 section range for broad coverage.
    """
    if db is None:
        return []

    metadata = chunk.get("metadata", {})
    doc_id = metadata.get("source")
    section = metadata.get("section")
    embedding_model = metadata.get("embedding_model", get_embedding_model_id())

    if not doc_id or section is None:
        return []

    try:
        neighbors = (
            db.query(DocumentChunk)
            .filter(
                DocumentChunk.tenant_id == tenant_id,
                DocumentChunk.embedding_model == embedding_model,
                DocumentChunk.doc_id == doc_id,
                DocumentChunk.section >= max(0, section - 20),
                DocumentChunk.section <= section + 20,
            )
            .order_by(DocumentChunk.section)
            .limit(max_chunks)
            .all()
        )
        return [_chunk_from_db_row(n) for n in neighbors]
    except Exception as e:
        print(f"[SectionRetriever] Section range expansion failed: {e}")
        return []


def _retrieve_by_heading_path(
    heading_path: List[str],
    tenant_id: str,
    db: Optional[Session],
    max_chunks: int,
) -> List[Dict[str, Any]]:
    """
    Retrieve chunks that share the same heading_path prefix.
    Falls back to section_title matching.
    """
    if not heading_path or db is None:
        return []

    embedding_model = get_embedding_model_id()

    # Try matching by section_title (most common field)
    section_title = heading_path[-1] if heading_path else ""
    if not section_title:
        return []

    try:
        rows = (
            db.query(DocumentChunk)
            .filter(
                DocumentChunk.tenant_id == tenant_id,
                DocumentChunk.embedding_model == embedding_model,
                DocumentChunk.section_title == section_title,
            )
            .order_by(DocumentChunk.section)
            .limit(max_chunks)
            .all()
        )

        if not rows:
            # Broader match: ILIKE on section_title
            rows = (
                db.query(DocumentChunk)
                .filter(
                    DocumentChunk.tenant_id == tenant_id,
                    DocumentChunk.embedding_model == embedding_model,
                    DocumentChunk.section_title.ilike(f"%{section_title}%"),
                )
                .order_by(DocumentChunk.section)
                .limit(max_chunks)
                .all()
            )

        return [_chunk_from_db_row(r) for r in rows]
    except Exception as e:
        print(f"[SectionRetriever] Heading path retrieval failed: {e}")
        return []
