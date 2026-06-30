import os
from collections import defaultdict
from typing import List, Dict, Any
from app.database import DocumentChunk
from app.rag.model_loader import cosine_similarity, encode_text, encode_texts, get_embedding_model_id
try:
    from app.rag.table_engine import assemble_html_table_from_chunks
    TABLE_ENGINE_AVAILABLE = True
except ImportError:
    TABLE_ENGINE_AVAILABLE = False

ENABLE_CONTEXT_EXPANSION = os.getenv("RAG_ENABLE_CONTEXT_EXPANSION", "true").lower() in {"1", "true", "yes", "on"}
BROAD_CONTEXT_SOURCE_LIMIT = int(os.getenv("RAG_BROAD_CONTEXT_SOURCE_LIMIT", "1"))
BROAD_CONTEXT_MAX_CHUNKS = int(os.getenv("RAG_BROAD_CONTEXT_MAX_CHUNKS", "32"))
BROAD_CONTEXT_MAX_CHARS = int(os.getenv("RAG_BROAD_CONTEXT_MAX_CHARS", "26000"))

def apply_mmr(query: str, chunks: list, diversity: float = 0.5) -> list:
    """
    Layer 5: Context Assembly - Apply MMR (Max Marginal Relevance)
    Removes redundant chunks to maximize information diversity.
    
    OPTIMIZED: Uses pre-computed embeddings from retrieval when available,
    falls back to batch encoding only when needed.
    """
    if not chunks or len(chunks) <= 1:
        return chunks
    
    query_emb = encode_text(query)
    
    # Batch encode only the texts (fast: single call to model)
    texts = [c.get("text", "") for c in chunks]
    chunk_embs = encode_texts(texts)
    
    if not chunk_embs:
        return chunks
    
    query_sim = [cosine_similarity(query_emb, chunk_emb) for chunk_emb in chunk_embs]
    
    selected = [query_sim.index(max(query_sim))]
    unselected = [i for i in range(len(chunks)) if i not in selected]
    
    # Limit MMR iterations — diminishing returns beyond top_k
    max_select = min(len(chunks), 20)
    
    while unselected and len(selected) < max_select:
        mmr_scores = []
        for i in unselected:
            sim_to_selected = max([cosine_similarity(chunk_embs[i], chunk_embs[j]) for j in selected])
            mmr_score = (1 - diversity) * query_sim[i] - diversity * sim_to_selected
            mmr_scores.append((mmr_score, i))
            
        best_idx = max(mmr_scores, key=lambda x: x[0])[1]
        selected.append(best_idx)
        unselected.remove(best_idx)
        
    return [chunks[i] for i in selected]

def compress_context(chunk: dict) -> dict:
    """
    Layer 5: Compress chunks > 350 tokens using extractive summarization.
    Ensures we pack the context window efficiently without exceeding limits.
    
    IMPORTANT: Table chunks are NEVER truncated — every column and row matters
    for lookup queries (e.g., "SNC2448L1125 door kit number").

    Limit raised to 3500 chars to preserve full multi-step procedure sequences
    from industrial manuals (numbered steps routinely exceed 2500 chars).
    """
    max_chars = 3500  # Raised from 2500: industrial procedures can exceed 2500 chars
    text = chunk.get("text", "")
    
    if "[TABLE" in text or "| " in text:
        return chunk
    
    if len(text) > max_chars:
        break_point = text.rfind(" ", 0, max_chars)
        if break_point > max_chars // 2:
            chunk["text"] = text[:break_point] + "..."
        else:
            chunk["text"] = text[:max_chars] + "..."
    return chunk

def assemble_context(query: str, reranked_chunks: list, db=None, broad_query: bool = False) -> list:
    """
    Layer 5: Context Assembly
    Target 60-70% fill of context window.
    Attach citation metadata [source, page, section] per chunk.
    Expanded Context: If db is provided, fetch neighboring chunks for top results.
    """
    if not reranked_chunks:
        return []
    
    # 1. Apply MMR to ensure diversity
    mmr_filtered = apply_mmr(query, reranked_chunks)
    
    # 2. Context Window Expansion (Neighboring chunks)
    # Fetch their immediate neighbors (+/- 5 for broad coverage)
    final_context = []
    expanded_ids = set()
    total_chars = 0
    MAX_CONTEXT_CHARS = 100000  # Hard ceiling to strictly protect llama3.1's 32768 token limit
    
    for i, chunk in enumerate(mmr_filtered):
        # Only expand top 5 chunks and only if db is available
        if ENABLE_CONTEXT_EXPANSION and db is not None and i < 5:
            metadata = chunk.get("metadata", {})
            doc_id = metadata.get("source")
            section = metadata.get("section")
            
            if doc_id and section is not None:
                tenant_id = metadata.get("tenant_id", "default")
                embedding_model = metadata.get("embedding_model", get_embedding_model_id())
                try:
                    neighbors = (
                        db.query(DocumentChunk)
                        .filter(
                            DocumentChunk.tenant_id == tenant_id,
                            DocumentChunk.embedding_model == embedding_model,
                            DocumentChunk.doc_id == doc_id,
                            DocumentChunk.section >= section - 5,
                            DocumentChunk.section <= section + 5,
                            DocumentChunk.id.notin_(expanded_ids)
                        )
                        .order_by(DocumentChunk.section)
                        .all()
                    )
                    for n in neighbors:
                        if n.id not in expanded_ids:
                            neighbor_chunk = _candidate_from_chunk(n)
                            compressed = compress_context(neighbor_chunk)
                            text_len = len(compressed.get("text", ""))
                            if total_chars + text_len <= MAX_CONTEXT_CHARS:
                                final_context.append(compressed)
                                expanded_ids.add(n.id)
                                total_chars += text_len
                except Exception as e:
                    print(f"[Context] Expansion failed: {e}")

        chunk_id = chunk.get("id")
        if chunk_id is None or chunk_id not in expanded_ids:
            compressed = compress_context(chunk)
            text_len = len(compressed.get("text", ""))
            if total_chars + text_len <= MAX_CONTEXT_CHARS:
                final_context.append(compressed)
                if chunk_id is not None:
                    expanded_ids.add(chunk_id)
                total_chars += text_len

    if broad_query and db is not None:
        _append_broad_document_context(final_context, expanded_ids, reranked_chunks, db)

    # 3. Attach structured source-label headers to each chunk so the LLM
    #    can anchor citations to exact document sections. We prepend a short
    #    header line to the text itself rather than relying on the LLM to
    #    infer the source from surrounding context.
    for chunk in final_context:
        metadata = chunk.get("metadata", {})
        source = os.path.basename(metadata.get("source", "unknown"))
        page = metadata.get("page_num", "?")
        section = metadata.get("section", "")
        section_title = metadata.get("section_title", "")
        # Build a concise but informative label
        label_parts = [f"Source: {source}", f"Page {page}"]
        if section_title:
            label_parts.append(f"Section: {section_title}")
        elif section is not None and section != "":
            label_parts.append(f"Section: {section}")
        label = "[" + ", ".join(label_parts) + "]"
        chunk["citation"] = label
        # Prepend the label to the text so it's visible to the LLM in the prompt
        existing_text = chunk.get("text", "")
        if not existing_text.startswith("[Source:"):
            chunk["text"] = f"{label}\n{existing_text}"

    return final_context


def build_table_html_context(chunks: list) -> str:
    """
    Given retrieved chunks, separate the table_row chunks, group them by
    table_id, sort by row_index, and return a clean HTML table string
    suitable for injection into the LLM prompt.

    Non-table chunks are returned as markdown text in a separate string.
    Returns (html_table_str, plain_text_str).
    """
    if not TABLE_ENGINE_AVAILABLE:
        return "", ""

    table_chunks = []
    text_chunks = []

    for chunk in chunks:
        meta = chunk.get("metadata", {})
        is_table = (
            meta.get("table_id")
            or meta.get("table_group")
            or meta.get("type") == "table_row"
            or "| " in chunk.get("text", "")[:100]
        )
        if is_table:
            table_chunks.append(chunk)
        else:
            text_chunks.append(chunk)

    html = ""
    if table_chunks:
        try:
            html = assemble_html_table_from_chunks(table_chunks)
        except Exception as e:
            print(f"[Context] HTML table assembly failed: {e}")
            html = ""

    plain = "\n\n".join(c.get("text", "") for c in text_chunks)
    return html, plain

def _append_broad_document_context(final_context: list, expanded_ids: set, seed_chunks: list, db) -> None:
    """For setup/overview questions, include ordered coverage from the best source document."""
    if len(final_context) >= BROAD_CONTEXT_MAX_CHUNKS:
        return

    source_order = []
    source_tenants = {}
    for chunk in seed_chunks:
        metadata = chunk.get("metadata", {})
        source = metadata.get("source")
        if not source or source in source_tenants:
            continue
        source_order.append(source)
        source_tenants[source] = metadata.get("tenant_id", "default")
        if len(source_order) >= BROAD_CONTEXT_SOURCE_LIMIT:
            break

    if not source_order:
        return

    embedding_model = get_embedding_model_id()
    current_chars = sum(len(chunk.get("text", "")) for chunk in final_context)
    remaining_slots = max(0, BROAD_CONTEXT_MAX_CHUNKS - len(final_context))

    for source in source_order:
        if remaining_slots <= 0 or current_chars >= BROAD_CONTEXT_MAX_CHARS:
            break
        tenant_id = source_tenants[source]
        try:
            rows = (
                db.query(DocumentChunk)
                .filter(
                    DocumentChunk.tenant_id == tenant_id,
                    DocumentChunk.embedding_model == embedding_model,
                    DocumentChunk.doc_id == source,
                )
                .order_by(DocumentChunk.section)
                .limit(BROAD_CONTEXT_MAX_CHUNKS * 2)
                .all()
            )
        except Exception as exc:
            print(f"[Context] Broad document coverage failed: {exc}")
            return

        for row in rows:
            if remaining_slots <= 0 or current_chars >= BROAD_CONTEXT_MAX_CHARS:
                return
            if row.id in expanded_ids:
                continue

            candidate = compress_context(_candidate_from_chunk(row))
            text = candidate.get("text", "")
            if current_chars + len(text) > BROAD_CONTEXT_MAX_CHARS and final_context:
                continue

            final_context.append(candidate)
            expanded_ids.add(row.id)
            current_chars += len(text)
            remaining_slots -= 1

def _candidate_from_chunk(chunk) -> dict:
    metadata = chunk.doc_metadata or {}
    return {
        "id": chunk.id,
        "text": chunk.text_content,
        "metadata": {
            "source": chunk.doc_id,
            "section": chunk.section,
            "page_num": metadata.get("page_num"),
            "tenant_id": chunk.tenant_id,
            "type": metadata.get("type", "text"),
            "file_type": metadata.get("file_type", chunk.file_type),
            "embedding_model": chunk.embedding_model,
            "entities": metadata.get("entities", []),
        },
        "quantized_embedding": chunk.quantized_embedding if hasattr(chunk, "quantized_embedding") else None,
    }

def _context_sources(final_context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sources = []
    seen = set()
    for item in final_context:
        metadata = item.get("metadata") or {}
        source_path = metadata.get("source") or "unknown_source"
        source_name = os.path.basename(str(source_path)) or str(source_path)
        page_num = metadata.get("page_num")
        key = (source_name, page_num)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "source": source_name,
                "page": page_num,
                "citation": item.get("citation"),
                "metadata": metadata,
            }
        )
    return sources
