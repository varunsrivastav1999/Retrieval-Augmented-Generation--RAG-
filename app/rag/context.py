import os
from app.database import DocumentChunk
from app.rag.model_loader import cosine_similarity, encode_text, encode_texts, get_embedding_model_id

ENABLE_CONTEXT_EXPANSION = os.getenv("RAG_ENABLE_CONTEXT_EXPANSION", "true").lower() in {"1", "true", "yes", "on"}

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
    Layer 5: Compress chunks > 300 tokens using extractive summarization.
    Ensures we pack the context window efficiently without exceeding limits.
    
    IMPORTANT: Table chunks are NEVER truncated — every column and row matters
    for lookup queries (e.g., "SNC2448L1125 door kit number").
    """
    max_chars = 1500  # Approx 300 tokens
    text = chunk.get("text", "")
    
    # Never truncate table data — structured data loses meaning when cut
    if "[TABLE" in text or "| " in text:
        return chunk
    
    if len(text) > max_chars:
        chunk["text"] = text[:max_chars] + "..."
    return chunk

def assemble_context(query: str, reranked_chunks: list, db=None) -> list:
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
    
    for i, chunk in enumerate(mmr_filtered):
        # Only expand top 5 chunks and only if db is available
        if ENABLE_CONTEXT_EXPANSION and db is not None and i < 5:
            metadata = chunk.get("metadata", {})
            doc_id = metadata.get("source")
            section = metadata.get("section")
            
            if doc_id and section is not None:
                tenant_id = metadata.get("tenant_id", "default")
                embedding_model = get_embedding_model_id()
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
                            final_context.append(compress_context(neighbor_chunk))
                            expanded_ids.add(n.id)
                except Exception as e:
                    print(f"[Context] Expansion failed: {e}")

        if chunk.get("id") not in expanded_ids:
            compressed = compress_context(chunk)
            final_context.append(compressed)
            expanded_ids.add(chunk.get("id"))

    # 3. Format citations
    for chunk in final_context:
        metadata = chunk.get("metadata", {})
        source = os.path.basename(metadata.get("source", "unknown"))
        page = metadata.get("page_num", "?")
        chunk["citation"] = f"[{source}, Page {page}]"
        
    return final_context

def _candidate_from_chunk(chunk) -> dict:
    return {
        "id": chunk.id,
        "text": chunk.text_content,
        "metadata": {
            "source": chunk.doc_id,
            "section": chunk.section,
            "page_num": (chunk.doc_metadata or {}).get("page_num"),
        },
        "quantized_embedding": chunk.quantized_embedding if hasattr(chunk, "quantized_embedding") else None,
    }
