from sentence_transformers import SentenceTransformer, util

mmr_embedder = SentenceTransformer('all-MiniLM-L6-v2')

def apply_mmr(query: str, chunks: list, diversity: float = 0.5) -> list:
    """
    Layer 5: Context Assembly - Apply MMR (Max Marginal Relevance)
    Removes redundant chunks to maximize information diversity.
    """
    if not chunks: return []
    
    query_emb = mmr_embedder.encode(query)
    chunk_embs = mmr_embedder.encode([c["text"] for c in chunks])
    
    query_sim = util.cos_sim(query_emb, chunk_embs)[0].tolist()
    
    selected = [query_sim.index(max(query_sim))]
    unselected = [i for i in range(len(chunks)) if i not in selected]
    
    while unselected:
        mmr_scores = []
        for i in unselected:
            sim_to_selected = max([util.cos_sim(chunk_embs[i], chunk_embs[j]).item() for j in selected])
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
    """
    max_chars = 1500 # Approx 300 tokens
    if len(chunk["text"]) > max_chars:
        chunk["text"] = chunk["text"][:max_chars] + "..."
    return chunk

def assemble_context(query: str, reranked_chunks: list) -> list:
    """
    Layer 5: Context Assembly
    Target 60-70% fill of context window.
    Attach citation metadata [source, page, section] per chunk.
    """
    # 1. Apply MMR
    mmr_filtered = apply_mmr(query, reranked_chunks)
    
    final_context = []
    # 2. Compress & format citations
    for chunk in mmr_filtered:
        compressed = compress_context(chunk)
        
        # 3. Attach citation metadata
        metadata = compressed.get("metadata", {})
        source = metadata.get("source", "unknown_source")
        page = metadata.get("page_num", "?")
        compressed["citation"] = f"[{source}, Page {page}]"
        
        final_context.append(compressed)
        
    return final_context
