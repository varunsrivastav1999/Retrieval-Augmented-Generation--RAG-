from app.rag.model_loader import get_reranker_model

def rerank_results(query: str, retrieved_chunks: list, top_n: int = 5) -> list:
    """
    Layer 4: Reranking
    Cross-encoder reranker (e.g., ms-marco-MiniLM).
    Input: top-20 fused results -> Output: top-5 ranked by relevance.
    Reranker runs on CPU (cheap, fast, high-impact).
    """
    print(f"[Reranker] Reranking {len(retrieved_chunks)} results for query: '{query}'")
    if not retrieved_chunks:
        return []
        
    pairs = [[query, chunk["text"]] for chunk in retrieved_chunks]
    scores = get_reranker_model().predict(pairs)
    
    for i, chunk in enumerate(retrieved_chunks):
        chunk["rerank_score"] = float(scores[i])
        
    sorted_chunks = sorted(retrieved_chunks, key=lambda x: x["rerank_score"], reverse=True)
    return sorted_chunks[:top_n]
