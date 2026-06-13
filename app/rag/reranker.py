from app.rag.model_loader import get_reranker_model

def rerank_results(query: str, retrieved_chunks: list, top_n: int = 5) -> list:
    """
    Layer 4/ColBERT: Reranking
    ColBERT Late-Interaction or Cross-encoder reranker.
    Input: top-20 fused results -> Output: top-N ranked by relevance.
    """
    print(f"[Reranker] Reranking {len(retrieved_chunks)} results for query: '{query}'")
    if not retrieved_chunks:
        return []
        
    model = get_reranker_model()
    docs = [chunk["text"] or "" for chunk in retrieved_chunks]
    
    # RAGatouille (ColBERT) has a .rerank() method, CrossEncoder has .predict()
    if hasattr(model, "rerank"):
        # ColBERT reranking
        results = model.rerank(query=query, documents=docs, k=top_n)
        # results is a list of dicts: [{"content": "...", "score": 23.5, "rank": 1}, ...]
        # Map scores back to the original chunks
        ranked_chunks = []
        for res in results:
            content = res["content"]
            # Find the original chunk
            for chunk in retrieved_chunks:
                if chunk["text"] == content:
                    chunk["rerank_score"] = res["score"]
                    ranked_chunks.append(chunk)
                    break
        return ranked_chunks[:top_n]
    else:
        # CrossEncoder reranking fallback
        pairs = [[query, doc] for doc in docs]
        scores = model.predict(pairs)
        
        for i, chunk in enumerate(retrieved_chunks):
            chunk["rerank_score"] = float(scores[i])
            
        sorted_chunks = sorted(retrieved_chunks, key=lambda x: x["rerank_score"], reverse=True)
        return sorted_chunks[:top_n]
