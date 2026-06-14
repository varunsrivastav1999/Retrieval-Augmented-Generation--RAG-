from app.rag.model_loader import get_reranker_model

def rerank_results(query: str, retrieved_chunks: list, top_n: int = 5) -> list:
    """
    Layer 4/ColBERT: Reranking
    ColBERT Late-Interaction or Cross-encoder reranker.
    Input: top-20 fused results -> Output: top-N ranked by relevance.
    
    FIXED: Uses index-based matching instead of fragile text equality
    for ColBERT results mapping.
    """
    if not retrieved_chunks:
        return []
    
    print(f"[Reranker] Reranking {len(retrieved_chunks)} results for query: '{query[:80]}'")
    model = get_reranker_model()
    docs = [chunk.get("text", "") or "" for chunk in retrieved_chunks]
    
    # RAGatouille (ColBERT) has a .rerank() method, CrossEncoder has .predict()
    if hasattr(model, "rerank"):
        # ColBERT reranking
        try:
            results = model.rerank(query=query, documents=docs, k=top_n)
        except Exception as e:
            print(f"[Reranker] ColBERT rerank failed: {e}, falling back to original order")
            for chunk in retrieved_chunks[:top_n]:
                chunk["rerank_score"] = chunk.get("hybrid_score", chunk.get("score", 0.0))
            return retrieved_chunks[:top_n]
        
        # Build a lookup from content -> original chunk (using index-based matching)
        # Create a map of doc text to list of chunk indices to handle duplicates
        doc_to_indices = {}
        for idx, doc in enumerate(docs):
            doc_key = doc[:500]  # Use first 500 chars as key for matching
            if doc_key not in doc_to_indices:
                doc_to_indices[doc_key] = []
            doc_to_indices[doc_key].append(idx)
        
        ranked_chunks = []
        used_indices = set()
        for res in results:
            content = res.get("content", "")
            content_key = content[:500]
            indices = doc_to_indices.get(content_key, [])
            
            matched_idx = None
            for idx in indices:
                if idx not in used_indices:
                    matched_idx = idx
                    used_indices.add(idx)
                    break
            
            if matched_idx is not None:
                chunk = retrieved_chunks[matched_idx]
                chunk["rerank_score"] = float(res.get("score", 0.0))
                ranked_chunks.append(chunk)
        
        return ranked_chunks[:top_n]
    else:
        # CrossEncoder reranking (LexicalReranker also uses .predict())
        try:
            pairs = [[query, doc] for doc in docs]
            scores = model.predict(pairs)
        except Exception as e:
            print(f"[Reranker] Predict failed: {e}, falling back to original order")
            for chunk in retrieved_chunks[:top_n]:
                chunk["rerank_score"] = chunk.get("hybrid_score", chunk.get("score", 0.0))
            return retrieved_chunks[:top_n]
        
        for i, chunk in enumerate(retrieved_chunks):
            chunk["rerank_score"] = float(scores[i])
            
        sorted_chunks = sorted(retrieved_chunks, key=lambda x: x["rerank_score"], reverse=True)
        return sorted_chunks[:top_n]
