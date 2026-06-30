from app.rag.model_loader import get_reranker_model
import re


# Rare marker used to embed original index in documents for collision-free
# reranker round-trip. Choose something unlikely in real text but safe for
# BERT tokenizers (which break on spaces/punctuation but keep word chars).
_IDX_MARKER = "__IDX_"


def rerank_results(query: str, retrieved_chunks: list, top_n: int = 8) -> list:
    """
    Layer 4/ColBERT: Reranking
    ColBERT Late-Interaction or Cross-encoder reranker.
    Input: top-20 fused results -> Output: top-N ranked by relevance.

    Uses index-embedded document markers to prevent content-collision bugs
    when two chunks share the same prefix text.
    """
    if not retrieved_chunks:
        return []
    
    print(f"[Reranker] Reranking {len(retrieved_chunks)} results for query: '{query[:80]}'")
    model = get_reranker_model()
    raw_docs = [chunk.get("text", "") or "" for chunk in retrieved_chunks]

    # Embed original index in document text for collision-free round-trip.
    # Use a marker that survives BERT/ColBERT tokenization (word chars only).
    indexed_docs = [f"{_IDX_MARKER}{idx}{_IDX_MARKER}{doc}" for idx, doc in enumerate(raw_docs)]

    if hasattr(model, "rerank"):
        try:
            results = model.rerank(query=query, documents=indexed_docs, k=top_n)
        except Exception as e:
            print(f"[Reranker] ColBERT rerank failed: {e}, falling back to original order")
            for chunk in retrieved_chunks[:top_n]:
                chunk["rerank_score"] = chunk.get("hybrid_score", chunk.get("score", 0.0))
            return retrieved_chunks[:top_n]

        ranked_chunks = []
        used_indices = set()
        for res in results:
            content = res.get("content", "")
            # Try marker-based extraction first (fast path)
            match = re.search(re.escape(_IDX_MARKER) + r'(\d+)' + re.escape(_IDX_MARKER), content)
            if match:
                orig_idx = int(match.group(1))
                if orig_idx < len(retrieved_chunks) and orig_idx not in used_indices:
                    chunk = retrieved_chunks[orig_idx]
                    chunk["rerank_score"] = float(res.get("score", 0.0))
                    ranked_chunks.append(chunk)
                    used_indices.add(orig_idx)
                    continue
            
            # Marker lost (e.g., tokenizer stripped it): fall back to content matching
            # Match by exact content against unused chunks
            for idx, doc in enumerate(raw_docs):
                if idx in used_indices:
                    continue
                if doc and doc == content:
                    chunk = retrieved_chunks[idx]
                    chunk["rerank_score"] = float(res.get("score", 0.0))
                    ranked_chunks.append(chunk)
                    used_indices.add(idx)
                    break
            else:
                # Last resort: match by truncated prefix (500 chars)
                for idx, doc in enumerate(raw_docs):
                    if idx in used_indices:
                        continue
                    if doc and doc[:500] == content[:500]:
                        chunk = retrieved_chunks[idx]
                        chunk["rerank_score"] = float(res.get("score", 0.0))
                        ranked_chunks.append(chunk)
                        used_indices.add(idx)
                        break

        return ranked_chunks[:top_n]
    else:
        try:
            pairs = [[query, doc] for doc in raw_docs]
            # Predict with limited batch size to avoid PyTorch OOM spikes
            scores = model.predict(pairs, batch_size=16)
        except Exception as e:
            print(f"[Reranker] Predict failed: {e}, falling back to original order")
            for chunk in retrieved_chunks[:top_n]:
                chunk["rerank_score"] = chunk.get("hybrid_score", chunk.get("score", 0.0))
            return retrieved_chunks[:top_n]
        
        for i, chunk in enumerate(retrieved_chunks):
            if i < len(scores):
                chunk["rerank_score"] = float(scores[i])
            else:
                chunk["rerank_score"] = chunk.get("hybrid_score", chunk.get("score", 0.0))
            
        sorted_chunks = sorted(retrieved_chunks, key=lambda x: x["rerank_score"], reverse=True)
        
        # Free VRAM immediately after prediction
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
            
        return sorted_chunks[:top_n]
