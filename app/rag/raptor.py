import hashlib
import json
import requests
import numpy as np
from sqlalchemy.orm import Session
from app.database import DocumentChunk
from app.rag.model_loader import get_ollama_generate_url, OLLAMA_MODEL, get_embedding_model_id, encode_text

def get_chunks_at_level(db: Session, tenant_id: str, level: int):
    return db.query(DocumentChunk).filter(
        DocumentChunk.tenant_id == tenant_id,
        DocumentChunk.raptor_level == level
    ).all()

def generate_cluster_summary(texts: list[str]) -> str:
    combined_text = "\n\n".join(texts)
    # Ensure it doesn't exceed context window
    if len(combined_text) > 100000:
        combined_text = combined_text[:100000]
        
    prompt = f"""
    You are an expert summarizer. Synthesize the following texts into a comprehensive, high-level summary.
    Extract the key themes, relationships, and overarching concepts.
    
    TEXTS:
    {combined_text}
    
    SUMMARY:
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1}
    }
    try:
        response = requests.post(get_ollama_generate_url(), json=payload, timeout=60)
        if response.status_code == 200:
            return response.json().get("response", "").strip()
    except Exception as e:
        print(f"[RAPTOR] Summarization failed: {e}")
    return ""

def build_raptor_tree(db: Session, tenant_id: str, max_levels: int = 3, n_clusters: int = 10):
    """
    Recursively clusters and summarizes document chunks to build a RAPTOR tree.
    """
    print(f"[RAPTOR] Starting tree generation for tenant {tenant_id}")
    try:
        import umap
        from sklearn.mixture import GaussianMixture
    except ImportError:
        print("[RAPTOR] Missing umap-learn or scikit-learn. Cannot build tree.")
        return

    level = 0
    while level < max_levels:
        chunks = get_chunks_at_level(db, tenant_id, level)
        if len(chunks) <= n_clusters:
            print(f"[RAPTOR] Reached root at level {level} with {len(chunks)} chunks.")
            break
            
        print(f"[RAPTOR] Processing Level {level} ({len(chunks)} chunks)")
        
        # Extract embeddings
        embeddings = []
        valid_chunks = []
        for c in chunks:
            if c.embedding is not None:
                embeddings.append(c.embedding)
                valid_chunks.append(c)
                
        if not embeddings:
            break
            
        X = np.array(embeddings)
        
        # Reduce dimensionality with UMAP (better for clustering high-dim vectors)
        n_components = min(10, len(embeddings) - 1)
        if n_components < 2:
            break
            
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        try:
            X_reduced = reducer.fit_transform(X)
        except Exception as e:
            print(f"[RAPTOR] UMAP failed: {e}")
            break
            
        # Cluster with Gaussian Mixture
        k = min(n_clusters, len(embeddings) // 2)
        if k < 2:
            break
            
        gmm = GaussianMixture(n_components=k, random_state=42)
        labels = gmm.fit_predict(X_reduced)
        
        # Group texts by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(valid_chunks[i].text_content)
            
        # Summarize each cluster and insert as next level
        new_level_chunks = []
        embedding_model_id = get_embedding_model_id()
        
        for cluster_id, texts in clusters.items():
            summary = generate_cluster_summary(texts)
            if summary:
                # Create a pseudo-hash
                chunk_hash = hashlib.sha256(summary.encode("utf-8")).hexdigest()
                # Encode summary
                vec = encode_text(summary)
                
                new_chunk = DocumentChunk(
                    tenant_id=tenant_id,
                    doc_id=f"raptor_summary_lvl_{level+1}",
                    chunk_hash=chunk_hash,
                    text_content=summary,
                    embedding_model=embedding_model_id,
                    embedding=vec,
                    raptor_level=level + 1,
                    file_type="raptor_summary"
                )
                db.add(new_chunk)
                new_level_chunks.append(new_chunk)
                
        db.commit()
        print(f"[RAPTOR] Created {len(new_level_chunks)} summaries for Level {level + 1}")
        level += 1

    print(f"[RAPTOR] Tree generation complete for tenant {tenant_id}")
