import hashlib
import json
import os
import requests
import numpy as np
from sqlalchemy.orm import Session
from app.database import DocumentChunk
from app.rag.model_loader import get_ollama_generate_url, OLLAMA_MODEL, get_embedding_model_id, encode_text
from app.rag.qdrant_client import get_qdrant_client, insert_qdrant_points
from qdrant_client.http import models

def get_chunks_at_level(db: Session, tenant_id: str, level: int):
    return db.query(DocumentChunk).filter(
        DocumentChunk.tenant_id == tenant_id,
        DocumentChunk.raptor_level == level
    ).all()

def _fetch_embeddings_from_qdrant(chunks):
    """Fetch embeddings from Qdrant for a list of DocumentChunk objects."""
    if not chunks:
        return [], []
    client = get_qdrant_client()
    chunk_ids = [c.id for c in chunks]
    results = client.retrieve(
        collection_name="document_chunks",
        ids=chunk_ids,
        with_vectors=True,
    )
    id_to_vec = {r.id: r.vector for r in results}
    embeddings = []
    valid_chunks = []
    for c in chunks:
        vec = id_to_vec.get(c.id)
        if vec is not None:
            if isinstance(vec, dict):
                vec = list(vec.values())[0] if vec else None
            if vec is not None:
                embeddings.append(vec)
                valid_chunks.append(c)
    return embeddings, valid_chunks


def generate_cluster_summary(texts: list[str], retries: int = 2) -> str:
    combined_text = "\n\n".join(texts)
    if len(combined_text) > 100000:
        combined_text = combined_text[:100000]
        
    prompt = f"""
    You are a faithful summarizer. Synthesize ONLY the information provided below.
    Do NOT add any external knowledge, examples, or explanations.
    Preserve all specific numbers, part numbers, and technical terms exactly as stated.
    Extract the key themes, relationships, and overarching concepts found in the texts.
    
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
    for attempt in range(retries + 1):
        try:
            response = requests.post(get_ollama_generate_url(), json=payload, timeout=120)
            if response.status_code == 200:
                summary = response.json().get("response", "").strip()
                if summary:
                    return summary
        except Exception as e:
            if attempt < retries:
                print(f"[RAPTOR] Summarization attempt {attempt + 1} failed: {e}, retrying...")
            else:
                print(f"[RAPTOR] Summarization failed after {retries + 1} attempts: {e}")
    return ""


def _delete_previous_tree(db: Session, tenant_id: str) -> None:
    """Remove all existing RAPTOR summaries for a tenant before rebuilding."""
    client = get_qdrant_client()
    try:
        client.delete(
            collection_name="document_chunks",
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id)),
                        models.FieldCondition(key="file_type", match=models.MatchValue(value="raptor_summary")),
                    ]
                )
            ),
            wait=True,
        )
    except Exception as e:
        print(f"[RAPTOR] Qdrant cleanup error: {e}")

    deleted = db.query(DocumentChunk).filter(
        DocumentChunk.tenant_id == tenant_id,
        DocumentChunk.file_type == "raptor_summary"
    ).delete(synchronize_session=False)
    db.commit()
    if deleted:
        print(f"[RAPTOR] Deleted {deleted} previous summary chunks for tenant '{tenant_id}'")


def build_raptor_tree(db: Session, tenant_id: str, max_levels: int = 3, n_clusters: int = 10):
    """
    Recursively clusters and summarizes document chunks to build a RAPTOR tree.
    Uses Qdrant for vector storage (not the DocumentChunk.embedding column).
    """
    print(f"[RAPTOR] Starting tree generation for tenant {tenant_id}")
    _delete_previous_tree(db, tenant_id)
    try:
        os.environ["NUMBA_DISABLE_CACHE"] = "1"
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
        
        # Fetch embeddings from Qdrant
        embeddings, valid_chunks = _fetch_embeddings_from_qdrant(chunks)
        if not embeddings:
            print(f"[RAPTOR] No embeddings found in Qdrant for level {level}.")
            break
            
        X = np.array(embeddings)
        
        # Reduce dimensionality with UMAP (better for clustering high-dim vectors)
        n_components = min(10, len(embeddings) - 1)
        if n_components < 2:
            break
            
        reducer = umap.UMAP(n_components=n_components, random_state=42, memory=None)
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
        points = []
        new_level_chunks = []
        embedding_model_id = get_embedding_model_id()
        
        for cluster_id, texts in clusters.items():
            summary = generate_cluster_summary(texts)
            if summary:
                chunk_hash = hashlib.sha256(summary.encode("utf-8")).hexdigest()
                vec = encode_text(summary)
                
                new_chunk = DocumentChunk(
                    tenant_id=tenant_id,
                    doc_id=f"raptor_summary_lvl_{level+1}",
                    chunk_hash=chunk_hash,
                    text_content=summary,
                    embedding_model=embedding_model_id,
                    raptor_level=level + 1,
                    file_type="raptor_summary"
                )
                db.add(new_chunk)
                db.flush()

                points.append(
                    models.PointStruct(
                        id=new_chunk.id,
                        vector=vec,
                        payload={
                            "tenant_id": tenant_id,
                            "doc_id": f"raptor_summary_lvl_{level+1}",
                            "file_type": "raptor_summary",
                            "text_content": summary,
                            "section": 0,
                            "metadata": {
                                "type": "raptor_summary",
                                "level": level + 1,
                                "tenant_id": tenant_id,
                                "source": f"raptor_summary_lvl_{level+1}",
                                "embedding_model": embedding_model_id,
                            }
                        }
                    )
                )
                new_level_chunks.append(new_chunk)
        
        if points:
            try:
                insert_qdrant_points("document_chunks", points)
            except Exception as e:
                print(f"[RAPTOR] Qdrant insert failed: {e}")
                db.rollback()
                for c in new_level_chunks:
                    db.delete(c)
                break
        
        db.commit()
        print(f"[RAPTOR] Created {len(new_level_chunks)} summaries for Level {level + 1}")
        level += 1

    print(f"[RAPTOR] Tree generation complete for tenant {tenant_id}")
