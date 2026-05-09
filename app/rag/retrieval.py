from sqlalchemy.orm import Session
from app.database import DocumentChunk
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def perform_hybrid_search(db: Session, query: str, tenant_id: str, top_k: int = 20) -> list:
    """
    Layer 3: Dense ANN search using pgvector.
    Note: Full production would combine this with pg_trgm or TSVECTOR for BM25.
    Here we implement the core pgvector ANN search via cosine distance.
    """
    print(f"[Retrieval] Executing pgvector ANN search for query: '{query}'")
    query_vector = embedder.encode(query).tolist()
    
    # Using pgvector cosine distance sorting natively in PostgreSQL
    results = db.query(DocumentChunk)\
                .order_by(DocumentChunk.embedding.cosine_distance(query_vector))\
                .limit(top_k)\
                .all()
    
    formatted_results = []
    for r in results:
        formatted_results.append({
            "id": r.id,
            "text": r.text_content,
            "score": 1.0, # Normalization placeholder for hybrid fusion
            "metadata": {
                "source": r.doc_id,
                "section": r.section
            }
        })
    return formatted_results
