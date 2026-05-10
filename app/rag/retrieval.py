from sqlalchemy.orm import Session
from sqlalchemy import text

from app.database import DocumentChunk
from app.rag.model_loader import encode_text, get_embedding_model_id


RRF_K = 60
DENSE_WEIGHT = 0.7
LEXICAL_WEIGHT = 0.3


def _candidate_from_chunk(chunk: DocumentChunk) -> dict:
    metadata = chunk.doc_metadata or {}
    return {
        "id": chunk.id,
        "text": chunk.text_content,
        "score": 0.0,
        "hybrid_score": 0.0,
        "dense_score": 0.0,
        "lexical_score": 0.0,
        "metadata": {
            "tenant_id": chunk.tenant_id,
            "source": chunk.doc_id,
            "section": chunk.section,
            "type": metadata.get("type", "text"),
            "page_num": metadata.get("page_num"),
            "embedding_model": chunk.embedding_model,
        }
    }


def _rrf_score(rank: int) -> float:
    return 1.0 / (RRF_K + rank)

def perform_hybrid_search(db: Session, query: str, tenant_id: str, top_k: int = 20) -> list:
    """
    Layer 3: Hybrid search using pgvector dense retrieval plus PostgreSQL full text.
    """
    print(f"[Retrieval] Executing hybrid search for tenant={tenant_id!r}, query={query!r}")
    embedding_model = get_embedding_model_id()
    query_vector = encode_text(query)
    candidate_limit = max(top_k * 4, 50)
    candidates = {}

    distance_expr = DocumentChunk.embedding.cosine_distance(query_vector).label("distance")
    dense_rows = (
        db.query(DocumentChunk, distance_expr)
        .filter(
            DocumentChunk.tenant_id == tenant_id,
            DocumentChunk.embedding_model == embedding_model,
        )
        .order_by(distance_expr)
        .limit(candidate_limit)
        .all()
    )

    for rank, (chunk, distance) in enumerate(dense_rows, start=1):
        candidate = candidates.setdefault(chunk.id, _candidate_from_chunk(chunk))
        dense_score = 1.0 / (1.0 + float(distance or 0.0))
        candidate["dense_score"] = max(candidate["dense_score"], dense_score)
        candidate["hybrid_score"] += DENSE_WEIGHT * _rrf_score(rank)
        candidate["metadata"]["dense_rank"] = rank

    lexical_rows = db.execute(
        text(
            "SELECT id, "
            "ts_rank_cd("
            "  to_tsvector('simple', coalesce(text_content, '')), "
            "  plainto_tsquery('simple', :query)"
            ") AS lexical_score "
            "FROM document_chunks "
            "WHERE tenant_id = :tenant_id "
            "AND embedding_model = :embedding_model "
            "AND to_tsvector('simple', coalesce(text_content, '')) "
            "    @@ plainto_tsquery('simple', :query) "
            "ORDER BY lexical_score DESC "
            "LIMIT :limit"
        ),
        {
            "query": query,
            "tenant_id": tenant_id,
            "embedding_model": embedding_model,
            "limit": candidate_limit,
        },
    ).mappings().all()

    if lexical_rows:
        lexical_ids = [row["id"] for row in lexical_rows]
        lexical_chunks = (
            db.query(DocumentChunk)
            .filter(DocumentChunk.id.in_(lexical_ids))
            .all()
        )
        chunks_by_id = {chunk.id: chunk for chunk in lexical_chunks}

        for rank, row in enumerate(lexical_rows, start=1):
            chunk = chunks_by_id.get(row["id"])
            if not chunk:
                continue
            candidate = candidates.setdefault(chunk.id, _candidate_from_chunk(chunk))
            candidate["lexical_score"] = max(
                candidate["lexical_score"],
                float(row["lexical_score"] or 0.0),
            )
            candidate["hybrid_score"] += LEXICAL_WEIGHT * _rrf_score(rank)
            candidate["metadata"]["lexical_rank"] = rank

    ranked = sorted(
        candidates.values(),
        key=lambda item: item["hybrid_score"],
        reverse=True,
    )
    for item in ranked:
        item["score"] = item["hybrid_score"]

    return ranked[:top_k]
