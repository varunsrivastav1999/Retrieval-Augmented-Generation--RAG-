import requests
from sqlalchemy.orm import Session
from sqlalchemy import text, cast, String

from app.database import DocumentChunk
from app.rag.model_loader import (
    encode_text,
    get_embedding_model_id,
    encode_image_text_query,
    extract_entities,
    get_ollama_generate_url,
    OLLAMA_MODEL,
)

RRF_K = 60
DENSE_WEIGHT = 0.5
LEXICAL_WEIGHT = 0.5
HYDE_WEIGHT = 0.3

# Use the same OLLAMA_URL resolution as model_loader.py
OLLAMA_URL = get_ollama_generate_url()

def _candidate_from_chunk(chunk: DocumentChunk) -> dict:
    metadata = chunk.doc_metadata or {}
    cand = {
        "id": chunk.id,
        "text": chunk.text_content or "",
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
            "entities": metadata.get("entities", []),
        }
    }
    if chunk.quantized_embedding:
        cand["quantized_embedding"] = chunk.quantized_embedding
    return cand

def _rrf_score(rank: int) -> float:
    return 1.0 / (RRF_K + rank)

def _generate_hyde(query: str) -> str:
    """HyDE: Generate a hypothetical answer to embed for dense retrieval."""
    prompt = f"Please write a very short, single sentence factual answer to the following question: {query}"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "30m",
        "options": {"num_predict": 30, "temperature": 0.3}
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=15.0)
        if response.status_code == 200:
            return response.json().get("response", "").strip()
    except Exception as e:
        print(f"[HyDE] Failed to generate: {e}")
    return ""

def perform_hybrid_search(db: Session, query: str, tenant_id: str, top_k: int = 20, metadata_filters: dict = None, fast_path: bool = False) -> list:
    """
    Layer 3: Hybrid search using pgvector dense retrieval plus PostgreSQL full text.
    Includes HyDE (Hypothetical Document Embeddings) and Text-to-SQL (metadata filters).

    When fast_path=True: skip HyDE, BM25, Vision, entity extraction — uses pure
    HNSW dense search only. Latency: ~2-5ms instead of ~200-2000ms.
    """
    print(f"[Retrieval] Executing {'fast ' if fast_path else ''}hybrid search for tenant={tenant_id!r}, query={query!r}, filters={metadata_filters}")
    embedding_model = get_embedding_model_id()
    query_vector = encode_text(query)
    
    hyde_text = None
    hyde_vector = None
    vision_vector = None
    query_entities = []
    
    if not fast_path:
        hyde_text = _generate_hyde(query)
        hyde_vector = encode_text(hyde_text) if hyde_text else None
        vision_vector = encode_image_text_query(query)
        query_entities = extract_entities(query)
    
    candidate_limit = max(top_k * 8, 100)
    candidates = {}

    distance_expr = DocumentChunk.embedding.cosine_distance(query_vector).label("distance")
    
    base_filters = [
        DocumentChunk.tenant_id == tenant_id,
        DocumentChunk.embedding_model == embedding_model,
    ]
    if metadata_filters:
        if "file_type" in metadata_filters and metadata_filters["file_type"]:
            base_filters.append(DocumentChunk.file_type == metadata_filters["file_type"])
        if "page" in metadata_filters and metadata_filters["page"]:
            base_filters.append(cast(DocumentChunk.doc_metadata['page_num'], String) == str(metadata_filters["page"]))

    dense_rows = (
        db.query(DocumentChunk, distance_expr)
        .filter(*base_filters)
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

    if hyde_vector:
        hyde_distance_expr = DocumentChunk.embedding.cosine_distance(hyde_vector).label("hyde_distance")
        hyde_rows = (
            db.query(DocumentChunk, hyde_distance_expr)
            .filter(*base_filters)
            .order_by(hyde_distance_expr)
            .limit(candidate_limit)
            .all()
        )
        for rank, (chunk, distance) in enumerate(hyde_rows, start=1):
            candidate = candidates.setdefault(chunk.id, _candidate_from_chunk(chunk))
            hyde_score = 1.0 / (1.0 + float(distance or 0.0))
            candidate["dense_score"] = max(candidate["dense_score"], hyde_score)
            candidate["hybrid_score"] += HYDE_WEIGHT * _rrf_score(rank)
            candidate["metadata"]["hyde_rank"] = rank

    if not fast_path:
        lexical_query = (
            "SELECT id, "
            "ts_rank_cd("
            "  to_tsvector('english', coalesce(text_content, '')), "
            "  websearch_to_tsquery('english', :query)"
            ") AS lexical_score "
            "FROM document_chunks "
            "WHERE tenant_id = :tenant_id "
            "AND embedding_model = :embedding_model "
        )
        
        lexical_params = {
            "query": query,
            "tenant_id": tenant_id,
            "embedding_model": embedding_model,
            "limit": candidate_limit,
        }

        if metadata_filters:
            if "file_type" in metadata_filters and metadata_filters["file_type"]:
                lexical_query += " AND file_type = :file_type "
                lexical_params["file_type"] = metadata_filters["file_type"]
            if "page" in metadata_filters and metadata_filters["page"]:
                lexical_query += " AND doc_metadata->>'page_num' = :page "
                lexical_params["page"] = str(metadata_filters["page"])

        lexical_query += (
            "AND to_tsvector('english', coalesce(text_content, '')) "
            "    @@ websearch_to_tsquery('english', :query) "
            "ORDER BY lexical_score DESC "
            "LIMIT :limit"
        )

        lexical_rows = db.execute(text(lexical_query), lexical_params).mappings().all()

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

    if not fast_path and vision_vector:
        vision_distance = DocumentChunk.image_embedding.cosine_distance(vision_vector).label("vision_distance")
        vision_rows = (
            db.query(DocumentChunk, vision_distance)
            .filter(
                DocumentChunk.tenant_id == tenant_id,
                DocumentChunk.embedding_model == embedding_model,
                DocumentChunk.image_embedding.is_not(None)
            )
            .order_by(vision_distance)
            .limit(candidate_limit)
            .all()
        )
        for rank, (chunk, distance) in enumerate(vision_rows, start=1):
            candidate = candidates.setdefault(chunk.id, _candidate_from_chunk(chunk))
            vision_score = 1.0 / (1.0 + float(distance or 0.0))
            candidate["dense_score"] = max(candidate["dense_score"], vision_score)
            candidate["hybrid_score"] += DENSE_WEIGHT * _rrf_score(rank)
            candidate["metadata"]["vision_rank"] = rank

    if not fast_path:
        for candidate in candidates.values():
            chunk_entities = candidate["metadata"].get("entities", [])
            if chunk_entities and query_entities:
                overlap = set(query_entities).intersection(set(chunk_entities))
                if overlap:
                    boost = len(overlap) * 0.2
                    candidate["hybrid_score"] += boost
                    candidate["metadata"]["entity_overlap"] = list(overlap)

    ranked = sorted(
        candidates.values(),
        key=lambda item: item["hybrid_score"],
        reverse=True,
    )
    for item in ranked:
        item["score"] = item["hybrid_score"]

    return ranked[:top_k]

def perform_multi_query_search(db: Session, queries: list, tenant_id: str, top_k: int = 20, metadata_filters: dict = None, fast_path: bool = False) -> list:
    """
    Layer 13 (Extension): Sub-Query RRF Fusion
    Runs hybrid search for multiple sub-queries independently and fuses results with RRF.
    When fast_path=True, uses single-query dense-only HNSW (sub-5ms).
    """
    if not queries:
        return []

    if fast_path:
        results = perform_hybrid_search(db, queries[0], tenant_id, top_k=top_k, metadata_filters=metadata_filters, fast_path=True)
        return results

    all_candidates = {}
    
    for q in queries:
        results = perform_hybrid_search(db, q, tenant_id, top_k=top_k, metadata_filters=metadata_filters)
        for rank, res in enumerate(results, start=1):
            cid = res["id"]
            if cid not in all_candidates:
                all_candidates[cid] = res
                all_candidates[cid]["fused_score"] = 0.0
            all_candidates[cid]["fused_score"] += _rrf_score(rank)
            
    ranked = sorted(
        all_candidates.values(),
        key=lambda item: item["fused_score"],
        reverse=True,
    )
    for item in ranked:
        item["score"] = item["fused_score"]
        
    return ranked[:top_k]

