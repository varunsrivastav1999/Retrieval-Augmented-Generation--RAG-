import requests
from sqlalchemy.orm import Session
from qdrant_client.http import models

from app.database import DocumentChunk
from app.rag.qdrant_client import get_qdrant_client
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
HYDE_WEIGHT = 0.3

def _candidate_from_payload(point_id: str, payload: dict, distance: float) -> dict:
    metadata = payload.get("metadata", {})
    return {
        "id": point_id,
        "text": payload.get("text_content", ""),
        "score": 0.0,
        "hybrid_score": 0.0,
        "dense_score": distance,
        "file_type": payload.get("file_type", metadata.get("file_type", "unknown")),
        "table_group": payload.get("table_group", metadata.get("table_group")),
        "metadata": {
            "tenant_id": payload.get("tenant_id"),
            "source": payload.get("doc_id"),
            "section": payload.get("section"),
            "type": metadata.get("type", "text"),
            "page_num": metadata.get("page_num"),
            "embedding_model": metadata.get("embedding_model"),
            "file_type": payload.get("file_type", metadata.get("file_type", "unknown")),
            "entities": metadata.get("entities", []),
            "table_group": payload.get("table_group", metadata.get("table_group")),
        }
    }

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
        response = requests.post(get_ollama_generate_url(), json=payload, timeout=45.0)
        if response.status_code == 200:
            return response.json().get("response", "").strip()
    except Exception as e:
        print(f"[HyDE] Failed to generate: {e}")
    return ""

def perform_hybrid_search(db: Session, query: str, tenant_id: str, top_k: int = 20, metadata_filters: dict = None, fast_path: bool = False) -> list:
    print(f"[Retrieval] Executing {'fast ' if fast_path else ''}hybrid search for tenant={tenant_id!r}, query={query!r}, filters={metadata_filters}")
    
    qdrant = get_qdrant_client()
    
    hyde_text = None
    hyde_vector = None
    vision_vector = None
    query_entities = []
    
    if fast_path:
        query_vector = encode_text(query)
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as executor:
            fut_dense = executor.submit(encode_text, query)
            fut_hyde = executor.submit(_generate_hyde, query)
            fut_vision = executor.submit(encode_image_text_query, query)
            fut_entities = executor.submit(extract_entities, query)
            
            query_vector = fut_dense.result()
            hyde_text = fut_hyde.result()
            hyde_vector = encode_text(hyde_text) if hyde_text else None
            vision_vector = fut_vision.result()
            query_entities = fut_entities.result()
    
    
    candidate_limit = max(top_k * 8, 100)
    candidates = {}

    # Qdrant filters: always scope by tenant AND embedding model to prevent
    # mixing vectors from different model versions (e.g., 384-dim vs 1024-dim)
    must_conditions = [
        models.FieldCondition(
            key="tenant_id",
            match=models.MatchValue(value=tenant_id)
        ),
        models.FieldCondition(
            key="metadata.embedding_model",
            match=models.MatchValue(value=get_embedding_model_id())
        ),
    ]
    
    if metadata_filters:
        if "file_type" in metadata_filters and metadata_filters["file_type"]:
            must_conditions.append(models.FieldCondition(key="file_type", match=models.MatchValue(value=metadata_filters["file_type"])))
        if "page" in metadata_filters and metadata_filters["page"]:
            try:
                page_val = int(metadata_filters["page"])
            except (ValueError, TypeError):
                page_val = None
            if page_val is not None:
                must_conditions.append(models.FieldCondition(key="metadata.page_num", match=models.MatchValue(value=page_val)))

    qdrant_filter = models.Filter(must=must_conditions)

    # Base dense search
    try:
        dense_results = qdrant.search(
            collection_name="document_chunks",
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=candidate_limit
        )
        for rank, point in enumerate(dense_results, start=1):
            candidate = candidates.setdefault(point.id, _candidate_from_payload(point.id, point.payload, point.score))
            candidate["dense_score"] = max(candidate["dense_score"], point.score)
            candidate["hybrid_score"] += DENSE_WEIGHT * _rrf_score(rank)
            candidate["metadata"]["dense_rank"] = rank
    except Exception as e:
        print(f"[Retrieval] Qdrant Dense Search Failed: {e}")

    # HyDE search
    if hyde_vector:
        try:
            hyde_results = qdrant.search(
                collection_name="document_chunks",
                query_vector=hyde_vector,
                query_filter=qdrant_filter,
                limit=candidate_limit
            )
            for rank, point in enumerate(hyde_results, start=1):
                candidate = candidates.setdefault(point.id, _candidate_from_payload(point.id, point.payload, point.score))
                candidate["dense_score"] = max(candidate["dense_score"], point.score)
                candidate["hybrid_score"] += HYDE_WEIGHT * _rrf_score(rank)
                candidate["metadata"]["hyde_rank"] = rank
        except Exception as e:
            print(f"[Retrieval] HyDE search failed: {e}")

    # Vision search
    if not fast_path and vision_vector:
        try:
            vision_results = qdrant.search(
                collection_name="image_chunks",
                query_vector=vision_vector,
                query_filter=qdrant_filter,
                limit=candidate_limit
            )
            for rank, point in enumerate(vision_results, start=1):
                candidate = candidates.setdefault(point.id, _candidate_from_payload(point.id, point.payload, point.score))
                candidate["dense_score"] = max(candidate["dense_score"], point.score)
                candidate["hybrid_score"] += DENSE_WEIGHT * _rrf_score(rank)
                candidate["metadata"]["vision_rank"] = rank
        except Exception as e:
            print(f"[Retrieval] Vision search failed: {e}")

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

    # Table-aware expansion: if top results include table chunks, fetch all
    # sibling chunks from the same table_group so the LLM sees the complete table
    top_results = ranked[:top_k]
    table_groups = set()
    for item in top_results:
        tg = item.get("table_group")
        if tg is not None:
            table_groups.add(tg)

    if table_groups:
        seen_ids = {item["id"] for item in top_results}
        # Table sibling expansion uses only tenant+model filters (not user's page/file_type filter)
        base_conditions = [
            models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id)),
            models.FieldCondition(key="metadata.embedding_model", match=models.MatchValue(value=get_embedding_model_id())),
        ]
        for tg in table_groups:
            try:
                sibling_results = qdrant.scroll(
                    collection_name="document_chunks",
                    scroll_filter=models.Filter(
                        must=base_conditions + [
                            models.FieldCondition(
                                key="table_group",
                                match=models.MatchValue(value=tg),
                            )
                        ]
                    ),
                    limit=200,
                )[0]
                for point in sibling_results:
                    if point.id not in seen_ids:
                        seen_ids.add(point.id)
                        candidate = _candidate_from_payload(point.id, point.payload, 0.0)
                        candidate["hybrid_score"] = 0.01  # low but non-zero to include
                        ranked.append(candidate)
            except Exception as e:
                print(f"[Retrieval] Table sibling expansion failed for group {tg}: {e}")

    return ranked[:top_k]

def perform_multi_query_search(db: Session, queries: list, tenant_id: str, top_k: int = 20, metadata_filters: dict = None, fast_path: bool = False) -> list:
    if not queries:
        return []

    if fast_path:
        results = perform_hybrid_search(db, queries[0], tenant_id, top_k=top_k, metadata_filters=metadata_filters, fast_path=True)
        return results

    all_candidates = {}
    
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(len(queries), 4)) as executor:
        futures = [executor.submit(perform_hybrid_search, db, q, tenant_id, top_k, metadata_filters) for q in queries]
        
        for future in futures:
            results = future.result()
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

