import requests
import os
import re
from sqlalchemy.orm import Session
from sqlalchemy import text
from qdrant_client.http import models

from app.rag.qdrant_client import get_qdrant_client
from app.rag.model_loader import (
    encode_text,
    get_embedding_model_id,
    get_ollama_generate_url,
    OLLAMA_MODEL,
)
try:
    from app.rag.table_engine import extract_catalogue_patterns
    TABLE_ENGINE_AVAILABLE = True
except ImportError:
    TABLE_ENGINE_AVAILABLE = False
    def extract_catalogue_patterns(q): return []

RRF_K = 60
DENSE_WEIGHT = 0.5
HYDE_WEIGHT = 0.3
BM25_WEIGHT = 0.4

def _candidate_from_payload(point_id: str, payload: dict, distance: float) -> dict:
    metadata = payload.get("metadata", {})
    
    # NeMo-style parent-child: swap child text with full parent block for better LLM context
    text_content = payload.get("text_content", "")
    parent_text = metadata.get("parent_text")
    if parent_text:
        text_content = parent_text

    return {
        "id": point_id,
        "text": text_content,
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
            # --- table-aware fields (v5.0) ---
            "table_id": payload.get("table_id", metadata.get("table_id")),
            "section_title": payload.get("section_title", metadata.get("section_title", "")),
            "cell_values": payload.get("cell_values", metadata.get("cell_values")),
            "header_path": payload.get("header_path", metadata.get("header_path", [])),
            "row_index": payload.get("row_index", metadata.get("row_index")),
        }
    }

def _rrf_score(rank: int) -> float:
    return 1.0 / (RRF_K + rank)

def _generate_hyde(query: str) -> str:
    """HyDE: Generate a hypothetical answer to embed for dense retrieval.

    Uses an industrial-documentation-aware prompt so the generated hypothesis
    matches the vocabulary, structure, and conventions found in technical manuals
    (step numbers, part numbers, section headings, safety notes, etc.).
    """
    prompt = (
        "Write a short technical answer as it would appear verbatim in an industrial "
        "equipment manual, installation guide, or maintenance procedure document. "
        "Use precise technical language. Include any relevant step numbers, part numbers, "
        "section references, or safety warnings that would typically appear in such a manual. "
        f"Answer the following question in 1-3 sentences: {query}"
    )
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "num_predict": 60,
            "temperature": 0.2,
            "num_ctx": int(os.getenv("OLLAMA_CONTEXT_LENGTH", "32768"))
        }
    }
    try:
        response = requests.post(get_ollama_generate_url(), json=payload, timeout=45.0)
        if response.status_code == 200:
            return response.json().get("response", "").strip()
    except Exception as e:
        print(f"[HyDE] Failed to generate: {e}")
    return ""

def _perform_postgres_bm25(query: str, tenant_id: str, limit: int, metadata_filters: dict = None) -> list:
    """Uses PostgreSQL's native Full-Text Search (tsvector) for sparse keyword match.
    Table-aware: strips markdown pipe characters before tokenization.
    """
    # Strip table-markdown noise so BM25 doesn't treat | and --- as tokens
    clean_query = re.sub(r'[|\-]{2,}', ' ', query).strip()
    
    target_file_clause = ""
    params = {"query": clean_query, "tenant_id": tenant_id, "limit": limit}
    
    if metadata_filters and "target_file" in metadata_filters and metadata_filters["target_file"]:
        target_file_clause = "AND doc_id ILIKE :target_file"
        params["target_file"] = f"%{metadata_filters['target_file']}%"

    sql = text(f"""
        SELECT id, doc_id, text_content, section, file_type, doc_metadata,
               ts_rank(tsvector_content, plainto_tsquery('english', :query)) as rank_score
        FROM document_chunks
        WHERE tenant_id = :tenant_id
          AND text_content IS NOT NULL
          AND plainto_tsquery('english', :query) @@ tsvector_content
          {target_file_clause}
        ORDER BY rank_score DESC
        LIMIT :limit
    """)
    candidates = []
    try:
        from app.database import SessionLocal
        with SessionLocal() as thread_db:
            results = thread_db.execute(sql, params).fetchall()
            for row in results:
                doc_metadata = row.doc_metadata or {}
                # NeMo-style replacement for BM25 candidates
                text_content = doc_metadata.get("parent_text") or row.text_content
                candidates.append({
                    "id": str(row.id),
                    "text": text_content,
                    "score": 0.0,
                    "hybrid_score": 0.0,
                    "dense_score": 0.0,
                    "sparse_score": row.rank_score,
                    "file_type": row.file_type or "unknown",
                    "table_group": doc_metadata.get("table_group"),
                    "metadata": {
                        "tenant_id": tenant_id,
                        "source": row.doc_id,
                        "section": row.section,
                        "type": doc_metadata.get("type", "text"),
                        "page_num": doc_metadata.get("page_num"),
                        "embedding_model": "bm25_postgres",
                        "file_type": row.file_type or "unknown",
                        "entities": doc_metadata.get("entities", []),
                        "table_group": doc_metadata.get("table_group"),
                        # table-aware fields
                        "table_id": doc_metadata.get("table_id"),
                        "section_title": doc_metadata.get("section_title", ""),
                        "cell_values": doc_metadata.get("cell_values"),
                        "header_path": doc_metadata.get("header_path", []),
                        "row_index": doc_metadata.get("row_index"),
                    }
                })
    except Exception as e:
        print(f"[Retrieval] PostgreSQL BM25 failed: {e}")
    return candidates


def exact_catalogue_lookup(db: Session, query: str, tenant_id: str) -> list:
    """
    Extract catalogue/model number patterns from the query and do a direct
    SQL text_content LIKE search — no vector search needed for exact strings.
    Handles patterns like ECL2412SD, SNC2448L1125, EQL40200D, DK10-1A.
    """
    patterns = extract_catalogue_patterns(query)
    if not patterns:
        return []

    candidates = []
    seen_ids: set = set()
    for pattern in patterns[:3]:  # Cap to 3 patterns to avoid query sprawl
        sql = text("""
            SELECT id, doc_id, text_content, section, file_type, doc_metadata
            FROM document_chunks
            WHERE tenant_id = :tenant_id
              AND text_content ILIKE :pattern
            LIMIT 20
        """)
        try:
            results = db.execute(sql, {
                "tenant_id": tenant_id,
                "pattern": f"%{pattern}%",
            }).fetchall()
            for row in results:
                if row.id in seen_ids:
                    continue
                seen_ids.add(row.id)
                doc_metadata = row.doc_metadata or {}
                candidates.append({
                    "id": str(row.id),
                    "text": row.text_content,
                    "score": 2.0,           # Exact match gets highest score
                    "hybrid_score": 2.0,
                    "dense_score": 0.0,
                    "file_type": row.file_type or "unknown",
                    "table_group": doc_metadata.get("table_group"),
                    "metadata": {
                        "tenant_id": tenant_id,
                        "source": row.doc_id,
                        "section": row.section,
                        "type": doc_metadata.get("type", "text"),
                        "page_num": doc_metadata.get("page_num"),
                        "embedding_model": "exact_lookup",
                        "file_type": row.file_type or "unknown",
                        "entities": [],
                        "table_group": doc_metadata.get("table_group"),
                        "table_id": doc_metadata.get("table_id"),
                        "section_title": doc_metadata.get("section_title", ""),
                        "cell_values": doc_metadata.get("cell_values"),
                        "header_path": doc_metadata.get("header_path", []),
                        "row_index": doc_metadata.get("row_index"),
                        "match_pattern": pattern,
                    }
                })
        except Exception as e:
            print(f"[Retrieval] Exact catalogue lookup failed for {pattern}: {e}")

    if candidates:
        print(f"[Retrieval] Exact catalogue lookup found {len(candidates)} matches for patterns {patterns}")
    return candidates

def perform_hybrid_search(db: Session, query: str, tenant_id: str, top_k: int = 20, metadata_filters: dict = None, fast_path: bool = False) -> list:
    print(f"[Retrieval] Executing {'fast ' if fast_path else ''}hybrid search for tenant={tenant_id!r}, query={query!r}, filters={metadata_filters}")

    qdrant = get_qdrant_client()

    hyde_text = None
    hyde_vector = None
    vision_vector = None
    query_entities = []
    bm25_results = []
    exact_results = []  # Catalogue number exact matches

    # --- Exact catalogue lookup (fast, zero-latency SQL path) ---
    if TABLE_ENGINE_AVAILABLE and not fast_path:
        exact_results = exact_catalogue_lookup(db, query, tenant_id)

    if fast_path:
        query_vector = encode_text(query)
    else:
        from concurrent.futures import ThreadPoolExecutor, TimeoutError
        with ThreadPoolExecutor(max_workers=3) as executor:
            fut_dense = executor.submit(encode_text, query)
            fut_hyde = executor.submit(_generate_hyde, query)
            fut_bm25 = executor.submit(_perform_postgres_bm25, query, tenant_id, max(top_k * 4, 50), metadata_filters)

            query_vector = fut_dense.result()
            
            try:
                # Add strict 0.3s timeout to HyDE to prevent it from bottlenecking instant retrieval
                hyde_text = fut_hyde.result(timeout=0.3)
            except TimeoutError:
                hyde_text = None
                print("[Retrieval] HyDE generation timed out (exceeded 300ms) - bypassing to guarantee low latency.")
                
            hyde_vector = encode_text(hyde_text) if hyde_text else None
            bm25_results = fut_bm25.result()
    
    
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
        if "target_file" in metadata_filters and metadata_filters["target_file"]:
            must_conditions.append(models.FieldCondition(key="metadata.source", match=models.MatchText(text=metadata_filters["target_file"])))
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



    # BM25 Sparse Search (Postgres)
    if not fast_path and bm25_results:
        for rank, bm25_candidate in enumerate(bm25_results, start=1):
            point_id = bm25_candidate["id"]
            if point_id not in candidates:
                candidates[point_id] = bm25_candidate
            else:
                candidates[point_id]["sparse_score"] = bm25_candidate["sparse_score"]
            candidates[point_id]["hybrid_score"] += BM25_WEIGHT * _rrf_score(rank)
            candidates[point_id]["metadata"]["bm25_rank"] = rank

    # Inject exact catalogue lookup results at the top of the candidate pool
    if exact_results:
        for exact_candidate in exact_results:
            cid = exact_candidate["id"]
            if cid not in candidates:
                candidates[cid] = exact_candidate
            else:
                # Boost existing candidate's score
                candidates[cid]["hybrid_score"] = max(
                    candidates[cid]["hybrid_score"] + 1.5, exact_candidate["hybrid_score"]
                )



    ranked = sorted(
        candidates.values(),
        key=lambda item: item["hybrid_score"],
        reverse=True,
    )
    
    # Deduplicate expanded parent blocks so the LLM doesn't see identical text multiple times
    dedup_ranked = []
    seen_texts = set()
    for item in ranked:
        if item["text"] not in seen_texts:
            seen_texts.add(item["text"])
            item["score"] = item["hybrid_score"]
            dedup_ranked.append(item)

    # Table-aware expansion: if top results include table chunks, fetch all
    # sibling chunks from the same table_group so the LLM sees the complete table
    top_results = dedup_ranked[:top_k]
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

