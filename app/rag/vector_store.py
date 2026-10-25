"""Qdrant access: one collection with a dense vector + a BM25 sparse vector.

Sized for 1M+ chunks on modest hardware:
  * int8 scalar quantization kept in RAM, full-precision vectors on disk
    (~0.4 GB RAM per 1M x 384-d instead of 1.5 GB) with rescoring;
  * BM25 as a server-side sparse index with IDF weighting (no Postgres FTS);
  * small keyword payload (tenant, document, collection, identifiers) with
    payload indexes so filters stay fast. Chunk text lives in Postgres.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from qdrant_client import QdrantClient, models

from app.config import get_settings

log = logging.getLogger(__name__)
DENSE = "dense"
SPARSE = "bm25"
_client: Optional[QdrantClient] = None
_client_lock = threading.Lock()


def client() -> QdrantClient:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                s = get_settings()
                _client = QdrantClient(url=s.qdrant_url, timeout=s.qdrant_timeout, prefer_grpc=False)
    return _client


def ensure_collection() -> None:
    s = get_settings()
    c = client()
    name = s.qdrant_collection
    if c.collection_exists(name):
        info = c.get_collection(name)
        vectors = info.config.params.vectors
        size = vectors[DENSE].size if isinstance(vectors, dict) and DENSE in vectors else None
        if size != s.embedding_dim:
            raise RuntimeError(
                f"Qdrant collection '{name}' has dense size {size}, expected {s.embedding_dim}. "
                "Set RAG_QDRANT_COLLECTION to a new name (and re-index) after changing the embedding model.")
    else:
        c.create_collection(
            collection_name=name,
            vectors_config={DENSE: models.VectorParams(size=s.embedding_dim, distance=models.Distance.COSINE, on_disk=True)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(
                modifier=models.Modifier.IDF, index=models.SparseIndexParams(on_disk=False))},
            quantization_config=models.ScalarQuantization(scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8, quantile=0.99, always_ram=True)),
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
            on_disk_payload=True,
        )
        log.info("created qdrant collection", extra={"collection": name})
    for field, tenant in (("tenant_id", True), ("document_id", False), ("collection", False),
                          ("file_type", False), ("content_type", False), ("ids", False)):
        try:
            c.create_payload_index(name, field, field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD, is_tenant=tenant))
        except Exception:  # index already exists
            pass


@dataclass
class PointIn:
    id: int
    dense: List[float]
    sparse: object  # fastembed SparseEmbedding
    payload: Dict


def upsert(points: Sequence[PointIn], wait: bool = True) -> None:
    s = get_settings()
    client().upsert(
        collection_name=s.qdrant_collection,
        wait=wait,
        points=[models.PointStruct(
            id=p.id,
            vector={DENSE: p.dense, SPARSE: models.SparseVector(
                indices=p.sparse.indices.tolist(), values=p.sparse.values.tolist())},
            payload=p.payload) for p in points],
    )


def delete_document(document_id: str) -> None:
    s = get_settings()
    client().delete(
        collection_name=s.qdrant_collection,
        points_selector=models.FilterSelector(filter=models.Filter(must=[
            models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))])),
        wait=True,
    )


def count(tenant_id: Optional[str] = None) -> int:
    s = get_settings()
    flt = None
    if tenant_id:
        flt = models.Filter(must=[models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id))])
    return client().count(s.qdrant_collection, count_filter=flt, exact=False).count


def ping() -> bool:
    try:
        client().get_collections()
        return True
    except Exception:
        return False


def _filter(tenant_id: str, collections: Iterable[str] = (), file_types: Iterable[str] = (),
            document_ids: Iterable[str] = (), ids: Iterable[str] = ()) -> models.Filter:
    must = [models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id))]
    for key, values in (("collection", collections), ("file_type", file_types),
                        ("document_id", document_ids), ("ids", ids)):
        values = [v for v in values if v]
        if values:
            must.append(models.FieldCondition(key=key, match=models.MatchAny(any=list(values))))
    return models.Filter(must=must)


def hybrid_search(
    dense: List[float], sparse, tenant_id: str, *, identifiers: Sequence[str] = (),
    collections: Sequence[str] = (), file_types: Sequence[str] = (), document_ids: Sequence[str] = (),
    limit: int = 40,
) -> Dict[str, List[tuple]]:
    """Run dense, sparse and exact-identifier retrieval in ONE batched request.

    Returns {"dense": [(id, score)], "sparse": [...], "ids:<code>": [...]}; fusion is
    done by the caller so each signal can be weighted.
    """
    s = get_settings()
    base = _filter(tenant_id, collections, file_types, document_ids)
    params = models.SearchParams(hnsw_ef=s.hnsw_ef, quantization=models.QuantizationSearchParams(
        rescore=True, oversampling=2.0))
    requests = [
        models.QueryRequest(query=dense, using=DENSE, filter=base, limit=limit, params=params, with_payload=False),
        models.QueryRequest(query=models.SparseVector(indices=sparse.indices.tolist(), values=sparse.values.tolist()),
                            using=SPARSE, filter=base, limit=limit, with_payload=False),
    ]
    # One exact-identifier lookup per code, so a rare code ("DB372.DBD520") is never
    # crowded out by a frequent one ("PLC-04") asked about in the same question.
    id_list = list(dict.fromkeys(identifiers))[:4]
    for ident in id_list:
        requests.append(models.QueryRequest(
            query=dense, using=DENSE, limit=limit, params=params, with_payload=False,
            filter=_filter(tenant_id, collections, file_types, document_ids, [ident])))
    results = client().query_batch_points(s.qdrant_collection, requests=requests)
    keys = ["dense", "sparse"] + [f"ids:{i}" for i in id_list]
    return {keys[i]: [(int(p.id), float(p.score)) for p in r.points] for i, r in enumerate(results)}
