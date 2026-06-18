import os
from typing import Iterable, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

def get_qdrant_client():
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def init_qdrant_collections(dim: Optional[int] = None):
    if dim is None:
        dim = int(os.getenv("RAG_EMBEDDING_DIM", "1024"))
    client = get_qdrant_client()
    collections = ["document_chunks", "image_chunks"]
    for col in collections:
        existing_collections = [c.name for c in client.get_collections().collections]
        if col not in existing_collections:
            client.create_collection(
                collection_name=col,
                vectors_config=models.VectorParams(
                    size=dim if col == "document_chunks" else 768, # CLIP dimension is 768
                    distance=models.Distance.COSINE
                )
            )

def insert_qdrant_points(collection_name: str, points: list):
    client = get_qdrant_client()
    client.upsert(
        collection_name=collection_name,
        points=points
    )

def delete_qdrant_points_by_source(
    tenant_id: str,
    doc_ids: Iterable[str],
    collections: tuple[str, ...] = ("document_chunks", "image_chunks"),
) -> None:
    """Delete all Qdrant vectors for source documents before a forced reindex."""
    client = get_qdrant_client()
    doc_id_list = sorted(set(doc_ids))
    if not doc_id_list:
        return
        
    source_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="tenant_id",
                match=models.MatchValue(value=tenant_id),
            ),
            models.FieldCondition(
                key="doc_id",
                match=models.MatchAny(any=doc_id_list),
            ),
        ]
    )
    selector = models.FilterSelector(filter=source_filter)
    for collection_name in collections:
        client.delete(
            collection_name=collection_name,
            points_selector=selector,
            wait=True,
        )

def delete_qdrant_points(tenant_id: str, doc_id: str):
    """Deletes vectors from Qdrant by tenant_id and doc_id for both collections."""
    client = get_qdrant_client()
    for col in ["document_chunks", "image_chunks"]:
        try:
            client.delete(
                collection_name=col,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id)),
                            models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
                        ]
                    )
                )
            )
            print(f"[Qdrant] Deleted stale points for {doc_id} in {col}")
        except Exception as e:
            print(f"[Qdrant] Failed to delete points in {col} for {doc_id}: {e}")
