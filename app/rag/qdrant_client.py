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
    collections_config = {
        "document_chunks": dim,
        "image_chunks": 768,  # CLIP dimension
        "section_embeddings": dim,  # v6.0: section-level search
    }
    existing_collections = [c.name for c in client.get_collections().collections]
    for col, vec_dim in collections_config.items():
        if col not in existing_collections:
            client.create_collection(
                collection_name=col,
                vectors_config=models.VectorParams(
                    size=vec_dim,
                    distance=models.Distance.COSINE
                )
            )
            print(f"[Qdrant] Created collection '{col}' with dim={vec_dim}")

    # Create payload indexes for section-aware retrieval (v6.0)
    _ensure_payload_index(client, "document_chunks", "metadata.section_id", models.PayloadSchemaType.KEYWORD)
    _ensure_payload_index(client, "section_embeddings", "tenant_id", models.PayloadSchemaType.KEYWORD)
    _ensure_payload_index(client, "section_embeddings", "embedding_model", models.PayloadSchemaType.KEYWORD)
    _ensure_payload_index(client, "section_embeddings", "section_id", models.PayloadSchemaType.KEYWORD)
    _ensure_payload_index(client, "section_embeddings", "document_id", models.PayloadSchemaType.KEYWORD)
    _ensure_payload_index(client, "section_embeddings", "level", models.PayloadSchemaType.KEYWORD)


def _ensure_payload_index(
    client,
    collection_name: str,
    field_name: str,
    field_schema,
):
    """Create a payload index if the collection exists, ignoring errors."""
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema,
        )
    except Exception:
        pass  # Index already exists or collection doesn't exist yet

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
