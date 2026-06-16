import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

def get_qdrant_client():
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def init_qdrant_collections(dim: int = 1024):
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
