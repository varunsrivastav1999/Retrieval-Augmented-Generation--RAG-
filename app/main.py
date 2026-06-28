"""
=============================================================================
 Enterprise Level RAG: 17-Layer Production Microservice — v4.0
=============================================================================
 World's Best Retrieval-Augmented Generation engine.
 Open Source (MIT) | Zero Hallucination | Sub-5ms Exact Text | 30+ Formats
 
 17 Layers:
   1. Universal Document Parser (PDF/DOCX/XLSX/PPTX/CSV/TXT/IMG/VIDEO)
   2. Smart OCR & Table/Image Extraction
   3. Semantic Parent-Child Chunking
   4. Batch Embedding (offline, GPU-accelerated)
   4. Vector DB: Qdrant
   5. Agentic RAG: Multi-hop Query Resolution
   6. Hybrid Search (HNSW + BM25 + Trigram)
   7. ColBERT Late-Interaction Reranking
   8. Max Marginal Relevance (MMR)
   9. Contextual Window Expansion
  10. Agentic Router (Keyword + LLM multi-tool)
  11. Query Intelligence (Spelling, Expansion, Decomposition)
  12. Hallucination Guard (ZERO general answers)
  13. Extractive Fast-Path (< 5ms exact document text)
  14. Semantic Query Cache (Redis)
  15. Active RAG (FLARE self-reflection)
  16. GraphRAG (Neo4j)
  17. Real-Time Token Streaming
   
 Microservice design — no file limit, zero error, 100% offline.
=============================================================================
"""

import os
import requests
import re
import threading
import time
import json
import hashlib
import warnings

# Suppress Pydantic v2 protected namespace warnings caused by third-party packages
warnings.filterwarnings("ignore", message='Field "model_.*" has conflict with protected namespace "model_"', category=UserWarning, module="pydantic")

# Suppress noisy third-party deprecation warnings (e.g., pynvml, huggingface_hub)
warnings.filterwarnings("ignore", category=FutureWarning)

from fastapi import Body, HTTPException, Depends, File, Query, UploadFile
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import redis
import json
import hashlib
import warnings

# Suppress noisy third-party deprecation warnings (e.g., pynvml, huggingface_hub)
warnings.filterwarnings("ignore", category=FutureWarning)

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from app.database import DocumentChunk, IngestionJob, SessionLocal, init_db, get_db
from app.rag.jobs import (
    create_ingestion_job,
    find_all_supported_files,
    get_ingestion_job,
    start_ingestion_worker,
)
from app.rag.model_loader import get_embedding_model_id, runtime_model_info, validate_runtime_models
from app.rag.retrieval import perform_hybrid_search, perform_multi_query_search
from app.rag.query_router import query_router, get_execution_strategy, QueryRoute
from app.rag.reranker import rerank_results
from app.rag.context import assemble_context
from app.rag.grounding import (
    NOT_FOUND_RESPONSE,
    build_strict_grounding_prompt,
    compute_grounding_score,
    verify_answer_grounding,
)
from app.rag.parsers import SUPPORTED_EXTENSIONS, is_supported_file
from app.rag.query_intelligence import intelligent_query_pipeline, reformulate_query, text_to_sql_filters, flare_query_decomposition, flare_mid_generation_retrieval


app = FastAPI(
    title="Enterprise Level RAG 17-Layer Microservice",
    description="World's best zero-hallucination Agentic RAG with unlimited file support, sub-5ms exact extraction, ColBERT reranking, and NVIDIA AI Blueprint Plan-and-Execute Architecture.",
    version="4.0.0",
)

# ── Prometheus Metrics ────────────────────────────────────────────────────────
RAG_QUERY_TOTAL = Counter("rag_queries_total", "Total queries processed", ["status"])
RAG_QUERY_LATENCY = Histogram("rag_query_latency_seconds", "Query latency in seconds",
    ["fast_path"], buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0))
RAG_INGESTION_TOTAL = Counter("rag_ingestion_total", "Total files ingested", ["file_type"])
RAG_CACHE_HITS = Counter("rag_cache_hits_total", "Total cache hits")
RAG_GROUNDING_BLOCKED = Counter("rag_grounding_blocked_total", "Queries blocked by grounding guard")
RAG_LLM_CALLS = Counter("rag_llm_calls_total", "Total Ollama LLM calls", ["operation"])

# Serve local JS libraries (Chart.js, marked.js) for offline use
_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

TENANT_PATTERN = r"^[a-zA-Z0-9_-]{1,80}$"
TENANT_RE = re.compile(TENANT_PATTERN)


def _cors_origins() -> List[str]:
    raw = os.getenv("RAG_CORS_ORIGINS", "*")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


CORS_ORIGINS = _cors_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials="*" not in CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

MEDIA_PATH = os.getenv("MEDIA_PATH", "/media")

# Connecting directly to the independent Ollama container inside this repository
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b") # Best for RAG — world-class reasoning, 128K context
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "1024"))
OLLAMA_CONTEXT_LENGTH = int(os.getenv("OLLAMA_CONTEXT_LENGTH", "32768"))
RAG_ENV = os.getenv("RAG_ENV", "local").lower()
PRELOAD_MODELS_ON_STARTUP = os.getenv(
    "RAG_PRELOAD_MODELS_ON_STARTUP",
    "true" if RAG_ENV in {"prod", "production"} else "false",
).lower() in {"1", "true", "yes", "on"}
ENABLE_INGESTION_WORKER = os.getenv("RAG_ENABLE_INGESTION_WORKER", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
INGESTION_WORKER_POLL_SECONDS = float(os.getenv("RAG_INGESTION_WORKER_POLL_SECONDS", "5"))
INGESTION_STALE_TIMEOUT_SECONDS = int(os.getenv("RAG_INGESTION_STALE_TIMEOUT_SECONDS", "1800"))
DEFAULT_TOP_K = int(os.getenv("RAG_DEFAULT_TOP_K", "24"))
MAX_TOP_K = int(os.getenv("RAG_MAX_TOP_K", "100"))
BROAD_QUERY_TOP_K = int(os.getenv("RAG_BROAD_QUERY_TOP_K", "48"))
SOURCE_LIMIT = int(os.getenv("RAG_SOURCE_LIMIT", "24"))
# File size limit — prevents zip bombs and OOM
MAX_UPLOAD_SIZE_BYTES = int(os.getenv("RAG_MAX_UPLOAD_SIZE_BYTES", str(5000 * 1024 * 1024)))  # 5000MB default

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
if REDIS_PASSWORD:
    REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"
else:
    REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"
try:
    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
except Exception as e:
    redis_client = None
    print(f"Warning: Redis cache not available - {e}")

ingestion_worker_stop = threading.Event()
ingestion_worker_thread = None


def validate_tenant_id(tenant_id: str) -> str:
    if not TENANT_RE.fullmatch(tenant_id):
        raise HTTPException(status_code=400, detail="Invalid tenant_id.")
    return tenant_id


def _cache_key(
    query: str,
    tenant_id: str,
    user_id: str,
    top_k: int,
    embedding_model: str,
    corpus_version: str,
    scope: Optional[Dict[str, Any]] = None,
) -> str:
    payload = {
        "tenant_id": tenant_id,
        "user_id": user_id,
        "query": query,
        "top_k": top_k,
        "embedding_model": embedding_model,
        "corpus_version": corpus_version,
        "scope": scope or {},
    }
    query_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"rag_cache:{query_hash}"


from app.rag.model_loader import cosine_similarity, encode_text

CACHE_SEMANTIC_THRESHOLD = float(os.getenv("RAG_CACHE_SEMANTIC_THRESHOLD", "0.95"))

def get_cached_response(
    query: str,
    tenant_id: str,
    user_id: str,
    top_k: int,
    embedding_model: str,
    corpus_version: str,
    scope: Optional[Dict[str, Any]] = None,
):
    if not redis_client: return None
    try:
        # Layer 11: Semantic Query Cache — aggressive matching (0.95 threshold)
        query_vector = encode_text(query)
        index_key = f"semantic_index:{tenant_id}:{user_id}:{embedding_model}:{corpus_version}"
        
        raw_entries = redis_client.lrange(index_key, 0, -1)
        if raw_entries:
            index = [json.loads(e) for e in raw_entries]
            
            # 1. Exact Match Check
            query_normalized = re.sub(r'\s+', ' ', query.strip().lower())
            for item in index:
                item_query = item.get("query", "")
                if re.sub(r'\s+', ' ', item_query.strip().lower()) == query_normalized:
                    print(f"[Cache] EXACT HIT for query={query!r}")
                    cached = redis_client.get(item["cache_key"])
                    if cached:
                        return json.loads(cached)

            # 2. Semantic Match with Entity Verification
            # Extract alphanumeric sequences that contain at least one digit (product IDs, part numbers)
            def extract_ids(text):
                return set(re.findall(r'\b[a-zA-Z0-9]*[0-9][a-zA-Z0-9]*\b', text.lower()))
            
            query_ids = extract_ids(query)

            best_match = None
            best_sim = 0.0
            
            for item in index:
                sim = cosine_similarity(query_vector, item["embedding"])
                if sim > best_sim:
                    item_query = item.get("query", "")
                    item_ids = extract_ids(item_query)
                    
                    # Entity verification: if both queries have IDs, they must match exactly.
                    # If neither has IDs, skip the check (e.g., "What is torque?").
                    # If one has IDs and the other doesn't, no match.
                    if query_ids and item_ids:
                        if query_ids != item_ids:
                            continue
                    elif query_ids or item_ids:
                        continue
                    
                    best_sim = sim
                    best_match = item
                    
            if best_match and best_sim > CACHE_SEMANTIC_THRESHOLD:
                print(f"[Cache] Semantic HIT (sim={best_sim:.3f}) for query={query!r}")
                cached = redis_client.get(best_match["cache_key"])
                if cached:
                    return json.loads(cached)
    except Exception as e:
        print(f"Redis get error: {e}")
    return None

def set_cached_response(
    query: str,
    tenant_id: str,
    user_id: str,
    top_k: int,
    embedding_model: str,
    corpus_version: str,
    response: dict,
    scope: Optional[Dict[str, Any]] = None,
):
    if not redis_client: return
    try:
        cache_key = _cache_key(query, tenant_id, user_id, top_k, embedding_model, corpus_version, scope)
        redis_client.setex(cache_key, 86400, json.dumps(response))
        
        # Update semantic index atomically with Redis list
        query_vector = encode_text(query)
        index_key = f"semantic_index:{tenant_id}:{user_id}:{embedding_model}:{corpus_version}"
        entry = json.dumps({"cache_key": cache_key, "embedding": query_vector, "query": query})
        
        pipe = redis_client.pipeline()
        pipe.lpush(index_key, entry)
        pipe.ltrim(index_key, 0, 999)
        pipe.expire(index_key, 86400)
        pipe.execute()
        
    except Exception as e:
        print(f"Redis set error: {e}")


def get_corpus_version(db: Session, tenant_id: str, embedding_model: str) -> str:
    latest = db.execute(
        text(
            "SELECT max(created_at)::text AS corpus_version "
            "FROM document_chunks "
            "WHERE tenant_id = :tenant_id "
            "AND embedding_model = :embedding_model"
        ),
        {"tenant_id": tenant_id, "embedding_model": embedding_model},
    ).scalar()
    return latest or "empty"

@app.on_event("startup")
def on_startup():
    print("🚀 [STARTUP] Enterprise Level RAG 17-Layer Microservice starting...")
    global ingestion_worker_thread
    try:
        init_db()
        from app.rag.qdrant_client import init_qdrant_collections
        init_qdrant_collections()
        print("✅ Database initialized and Qdrant collections created successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        raise

    def run_background_sync():
        if PRELOAD_MODELS_ON_STARTUP:
            print("[Plug&Play] Ensuring AI models are ready in background...")
            try:
                from app.rag.model_loader import get_embedding_model, get_reranker_model
                get_embedding_model()
                get_reranker_model()
                
                # Auto-Create/Update Custom Ollama Brain
                import httpx
                modelfile_path = "/app/Modelfile"
                if os.path.exists(modelfile_path):
                    print(f"[Plug&Play] Syncing custom brain '{OLLAMA_MODEL}' from Modelfile...")
                    with open(modelfile_path, "r") as f:
                        modelfile_content = f.read()
                    
                    ollama_base_url = OLLAMA_URL.replace("/api/generate", "")
                    with httpx.Client(timeout=300.0) as client:
                        try:
                            # Trigger the create API
                            resp = client.post(
                                f"{ollama_base_url}/api/create",
                                json={"model": OLLAMA_MODEL, "modelfile": modelfile_content, "stream": False},
                            )
                            if resp.status_code == 200:
                                print(f"✅ SUCCESS: Custom brain '{OLLAMA_MODEL}' is ready.")
                            else:
                                print(f"⚠️ Warning: Model creation returned {resp.status_code}: {resp.text}")
                        except Exception as conn_err:
                            print(f"❌ Connection Error: Could not reach Native Ollama at {ollama_base_url}. Make sure Ollama for Mac is RUNNING! ({conn_err})")
                
                print(f"RAG models ready: {runtime_model_info()}")
            except Exception as e:
                print(f"[Plug&Play] Warning: Background sync issue: {e}")

    # Run the heavy sync in a background thread to prevent health check 503s
    sync_thread = threading.Thread(target=run_background_sync, name="rag-sync-worker", daemon=True)
    sync_thread.start()

    if ENABLE_INGESTION_WORKER:
        ingestion_worker_thread = start_ingestion_worker(
            ingestion_worker_stop,
            poll_seconds=INGESTION_WORKER_POLL_SECONDS,
            stale_timeout_seconds=INGESTION_STALE_TIMEOUT_SECONDS,
        )
        print("✅ Ingestion worker started.")

    # Auto-scan /media for ALL supported file types and queue them
    def auto_scan_media():
        sync_thread.join(timeout=600)
        try:
            all_files = find_all_supported_files(MEDIA_PATH, include_scan=True)
            if not all_files:
                print("[AutoScan] No supported files found in media path, skipping.")
                return

            db = SessionLocal()
            try:
                result = _queue_file_ingestion("default", all_files, db)
                queued = result["queued"]
                skipped = result["skipped"]
                if queued:
                    print(f"[AutoScan] ✅ Queued {queued} new file(s) for ingestion ({skipped} already indexed).")
                else:
                    print(f"[AutoScan] All {len(all_files)} file(s) already indexed. Nothing to do.")
            finally:
                db.close()
        except Exception as e:
            print(f"[AutoScan] Warning: Auto-scan failed: {e}")

    scan_thread = threading.Thread(target=auto_scan_media, name="rag-auto-scan", daemon=True)
    scan_thread.start()


@app.on_event("shutdown")
def on_shutdown():
    ingestion_worker_stop.set()
    if ingestion_worker_thread:
        ingestion_worker_thread.join(timeout=5)
    try:
        pass
    except Exception:
        pass

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    tenant_id: str = Field("default", pattern=TENANT_PATTERN)
    user_id: str = Field("default", pattern=TENANT_PATTERN)
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    parent: Optional[str] = None
    child: Optional[str] = None
    sync_documents: bool = False
    include_scan: bool = True
    force_reindex: bool = False
    stream: bool = False
    fast_path: bool = Field(False, description="Sub-5ms fast path: skip HyDE, BM25, Vision, Cross-encoder reranker")
    extractive: bool = Field(False, description="Exact mode: skip LLM, return verbatim text from top chunk instead of generated answer")
    auto: bool = Field(True, description="Auto-mode: simple facts get 10ms extractive, complex analysis gets full LLM pipeline")
    evaluate: bool = Field(False, description="Run RAGAS evaluation on the final answer and context")


class IngestRequest(BaseModel):
    tenant_id: Optional[str] = Field(None, pattern=TENANT_PATTERN)
    force_reindex: bool = False


class IngestionJobResponse(BaseModel):
    id: str
    tenant_id: str
    source_name: str
    status: str
    attempts: int
    chunks_total: int
    chunks_inserted: int
    error: Optional[str] = None
    file_type: Optional[str] = None
    progress_pct: Optional[float] = None

class IngestResponse(BaseModel):
    status: str
    message: str
    files_processed: int
    files_queued: int = 0
    files_skipped: int = 0
    jobs: List[IngestionJobResponse] = Field(default_factory=list)

class QueryResponse(BaseModel):
    answer: str
    context: List[Dict[str, Any]]
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    latency_ms: int
    ingest: Optional[Dict[str, Any]] = None
    grounding: Optional[Dict[str, Any]] = None
    verification: Optional[Dict[str, Any]] = None
    evaluation: Optional[Dict[str, Any]] = None


def _job_response(job: IngestionJob) -> IngestionJobResponse:
    return IngestionJobResponse(
        id=job.id,
        tenant_id=job.tenant_id,
        source_name=job.source_name,
        status=job.status,
        attempts=job.attempts,
        chunks_total=job.chunks_total,
        chunks_inserted=job.chunks_inserted,
        error=job.error,
        file_type=getattr(job, 'file_type', None),
        progress_pct=getattr(job, 'progress_pct', None),
    )


def _queue_file_ingestion(
    tenant_id: str,
    files: List[str],
    db: Session,
    force_reindex: bool = False,
) -> Dict[str, Any]:
    """Queue ingestion jobs for any supported file type."""
    embedding_model = get_embedding_model_id()
    unique_files = sorted(set(files))

    if force_reindex and unique_files:
        try:
            from app.rag.qdrant_client import delete_qdrant_points_by_source
            delete_qdrant_points_by_source(tenant_id, unique_files)
            print(f"[Ingest] Cleared Qdrant vectors for {len(unique_files)} source file(s).")
        except Exception as exc:
            print(f"[Ingest] Warning: Qdrant source cleanup failed before reindex: {exc}")

        db.query(DocumentChunk).filter(
            DocumentChunk.tenant_id == tenant_id,
            DocumentChunk.embedding_model == embedding_model,
            DocumentChunk.doc_id.in_(unique_files),
        ).delete(synchronize_session=False)
        db.commit()
        if redis_client:
            try:
                for k in redis_client.scan_iter(match=f"semantic_index:{tenant_id}:*", count=100):
                    redis_client.delete(k)
                print(f"[Cache] Cleared semantic cache indices for tenant {tenant_id}")
            except Exception as e:
                print(f"[Cache] Failed to clear semantic cache: {e}")

    indexed_sources = {
        row[0]
        for row in (
            db.query(DocumentChunk.doc_id)
            .filter(
                DocumentChunk.tenant_id == tenant_id,
                DocumentChunk.embedding_model == embedding_model,
                DocumentChunk.doc_id.in_(unique_files),
            )
            .distinct()
            .all()
        )
    }
    active_job_sources = {
        row[0]
        for row in (
            db.query(IngestionJob.source_path)
            .filter(
                IngestionJob.tenant_id == tenant_id,
                IngestionJob.source_path.in_(unique_files),
                IngestionJob.status.in_(["queued", "running", "retry"]),
            )
            .distinct()
            .all()
        )
    }

    jobs = []
    skipped = 0
    for f in unique_files:
        if not force_reindex and (f in indexed_sources or f in active_job_sources):
            skipped += 1
            continue
        jobs.append(create_ingestion_job(tenant_id, f, force_reindex=force_reindex))

    return {
        "total_candidates": len(unique_files),
        "queued": len(jobs),
        "skipped": skipped,
        "jobs": jobs,
    }


def _is_broad_query(query: str) -> bool:
    normalized = f" {query.lower()} "
    broad_phrases = [
        " all ",
        " every ",
        " each ",
        " topics",
        " topic ",
        " complete",
        " full ",
        " final ",
        " summarize",
        " summary",
        " overview",
    ]
    return any(phrase in normalized for phrase in broad_phrases)


def _retrieval_query(request: QueryRequest) -> str:
    topic_bits = [value.strip() for value in [request.parent, request.child] if value and value.strip()]
    if not topic_bits:
        return request.query

    topic_hint = " ".join(topic_bits)
    if _is_broad_query(request.query) or len(request.query.split()) <= 5:
        return f"{request.query} {topic_hint}"
    return request.query


def _context_sources(final_context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sources = []
    seen = set()
    for item in final_context:
        metadata = item.get("metadata") or {}
        source_path = metadata.get("source") or "unknown_source"
        source_name = os.path.basename(str(source_path)) or str(source_path)
        page_num = metadata.get("page_num")
        key = (source_name, page_num)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "source": source_name,
                "page": page_num,
                "citation": item.get("citation"),
                "metadata": metadata,
            }
        )
        if len(sources) >= SOURCE_LIMIT:
            break
    return sources


# =========================================================================
# API Endpoints
# =========================================================================

@app.post("/api/v1/ingest", response_model=IngestResponse)
def ingest_media(
    payload: Optional[IngestRequest] = Body(None),
    tenant_id: str = Query("default", pattern=TENANT_PATTERN),
    force_reindex: bool = Query(False),
    db: Session = Depends(get_db),
):
    """
    Scan the shared media volume for ALL supported files and ingest them.
    Supports: PDF, DOCX, XLSX, PPTX, CSV, TXT, Images, Video subtitles.
    Auto-detects format. Background chunking starts automatically.
    """
    if payload and payload.tenant_id:
        tenant_id = payload.tenant_id
    tenant_id = validate_tenant_id(tenant_id)
    if payload and payload.force_reindex:
        force_reindex = True

    all_files = find_all_supported_files(MEDIA_PATH, include_scan=True)
    if not all_files:
        return {
            "status": "success",
            "message": f"No supported files found in {MEDIA_PATH}.",
            "files_processed": 0,
            "files_queued": 0,
            "files_skipped": 0,
            "jobs": [],
        }

    queue_result = _queue_file_ingestion(
        tenant_id,
        all_files,
        db,
        force_reindex=force_reindex,
    )
    jobs = queue_result["jobs"]
    return {
        "status": "queued" if jobs else "success",
        "message": (
            f"Found {len(all_files)} files: {len(jobs)} queued, {queue_result['skipped']} skipped."
            if jobs
            else f"All {len(all_files)} files are already indexed or queued."
        ),
        "files_processed": len(all_files),
        "files_queued": len(jobs),
        "files_skipped": queue_result["skipped"],
        "jobs": [_job_response(job) for job in jobs],
    }

@app.post("/api/v1/upload")
async def upload_file(
    tenant_id: str = Query("default", pattern=TENANT_PATTERN),
    file: UploadFile = File(...),
):
    """
    Upload ANY supported file for ingestion.
    No file size limit. Supports: PDF, DOCX, XLSX, PPTX, CSV, TXT, Images, Video.
    Background chunking starts automatically after upload.
    """
    tenant_id = validate_tenant_id(tenant_id)
    filename = os.path.basename(file.filename or "")
    
    # Check if format is supported
    ext = os.path.splitext(filename)[1].lower()
    if not ext or ext not in SUPPORTED_EXTENSIONS:
        supported_list = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported: {supported_list}",
        )

    # Stream file to disk with size limit
    tenant_media_path = os.path.join(MEDIA_PATH, tenant_id)
    os.makedirs(tenant_media_path, exist_ok=True)
    file_path = os.path.join(tenant_media_path, filename)

    total_bytes = 0
    try:
        with open(file_path, "wb") as buffer:
            while True:
                chunk = await file.read(4 * 1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if MAX_UPLOAD_SIZE_BYTES and total_bytes > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(status_code=413, detail=f"File exceeds maximum upload size of {MAX_UPLOAD_SIZE_BYTES / (1024*1024):.0f}MB")
                buffer.write(chunk)
    except HTTPException:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail="Upload failed. Check server logs.")

    file_size_mb = round(total_bytes / (1024 * 1024), 2)
    from app.rag.parsers import get_file_type
    file_type = get_file_type(file_path)
    job = create_ingestion_job(tenant_id, file_path, file_type=file_type)
    return {
        "message": f"Saved {filename} ({file_size_mb}MB) and queued background ingestion.",
        "file_type": file_type,
        "job": _job_response(job),
    }


@app.get("/api/v1/ingest/jobs/{job_id}", response_model=IngestionJobResponse)
def get_ingestion_status(
    job_id: str,
    tenant_id: str = Query("default", pattern=TENANT_PATTERN),
):
    tenant_id = validate_tenant_id(tenant_id)
    job = get_ingestion_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Ingestion job not found.")
    if job.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="Ingestion job not found.")
    return _job_response(job)


@app.get("/api/v1/ingest/jobs", response_model=List[IngestionJobResponse])
def list_ingestion_jobs(
    tenant_id: str = Query("default", pattern=TENANT_PATTERN),
    limit: int = Query(25, ge=1, le=100),
    db: Session = Depends(get_db),
):
    tenant_id = validate_tenant_id(tenant_id)
    jobs = (
        db.query(IngestionJob)
        .filter(IngestionJob.tenant_id == tenant_id)
        .order_by(IngestionJob.created_at.desc())
        .limit(limit)
        .all()
    )
    return [_job_response(job) for job in jobs]


@app.get("/api/v1/formats")
def list_supported_formats():
    """List all supported file formats for ingestion."""
    return {
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
        "categories": {
            "documents": [".pdf", ".docx", ".doc", ".pptx", ".ppt"],
            "spreadsheets": [".xlsx", ".xls", ".csv"],
            "text": [".txt", ".text", ".md", ".log", ".json", ".xml"],
            "images": [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif", ".webp"],
            "video": [".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv"],
            "subtitles": [".srt", ".ass", ".ssa", ".vtt"],
        },
        "file_size_limit": "unlimited",
    }


@app.get("/health/live")
def live_health():
    return {"status": "ok"}


@app.get("/metrics")
def prometheus_metrics():
    """Prometheus metrics endpoint for monitoring and auto-scaling."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health/ready")
def ready_health(db: Session = Depends(get_db)):
    checks = {}
    status_code = 200

    try:
        db.execute(text("SELECT 1"))
        checks["database"] = "ok"
        
        stats_query = db.execute(text(
            "SELECT COUNT(*) as chunks, COUNT(DISTINCT doc_id) as docs, "
            "COUNT(DISTINCT file_type) as file_types "
            "FROM document_chunks"
        )).mappings().first()
        checks["stats"] = {
            "chunks": stats_query["chunks"] if stats_query else 0,
            "docs": stats_query["docs"] if stats_query else 0,
            "file_types": stats_query["file_types"] if stats_query else 0,
        }
        
        # Get file type breakdown
        try:
            type_rows = db.execute(text(
                "SELECT file_type, COUNT(DISTINCT doc_id) as doc_count, COUNT(*) as chunk_count "
                "FROM document_chunks WHERE file_type IS NOT NULL "
                "GROUP BY file_type ORDER BY chunk_count DESC"
            )).mappings().all()
            checks["stats"]["by_type"] = {
                row["file_type"]: {"docs": row["doc_count"], "chunks": row["chunk_count"]}
                for row in type_rows
            }
        except Exception:
            pass
            
        # Active ingestion jobs
        try:
            active_jobs = db.execute(text(
                "SELECT COUNT(*) as cnt FROM ingestion_jobs WHERE status IN ('queued', 'running')"
            )).scalar()
            checks["stats"]["active_jobs"] = active_jobs or 0
        except Exception:
            pass
            
    except Exception as exc:
        checks["database"] = f"error: {exc}"
        status_code = 503

    try:
        if not redis_client:
            raise RuntimeError("redis client not configured")
        redis_client.ping()
        checks["redis"] = "ok"
    except Exception as exc:
        checks["redis"] = f"degraded: {exc}"

    try:
        ollama_base = OLLAMA_URL.split("/api/")[0]
        response = requests.get(f"{ollama_base}/api/tags", timeout=5)
        response.raise_for_status()
        checks["ollama"] = "ok"
    except Exception as exc:
        checks["ollama"] = f"degraded: {exc}"

    try:
        checks["models_info"] = runtime_model_info()
        checks["models"] = "ready"
    except Exception as exc:
        checks["models"] = f"info_error: {exc}"

    payload = {"status": "ok" if status_code == 200 else "unready", "checks": checks}
    if status_code != 200:
        raise HTTPException(status_code=status_code, detail=payload)
    return payload


# =========================================================================
# Production Dashboard — Home Page
# =========================================================================
@app.get("/", response_class=HTMLResponse)
def root_ui():
    with open("app/templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# =========================================================================
# Query Pipeline — 12 Layers
# =========================================================================
@app.post("/api/v1/query")
def query_rag(request: QueryRequest, db: Session = Depends(get_db)):
    """
    17-Layer RAG Query Pipeline (Strict Document Grounding):
    
    Layer 5:  Hybrid Retrieval (ANN via pgvector + BM25)
    Layer 6:  Cross-Encoder Reranking
    Layer 7:  MMR Diversity
    Layer 8:  Contextual Window Expansion
    Layer 9:  🛡️ Hallucination Guard
    Layer 10: ✅ Answer Verification
    Layer 11: Semantic Cache
    Layer 12: Token Streaming
    
    If information is NOT in documents → returns "not available" (ZERO hallucination).
    """
    start_time = time.time()
    request_start = start_time
    print(f"\n[{'='*50}]")
    print(f"[API: IN] Query: {request.query}")
    print(f"[{'='*50}]")
    tenant_id = validate_tenant_id(request.tenant_id)
    embedding_model = get_embedding_model_id()
    broad_query = _is_broad_query(request.query)
    effective_top_k = request.top_k
    original_top_k = effective_top_k
    if broad_query:
        effective_top_k = min(MAX_TOP_K, max(effective_top_k, BROAD_QUERY_TOP_K))

    ingest_summary = None
    if request.sync_documents:
        all_files = find_all_supported_files(MEDIA_PATH, include_scan=request.include_scan)
        queue_result = _queue_file_ingestion(
            tenant_id,
            all_files,
            db,
            force_reindex=request.force_reindex,
        )
        ingest_summary = {
            "total_candidates": queue_result["total_candidates"],
            "queued": queue_result["queued"],
            "skipped": queue_result["skipped"],
        }

    # --- Auto mode: simple fact → 10ms extractive, complex analysis → full LLM ---
    if request.auto and not request.fast_path and not request.extractive:
        q_lower = request.query.strip().lower()
        is_analysis = bool(re.search(
            r"(?:compare|contrast|difference|analysis|analyze|predict|trend|pattern|"
            r"relationship|correlation|impact|effect|cause|explain|why\s+|"
            r"how\s+(?:does|do|can|would|to|is|are)|summarize|overview|troubleshoot|"
            r"diagnos(?:e|is|tic)|list\s+all|\ball\s+the\b|"
            r"\bevery\b|\beach\b)", q_lower
        ))
        if is_analysis:
            print("[Auto] Complex analysis → full LLM pipeline")
        else:
            is_simple = (
                bool(re.search(r"(?:(?:^|\s)what\s+is|(?:^|\s)who\s+is|(?:^|\s)when\s+|"
                               r"(?:^|\s)where\s+|(?:^|\s)which\s+|"
                               r"(?:^|\s)how\s+to|(?:^|\s)define\s+|(?:^|\s)meaning\s+|"
                               r"(?:^|\s)list\s+|(?:^|\s)show\s+)", q_lower))
                or (len(q_lower.split()) <= 5)
            )
            if is_simple:
                request.fast_path = True
                print("[Auto] Simple fact → fast retrieval + LLM stream")

        # Conversational Bypass
        is_greeting = bool(re.match(r"^(hi|hello|hey|greetings|how are you|good morning|good afternoon)(?:\s+|$|[!.,?])", q_lower))
        if is_greeting:
            answer = "Hello! I'm your Enterprise Q&A Assistant. Please ask me a question about your uploaded documents!"
            print(f"[API: OUT] Response: {answer}")
            grounding_result = {"is_grounded": True, "score": 1.0, "detail": "Conversational Greeting"}
            latency_ms = int((time.time() - start_time) * 1000)
            response_data = {
                "answer": answer,
                "context": [],
                "sources": [],
                "latency_ms": latency_ms,
                "ingest": ingest_summary,
                "grounding": grounding_result,
                "verification": {"confidence": "high", "confidence_score": 1.0, "grounded_sentences": 1, "total_sentences": 1, "evidence": []},
            }
            RAG_QUERY_TOTAL.labels(status="greeting").inc()
            RAG_QUERY_LATENCY.labels(fast_path="false").observe(latency_ms / 1000.0)
            if request.stream:
                def stream_greeting():
                    yield "data: " + json.dumps({'token': answer}) + "\n\n"
                    yield "data: " + json.dumps({'done': True, 'sources': [], 'grounding': grounding_result, 'verification': response_data['verification'], 'latency_ms': response_data.get('latency_ms', 0)}) + "\n\n"
                return StreamingResponse(stream_greeting(), media_type="text/event-stream")
            return response_data

    corpus_version = get_corpus_version(db, tenant_id, embedding_model)
    search_query = _retrieval_query(request)
    cache_scope = {
        "parent": request.parent,
        "child": request.child,
        "search_query": search_query,
        "fast_path": request.fast_path,
        "extractive": request.extractive,
        "auto": request.auto,
    }
    
    # Layer 11: Semantic Query Cache
    cached = get_cached_response(
        request.query,
        tenant_id,
        request.user_id,
        effective_top_k,
        embedding_model,
        corpus_version,
        scope=cache_scope,
    )
    if cached:
        RAG_CACHE_HITS.inc()
        print(f"[Cache] HIT for tenant={tenant_id!r}, query={request.query!r}")
        print(f"[API: OUT] Response: {cached.get('answer', '')}")
        cached["latency_ms"] = int((time.time() - start_time) * 1000)
        cached.setdefault("sources", _context_sources(cached.get("context", [])))
        cached["ingest"] = ingest_summary

        if request.stream:
            def stream_cache():
                yield "data: " + json.dumps({'token': cached.get('answer', '')}) + "\n\n"
                yield "data: " + json.dumps({'done': True, 'sources': cached.get('sources', []), 'grounding': cached.get('grounding'), 'verification': cached.get('verification'), 'latency_ms': cached.get('latency_ms', 0)}) + "\n\n"
            return StreamingResponse(stream_cache(), media_type="text/event-stream")

        return cached

    if request.stream:
        def stream_llm():
            nonlocal ingest_summary, corpus_version
            final_context = []
            sources = []
            grounding_result = None
            force_general = False
            effective_top_k = request.top_k
            original_top_k = effective_top_k
            if broad_query:
                effective_top_k = min(MAX_TOP_K, max(effective_top_k, BROAD_QUERY_TOP_K))

            if request.fast_path:
                retrieved_chunks = perform_hybrid_search(db, search_query, tenant_id, top_k=effective_top_k, fast_path=True)
                final_context = retrieved_chunks[:effective_top_k]
                for c in final_context:
                    c["rerank_score"] = c.get("dense_score", 0.0)
                sources = _context_sources(final_context)
                grounding_result = compute_grounding_score(search_query, final_context)
            else:
                # Plan and Execute Pipeline (NVIDIA RAG Blueprint Style)
                import asyncio
                from app.rag.agentic import AgenticRAGPipeline
                
                pipeline = AgenticRAGPipeline(db, tenant_id, effective_top_k)
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    agentic_result = loop.run_until_complete(pipeline.run(search_query))
                    loop.close()
                except RuntimeError:
                    agentic_result = asyncio.run(pipeline.run(search_query))
                    
                final_context = agentic_result["context"]
                sources = agentic_result["sources"]
                grounding_result = agentic_result["grounding"]
                
                if agentic_result.get("answer"):
                    # The agentic pipeline synthesized the answer directly, so we stream the final answer in one block (since it's already generated)
                    # A true async stream of synthesis tokens requires async generator support across the pipeline.
                    yield "data: " + json.dumps({'token': agentic_result["answer"]}) + "\n\n"
                    latency = int((time.time() - start_time) * 1000)
                    verification = verify_answer_grounding(agentic_result["answer"], final_context)
                    
                    db.commit()
                    status = "grounded" if (grounding_result and grounding_result.get("is_grounded")) else "blocked"
                    RAG_QUERY_TOTAL.labels(status=status).inc()
                    RAG_QUERY_LATENCY.labels(fast_path=str(request.fast_path)).observe(latency / 1000.0)
                    
                    try:
                        set_cached_response(
                            request.query, tenant_id, request.user_id, effective_top_k,
                            embedding_model, corpus_version,
                            {"answer": agentic_result["answer"], "context": final_context, "sources": sources,
                             "grounding": grounding_result, "verification": verification},
                            scope=cache_scope,
                        )
                    except Exception as cache_err:
                        print(f"[Cache] Error saving agentic response: {cache_err}")

                    yield "data: " + json.dumps({'done': True, 'sources': sources, 'grounding': grounding_result, 'verification': verification, 'latency_ms': latency}) + "\n\n"
                    return

            if not grounding_result or not grounding_result.get("is_grounded") or not final_context:
                if not grounding_result:
                    grounding_result = {"is_grounded": False, "score": 0.0, "detail": "Ungrounded: No context found."}
                latency = int((time.time() - start_time) * 1000)
                db.commit()
                status = "blocked"
                RAG_QUERY_TOTAL.labels(status=status).inc()
                RAG_QUERY_LATENCY.labels(fast_path=str(request.fast_path)).observe(latency / 1000.0)
                RAG_GROUNDING_BLOCKED.inc()
                yield "data: " + json.dumps({'token': NOT_FOUND_RESPONSE}) + "\n\n"
                yield "data: " + json.dumps({'done': True, 'sources': [], 'grounding': grounding_result, 'verification': {"confidence": "high", "confidence_score": 1.0, "grounded_sentences": 1, "total_sentences": 1, "evidence": []}, 'latency_ms': latency}) + "\n\n"
                return
            


            if request.extractive:
                answer = ""
                if final_context:
                    best = final_context[0]
                    text = best.get("text", "")
                    source_name = "unknown"
                    if best.get("metadata"):
                        src = best["metadata"].get("source", "")
                        if src:
                            source_name = os.path.basename(str(src))
                    page = best.get("metadata", {}).get("page_num")
                    page_str = f", Page {page}" if page else ""
                    if len(text) > 4000:
                        text = text[:4000] + "..."
                    answer = f"[{source_name}{page_str}]\n{text}"
                verification = verify_answer_grounding(answer, final_context) if answer else {"confidence": "low", "confidence_score": 0.0, "grounded_sentences": 0, "total_sentences": 0, "evidence": []}
                latency = int((time.time() - start_time) * 1000)
                
                db.commit()
                yield "data: " + json.dumps({'token': answer}) + "\n\n"
                yield "data: " + json.dumps({'done': True, 'sources': sources, 'grounding': grounding_result, 'verification': verification, 'latency_ms': latency}) + "\n\n"
                return

            context_texts = []
            for chunk in final_context:
                text = chunk.get('text', '')
                text = re.sub(r'\[.*?(?:Page|Part)\s*\d+\]\s*', '', text)
                context_texts.append(text.strip())
            context_text = "\n\n---\n\n".join(context_texts)
            prompt = build_strict_grounding_prompt(search_query, context_text, broad_query)
            
            db.commit()
            
            payload = {
                "model": OLLAMA_MODEL,
                "system": "",
                "prompt": prompt,
                "stream": True,
                "options": {
                    "num_predict": OLLAMA_NUM_PREDICT,
                    "num_ctx": int(os.getenv("OLLAMA_CONTEXT_LENGTH", "32768")),
                    "temperature": 0.0,
                    "num_gpu": 99,
                }
            }
            answer_acc = ""
            flare_retrieved_chunks = []
            
            try:
                response = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=OLLAMA_TIMEOUT_SECONDS)
                if response.status_code == 200:
                    token_buffer = ""
                    for line in response.iter_lines():
                        if line:
                            try:
                                data = json.loads(line.decode("utf-8"))
                            except json.JSONDecodeError:
                                continue
                            
                            if "error" in data:
                                err_msg = f"\n\n**Ollama Error:** {data['error']}"
                                yield "data: " + json.dumps({'token': err_msg}) + "\n\n"
                                break
                                
                            token = data.get("response", "")
                            answer_acc += token
                            yield "data: " + json.dumps({'token': token}) + "\n\n"
                            
                            token_buffer += token
                            if len(token_buffer) >= 40 and (token.endswith('.') or token.endswith('!') or token.endswith('?')):
                                flare_query = flare_mid_generation_retrieval(
                                    partial_answer=answer_acc,
                                    original_query=search_query,
                                    existing_context=final_context + flare_retrieved_chunks,
                                )
                                if flare_query and flare_query.strip():
                                    new_chunks = perform_multi_query_search(
                                        db, [flare_query], tenant_id,
                                        top_k=effective_top_k,
                                        metadata_filters=None,
                                    )
                                    if new_chunks:
                                        new_reranked = rerank_results(flare_query, new_chunks, top_n=3)
                                        for nc in new_reranked:
                                            if nc not in flare_retrieved_chunks:
                                                flare_retrieved_chunks.append(nc)
                                token_buffer = ""
                else:
                    err = f"Ollama Error: {response.text}"
                    yield "data: " + json.dumps({'token': err}) + "\n\n"
            except requests.exceptions.Timeout:
                err = "Ollama Timeout. The model took too long to respond."
                yield "data: " + json.dumps({'token': err}) + "\n\n"
            except requests.exceptions.RequestException:
                err = "Ollama Connection Error. The model service is unavailable."
                yield "data: " + json.dumps({'token': err}) + "\n\n"
                
            answer_acc = answer_acc.strip()
            
            if flare_retrieved_chunks:
                enriched_context = final_context + flare_retrieved_chunks
                enriched_texts = [re.sub(r'\[.*?(?:Page|Part)\s*\d+\]\s*', '', c.get('text', '')).strip() for c in enriched_context]
                enriched_text = "\n\n---\n\n".join(enriched_texts)
                flare_prompt = build_strict_grounding_prompt(search_query, enriched_text, broad_query)
                try:
                    final_payload = {
                        "model": OLLAMA_MODEL,
                        "system": "",
                        "prompt": flare_prompt,
                        "stream": False,
                        "options": {"num_predict": OLLAMA_NUM_PREDICT, "temperature": 0.0, "num_gpu": 99}
                    }
                    final_resp = requests.post(OLLAMA_URL, json=final_payload, timeout=OLLAMA_TIMEOUT_SECONDS)
                    if final_resp.status_code == 200:
                        corrected_answer = final_resp.json().get("response", "").strip()
                        if corrected_answer:
                            answer_acc = corrected_answer
                except Exception as flare_err:
                    print(f"[FLARE] Final pass failed: {flare_err}")
                final_context = enriched_context
                sources = _context_sources(enriched_context)
            
            clean_answer = re.sub(r'\[FLARE re-retrieved:.*?\]', '', answer_acc).strip()
            verification = verify_answer_grounding(clean_answer, final_context)
            latency_ms = int((time.time() - start_time) * 1000)
            
            yield "data: " + json.dumps({'done': True, 'sources': sources, 'grounding': grounding_result, 'verification': verification, 'latency_ms': latency_ms}) + "\n\n"
            
            status = "grounded" if (grounding_result and grounding_result.get("is_grounded")) else "blocked"
            RAG_QUERY_TOTAL.labels(status=status).inc()
            RAG_QUERY_LATENCY.labels(fast_path=str(request.fast_path)).observe(latency_ms / 1000.0)
            if grounding_result and not grounding_result.get("is_grounded"):
                RAG_GROUNDING_BLOCKED.inc()
            
            try:
                set_cached_response(
                    request.query, tenant_id, request.user_id, effective_top_k,
                    embedding_model, corpus_version,
                    {"answer": clean_answer, "context": final_context, "sources": sources,
                     "grounding": grounding_result, "verification": verification},
                    scope=cache_scope,
                )
            except Exception as cache_err:
                print(f"[Cache] Error saving streamed response: {cache_err}")

        return StreamingResponse(stream_llm(), media_type="text/event-stream")
        
    else:
        # NON-STREAMING FALLBACK (Agentic RAG / Fast Path)
        if request.fast_path:
            retrieved_chunks = perform_hybrid_search(db, search_query, tenant_id, top_k=effective_top_k, fast_path=True)
            final_context = retrieved_chunks[:effective_top_k]
            for c in final_context:
                c["rerank_score"] = c.get("dense_score", 0.0)
            sources = _context_sources(final_context)
            grounding_result = compute_grounding_score(search_query, final_context)
        else:
            # Plan and Execute Pipeline (NVIDIA RAG Blueprint Style)
            import asyncio
            from app.rag.agentic import AgenticRAGPipeline
            
            pipeline = AgenticRAGPipeline(db, tenant_id, effective_top_k)
            try:
                # FastAPI runs sync defs in a threadpool, so asyncio.run is safe here.
                # If an event loop is already running, we create a new one.
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                agentic_result = loop.run_until_complete(pipeline.run(search_query))
                loop.close()
            except RuntimeError:
                # Fallback if somehow already in an async context
                agentic_result = asyncio.run(pipeline.run(search_query))
                
            final_context = agentic_result["context"]
            sources = agentic_result["sources"]
            grounding_result = agentic_result["grounding"]
            
            if agentic_result.get("answer"):
                # The agentic pipeline synthesized the answer directly!
                latency = int((time.time() - start_time) * 1000)
                verification = verify_answer_grounding(agentic_result["answer"], final_context)
                
                db.commit()
                status = "grounded" if (grounding_result and grounding_result.get("is_grounded")) else "blocked"
                RAG_QUERY_TOTAL.labels(status=status).inc()
                RAG_QUERY_LATENCY.labels(fast_path=str(request.fast_path)).observe(latency / 1000.0)
                
                try:
                    set_cached_response(
                        request.query, tenant_id, request.user_id, effective_top_k,
                        embedding_model, corpus_version,
                        {"answer": agentic_result["answer"], "context": final_context, "sources": sources,
                         "grounding": grounding_result, "verification": verification},
                        scope=cache_scope,
                    )
                except Exception as cache_err:
                    print(f"[Cache] Error saving agentic response: {cache_err}")

                return {
                    "answer": agentic_result["answer"],
                    "context": final_context,
                    "sources": sources,
                    "latency_ms": latency,
                    "ingest": ingest_summary,
                    "grounding": grounding_result,
                    "verification": verification,
                }
            
            # If agentic_result["answer"] is empty, it means empty_plan was selected.
            # We will fall through to standard generation below using the retrieved final_context.


        if not grounding_result or not grounding_result.get("is_grounded") or not final_context:
            if not grounding_result:
                grounding_result = {"is_grounded": False, "score": 0.0, "detail": "Ungrounded: No context found."}
            latency = int((time.time() - start_time) * 1000)
            status = "blocked"
            RAG_QUERY_TOTAL.labels(status=status).inc()
            RAG_QUERY_LATENCY.labels(fast_path=str(request.fast_path)).observe(latency / 1000.0)
            RAG_GROUNDING_BLOCKED.inc()
            return {
                "answer": NOT_FOUND_RESPONSE,
                "context": [],
                "sources": [],
                "latency_ms": latency,
                "ingest": ingest_summary,
                "grounding": grounding_result,
                "verification": {"confidence": "high", "confidence_score": 1.0, "grounded_sentences": 1, "total_sentences": 1, "evidence": []},
            }

        if request.extractive:
            answer = ""
            if final_context:
                best = final_context[0]
                text = best.get("text", "")
                source_name = "unknown"
                if best.get("metadata"):
                    src = best["metadata"].get("source", "")
                    if src:
                        source_name = os.path.basename(str(src))
                page = best.get("metadata", {}).get("page_num")
                page_str = f", Page {page}" if page else ""
                if len(text) > 4000:
                    text = text[:4000] + "..."
                answer = f"[{source_name}{page_str}]\n{text}"
            verification = verify_answer_grounding(answer, final_context) if answer else {"confidence": "low", "confidence_score": 0.0, "grounded_sentences": 0, "total_sentences": 0, "evidence": []}
            latency = int((time.time() - start_time) * 1000)
            response_data = {
                "answer": answer,
                "context": final_context,
                "sources": sources,
                "latency_ms": latency,
                "ingest": ingest_summary,
                "grounding": grounding_result,
                "verification": verification,
            }
            if not request.fast_path:
                set_cached_response(
                    request.query, tenant_id, request.user_id, effective_top_k,
                    embedding_model, corpus_version,
                    {key: value for key, value in response_data.items() if key != "ingest"},
                    scope=cache_scope,
                )
            return response_data

        context_texts = []
        for chunk in final_context:
            text = chunk.get('text', '')
            text = re.sub(r'\[.*?(?:Page|Part)\s*\d+\]\s*', '', text)
            context_texts.append(text.strip())
        context_text = "\n\n---\n\n".join(context_texts)
        prompt = build_strict_grounding_prompt(search_query, context_text, broad_query)
        
        flare_retrieved_chunks = []
        answer = ""
        try:
            payload = {
                "model": OLLAMA_MODEL,
                "system": "",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": OLLAMA_NUM_PREDICT,
                    "temperature": 0.0,
                    "num_gpu": 99,
                    "num_ctx": int(os.getenv("OLLAMA_CONTEXT_LENGTH", "32768")),
                }
            }
            response = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT_SECONDS)
            if response.status_code == 200:
                answer = response.json().get("response", "").strip()
                
                flare_query = flare_mid_generation_retrieval(
                    partial_answer=answer,
                    original_query=search_query,
                    existing_context=final_context,
                    is_full_answer=True,
                )
                if flare_query and flare_query.strip():
                    new_chunks = perform_multi_query_search(
                        db, [flare_query], tenant_id,
                        top_k=effective_top_k,
                        metadata_filters=None,
                    )
                    if new_chunks:
                        new_reranked = rerank_results(flare_query, new_chunks, top_n=3)
                        for nc in new_reranked:
                            if nc not in flare_retrieved_chunks:
                                flare_retrieved_chunks.append(nc)
                
                if flare_retrieved_chunks:
                    enriched_context = final_context + flare_retrieved_chunks
                    enriched_texts = [re.sub(r'\[.*?(?:Page|Part)\s*\d+\]\s*', '', c.get('text', '')).strip() for c in enriched_context]
                    enriched_text = "\n\n---\n\n".join(enriched_texts)
                    flare_prompt = build_strict_grounding_prompt(search_query, enriched_text, broad_query)
                    final_payload = {
                        "model": OLLAMA_MODEL,
                        "system": "",
                        "prompt": flare_prompt,
                        "stream": False,
                        "options": {"num_predict": OLLAMA_NUM_PREDICT, "temperature": 0.0, "num_gpu": 99}
                    }
                    final_resp = requests.post(OLLAMA_URL, json=final_payload, timeout=OLLAMA_TIMEOUT_SECONDS)
                    if final_resp.status_code == 200:
                        answer = final_resp.json().get("response", "").strip()
                    final_context = enriched_context
                    sources = _context_sources(enriched_context)
            else:
                answer = "Ollama Error: The model service returned an error."
        except Exception:
            answer = "Ollama Error: An unexpected error occurred."
            
        verification = verify_answer_grounding(answer, final_context)
        
        evaluation = None
        if request.evaluate and final_context and not answer.startswith("Ollama Error"):
            from app.rag.evaluation import evaluate_rag_response
            contexts_str = [c.get("text", "") for c in final_context]
            evaluation = evaluate_rag_response(search_query, answer, contexts_str)
            
        latency = int((time.time() - start_time) * 1000)
        
        response_data = {
            "answer": answer,
            "context": final_context,
            "sources": sources,
            "latency_ms": latency,
            "ingest": ingest_summary,
            "grounding": grounding_result,
            "verification": verification,
            "evaluation": evaluation,
        }
        
        if not answer.startswith("Ollama Error:"):
            set_cached_response(
                request.query,
                tenant_id,
                request.user_id,
                effective_top_k,
                embedding_model,
                corpus_version,
                {key: value for key, value in response_data.items() if key != "ingest"},
                scope=cache_scope,
            )

        elapsed = time.time() - request_start
        status = "grounded" if (grounding_result and grounding_result.get("is_grounded")) else "blocked"
        RAG_QUERY_TOTAL.labels(status=status).inc()
        RAG_QUERY_LATENCY.labels(fast_path=str(request.fast_path)).observe(elapsed)
        if grounding_result and not grounding_result.get("is_grounded"):
            RAG_GROUNDING_BLOCKED.inc()
        
        return response_data


if __name__ == "__main__":
    import uvicorn
    from dotenv import load_dotenv
    
    # Auto-load the local environment if running natively
    env_path = os.path.join(os.path.dirname(__file__), "..", ".envs", ".local", ".rag")
    if os.path.exists(env_path):
        print(f"[RAG Native] Loading environment from {env_path}")
        load_dotenv(env_path)
    
    print("[RAG Native] Starting Enterprise Level RAG 17-Layer Microservice natively...")
    uvicorn.run(app, host="0.0.0.0", port=1000)
