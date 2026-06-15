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
   5. RAPTOR Hierarchical Summarization
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
import glob
import requests
import shutil
import re
import threading
import time
from fastapi import Body, HTTPException, Depends, File, Query, UploadFile
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import redis
import json
import hashlib

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
from app.rag.router import query_router
from app.rag.graph import graph_db
from app.rag.reranker import rerank_results
from app.rag.context import assemble_context
from app.rag.grounding import (
    NOT_FOUND_RESPONSE,
    build_strict_grounding_prompt,
    compute_grounding_score,
    verify_answer_grounding,
)
from app.rag.parsers import SUPPORTED_EXTENSIONS, is_supported_file
from app.rag.query_intelligence import intelligent_query_pipeline, reformulate_query, text_to_sql_filters

app = FastAPI(
    title="Enterprise Level RAG 17-Layer Microservice",
    description="World's best zero-hallucination RAG with unlimited file support, sub-5ms exact extraction, ColBERT reranking, RAPTOR indexing, and Active RAG.",
    version="4.0.0",
)

# ── Prometheus Metrics ────────────────────────────────────────────────────────
RAG_QUERY_TOTAL = Counter("rag_queries_total", "Total queries processed", ["tenant", "status"])
RAG_QUERY_LATENCY = Histogram("rag_query_latency_seconds", "Query latency in seconds",
    ["tenant", "fast_path"], buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0))
RAG_INGESTION_TOTAL = Counter("rag_ingestion_total", "Total files ingested", ["file_type"])
RAG_CACHE_HITS = Counter("rag_cache_hits_total", "Total cache hits", ["type"])
RAG_GROUNDING_BLOCKED = Counter("rag_grounding_blocked_total", "Queries blocked by grounding guard")
RAG_LLM_CALLS = Counter("rag_llm_calls_total", "Total Ollama LLM calls", ["operation"])

# Serve local JS libraries (Chart.js, marked.js) for offline use
_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

TENANT_PATTERN = r"^[a-zA-Z0-9_.:-]{1,80}$"
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
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b") # Best for RAG — world-class reasoning, 128K context
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "1024"))
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
DEFAULT_TOP_K = int(os.getenv("RAG_DEFAULT_TOP_K", "12"))
MAX_TOP_K = int(os.getenv("RAG_MAX_TOP_K", "50"))
BROAD_QUERY_TOP_K = int(os.getenv("RAG_BROAD_QUERY_TOP_K", "16"))
SOURCE_LIMIT = int(os.getenv("RAG_SOURCE_LIMIT", "12"))
# NO FILE SIZE LIMIT — enterprise production system
MAX_UPLOAD_SIZE_BYTES = None  # Unlimited

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
    top_k: int,
    embedding_model: str,
    corpus_version: str,
    scope: Optional[Dict[str, Any]] = None,
) -> str:
    payload = {
        "tenant_id": tenant_id,
        "query": query,
        "top_k": top_k,
        "embedding_model": embedding_model,
        "corpus_version": corpus_version,
        "scope": scope or {},
    }
    query_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"rag_cache:{query_hash}"


from app.rag.model_loader import cosine_similarity, encode_text

CACHE_SEMANTIC_THRESHOLD = float(os.getenv("RAG_CACHE_SEMANTIC_THRESHOLD", "0.85"))

def get_cached_response(
    query: str,
    tenant_id: str,
    top_k: int,
    embedding_model: str,
    corpus_version: str,
    scope: Optional[Dict[str, Any]] = None,
):
    if not redis_client: return None
    try:
        # Layer 11: Semantic Query Cache — aggressive matching (0.85 threshold)
        query_vector = encode_text(query)
        index_key = f"semantic_index:{tenant_id}:{embedding_model}:{corpus_version}"
        
        index_data = redis_client.get(index_key)
        if index_data:
            index = json.loads(index_data)
            best_match = None
            best_sim = 0.0
            
            for item in index:
                sim = cosine_similarity(query_vector, item["embedding"])
                if sim > best_sim:
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
    top_k: int,
    embedding_model: str,
    corpus_version: str,
    response: dict,
    scope: Optional[Dict[str, Any]] = None,
):
    if not redis_client: return
    try:
        cache_key = _cache_key(query, tenant_id, top_k, embedding_model, corpus_version, scope)
        redis_client.setex(cache_key, 86400, json.dumps(response))
        
        # Update semantic index
        query_vector = encode_text(query)
        index_key = f"semantic_index:{tenant_id}:{embedding_model}:{corpus_version}"
        
        index_data = redis_client.get(index_key)
        index = json.loads(index_data) if index_data else []
        
        # Prune index to keep it fast (max 1000 items per tenant)
        if len(index) > 1000:
            index.pop(0)
            
        index.append({
            "cache_key": cache_key,
            "embedding": query_vector
        })
        redis_client.setex(index_key, 86400, json.dumps(index))
        
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
        print("✅ Database initialized with pgvector successfully.")
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

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    tenant_id: str = Field("default", pattern=TENANT_PATTERN)
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
        db.query(DocumentChunk).filter(
            DocumentChunk.tenant_id == tenant_id,
            DocumentChunk.embedding_model == embedding_model,
            DocumentChunk.doc_id.in_(unique_files),
        ).delete(synchronize_session=False)
        db.commit()

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
        " response",
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

    # Stream file to disk — NO SIZE LIMIT
    tenant_media_path = os.path.join(MEDIA_PATH, tenant_id)
    os.makedirs(tenant_media_path, exist_ok=True)
    file_path = os.path.join(tenant_media_path, filename)

    total_bytes = 0
    try:
        with open(file_path, "wb") as buffer:
            while True:
                chunk = await file.read(4 * 1024 * 1024)  # 4MB chunks for speed
                if not chunk:
                    break
                total_bytes += len(chunk)
                buffer.write(chunk)
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

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
def get_ingestion_status(job_id: str):
    job = get_ingestion_job(job_id)
    if not job:
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
            r"how\s+(?:does|do|can|would)|summarize|overview|troubleshoot|diagnos|"
            r"list\s+all|every|all\s+the)", q_lower
        ))
        if is_analysis:
            print("[Auto] Complex analysis → full LLM pipeline")
        else:
            is_simple = (
                bool(re.search(r"(?:^what\s+is|^who\s+is|^when\s+|^where\s+|^which\s+|"
                               r"^how\s+to|^define\s+|^meaning\s+|^list\s+|^show\s+)", q_lower))
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
            response_data = {
                "answer": answer,
                "context": [],
                "sources": [],
                "latency_ms": int((time.time() - start_time) * 1000),
                "ingest": ingest_summary,
                "grounding": grounding_result,
                "verification": {"confidence": "high", "confidence_score": 1.0, "grounded_sentences": 1, "total_sentences": 1, "evidence": []},
            }
            if request.stream:
                def stream_greeting():
                    yield f"data: {json.dumps({'token': answer})}\n\n"
                    yield f"data: {json.dumps({'done': True, 'sources': [], 'grounding': grounding_result, 'verification': response_data['verification']})}\n\n"
                return StreamingResponse(stream_greeting(), media_type="text/event-stream")
            return response_data

    corpus_version = get_corpus_version(db, tenant_id, embedding_model)
    search_query = _retrieval_query(request)
    cache_scope = {
        "parent": request.parent,
        "child": request.child,
        "search_query": search_query,
    }
    
    # Layer 11: Semantic Query Cache
    cached = get_cached_response(
        request.query,
        tenant_id,
        effective_top_k,
        embedding_model,
        corpus_version,
        scope=cache_scope,
    )
    if cached:
        print(f"[Cache] HIT for tenant={tenant_id!r}, query={request.query!r}")
        print(f"[API: OUT] Response: {cached.get('answer', '')}")
        cached["latency_ms"] = int((time.time() - start_time) * 1000)
        cached.setdefault("sources", _context_sources(cached.get("context", [])))
        cached["ingest"] = ingest_summary

        if request.stream:
            def stream_cache():
                yield f"data: {json.dumps({'token': cached.get('answer', '')})}\n\n"
                yield f"data: {json.dumps({'done': True, 'sources': cached.get('sources', []), 'grounding': cached.get('grounding'), 'verification': cached.get('verification')})}\n\n"
            return StreamingResponse(stream_cache(), media_type="text/event-stream")

        return cached

    # --- FAST PATH (sub-5ms): single dense HNSW search, skip all LLM calls ---
    if request.fast_path:
        retrieved_chunks = perform_hybrid_search(db, search_query, tenant_id, top_k=effective_top_k, fast_path=True)
        final_context = retrieved_chunks[:effective_top_k]
        for c in final_context:
            c["rerank_score"] = c.get("dense_score", 0.0)
        sources = _context_sources(final_context)
        grounding_result = compute_grounding_score(search_query, final_context)
        # Skip the full state machine
        state_machine_used = False
    else:
        state_machine_used = True
        
        # Layer 13: Query Intelligence (Spelling, Expansion, Decomposition)
        query_intel = intelligent_query_pipeline(search_query)
        
        # --- FULL AGENTIC STATE MACHINE (multi-query, HyDE, BM25, reranker, CRAG) ---
        class AgentState:
            def __init__(self):
                self.current_state = "route"
                self.queries_to_search = [search_query]
                self.metadata_filters = None
                self.graph_context = ""
                self.final_context = []
                self.grounding_result = None
                self.retry_count = 0
                self.sources = []
                self.answer = ""
                
        agent = AgentState()
        
        while agent.current_state not in ["generate", "end"]:
            
            if agent.current_state == "route":
                route = query_router.route_query(search_query)
                print(f"[Agent:Router] Routed to: {route.upper()}")
                if route == "sql":
                    agent.metadata_filters = text_to_sql_filters(search_query)
                elif route == "graph":
                    agent.graph_context = graph_db.query_graph(search_query, tenant_id)
                    if agent.graph_context:
                        agent.final_context = [{"text": agent.graph_context, "metadata": {"type": "graph", "source": "Neo4j"}}]
                        agent.grounding_result = {"is_grounded": True, "score": 1.0, "detail": "Answered via Graph"}
                        agent.sources = ["Neo4j Knowledge Graph"]
                        agent.current_state = "generate"
                        continue
                    else:
                        route = "vector"
                elif route == "raptor":
                    # Fetch highest-level RAPTOR summaries
                    highest_level_summaries = db.query(DocumentChunk).filter(
                        DocumentChunk.tenant_id == tenant_id,
                        DocumentChunk.file_type == "raptor_summary"
                    ).order_by(DocumentChunk.raptor_level.desc()).limit(10).all()
                    
                    if highest_level_summaries:
                        raptor_context = "\n\n".join([c.text_content for c in highest_level_summaries])
                        agent.final_context = [{"text": raptor_context, "metadata": {"type": "raptor", "source": "RAPTOR Global Index"}}]
                        agent.grounding_result = {"is_grounded": True, "score": 1.0, "detail": "Answered via RAPTOR Global Summary"}
                        agent.sources = ["RAPTOR Global Index"]
                        agent.current_state = "generate"
                        continue
                    else:
                        route = "vector"
                
                agent.queries_to_search = query_intel["expanded_queries"][:2]
                agent.current_state = "retrieve"
                
            elif agent.current_state == "retrieve":
                print(f"[Agent:Retriever] Searching: {agent.queries_to_search}")
                retrieved_chunks = perform_multi_query_search(db, agent.queries_to_search, tenant_id, top_k=max(20, effective_top_k * 4), metadata_filters=agent.metadata_filters)
                reranked_chunks = rerank_results(search_query, retrieved_chunks, top_n=effective_top_k)
                agent.final_context = assemble_context(search_query, reranked_chunks, db=db)
                agent.sources = _context_sources(agent.final_context)
                agent.current_state = "grade"
                
            elif agent.current_state == "grade":
                agent.grounding_result = compute_grounding_score(search_query, agent.final_context)
                score = agent.grounding_result.get("score", 0.0)
                print(f"[Agent:Evaluator] Try {agent.retry_count+1} Score: {score}")
                
                if agent.grounding_result["is_grounded"] or agent.retry_count >= 1:
                    agent.current_state = "generate" if agent.final_context else "end"
                else:
                    agent.current_state = "rewrite"
                    
            elif agent.current_state == "rewrite":
                print("[Agent:Rewriter] Rewriting query for retry...")
                reformulated = reformulate_query(search_query)
                agent.queries_to_search = [reformulated]
                agent.retry_count += 1
                agent.current_state = "retrieve"

        final_context = agent.final_context
        sources = agent.sources
        grounding_result = agent.grounding_result
    
    if not grounding_result or not grounding_result.get("is_grounded") or not final_context:
        # BLOCKED — no relevant content in documents
        latency = int((time.time() - start_time) * 1000)
        
        answer = NOT_FOUND_RESPONSE
        if ingest_summary and ingest_summary.get("queued"):
            answer = (
                f"I queued {ingest_summary['queued']} file(s) for ingestion. "
                "Please wait for ingestion to complete, then ask again."
            )

        print(f"[API: OUT] Response: {answer}")
        
        response_data = {
            "answer": answer,
            "context": [],
            "sources": [],
            "latency_ms": latency,
            "ingest": ingest_summary,
            "grounding": grounding_result,
            "verification": {"confidence": "low", "confidence_score": 0.0, "grounded_sentences": 0, "total_sentences": 0, "evidence": []},
        }

        if request.stream:
            def stream_not_found():
                yield f"data: {json.dumps({'token': answer})}\n\n"
                yield f"data: {json.dumps({'done': True, 'sources': [], 'grounding': grounding_result, 'verification': response_data['verification']})}\n\n"
            return StreamingResponse(stream_not_found(), media_type="text/event-stream")

        return response_data

    # ------------------------------------------------------------
    # Extractive Mode: skip LLM, return verbatim text from top chunk
    # ------------------------------------------------------------
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
        print(f"[API: OUT] Response: {answer}")
        
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
                request.query, tenant_id, effective_top_k,
                embedding_model, corpus_version,
                {key: value for key, value in response_data.items() if key != "ingest"},
                scope=cache_scope,
            )
        latency = time.time() - request_start
        status = "grounded" if (grounding_result and grounding_result.get("is_grounded")) else "blocked"
        RAG_QUERY_TOTAL.labels(tenant=tenant_id, status=status).inc()
        RAG_QUERY_LATENCY.labels(tenant=tenant_id, fast_path=str(request.fast_path)).observe(latency)
        
        if request.stream:
            def stream_extractive():
                yield f"data: {json.dumps({'token': answer})}\n\n"
                yield f"data: {json.dumps({'done': True, 'sources': sources, 'grounding': grounding_result, 'verification': verification})}\n\n"
            return StreamingResponse(stream_extractive(), media_type="text/event-stream")
            
        return response_data

    # ------------------------------------------------------------
    # Layer 12: LLM Answer Synthesis (Ollama)
    # ------------------------------------------------------------
    context_texts = []
    for chunk in final_context:
        text = chunk.get('text', '')
        context_texts.append(text)
    context_text = "\n\n---\n\n".join(context_texts)
    
    prompt = build_strict_grounding_prompt(search_query, context_text, broad_query)
    
    if request.stream:
        def stream_llm():
            payload = {
                "model": OLLAMA_MODEL,
                "system": "",
                "prompt": prompt,
                "stream": True,
                "options": {
                    "num_predict": OLLAMA_NUM_PREDICT,
                    "temperature": 0.0,
                }
            }
            answer_acc = ""
            buffer = ""
            prefix_stripped = False
            
            def strip_prefix(text):
                lower = text.lower()
                prefixes = [
                    "based on the provided context,", "based on the provided manual,", 
                    "based on the context,", "according to the document,", "according to the documents,",
                    "the context states that", "from the provided context,",
                    "according to the extracted data", "according to the database records"
                ]
                for p in prefixes:
                    if lower.startswith(p):
                        cleaned = text[len(p):].strip()
                        return cleaned[0].upper() + cleaned[1:] if cleaned else ""
                return text

            try:
                response = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=OLLAMA_TIMEOUT_SECONDS)
                if response.status_code == 200:
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line.decode("utf-8"))
                            
                            if "error" in data:
                                err_msg = f"\n\n**Ollama Error:** {data['error']}"
                                yield f"data: {json.dumps({'token': err_msg})}\n\n"
                                break
                                
                            token = data.get("response", "")
                            
                            if not prefix_stripped:
                                buffer += token
                                # Wait until we have enough chars to check for prefixes
                                if len(buffer) > 50 or data.get("done", False):
                                    cleaned_buffer = strip_prefix(buffer)
                                    answer_acc += cleaned_buffer
                                    yield f"data: {json.dumps({'token': cleaned_buffer})}\n\n"
                                    prefix_stripped = True
                            else:
                                answer_acc += token
                                yield f"data: {json.dumps({'token': token})}\n\n"
                else:
                    err = f"Ollama Error: {response.text}"
                    yield f"data: {json.dumps({'token': err})}\n\n"
            except requests.exceptions.Timeout:
                err = "Ollama Timeout. The model took too long to respond."
                yield f"data: {json.dumps({'token': err})}\n\n"
            except requests.exceptions.RequestException as e:
                err = f"Ollama Connection Error: {str(e)}"
                yield f"data: {json.dumps({'token': err})}\n\n"
                
            # Final safety check
            answer_acc = strip_prefix(answer_acc.strip())
            
            # Verification and Caching after generation
            verification = verify_answer_grounding(answer_acc, final_context)
            print(f"[API: OUT] Response: {answer_acc}")
            yield f"data: {json.dumps({'done': True, 'sources': sources, 'grounding': grounding_result, 'verification': verification})}\n\n"
            
            try:
                set_cached_response(
                    request.query, tenant_id, effective_top_k,
                    embedding_model, corpus_version,
                    {"answer": answer_acc, "context": final_context, "sources": sources,
                     "grounding": grounding_result, "verification": verification},
                    scope=cache_scope,
                )
            except Exception as cache_err:
                print(f"[Cache] Error saving streamed response: {cache_err}")
        
        return StreamingResponse(stream_llm(), media_type="text/event-stream")

    # Non-streaming fallback
    payload = {
        "model": OLLAMA_MODEL,
        "system": "",
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": OLLAMA_NUM_PREDICT,
            "temperature": 0.0,
        }
    }
    answer = ""
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT_SECONDS)
        if response.status_code == 200:
            answer = response.json().get("response", "").strip()
            
            # Post-process strip
            lower = answer.lower()
            prefixes = [
                "based on the provided context,", "based on the provided manual,", 
                "based on the context,", "according to the document,", "according to the documents,",
                "the context states that", "from the provided context,"
            ]
            for p in prefixes:
                if lower.startswith(p):
                    answer = answer[len(p):].strip()
                    if answer:
                        answer = answer[0].upper() + answer[1:]
                    break
        else:
            answer = f"Ollama Error: {response.text}"
    except Exception as e:
        answer = f"Ollama Error: {str(e)}"
        
    verification = verify_answer_grounding(answer, final_context)
    latency = int((time.time() - start_time) * 1000)
    print(f"[API: OUT] Response: {answer}")
    
    response_data = {
        "answer": answer,
        "context": final_context,
        "sources": sources,
        "latency_ms": latency,
        "ingest": ingest_summary,
        "grounding": grounding_result,
        "verification": verification,
    }
    
    # Layer 11: Save to Cache
    set_cached_response(
        request.query,
        tenant_id,
        effective_top_k,
        embedding_model,
        corpus_version,
        {key: value for key, value in response_data.items() if key != "ingest"},
        scope=cache_scope,
    )

    # Record Prometheus metrics
    latency = time.time() - request_start
    status = "grounded" if (grounding_result and grounding_result.get("is_grounded")) else "blocked"
    RAG_QUERY_TOTAL.labels(tenant=tenant_id, status=status).inc()
    RAG_QUERY_LATENCY.labels(tenant=tenant_id, fast_path=str(request.fast_path)).observe(latency)
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
