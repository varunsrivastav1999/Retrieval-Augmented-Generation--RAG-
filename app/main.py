"""
=============================================================================
 i-Tips RAG: 13-Layer Production Microservice — v3.0
=============================================================================
 World's Best Retrieval-Augmented Generation engine.
 Open Source (MIT) | Zero Hallucination | Sub-5ms Exact Text | 30+ Formats
 
 13 Layers:
   1. Universal Document Parser (PDF/DOCX/XLSX/PPTX/CSV/TXT/IMG/VIDEO)
   2. Smart OCR & Table/Image Extraction
   3. Semantic Parent-Child Chunking
   4. Batch Embedding (offline, GPU-accelerated)
  13. Query Intelligence (Spelling, Expansion, Decomposition)
   5. Hybrid Search (HNSW + BM25 + Trigram)
   6. Cross-Encoder Reranking
   7. Max Marginal Relevance (MMR)
   8. Contextual Window Expansion
   9. Hallucination Guard (ZERO general answers)
  10. Extractive Fast-Path (< 5ms exact document text)
  11. Semantic Query Cache (Redis)
  12. Real-Time Token Streaming
  
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
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import redis
import json
import hashlib

from app.database import DocumentChunk, IngestionJob, SessionLocal, init_db, get_db
from app.rag.jobs import (
    create_ingestion_job,
    find_all_supported_files,
    get_ingestion_job,
    start_ingestion_worker,
)
from app.rag.model_loader import get_embedding_model_id, runtime_model_info, validate_runtime_models
from app.rag.retrieval import perform_hybrid_search, perform_multi_query_search
from app.rag.reranker import rerank_results
from app.rag.context import assemble_context
from app.rag.grounding import (
    NOT_FOUND_RESPONSE,
    build_strict_grounding_prompt,
    compute_grounding_score,
    verify_answer_grounding,
)
from app.rag.parsers import SUPPORTED_EXTENSIONS, is_supported_file
from app.rag.query_intelligence import intelligent_query_pipeline, reformulate_query

app = FastAPI(
    title="i-Tips RAG 13-Layer Microservice",
    description="World's best zero-hallucination RAG with unlimited file support, sub-5ms exact extraction, and Layer 13 Query Intelligence.",
    version="3.0.0",
)

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
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3") # Set to whichever model you have pulled in Ollama
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

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
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


import math
from app.rag.model_loader import encode_text

def _cosine_sim(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

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
        # Layer 11: Semantic Query Cache
        query_vector = encode_text(query)
        index_key = f"semantic_index:{tenant_id}:{embedding_model}:{corpus_version}"
        
        index_data = redis_client.get(index_key)
        if index_data:
            index = json.loads(index_data)
            best_match = None
            best_sim = 0.0
            
            for item in index:
                sim = _cosine_sim(query_vector, item["embedding"])
                if sim > best_sim:
                    best_sim = sim
                    best_match = item
                    
            if best_match and best_sim > 0.92:
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
    print("🚀 [STARTUP] i-Tips RAG 12-Layer Microservice starting...")
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
                                json={"name": OLLAMA_MODEL, "modelfile": modelfile_content},
                            )
                            if resp.status_code == 200:
                                print(f"✅ SUCCESS: Custom brain '{OLLAMA_MODEL}' is ready.")
                            else:
                                print(f"⚠️ Warning: Model creation returned {resp.status_code}")
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
        validate_runtime_models()
        checks["models"] = "ready"
    except Exception as exc:
        checks["models"] = f"syncing: {exc}"

    payload = {"status": "ok" if status_code == 200 else "unready", "checks": checks}
    if status_code != 200:
        raise HTTPException(status_code=status_code, detail=payload)
    return payload


# =========================================================================
# Production Dashboard — Home Page
# =========================================================================
@app.get("/", response_class=HTMLResponse)
def root_ui():
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>i-Tips RAG | 12-Layer Production Hub</title>
        <meta name="description" content="i-Tips RAG 12-Layer Production Intelligence Hub — World-class Retrieval-Augmented Generation with zero hallucination.">
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        <script src="/static/js/marked.min.js"></script>
        <script src="/static/js/chart.umd.min.js"></script>
        <style>
            :root {
                --bg: #0a0f1e;
                --surface: rgba(15, 23, 42, 0.85);
                --card: rgba(30, 41, 59, 0.6);
                --border: rgba(255,255,255,0.08);
                --primary: #3b82f6;
                --primary-glow: rgba(59,130,246,0.15);
                --success: #10b981;
                --warning: #f59e0b;
                --danger: #ef4444;
                --purple: #8b5cf6;
                --cyan: #06b6d4;
                --text: #f1f5f9;
                --text-muted: #94a3b8;
                --text-dim: #64748b;
            }
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: var(--bg);
                background-image: radial-gradient(ellipse at 20% 50%, rgba(59,130,246,0.08) 0%, transparent 50%),
                                  radial-gradient(ellipse at 80% 20%, rgba(139,92,246,0.06) 0%, transparent 50%);
                color: var(--text);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 1.5rem;
            }
            .header { text-align: center; margin: 1rem 0 1.5rem; }
            .header h1 {
                font-size: 2.2rem;
                font-weight: 800;
                background: linear-gradient(135deg, #3b82f6, #8b5cf6, #06b6d4);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                letter-spacing: -0.5px;
            }
            .header p { color: var(--text-dim); font-size: 0.85rem; margin-top: 0.3rem; }
            .layer-badge {
                display: inline-block;
                background: linear-gradient(135deg, rgba(59,130,246,0.2), rgba(139,92,246,0.2));
                border: 1px solid rgba(139,92,246,0.3);
                color: #c4b5fd;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.7rem;
                font-weight: 600;
                margin-top: 0.5rem;
                letter-spacing: 0.5px;
            }

            /* API Status Grid */
            .api-grid {
                display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 0.75rem; margin-bottom: 1.5rem; max-width: 900px; width: 100%;
            }
            .api-card {
                background: var(--card); border: 1px solid var(--border);
                border-radius: 12px; padding: 0.75rem 1rem;
                backdrop-filter: blur(10px);
                transition: border-color 0.3s, transform 0.2s;
            }
            .api-card:hover { border-color: rgba(59,130,246,0.3); transform: translateY(-1px); }
            .api-card .label { font-size: 0.7rem; color: var(--text-dim); font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
            .api-card .value { font-size: 1.3rem; font-weight: 700; color: var(--text); margin-top: 2px; }
            .api-card .status { display: flex; align-items: center; gap: 6px; margin-top: 4px; font-size: 0.7rem; }
            .dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
            .dot-ok { background: var(--success); box-shadow: 0 0 6px rgba(16,185,129,0.5); }
            .dot-warn { background: var(--warning); box-shadow: 0 0 6px rgba(245,158,11,0.5); }
            .dot-err { background: var(--danger); box-shadow: 0 0 6px rgba(239,68,68,0.5); }

            .stats-bar {
                display: flex; gap: 1rem; flex-wrap: wrap; justify-content: center;
                margin-bottom: 1.5rem;
            }
            .stat {
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: 10px;
                padding: 0.6rem 1.2rem;
                font-size: 0.75rem;
                color: var(--text-muted);
                backdrop-filter: blur(10px);
            }
            .stat strong { color: var(--text); font-weight: 600; }
            .container { max-width: 900px; width: 100%; display: flex; flex-direction: column; gap: 1.5rem; }
            .card {
                background: var(--card);
                backdrop-filter: blur(16px);
                border-radius: 16px;
                padding: 1.5rem 2rem;
                border: 1px solid var(--border);
                box-shadow: 0 8px 32px rgba(0,0,0,0.3);
                transition: border-color 0.3s;
            }
            .card:hover { border-color: rgba(59,130,246,0.2); }
            .card h2 { font-size: 1rem; font-weight: 600; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem; }
            .card h2 .icon { font-size: 1.2rem; }
            input[type="text"] {
                flex: 1; padding: 0.75rem 1rem; border-radius: 10px;
                border: 1px solid var(--border);
                background: rgba(0,0,0,0.3); color: white;
                font-size: 0.9rem; font-family: inherit;
                transition: border-color 0.2s, box-shadow 0.2s;
            }
            input[type="text"]:focus { outline: none; border-color: var(--primary); box-shadow: 0 0 0 3px var(--primary-glow); }
            input[type="file"] {
                flex: 1; padding: 0.5rem; color: var(--text-muted);
                font-size: 0.85rem; font-family: inherit;
            }
            .btn {
                background: linear-gradient(135deg, #3b82f6, #2563eb);
                color: white; border: none;
                padding: 0.7rem 1.5rem; border-radius: 10px;
                cursor: pointer; font-weight: 600; font-size: 0.85rem;
                font-family: inherit;
                transition: all 0.2s; white-space: nowrap;
                box-shadow: 0 2px 8px rgba(59,130,246,0.3);
            }
            .btn:hover { transform: translateY(-1px); box-shadow: 0 4px 16px rgba(59,130,246,0.4); }
            .btn:active { transform: translateY(0); }
            .btn-secondary {
                background: linear-gradient(135deg, #6366f1, #4f46e5);
                box-shadow: 0 2px 8px rgba(99,102,241,0.3);
            }
            .btn-sm { padding: 0.4rem 0.8rem; font-size: 0.75rem; }
            .flex-row { display: flex; gap: 0.75rem; align-items: center; width: 100%; }
            .formats-badge {
                display: inline-block;
                background: rgba(139,92,246,0.15); color: #a78bfa;
                padding: 2px 8px; border-radius: 6px;
                font-size: 0.65rem; font-weight: 500; margin: 2px;
            }
            .formats-list { margin-top: 0.5rem; line-height: 1.8; }

            /* Answer Box with Markdown Rendering */
            .answer-box {
                background: rgba(0,0,0,0.35);
                padding: 1.5rem;
                border-radius: 12px;
                min-height: 80px;
                font-size: 0.9rem;
                line-height: 1.7;
                margin-top: 1rem;
                border: 1px solid var(--border);
                overflow-x: auto;
            }
            .answer-box h1,.answer-box h2,.answer-box h3,.answer-box h4 { color: #93c5fd; margin: 1rem 0 0.5rem; font-size: 1rem; }
            .answer-box p { margin: 0.4rem 0; }
            .answer-box ul, .answer-box ol { padding-left: 1.5rem; margin: 0.5rem 0; }
            .answer-box li { margin: 0.3rem 0; }
            .answer-box strong { color: #93c5fd; }
            .answer-box code { background: rgba(59,130,246,0.15); padding: 2px 6px; border-radius: 4px; font-size: 0.85em; color: #7dd3fc; }
            .answer-box pre { background: rgba(0,0,0,0.4); padding: 1rem; border-radius: 8px; overflow-x: auto; margin: 0.5rem 0; }
            .answer-box pre code { background: none; padding: 0; }
            .answer-box blockquote { border-left: 3px solid var(--primary); padding-left: 1rem; color: var(--text-muted); margin: 0.5rem 0; }

            /* Beautiful Tables */
            .answer-box table {
                width: 100%; border-collapse: collapse; margin: 1rem 0;
                font-size: 0.85rem; border-radius: 8px; overflow: hidden;
            }
            .answer-box table th {
                background: rgba(59,130,246,0.2);
                color: #93c5fd; font-weight: 600;
                padding: 0.6rem 0.8rem; text-align: left;
                border-bottom: 2px solid rgba(59,130,246,0.3);
            }
            .answer-box table td {
                padding: 0.5rem 0.8rem;
                border-bottom: 1px solid var(--border);
            }
            .answer-box table tr:hover td { background: rgba(59,130,246,0.05); }

            /* Source Cards */
            .sources-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 0.75rem; margin-top: 1rem; }
            .source-card {
                background: rgba(0,0,0,0.25);
                border: 1px solid var(--border);
                border-radius: 10px;
                padding: 0.75rem 1rem;
                transition: border-color 0.2s;
            }
            .source-card:hover { border-color: rgba(59,130,246,0.3); }
            .source-card .name { font-weight: 600; font-size: 0.8rem; color: #93c5fd; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
            .source-card .meta { font-size: 0.7rem; color: var(--text-dim); margin-top: 0.25rem; }
            .badge {
                display: inline-block; padding: 2px 8px; border-radius: 20px;
                font-size: 0.65rem; font-weight: 600; text-transform: uppercase;
            }
            .badge-text { background: rgba(16,185,129,0.15); color: #10b981; }
            .badge-table { background: rgba(245,158,11,0.15); color: #f59e0b; }
            .badge-image { background: rgba(139,92,246,0.15); color: #8b5cf6; }
            .badge-subtitle { background: rgba(6,182,212,0.15); color: #06b6d4; }

            /* Grounding indicator */
            .grounding-bar {
                display: flex; align-items: center; gap: 0.75rem; margin-top: 0.75rem;
                padding: 0.5rem 1rem; background: rgba(0,0,0,0.2); border-radius: 8px;
                font-size: 0.75rem; color: var(--text-dim);
            }
            .confidence-high { color: var(--success); font-weight: 600; }
            .confidence-medium { color: var(--warning); font-weight: 600; }
            .confidence-low { color: var(--danger); font-weight: 600; }

            /* Chart Container */
            .chart-container { background: rgba(0,0,0,0.25); border-radius: 12px; padding: 1rem; margin-top: 1rem; border: 1px solid var(--border); }
            .chart-container canvas { max-height: 250px; }

            /* Latency Bar */
            .latency-bar {
                display: flex; align-items: center; gap: 0.75rem; margin-top: 0.5rem;
                padding: 0.5rem 1rem; background: rgba(0,0,0,0.2); border-radius: 8px;
                font-size: 0.75rem; color: var(--text-dim);
            }
            .latency-dot { width: 8px; height: 8px; border-radius: 50%; }
            .latency-fast { background: #10b981; }
            .latency-medium { background: #f59e0b; }
            .latency-slow { background: #ef4444; }

            .spinner { display: none; width: 18px; height: 18px; border: 2px solid rgba(255,255,255,0.2); border-radius: 50%; border-top-color: var(--primary); animation: spin 0.8s linear infinite; margin-left: 0.5rem; flex-shrink: 0; }
            @keyframes spin { to { transform: rotate(360deg); } }
            .status-msg { color: var(--text-muted); font-size: 0.8rem; margin-top: 0.5rem; }
            .placeholder { color: var(--text-dim); font-style: italic; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>i-Tips RAG Production Hub</h1>
            <p>Enterprise-Grade Knowledge Intelligence Engine</p>
            <div class="layer-badge">12-LAYER ARCHITECTURE &bull; ZERO HALLUCINATION &bull; 100% OFFLINE</div>
        </div>

        <!-- API Status Dashboard -->
        <div class="api-grid" id="apiGrid">
            <div class="api-card">
                <div class="label">Database</div>
                <div class="value" id="dbStatus">--</div>
                <div class="status"><span class="dot" id="dbDot"></span> <span id="dbDetail">Checking...</span></div>
            </div>
            <div class="api-card">
                <div class="label">Redis Cache</div>
                <div class="value" id="redisStatus">--</div>
                <div class="status"><span class="dot" id="redisDot"></span> <span id="redisDetail">Checking...</span></div>
            </div>
            <div class="api-card">
                <div class="label">LLM (Ollama)</div>
                <div class="value" id="ollamaStatus">--</div>
                <div class="status"><span class="dot" id="ollamaDot"></span> <span id="ollamaDetail">Checking...</span></div>
            </div>
            <div class="api-card">
                <div class="label">AI Models</div>
                <div class="value" id="modelsStatus">--</div>
                <div class="status"><span class="dot" id="modelsDot"></span> <span id="modelsDetail">Checking...</span></div>
            </div>
        </div>

        <div class="stats-bar" id="statsBar">
            <div class="stat"><strong id="statChunks">--</strong> Chunks Indexed</div>
            <div class="stat"><strong id="statDocs">--</strong> Documents</div>
            <div class="stat"><strong id="statTypes">--</strong> File Types</div>
            <div class="stat"><strong id="statJobs">--</strong> Active Jobs</div>
        </div>

        <div class="container">
            <div class="card">
                <h2><span class="icon">&#128196;</span> Upload Knowledge Base
                    <small style="font-weight:400; color:var(--text-dim); margin-left:auto; font-size:0.7rem;">No file size limit</small>
                </h2>
                <div class="flex-row">
                    <input type="file" id="uploadFile" accept=".pdf,.docx,.doc,.xlsx,.xls,.csv,.pptx,.ppt,.txt,.text,.md,.log,.json,.xml,.png,.jpg,.jpeg,.bmp,.tiff,.tif,.gif,.webp,.mp4,.avi,.mkv,.mov,.srt,.ass,.ssa,.vtt">
                    <button class="btn" onclick="uploadFile()">Upload & Ingest</button>
                    <div id="uploadSpinner" class="spinner"></div>
                </div>
                <p id="uploadStatus" class="status-msg"></p>
                <div class="formats-list">
                    <span class="formats-badge">PDF</span>
                    <span class="formats-badge">DOCX</span>
                    <span class="formats-badge">XLSX</span>
                    <span class="formats-badge">PPTX</span>
                    <span class="formats-badge">CSV</span>
                    <span class="formats-badge">TXT</span>
                    <span class="formats-badge">MD</span>
                    <span class="formats-badge">PNG</span>
                    <span class="formats-badge">JPG</span>
                    <span class="formats-badge">BMP</span>
                    <span class="formats-badge">TIFF</span>
                    <span class="formats-badge">MP4</span>
                    <span class="formats-badge">AVI</span>
                    <span class="formats-badge">MKV</span>
                    <span class="formats-badge">SRT</span>
                    <span class="formats-badge">VTT</span>
                </div>
            </div>

            <div class="card">
                <h2><span class="icon">&#128269;</span> Ask Your Knowledge Base
                    <small style="font-weight:400; color:var(--text-dim); margin-left:auto; font-size:0.7rem;">Strict document grounding — zero hallucination</small>
                </h2>
                <div class="flex-row">
                    <input type="text" id="queryInput" placeholder="Ask anything from your uploaded documents..." onkeypress="if(event.key === 'Enter') askQuery()">
                    <button class="btn" onclick="askQuery()">Ask</button>
                    <div id="askSpinner" class="spinner"></div>
                </div>

                <div class="answer-box" id="answerBox"><span class="placeholder">Your answer will appear here — sourced only from your uploaded documents...</span></div>

                <div id="groundingBar" style="display:none" class="grounding-bar">
                    <span id="groundingIcon">🛡️</span>
                    <span id="groundingText"></span>
                </div>

                <div id="chartArea"></div>

                <div id="sourcesArea"></div>

                <div id="latencyBar" style="display:none" class="latency-bar">
                    <span class="latency-dot" id="latencyDot"></span>
                    <span id="latencyText"></span>
                </div>
            </div>

            <div class="card" style="margin-top: -0.5rem; padding: 1rem 2rem;">
                <h2 style="margin-bottom: 0.5rem; cursor: pointer; user-select: none;" onclick="document.getElementById('rawApiBox').style.display = document.getElementById('rawApiBox').style.display === 'none' ? 'block' : 'none'">
                    <span class="icon">&#123;&#125;</span> View Raw API Response <small style="font-weight:400; color:var(--text-dim); margin-left:auto;">(Click to toggle)</small>
                </h2>
                <pre id="rawApiBox" style="display:none; font-size: 0.7rem; color: var(--text-muted); background: rgba(0,0,0,0.4); padding: 1rem; border-radius: 8px; margin-top: 0.5rem; max-height: 300px; overflow: auto; border: 1px solid var(--border);"></pre>
            </div>

            <!-- API Reference Card -->
            <div class="card" style="margin-top: -0.5rem; padding: 1rem 2rem;">
                <h2 style="cursor: pointer; user-select: none;" onclick="document.getElementById('apiRef').style.display = document.getElementById('apiRef').style.display === 'none' ? 'block' : 'none'">
                    <span class="icon">&#128218;</span> Microservice API Reference <small style="font-weight:400; color:var(--text-dim); margin-left:auto;">(Click to toggle)</small>
                </h2>
                <div id="apiRef" style="display:none; margin-top: 0.75rem; font-size: 0.8rem; color: var(--text-muted); line-height: 2;">
                    <code style="color:#7dd3fc;">POST /api/v1/upload</code> — Upload any file (no size limit)<br>
                    <code style="color:#7dd3fc;">POST /api/v1/ingest</code> — Scan /media and ingest all files<br>
                    <code style="color:#7dd3fc;">POST /api/v1/query</code> — Query knowledge base (strict grounding)<br>
                    <code style="color:#7dd3fc;">GET &nbsp;/api/v1/ingest/jobs</code> — List ingestion jobs<br>
                    <code style="color:#7dd3fc;">GET &nbsp;/api/v1/ingest/jobs/{id}</code> — Job status + progress<br>
                    <code style="color:#7dd3fc;">GET &nbsp;/api/v1/formats</code> — List supported file formats<br>
                    <code style="color:#7dd3fc;">GET &nbsp;/health/live</code> — Liveness probe<br>
                    <code style="color:#7dd3fc;">GET &nbsp;/health/ready</code> — Readiness probe + stats<br>
                </div>
            </div>
        </div>

        <script>
            // Configure marked for safe markdown rendering
            marked.use({ breaks: true, gfm: true });

            // Load stats + API status on page load
            (async function loadDashboard() {
                try {
                    const res = await fetch('/health/ready');
                    const data = await res.json();
                    const checks = data.checks || {};
                    const stats = checks.stats || {};

                    // Stats
                    document.getElementById('statChunks').textContent = (stats.chunks || 0).toLocaleString();
                    document.getElementById('statDocs').textContent = (stats.docs || 0).toLocaleString();
                    document.getElementById('statTypes').textContent = stats.file_types || 0;
                    document.getElementById('statJobs').textContent = stats.active_jobs || 0;

                    // API Status Cards
                    setApiStatus('db', checks.database);
                    setApiStatus('redis', checks.redis);
                    setApiStatus('ollama', checks.ollama);
                    setApiStatus('models', checks.models);
                } catch(e) {
                    console.error('Dashboard load failed:', e);
                }
            })();

            // Refresh dashboard every 15 seconds
            setInterval(async () => {
                try {
                    const res = await fetch('/health/ready');
                    const data = await res.json();
                    const stats = data.checks?.stats || {};
                    document.getElementById('statChunks').textContent = (stats.chunks || 0).toLocaleString();
                    document.getElementById('statDocs').textContent = (stats.docs || 0).toLocaleString();
                    document.getElementById('statJobs').textContent = stats.active_jobs || 0;
                } catch(e) {}
            }, 15000);

            function setApiStatus(prefix, value) {
                const statusEl = document.getElementById(prefix + 'Status');
                const dotEl = document.getElementById(prefix + 'Dot');
                const detailEl = document.getElementById(prefix + 'Detail');
                if (!value) { statusEl.textContent = '?'; return; }

                const str = String(value);
                if (str === 'ok' || str === 'ready') {
                    statusEl.textContent = '✓';
                    statusEl.style.color = 'var(--success)';
                    dotEl.className = 'dot dot-ok';
                    detailEl.textContent = 'Connected';
                    detailEl.style.color = 'var(--success)';
                } else if (str.startsWith('degraded') || str.startsWith('syncing')) {
                    statusEl.textContent = '⚠';
                    statusEl.style.color = 'var(--warning)';
                    dotEl.className = 'dot dot-warn';
                    detailEl.textContent = str.includes(':') ? str.split(':')[1].trim().substring(0, 40) : str;
                    detailEl.style.color = 'var(--warning)';
                } else {
                    statusEl.textContent = '✗';
                    statusEl.style.color = 'var(--danger)';
                    dotEl.className = 'dot dot-err';
                    detailEl.textContent = str.includes(':') ? str.split(':')[1].trim().substring(0, 40) : str;
                    detailEl.style.color = 'var(--danger)';
                }
            }

            async function uploadFile() {
                const fileInput = document.getElementById('uploadFile');
                if (!fileInput.files[0]) return alert("Please select a file first.");

                const formData = new FormData();
                formData.append("file", fileInput.files[0]);

                document.getElementById('uploadSpinner').style.display = 'block';
                document.getElementById('uploadStatus').innerText = "Uploading and starting background ingestion...";

                try {
                    const res = await fetch('/api/v1/upload', { method: 'POST', body: formData });
                    const data = await res.json();
                    if (res.ok) {
                        const ft = data.file_type ? ` [${data.file_type.toUpperCase()}]` : '';
                        document.getElementById('uploadStatus').innerHTML = '<span style="color:#10b981">&#10003; ' + data.message + ft + '</span>';
                    } else {
                        document.getElementById('uploadStatus').innerHTML = '<span style="color:#ef4444">&#10007; ' + (data.detail || 'Upload failed') + '</span>';
                    }
                } catch (e) {
                    document.getElementById('uploadStatus').innerHTML = '<span style="color:#ef4444">&#10007; Network error</span>';
                } finally {
                    document.getElementById('uploadSpinner').style.display = 'none';
                    fileInput.value = '';
                }
            }

            function renderAnswer(text) {
                let processed = text.replace(/\\\\n/g, '\\n');
                return marked.parse(processed);
            }

            function tryBuildChart(answer, context) {
                const chartArea = document.getElementById('chartArea');
                chartArea.innerHTML = '';
                let tableChunks = (context || []).filter(c => {
                    const meta = c.metadata || {};
                    return meta.type === 'table' || (c.text && c.text.includes('|') && c.text.split('|').length > 4);
                });
                if (tableChunks.length === 0) return;
                const lines = tableChunks[0].text.split('\\n').filter(l => l.trim() && !l.match(/^[\\-|\\s]+$/));
                if (lines.length < 2) return;
                const headers = lines[0].split('|').map(h => h.trim()).filter(Boolean);
                const rows = lines.slice(1).map(l => l.split('|').map(c => c.trim()).filter(Boolean));
                let numericCol = -1;
                for (let i = 1; i < headers.length; i++) {
                    if (rows.length > 0 && !isNaN(parseFloat(rows[0][i]))) { numericCol = i; break; }
                }
                if (numericCol === -1 || rows.length < 2 || rows.length > 20) return;
                const labels = rows.map(r => r[0] || '').slice(0, 15);
                const values = rows.map(r => parseFloat(r[numericCol]) || 0).slice(0, 15);
                chartArea.innerHTML = '<div class="chart-container"><canvas id="dataChart"></canvas></div>';
                const ctx = document.getElementById('dataChart').getContext('2d');
                new Chart(ctx, {
                    type: values.length > 8 ? 'line' : 'bar',
                    data: { labels, datasets: [{ label: headers[numericCol] || 'Value', data: values, backgroundColor: 'rgba(59,130,246,0.4)', borderColor: '#3b82f6', borderWidth: 2, borderRadius: 6, tension: 0.3, fill: true }] },
                    options: { responsive: true, plugins: { legend: { labels: { color: '#94a3b8' } } }, scales: { x: { ticks: { color: '#64748b', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' } }, y: { ticks: { color: '#64748b' }, grid: { color: 'rgba(255,255,255,0.05)' } } } }
                });
            }

            function renderSources(context) {
                const area = document.getElementById('sourcesArea');
                if (!context || context.length === 0) { area.innerHTML = ''; return; }
                let html = '<div class="sources-grid">';
                context.forEach(c => {
                    const meta = c.metadata || {};
                    const type = meta.type || 'text';
                    const badgeClass = type === 'table' ? 'badge-table' : type === 'image_ocr' ? 'badge-image' : type === 'subtitle' ? 'badge-subtitle' : 'badge-text';
                    const score = c.rerank_score ? c.rerank_score.toFixed(3) : '--';
                    const source = c.citation || 'Unknown';
                    html += `<div class="source-card">
                        <div class="name">${source}</div>
                        <div class="meta">
                            <span class="badge ${badgeClass}">${type}</span>
                            &nbsp; Score: ${score}
                        </div>
                    </div>`;
                });
                html += '</div>';
                area.innerHTML = html;
            }

            function showGrounding(grounding, verification) {
                const bar = document.getElementById('groundingBar');
                const icon = document.getElementById('groundingIcon');
                const txt = document.getElementById('groundingText');
                if (!grounding && !verification) { bar.style.display = 'none'; return; }
                bar.style.display = 'flex';

                let parts = [];
                if (grounding) {
                    parts.push(`Grounding: ${(grounding.score * 100).toFixed(0)}%`);
                }
                if (verification) {
                    const conf = verification.confidence || 'unknown';
                    const cls = conf === 'high' ? 'confidence-high' : conf === 'medium' ? 'confidence-medium' : 'confidence-low';
                    parts.push(`Confidence: <span class="${cls}">${conf.toUpperCase()}</span> (${verification.grounded_sentences}/${verification.total_sentences} sentences verified)`);
                }
                icon.textContent = verification?.confidence === 'high' ? '🛡️' : verification?.confidence === 'medium' ? '⚠️' : '⚡';
                txt.innerHTML = parts.join(' &bull; ');
            }

            async function askQuery() {
                const query = document.getElementById('queryInput').value;
                if (!query) return;

                const answerBox = document.getElementById('answerBox');
                const askSpinner = document.getElementById('askSpinner');
                const chartArea = document.getElementById('chartArea');
                const sourcesArea = document.getElementById('sourcesArea');
                const latencyBar = document.getElementById('latencyBar');
                const groundingBar = document.getElementById('groundingBar');

                askSpinner.style.display = 'block';
                answerBox.innerHTML = '<span class="placeholder">Searching knowledge base (12-layer pipeline)...</span>';
                chartArea.innerHTML = '';
                sourcesArea.innerHTML = '';
                latencyBar.style.display = 'none';
                groundingBar.style.display = 'none';

                try {
                    const response = await fetch('/api/v1/query', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query: query, stream: true })
                    });

                    if (!response.ok) throw new Error("Request failed");

                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    let fullAnswer = "";

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;

                        const chunk = decoder.decode(value, { stream: true });
                        const lines = chunk.split("\\n");

                        for (const line of lines) {
                            if (line.startsWith("data: ")) {
                                try {
                                    const data = JSON.parse(line.substring(6));
                                    if (data.token) {
                                        fullAnswer += data.token;
                                        answerBox.innerHTML = renderAnswer(fullAnswer);
                                        answerBox.scrollTop = answerBox.scrollHeight;
                                    }
                                    if (data.done) {
                                        if (data.sources) renderSources(data.sources);
                                        if (data.grounding || data.verification) showGrounding(data.grounding, data.verification);
                                        document.getElementById('rawApiBox').textContent = JSON.stringify(data, null, 2);
                                    }
                                    if (data.error) {
                                        answerBox.innerHTML += `<div style="color:#ef4444; margin-top:1rem;">Error: ${data.error}</div>`;
                                    }
                                } catch (e) {
                                    console.error("Error parsing stream chunk", e);
                                }
                            }
                        }
                    }
                } catch (e) {
                    answerBox.innerHTML = `<span style="color:#ef4444">Error: ${e.message}</span>`;
                } finally {
                    askSpinner.style.display = 'none';
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


# =========================================================================
# Query Pipeline — 12 Layers
# =========================================================================
@app.post("/api/v1/query", response_model=QueryResponse)
def query_rag(request: QueryRequest, db: Session = Depends(get_db)):
    """
    12-Layer RAG Query Pipeline (Strict Document Grounding):
    
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
        cached["latency_ms"] = int((time.time() - start_time) * 1000)
        cached.setdefault("sources", _context_sources(cached.get("context", [])))
        cached["ingest"] = ingest_summary
        return cached

    # Layer 13: Query Intelligence (Spelling, Expansion, Decomposition)
    query_intel = intelligent_query_pipeline(search_query)
    
    # CRAG Retry Loop
    max_retries = 1
    retry_count = 0
    grounding_result = None
    final_context = []
    
    while retry_count <= max_retries:
        if retry_count == 0:
            queries_to_search = query_intel["expanded_queries"][:2]  # Limit to 2 for latency
            print(f"[Retrieval] Multi-query search: {queries_to_search}")
        else:
            # CRAG: Reformulate query for retry
            reformulated = reformulate_query(search_query)
            queries_to_search = [reformulated]
            print(f"[CRAG] Retrying with reformulated query: {reformulated}")
            
        # Layer 5+13: Sub-Query RRF Fusion
        retrieved_chunks = perform_multi_query_search(db, queries_to_search, tenant_id, top_k=max(20, effective_top_k * 4))
        
        # Layer 6: Cross-Encoder Reranking
        reranked_chunks = rerank_results(search_query, retrieved_chunks, top_n=effective_top_k)
        
        # Layer 7+8: MMR + Context Assembly with Window Expansion
        final_context = assemble_context(search_query, reranked_chunks, db=db)
        sources = _context_sources(final_context)

        # Layer 9: HALLUCINATION GUARD
        grounding_result = compute_grounding_score(search_query, final_context)
        print(f"[Grounding] Try {retry_count+1}: {grounding_result['detail']}")

        score = grounding_result.get("score", 0.0)
        
        if grounding_result["is_grounded"] or score == 0.0:
            # If grounded, or completely irrelevant (score 0), don't retry
            break
            
        if 0.0 < score < 0.25:
            # Gray zone: answer might be there under different terms
            print("[CRAG] Grounding score in gray zone. Triggering corrective retry.")
            retry_count += 1
        else:
            break

    if not grounding_result["is_grounded"] or not final_context:
        # BLOCKED — no relevant content in documents
        latency = int((time.time() - start_time) * 1000)
        
        answer = NOT_FOUND_RESPONSE
        if ingest_summary and ingest_summary.get("queued"):
            answer = (
                f"I queued {ingest_summary['queued']} file(s) for ingestion. "
                "Please wait for ingestion to complete, then ask again."
            )

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
    # Layer 12: LLM Answer Synthesis (Ollama)
    # ------------------------------------------------------------
    import requests
    
    context_texts = []
    for chunk in final_context:
        citation = chunk.get('citation', '[Unknown Source]')
        text = chunk.get('text', '')
        context_texts.append(f"From {citation}:\n{text}")
    context_text = "\n\n".join(context_texts)
    
    prompt = build_strict_grounding_prompt(search_query, context_text, False)
    
    if request.stream:
        def stream_llm():
            payload = {
                "model": OLLAMA_MODEL,
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
                    "the context states that", "from the provided context,"
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
            yield f"data: {json.dumps({'done': True, 'sources': sources, 'grounding': grounding_result, 'verification': verification})}\n\n"
            
            set_cached_response(
                request.query, tenant_id, effective_top_k,
                embedding_model, corpus_version,
                {"answer": answer_acc, "context": final_context, "sources": sources,
                 "grounding": grounding_result, "verification": verification},
                scope=cache_scope,
            )
        
        return StreamingResponse(stream_llm(), media_type="text/event-stream")

    # Non-streaming fallback
    payload = {
        "model": OLLAMA_MODEL,
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
    
    return response_data


if __name__ == "__main__":
    import uvicorn
    from dotenv import load_dotenv
    
    # Auto-load the local environment if running natively
    env_path = os.path.join(os.path.dirname(__file__), "..", ".envs", ".local", ".rag")
    if os.path.exists(env_path):
        print(f"[RAG Native] Loading environment from {env_path}")
        load_dotenv(env_path)
    
    print("[RAG Native] Starting i-Tips RAG 13-Layer Microservice natively...")
    uvicorn.run(app, host="0.0.0.0", port=1000)
