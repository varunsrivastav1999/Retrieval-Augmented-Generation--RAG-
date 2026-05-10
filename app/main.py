import os
import glob
import requests
import shutil
import re
import threading
import time
from fastapi import Body, HTTPException, Depends, File, Query, UploadFile
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import redis
import json
import hashlib

from app.database import DocumentChunk, IngestionJob, init_db, get_db
from app.rag.jobs import create_ingestion_job, get_ingestion_job, start_ingestion_worker
from app.rag.model_loader import get_embedding_model_id, runtime_model_info, validate_runtime_models
from app.rag.retrieval import perform_hybrid_search
from app.rag.reranker import rerank_results
from app.rag.context import assemble_context

app = FastAPI(title="i-Tips RAG Production API", version="1.1.0")

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
MAX_UPLOAD_SIZE_MB = int(os.getenv("RAG_MAX_UPLOAD_SIZE_MB", "200"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

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
        cached = redis_client.get(
            _cache_key(query, tenant_id, top_k, embedding_model, corpus_version, scope)
        )
        if cached: return json.loads(cached)
    except Exception as e: print(f"Redis get error: {e}")
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
        redis_client.setex(
            _cache_key(query, tenant_id, top_k, embedding_model, corpus_version, scope),
            86400,
            json.dumps(response),
        )
    except Exception as e: print(f"Redis set error: {e}")


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
    global ingestion_worker_thread
    try:
        init_db()
        print("Database initialized with pgvector successfully.")
    except Exception as e:
        print(f"Failed to initialize database: {e}")
        raise

    if PRELOAD_MODELS_ON_STARTUP:
        validate_runtime_models()
        print(f"RAG models ready: {runtime_model_info()}")

    if ENABLE_INGESTION_WORKER:
        ingestion_worker_thread = start_ingestion_worker(
            ingestion_worker_stop,
            poll_seconds=INGESTION_WORKER_POLL_SECONDS,
            stale_timeout_seconds=INGESTION_STALE_TIMEOUT_SECONDS,
        )
        print("Ingestion worker started.")


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
    )


def _find_pdf_files(include_scan: bool = True) -> List[str]:
    if not include_scan:
        return []
    return sorted(set(glob.glob(os.path.join(MEDIA_PATH, "**/*.pdf"), recursive=True)))


def _queue_pdf_ingestion(
    tenant_id: str,
    pdf_files: List[str],
    db: Session,
    force_reindex: bool = False,
) -> Dict[str, Any]:
    embedding_model = get_embedding_model_id()
    unique_pdf_files = sorted(set(pdf_files))

    if force_reindex and unique_pdf_files:
        db.query(DocumentChunk).filter(
            DocumentChunk.tenant_id == tenant_id,
            DocumentChunk.embedding_model == embedding_model,
            DocumentChunk.doc_id.in_(unique_pdf_files),
        ).delete(synchronize_session=False)
        db.commit()

    indexed_sources = {
        row[0]
        for row in (
            db.query(DocumentChunk.doc_id)
            .filter(
                DocumentChunk.tenant_id == tenant_id,
                DocumentChunk.embedding_model == embedding_model,
                DocumentChunk.doc_id.in_(unique_pdf_files),
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
                IngestionJob.source_path.in_(unique_pdf_files),
                IngestionJob.status.in_(["queued", "running", "retry"]),
            )
            .distinct()
            .all()
        )
    }

    jobs = []
    skipped = 0
    for pdf_file in unique_pdf_files:
        if not force_reindex and (pdf_file in indexed_sources or pdf_file in active_job_sources):
            skipped += 1
            continue
        jobs.append(create_ingestion_job(tenant_id, pdf_file))

    return {
        "total_candidates": len(unique_pdf_files),
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


def _build_generation_prompt(
    question: str,
    context_text: str,
    broad_query: bool,
    parent: Optional[str] = None,
    child: Optional[str] = None,
    stream: bool = False,
) -> str:
    topic_hint = " / ".join([value for value in [parent, child] if value])
    topic_line = f"The user is currently looking at this topic area: {topic_hint}.\n" if topic_hint else ""
    broad_instruction = (
        "The user appears to want a complete or every-topic response. "
        "Cover every relevant topic present in the context, group the answer by topic, "
        "and do not stop after the first matching paragraph.\n"
        if broad_query
        else ""
    )
    return (
        "You are an expert technical document assistant for i-Tips. Your goal is to provide a highly detailed, "
        "accurate, and comprehensive answer based ON ONLY the provided Context.\n"
        f"{topic_line}"
        f"{broad_instruction}"
        "INSTRUCTIONS:\n"
        "1. Provide a comprehensive response. If multiple sections of the context are relevant, combine them logically.\n"
        "2. If the context contains specific technical parameters, values, or step-by-step instructions, include them all.\n"
        "3. Render any tables found in the context as Markdown Pipe Tables.\n"
        "4. Use bold text for key terms, error codes, and technical variables.\n"
        "5. Use bullet points for maintenance steps, checklists, or multi-part information.\n"
        "6. Include citations like [source, Page N] for every major fact or instruction.\n"
        "7. If the context is insufficient to give a detailed answer, explain what is missing instead of guessing.\n\n"
        f"Context:\n{context_text}\n\nQuery: {question}\n\nDetailed Technical Answer:"
    )

@app.post("/api/v1/ingest", response_model=IngestResponse)
def ingest_media(
    payload: Optional[IngestRequest] = Body(None),
    tenant_id: str = Query("default", pattern=TENANT_PATTERN),
    db: Session = Depends(get_db),
):
    """
    Scans the external media path for PDFs and ingests them into PostgreSQL (pgvector).
    """
    if payload and payload.tenant_id:
        tenant_id = payload.tenant_id
    tenant_id = validate_tenant_id(tenant_id)
    force_reindex = bool(payload.force_reindex) if payload else False
    pdf_files = _find_pdf_files(include_scan=True)
    if not pdf_files:
        return {
            "status": "success",
            "message": f"No PDFs found to ingest in {MEDIA_PATH}.",
            "files_processed": 0,
            "files_queued": 0,
            "files_skipped": 0,
            "jobs": [],
        }

    queue_result = _queue_pdf_ingestion(
        tenant_id,
        pdf_files,
        db,
        force_reindex=force_reindex,
    )
    jobs = queue_result["jobs"]
    return {
        "status": "queued" if jobs else "success",
        "message": (
            f"Queued ingestion for {len(jobs)} of {len(pdf_files)} PDFs."
            if jobs
            else f"All {len(pdf_files)} PDFs are already indexed or queued."
        ),
        "files_processed": len(pdf_files),
        "files_queued": len(jobs),
        "files_skipped": queue_result["skipped"],
        "jobs": [_job_response(job) for job in jobs],
    }

@app.post("/api/v1/upload")
async def upload_file(
    tenant_id: str = Query("default", pattern=TENANT_PATTERN),
    file: UploadFile = File(...),
):
    tenant_id = validate_tenant_id(tenant_id)
    filename = os.path.basename(file.filename or "")
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Stream file to disk in chunks to avoid loading entire PDF into RAM
    tenant_media_path = os.path.join(MEDIA_PATH, tenant_id)
    os.makedirs(tenant_media_path, exist_ok=True)
    file_path = os.path.join(tenant_media_path, filename)

    total_bytes = 0
    try:
        with open(file_path, "wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)  # Read 1MB at a time
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_SIZE_BYTES:
                    buffer.close()
                    os.remove(file_path)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum upload size is {MAX_UPLOAD_SIZE_MB}MB.",
                    )
                buffer.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

    file_size_mb = round(total_bytes / (1024 * 1024), 2)
    job = create_ingestion_job(tenant_id, file_path)
    return {
        "message": f"Saved {filename} ({file_size_mb}MB) and queued background ingestion.",
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
        
        # Add basic stats for the dashboard
        stats_query = db.execute(text(
            "SELECT COUNT(*) as chunks, COUNT(DISTINCT doc_id) as docs FROM document_chunks"
        )).mappings().first()
        checks["stats"] = {
            "chunks": stats_query["chunks"] if stats_query else 0,
            "docs": stats_query["docs"] if stats_query else 0
        }
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
        checks["ollama"] = f"error: {exc}"
        status_code = 503

    try:
        validate_runtime_models()
        checks["models"] = runtime_model_info()
    except Exception as exc:
        checks["models"] = f"error: {exc}"
        status_code = 503

    payload = {"status": "ok" if status_code == 200 else "unready", "checks": checks}
    if status_code != 200:
        raise HTTPException(status_code=status_code, detail=payload)
    return payload

@app.get("/", response_class=HTMLResponse)
def root_ui():
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>i-Tips RAG Production Hub</title>
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
                --text: #f1f5f9;
                --text-muted: #94a3b8;
                --text-dim: #64748b;
            }
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
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
            .header { text-align: center; margin: 1rem 0 2rem; }
            .header h1 {
                font-size: 2rem;
                font-weight: 700;
                background: linear-gradient(135deg, #3b82f6, #8b5cf6, #06b6d4);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                letter-spacing: -0.5px;
            }
            .header p { color: var(--text-dim); font-size: 0.85rem; margin-top: 0.25rem; }
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
            .btn-sm { padding: 0.4rem 0.8rem; font-size: 0.75rem; }
            .flex-row { display: flex; gap: 0.75rem; align-items: center; width: 100%; }

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

            /* Chart Container */
            .chart-container { background: rgba(0,0,0,0.25); border-radius: 12px; padding: 1rem; margin-top: 1rem; border: 1px solid var(--border); }
            .chart-container canvas { max-height: 250px; }

            /* Latency Bar */
            .latency-bar {
                display: flex; align-items: center; gap: 0.75rem; margin-top: 1rem;
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
            <h1>i-Tips Production RAG Hub</h1>
            <p>6-Layer Architecture | Hybrid Search | Cross-Encoder Reranking</p>
        </div>

        <div class="stats-bar" id="statsBar">
            <div class="stat"><strong id="statChunks">--</strong> Chunks Indexed</div>
            <div class="stat"><strong id="statPdfs">--</strong> PDFs Loaded</div>
            <div class="stat"><strong id="statModel">--</strong> LLM Model</div>
        </div>

        <div class="container">
            <div class="card">
                <h2><span class="icon">&#128196;</span> Upload Knowledge Base</h2>
                <div class="flex-row">
                    <input type="file" id="pdfFile" accept="application/pdf">
                    <button class="btn" onclick="uploadPdf()">Upload &amp; Ingest</button>
                    <div id="uploadSpinner" class="spinner"></div>
                </div>
                <p id="uploadStatus" class="status-msg"></p>
            </div>

            <div class="card">
                <h2><span class="icon">&#128269;</span> Ask Your Knowledge Base</h2>
                <div class="flex-row">
                    <input type="text" id="queryInput" placeholder="E.g. What is a DC Sensor? Show me the maintenance schedule..." onkeypress="if(event.key === 'Enter') askQuery()">
                    <button class="btn" onclick="askQuery()">Ask</button>
                    <div id="askSpinner" class="spinner"></div>
                </div>

                <div class="answer-box" id="answerBox"><span class="placeholder">Your answer will appear here with tables, charts, and formatted text...</span></div>

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
        </div>

        <script>
            // Configure marked for safe markdown rendering
            marked.setOptions({ breaks: true, gfm: true });

            // Load stats on page load
            (async function loadStats() {
                try {
                    const res = await fetch('/health/ready');
                    const data = await res.json();
                    const stats = data.checks?.stats || {};
                    document.getElementById('statChunks').textContent = stats.chunks?.toLocaleString() || '0';
                    document.getElementById('statPdfs').textContent = stats.docs?.toLocaleString() || '0';
                    document.getElementById('statModel').textContent = data.checks?.models?.embedding_model?.split('/').pop() || '--';
                } catch(e) {}
            })();

            async function uploadPdf() {
                const fileInput = document.getElementById('pdfFile');
                if (!fileInput.files[0]) return alert("Please select a PDF file first.");

                const formData = new FormData();
                formData.append("file", fileInput.files[0]);

                document.getElementById('uploadSpinner').style.display = 'block';
                document.getElementById('uploadStatus').innerText = "Uploading and embedding PDF chunks...";

                try {
                    const res = await fetch('/api/v1/upload', { method: 'POST', body: formData });
                    const data = await res.json();
                    if (res.ok) {
                        document.getElementById('uploadStatus').innerHTML = '<span style="color:#10b981">&#10003; ' + data.message + '</span>';
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
                // Convert plain text tables (pipe-separated) to markdown tables if needed
                let processed = text.replace(/\\n/g, '\\n');
                return marked.parse(processed);
            }

            function tryBuildChart(answer, context) {
                const chartArea = document.getElementById('chartArea');
                chartArea.innerHTML = '';

                // Detect table data in context for visualization
                let tableChunks = (context || []).filter(c => {
                    const meta = c.metadata || {};
                    return meta.type === 'table' || (c.text && c.text.includes('|') && c.text.split('|').length > 4);
                });

                if (tableChunks.length === 0) return;

                // Try to extract numeric data from first table chunk
                const lines = tableChunks[0].text.split('\\n').filter(l => l.trim() && !l.match(/^[\\-|\\s]+$/));
                if (lines.length < 2) return;

                const headers = lines[0].split('|').map(h => h.trim()).filter(Boolean);
                const rows = lines.slice(1).map(l => l.split('|').map(c => c.trim()).filter(Boolean));

                // Find numeric columns
                let numericCol = -1;
                let labelCol = 0;
                for (let i = 1; i < headers.length; i++) {
                    if (rows.length > 0 && !isNaN(parseFloat(rows[0][i]))) {
                        numericCol = i;
                        break;
                    }
                }
                if (numericCol === -1 || rows.length < 2 || rows.length > 20) return;

                const labels = rows.map(r => r[labelCol] || '').slice(0, 15);
                const values = rows.map(r => parseFloat(r[numericCol]) || 0).slice(0, 15);

                chartArea.innerHTML = '<div class="chart-container"><canvas id="dataChart"></canvas></div>';
                const ctx = document.getElementById('dataChart').getContext('2d');
                new Chart(ctx, {
                    type: values.length > 8 ? 'line' : 'bar',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: headers[numericCol] || 'Value',
                            data: values,
                            backgroundColor: 'rgba(59,130,246,0.4)',
                            borderColor: '#3b82f6',
                            borderWidth: 2,
                            borderRadius: 6,
                            tension: 0.3,
                            fill: true,
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            legend: { labels: { color: '#94a3b8', font: { family: 'Inter' } } },
                        },
                        scales: {
                            x: { ticks: { color: '#64748b', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
                            y: { ticks: { color: '#64748b' }, grid: { color: 'rgba(255,255,255,0.05)' } }
                        }
                    }
                });
            }

            function renderSources(context) {
                const area = document.getElementById('sourcesArea');
                if (!context || context.length === 0) { area.innerHTML = ''; return; }

                let html = '<div class="sources-grid">';
                context.forEach(c => {
                    const meta = c.metadata || {};
                    const type = meta.type || 'text';
                    const badgeClass = type === 'table' ? 'badge-table' : type === 'image_ocr' ? 'badge-image' : 'badge-text';
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

            function showLatency(ms) {
                const bar = document.getElementById('latencyBar');
                const dot = document.getElementById('latencyDot');
                const txt = document.getElementById('latencyText');
                bar.style.display = 'flex';

                dot.className = 'latency-dot ' + (ms < 5000 ? 'latency-fast' : ms < 30000 ? 'latency-medium' : 'latency-slow');
                const label = ms < 1000 ? ms + 'ms (Cached)' : (ms/1000).toFixed(1) + 's';
                txt.textContent = 'Response: ' + label + ' | Pipeline: Cache > Hybrid Search > Reranker > MMR > LLM';
            }

            async function askQuery() {
                const query = document.getElementById('queryInput').value;
                if (!query) return;

                const answerBox = document.getElementById('answerBox');
                const askSpinner = document.getElementById('askSpinner');
                const chartArea = document.getElementById('chartArea');
                const sourcesArea = document.getElementById('sourcesArea');
                const latencyBar = document.getElementById('latencyBar');

                askSpinner.style.display = 'block';
                answerBox.innerHTML = '<span class="placeholder">Searching knowledge base and generating response...</span>';
                chartArea.innerHTML = '';
                sourcesArea.innerHTML = '';
                latencyBar.style.display = 'none';

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
                        const lines = chunk.split("\n");

                        for (const line of lines) {
                            if (line.startsWith("data: ")) {
                                try {
                                    const data = JSON.parse(line.substring(6));
                                    if (data.token) {
                                        fullAnswer += data.token;
                                        answerBox.innerHTML = renderAnswer(fullAnswer);
                                        // Scroll to bottom
                                        answerBox.scrollTop = answerBox.scrollHeight;
                                    }
                                    if (data.done) {
                                        if (data.sources) renderSources(data.sources);
                                        // Update raw API box
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

@app.post("/api/v1/query", response_model=QueryResponse)
def query_rag(request: QueryRequest, db: Session = Depends(get_db)):
    """
    Production RAG Query Pipeline:
    1. Hybrid Retrieval (ANN via pgvector)
    2. Reranking (Cross-encoder)
    3. Context Assembly (MMR)
    4. Generation (Ollama)
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
        pdf_files = _find_pdf_files(include_scan=request.include_scan)
        queue_result = _queue_pdf_ingestion(
            tenant_id,
            pdf_files,
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
    
    # Layer 6: Semantic Query Cache
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
    
    # Layer 3: Hybrid Retrieval (ANN with pgvector + BM25)
    retrieved_chunks = perform_hybrid_search(db, search_query, tenant_id, top_k=max(20, effective_top_k * 4))
    
    # Layer 4: Reranking (Top 20 -> Top 5 using Cross-Encoder)
    reranked_chunks = rerank_results(search_query, retrieved_chunks, top_n=effective_top_k)
    
    # Layer 5: Context Assembly
    final_context = assemble_context(search_query, reranked_chunks, db=db)
    sources = _context_sources(final_context)
    
    # Formulate Prompt for Ollama
    context_text = "\n\n".join([f"Source: {c['citation']}\n{c['text']}" for c in final_context])
    prompt = _build_generation_prompt(
        request.query,
        context_text,
        broad_query=broad_query,
        parent=request.parent,
        child=request.child,
    )
    
    # Call Ollama API
    answer = "The knowledge base does not contain enough information to answer this question."
    if not final_context:
        if ingest_summary and ingest_summary.get("queued"):
            answer = (
                f"I queued {ingest_summary['queued']} PDF(s) for ingestion. "
                "The knowledge base is still processing them, so please ask again once ingestion finishes."
            )
        latency = int((time.time() - start_time) * 1000)
        response_data = {
            "answer": answer,
            "context": final_context,
            "sources": sources,
            "latency_ms": latency,
            "ingest": ingest_summary,
        }
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

    if request.stream:
        def stream_generator():
            try:
                with requests.post(OLLAMA_URL, json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "num_predict": OLLAMA_NUM_PREDICT,
                        "num_ctx": 8192,
                    }
                }, timeout=OLLAMA_TIMEOUT_SECONDS, stream=True) as response:
                    for line in response.iter_lines():
                        if line:
                            json_resp = json.loads(line)
                            token = json_resp.get("response", "")
                            if token:
                                yield f"data: {json.dumps({'token': token})}\n\n"
                            if json_resp.get("done"):
                                # Final payload with sources
                                yield f"data: {json.dumps({'done': True, 'sources': sources})}\n\n"
                                break
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    try:
        response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": OLLAMA_NUM_PREDICT,
                "num_ctx": 8192,
            }
        }, timeout=OLLAMA_TIMEOUT_SECONDS)
        
        if response.status_code == 200:
            answer = response.json().get("response", answer)
        else:
            print(f"Ollama error: {response.text}")
    except Exception as e:
        print(f"Error connecting to Ollama at {OLLAMA_URL}: {e}")
    
    latency = int((time.time() - start_time) * 1000)
    
    response_data = {
        "answer": answer,
        "context": final_context,
        "sources": sources,
        "latency_ms": latency,
        "ingest": ingest_summary,
    }
    
    # Layer 6: Save to Cache
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
    
    print("[RAG Native] Starting i-Tips RAG natively for GPU access...")
    uvicorn.run(app, host="0.0.0.0", port=1000)
