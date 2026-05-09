import os
import glob
import requests
import shutil
import re
import threading
import time
from fastapi import HTTPException, Depends, File, Query, UploadFile
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import redis
import json
import hashlib

from app.database import IngestionJob, init_db, get_db
from app.rag.jobs import create_ingestion_job, create_ingestion_jobs, get_ingestion_job, start_ingestion_worker
from app.rag.model_loader import get_embedding_model_id, runtime_model_info, validate_runtime_models
from app.rag.retrieval import perform_hybrid_search
from app.rag.reranker import rerank_results
from app.rag.context import assemble_context

app = FastAPI(title="i-Tips RAG Production API", version="1.1.0")

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
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))
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
) -> str:
    payload = {
        "tenant_id": tenant_id,
        "query": query,
        "top_k": top_k,
        "embedding_model": embedding_model,
        "corpus_version": corpus_version,
    }
    query_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"rag_cache:{query_hash}"


def get_cached_response(
    query: str,
    tenant_id: str,
    top_k: int,
    embedding_model: str,
    corpus_version: str,
):
    if not redis_client: return None
    try:
        cached = redis_client.get(
            _cache_key(query, tenant_id, top_k, embedding_model, corpus_version)
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
):
    if not redis_client: return
    try:
        redis_client.setex(
            _cache_key(query, tenant_id, top_k, embedding_model, corpus_version),
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
    top_k: int = Field(5, ge=1, le=20)


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
    jobs: List[IngestionJobResponse] = []

class QueryResponse(BaseModel):
    answer: str
    context: List[Dict[str, Any]]
    latency_ms: int


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

@app.post("/api/v1/ingest", response_model=IngestResponse)
def ingest_media(tenant_id: str = Query("default", pattern=TENANT_PATTERN)):
    """
    Scans the external media path for PDFs and ingests them into PostgreSQL (pgvector).
    """
    tenant_id = validate_tenant_id(tenant_id)
    pdf_files = glob.glob(os.path.join(MEDIA_PATH, "**/*.pdf"), recursive=True)
    if not pdf_files:
        return {
            "status": "success",
            "message": f"No PDFs found to ingest in {MEDIA_PATH}.",
            "files_processed": 0,
            "jobs": [],
        }

    jobs = create_ingestion_jobs(tenant_id, pdf_files)
    return {
        "status": "queued",
        "message": f"Queued ingestion for {len(pdf_files)} PDFs.",
        "files_processed": len(pdf_files),
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

    tenant_media_path = os.path.join(MEDIA_PATH, tenant_id)
    os.makedirs(tenant_media_path, exist_ok=True)
    file_path = os.path.join(tenant_media_path, filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    job = create_ingestion_job(tenant_id, file_path)
    return {
        "message": f"Saved {filename} and queued background ingestion.",
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
        <title>RAG</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
        <style>
            :root { --bg: #0f172a; --panel: rgba(30, 41, 59, 0.7); --primary: #3b82f6; --text: #f8fafc; --text-muted: #94a3b8; }
            body { margin: 0; font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); display: flex; flex-direction: column; align-items: center; padding: 2rem; min-height: 100vh;}
            .container { max-width: 800px; width: 100%; display: flex; flex-direction: column; gap: 2rem; margin-top: 2rem;}
            .card { background: var(--panel); backdrop-filter: blur(12px); border-radius: 16px; padding: 2rem; border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
            h1, h2 { margin-top: 0; font-weight: 600; }
            input[type="text"], input[type="file"] { width: 100%; padding: 0.75rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.2); background: rgba(0,0,0,0.2); color: white; box-sizing: border-box; }
            input[type="text"]:focus { outline: none; border-color: var(--primary); }
            button { background: var(--primary); color: white; border: none; padding: 0.75rem 1.5rem; border-radius: 8px; cursor: pointer; font-weight: 600; transition: all 0.2s; white-space: nowrap;}
            button:hover { background: #2563eb; transform: translateY(-1px); }
            .answer-box { background: rgba(0,0,0,0.3); padding: 1.5rem; border-radius: 8px; min-height: 100px; white-space: pre-wrap; font-size: 0.95rem; line-height: 1.6; margin-top: 1rem; border: 1px solid rgba(255,255,255,0.05);}
            .context-box { font-size: 0.8rem; color: var(--text-muted); margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.1); }
            .spinner { display: none; width: 20px; height: 20px; border: 3px solid rgba(255,255,255,0.3); border-radius: 50%; border-top-color: #fff; animation: spin 1s ease-in-out infinite; margin-left: 1rem; flex-shrink: 0;}
            @keyframes spin { to { transform: rotate(360deg); } }
            .flex-row { display: flex; gap: 1rem; align-items: center; width: 100%; }
        </style>
    </head>
    <body>
        <h1 style="background: -webkit-linear-gradient(#3b82f6, #60a5fa); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Welcome to Q&A </h1>
        <div class="container">
            <div class="card">
                <h2> Upload Knowledge Base (PDF)</h2>
                <div class="flex-row">
                    <input type="file" id="pdfFile" accept="application/pdf">
                    <button onclick="uploadPdf()">Upload & Ingest</button>
                    <div id="uploadSpinner" class="spinner"></div>
                </div>
                <p id="uploadStatus" style="color: var(--text-muted); font-size: 0.85rem; margin-bottom: 0;"></p>
            </div>
            
            <div class="card">
                <h2> Ask Me Anything</h2>
                <div class="flex-row">
                    <input type="text" id="queryInput" placeholder="E.g. What is the RAG?" onkeypress="if(event.key === 'Enter') askQuery()">
                    <button onclick="askQuery()">Ask Query</button>
                    <div id="askSpinner" class="spinner"></div>
                </div>
                
                <div class="answer-box" id="answerBox">Awaiting your question...</div>
                <div class="context-box" id="contextBox"></div>
            </div>
        </div>

        <script>
            async function uploadPdf() {
                const fileInput = document.getElementById('pdfFile');
                if (!fileInput.files[0]) return alert("Please select a PDF file first.");
                
                const formData = new FormData();
                formData.append("file", fileInput.files[0]);
                
                document.getElementById('uploadSpinner').style.display = 'block';
                document.getElementById('uploadStatus').innerText = "Uploading and embedding PDF chunks... This may take a moment.";
                
                try {
                    const res = await fetch('/api/v1/upload', { method: 'POST', body: formData });
                    const data = await res.json();
                    document.getElementById('uploadStatus').innerText = "Success! " + data.message;
                } catch (e) {
                    document.getElementById('uploadStatus').innerText = "Error uploading file. Check console.";
                } finally {
                    document.getElementById('uploadSpinner').style.display = 'none';
                    fileInput.value = '';
                }
            }
            
            async function askQuery() {
                const query = document.getElementById('queryInput').value;
                if (!query) return;
                
                document.getElementById('askSpinner').style.display = 'block';
                document.getElementById('answerBox').innerText = "Retrieving context and running Ollama inference...";
                document.getElementById('contextBox').innerHTML = "";
                
                try {
                    const res = await fetch('/api/v1/query', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query: query })
                    });
                    const data = await res.json();
                    
                    document.getElementById('answerBox').innerText = data.answer || "No answer generated.";
                    
                    if (data.context && data.context.length > 0) {
                        let ctxHtml = "<strong>Sources retrieved (Top 5):</strong><ul style='margin-bottom:0;'>";
                        data.context.forEach(c => {
                            ctxHtml += `<li>${c.citation} (Relevance Score: ${c.rerank_score.toFixed(3)})</li>`;
                        });
                        ctxHtml += "</ul><br><small>Response Time: " + data.latency_ms + "ms</small>";
                        document.getElementById('contextBox').innerHTML = ctxHtml;
                    }
                } catch (e) {
                    document.getElementById('answerBox').innerText = "Error connecting to backend.";
                } finally {
                    document.getElementById('askSpinner').style.display = 'none';
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
    corpus_version = get_corpus_version(db, tenant_id, embedding_model)
    
    # Layer 6: Semantic Query Cache
    cached = get_cached_response(
        request.query,
        tenant_id,
        request.top_k,
        embedding_model,
        corpus_version,
    )
    if cached:
        print(f"[Cache] HIT for tenant={tenant_id!r}, query={request.query!r}")
        cached["latency_ms"] = int((time.time() - start_time) * 1000)
        return cached
    
    # Layer 3: Hybrid Retrieval (ANN with pgvector + BM25)
    retrieved_chunks = perform_hybrid_search(db, request.query, tenant_id, top_k=max(20, request.top_k * 4))
    
    # Layer 4: Reranking (Top 20 -> Top 5 using Cross-Encoder)
    reranked_chunks = rerank_results(request.query, retrieved_chunks, top_n=request.top_k)
    
    # Layer 5: Context Assembly
    final_context = assemble_context(request.query, reranked_chunks)
    
    # Formulate Prompt for Ollama
    context_text = "\n\n".join([f"Source: {c['citation']}\n{c['text']}" for c in final_context])
    prompt = (
        "Use only the following context to answer the user's query. "
        "If the context is insufficient, say that the knowledge base does not contain enough information.\n\n"
        f"Context:\n{context_text}\n\nQuery: {request.query}\nAnswer:"
    )
    
    # Call Ollama API
    answer = "The knowledge base does not contain enough information to answer this question."
    if not final_context:
        latency = int((time.time() - start_time) * 1000)
        response_data = {
            "answer": answer,
            "context": final_context,
            "latency_ms": latency
        }
        set_cached_response(
            request.query,
            tenant_id,
            request.top_k,
            embedding_model,
            corpus_version,
            response_data,
        )
        return response_data

    try:
        response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
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
        "latency_ms": latency
    }
    
    # Layer 6: Save to Cache
    set_cached_response(
        request.query,
        tenant_id,
        request.top_k,
        embedding_model,
        corpus_version,
        response_data,
    )
    
    return response_data
