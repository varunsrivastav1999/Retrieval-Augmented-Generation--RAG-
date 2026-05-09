import os
import glob
import requests
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Dict, Any
import redis
import json
import hashlib

from app.database import init_db, get_db
from app.rag.ingestion import ingest_pdf
from app.rag.retrieval import perform_hybrid_search
from app.rag.reranker import rerank_results
from app.rag.context import assemble_context

app = FastAPI(title="i-Tips RAG Production API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MEDIA_PATH = os.getenv("MEDIA_PATH", "/media")

# Connecting directly to the independent Ollama container inside this repository
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3") # Set to whichever model you have pulled in Ollama

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
try:
    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
except Exception as e:
    redis_client = None
    print(f"Warning: Redis cache not available - {e}")

def get_cached_response(query: str):
    if not redis_client: return None
    try:
        query_hash = hashlib.sha256(query.encode('utf-8')).hexdigest()
        cached = redis_client.get(f"rag_cache:{query_hash}")
        if cached: return json.loads(cached)
    except Exception as e: print(f"Redis get error: {e}")
    return None

def set_cached_response(query: str, response: dict):
    if not redis_client: return
    try:
        query_hash = hashlib.sha256(query.encode('utf-8')).hexdigest()
        redis_client.setex(f"rag_cache:{query_hash}", 86400, json.dumps(response)) # 24h TTL
    except Exception as e: print(f"Redis set error: {e}")

@app.on_event("startup")
def on_startup():
    try:
        init_db()
        print("Database initialized with pgvector successfully.")
    except Exception as e:
        print(f"Failed to initialize database: {e}")

class QueryRequest(BaseModel):
    query: str
    tenant_id: str = "default"
    top_k: int = 5

class IngestResponse(BaseModel):
    status: str
    message: str
    files_processed: int

class QueryResponse(BaseModel):
    answer: str
    context: List[Dict[str, Any]]
    latency_ms: int

@app.post("/api/v1/ingest", response_model=IngestResponse)
def ingest_media(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    Scans the external media path for PDFs and ingests them into PostgreSQL (pgvector).
    """
    pdf_files = glob.glob(os.path.join(MEDIA_PATH, "**/*.pdf"), recursive=True)
    if not pdf_files:
        return {"status": "success", "message": f"No PDFs found to ingest in {MEDIA_PATH}.", "files_processed": 0}
    
    # Process offline in background
    for pdf_path in pdf_files:
        background_tasks.add_task(ingest_pdf, pdf_path)
        
    return {"status": "processing", "message": f"Started ingestion for {len(pdf_files)} PDFs.", "files_processed": len(pdf_files)}

@app.post("/api/v1/query", response_model=QueryResponse)
def query_rag(request: QueryRequest, db: Session = Depends(get_db)):
    """
    Production RAG Query Pipeline:
    1. Hybrid Retrieval (ANN via pgvector)
    2. Reranking (Cross-encoder)
    3. Context Assembly (MMR)
    4. Generation (Ollama)
    """
    import time
    start_time = time.time()
    
    # Layer 6: Semantic Query Cache
    cached = get_cached_response(request.query)
    if cached:
        print(f"[Cache] HIT for query: '{request.query}'")
        cached["latency_ms"] = int((time.time() - start_time) * 1000)
        return cached
    
    # Layer 3: Hybrid Retrieval (ANN with pgvector + BM25)
    retrieved_chunks = perform_hybrid_search(db, request.query, request.tenant_id, top_k=20)
    
    # Layer 4: Reranking (Top 20 -> Top 5 using Cross-Encoder)
    reranked_chunks = rerank_results(request.query, retrieved_chunks, top_n=request.top_k)
    
    # Layer 5: Context Assembly
    final_context = assemble_context(request.query, reranked_chunks)
    
    # Formulate Prompt for Ollama
    context_text = "\n\n".join([f"Source: {c['citation']}\n{c['text']}" for c in final_context])
    prompt = f"Use the following context to answer the user's query.\n\nContext:\n{context_text}\n\nQuery: {request.query}\nAnswer:"
    
    # Call Ollama API
    answer = "Failed to generate answer. Check if Ollama is running and the model is pulled."
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }, timeout=180)
        
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
    set_cached_response(request.query, response_data)
    
    return response_data
