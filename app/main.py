import os
import glob
import requests
import shutil
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends, File, UploadFile
from fastapi.responses import HTMLResponse
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

@app.post("/api/v1/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    file_path = os.path.join(MEDIA_PATH, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    background_tasks.add_task(ingest_pdf, file_path)
    return {"message": f"Saved {file.filename} and triggered background ingestion."}

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
