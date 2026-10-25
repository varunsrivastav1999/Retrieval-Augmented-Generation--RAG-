"""HTTP API + UI for the offline industrial maintenance RAG system."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select, text as sql

from app.config import get_settings
from app.db import Chunk, Conversation, Document, Message, SessionLocal, db_ping, init_db
from app.logging_setup import setup_logging
from app.metrics import INDEXED_CHUNKS
from app.rag import cache, jobs, llm, models, relations, vector_store
from app.rag.parsers import SUPPORTED_EXTENSIONS, is_supported_file
from app.rag.pipeline import QueryOptions, run, run_to_completion

settings = get_settings()
setup_logging(settings.log_level, settings.log_json)
log = logging.getLogger("api")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
TENANT_RE = r"^[A-Za-z0-9_-]{1,80}$"
_retrieval_slots = threading.BoundedSemaphore(max(2, (os.cpu_count() or 2)))


def _wait_for(name: str, fn, attempts: int = 60, delay: float = 2.0) -> None:
    for i in range(1, attempts + 1):
        try:
            fn()
            return
        except Exception as exc:
            if i == attempts:
                raise
            log.warning("%s not ready (%d/%d): %s", name, i, attempts, exc)
            time.sleep(delay)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    _wait_for("qdrant", vector_store.ensure_collection)
    models.load_models()
    os.makedirs(os.path.join(settings.media_path, "uploads"), exist_ok=True)
    threading.Thread(target=llm.warmup, name="llm-warmup", daemon=True).start()
    log.info("api ready", extra={"llm": settings.llm_model, "embedding": settings.embedding_model})
    yield


app = FastAPI(title="Industrial Maintenance RAG", version="7.0.0", lifespan=lifespan,
              docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json")
if settings.cors_origins:
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    if request.url.path.startswith("/static"):
        # Revalidate (ETag) on every load so an upgraded UI is never mixed with cached old files.
        response.headers["Cache-Control"] = "no-cache"
    if request.url.path == "/" or request.url.path.startswith("/static"):
        response.headers.setdefault("Content-Security-Policy",
                                    "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'")
    return response


def require_key(request: Request) -> None:
    if settings.api_key and request.headers.get("x-api-key") != settings.api_key:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key")


def tenant_param(tenant_id: str = Query("default", pattern=TENANT_RE)) -> str:
    return tenant_id


# ---------------------------------------------------------------------------
# Health, status, metrics
# ---------------------------------------------------------------------------
@app.get("/health/live", include_in_schema=False)
def live():
    return {"status": "ok"}


@app.get("/health/ready", include_in_schema=False)
def ready():
    checks = {"database": db_ping(), "vector_store": vector_store.ping(), "models": models.models_ready()}
    ok = all(checks.values())
    return JSONResponse({"status": "ready" if ok else "not_ready", "checks": checks}, status_code=200 if ok else 503)


@app.get("/metrics", include_in_schema=False)
def metrics():
    try:
        INDEXED_CHUNKS.set(vector_store.count())
    except Exception:
        pass
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/v1/status", dependencies=[Depends(require_key)])
def status(tenant_id: str = Depends(tenant_param)):
    started = time.perf_counter()
    services = {
        "api": {"status": "ok"},
        "database": {"status": "ok" if db_ping() else "down"},
        "vector_store": {"status": "ok" if vector_store.ping() else "down"},
        "cache": {"status": "ok" if cache.ping() else "down"},
        "llm": llm.health(),
        "models": models.model_info(),
    }
    index = {}
    try:
        with SessionLocal() as db:
            rows = db.execute(sql("""SELECT status, count(*) AS docs, coalesce(sum(chunk_count),0) AS chunks
                                     FROM documents WHERE tenant_id=:t GROUP BY status"""), {"t": tenant_id}).all()
        index["documents"] = {r.status: r.docs for r in rows}
        index["chunks"] = int(sum(r.chunks for r in rows))
        index["vectors"] = vector_store.count(tenant_id)
        index["queue"] = jobs.queue_depth()
    except Exception as exc:
        index["error"] = str(exc)[:200]
    overall = "ok"
    if any(services[k]["status"] != "ok" for k in ("database", "vector_store")) or not services["models"]["loaded"]:
        overall = "down"
    elif services["llm"]["status"] != "ok" or services["cache"]["status"] != "ok":
        overall = "degraded"
    return {"status": overall, "services": services, "index": index,
            "config": {"llm_model": settings.llm_model, "top_k": settings.top_k, "rerank": "full" if settings.rerank_enabled else f"top-{settings.rerank_gate_n} relevance gate",
                       "auth": bool(settings.api_key)},
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)}


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    tenant_id: str = Field("default", pattern=TENANT_RE)
    mode: str = Field("auto", pattern="^(auto|evidence|answer)$")
    top_k: Optional[int] = Field(None, ge=1, le=50)
    collections: List[str] = Field(default_factory=list, max_length=20)
    file_types: List[str] = Field(default_factory=list, max_length=20)
    document_ids: List[str] = Field(default_factory=list, max_length=50)
    conversation_id: Optional[str] = Field(None, max_length=36)
    stream: bool = False
    use_cache: bool = True


def _options(req: QueryRequest) -> QueryOptions:
    return QueryOptions(query=req.query, tenant_id=req.tenant_id, mode=req.mode, top_k=req.top_k,
                        collections=req.collections, file_types=req.file_types, document_ids=req.document_ids,
                        conversation_id=req.conversation_id, use_cache=req.use_cache)


@app.post("/api/v1/query", dependencies=[Depends(require_key)])
def query(req: QueryRequest):
    opts = _options(req)
    if req.stream:
        def events():
            with _retrieval_slots:
                for event in run(opts):
                    yield f"data: {json.dumps(event, default=str)}\n\n"
        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    with _retrieval_slots:
        result = run_to_completion(opts)
    if result.get("status") == "error":
        raise HTTPException(status_code=503, detail=result.get("message", "service unavailable"))
    return result


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    tenant_id: str = Field("default", pattern=TENANT_RE)
    top_k: int = Field(10, ge=1, le=50)
    collections: List[str] = Field(default_factory=list)
    file_types: List[str] = Field(default_factory=list)
    rerank: bool = True


@app.post("/api/v1/search", dependencies=[Depends(require_key)])
def search_only(req: SearchRequest):
    """Retrieval only (no answer generation) — for integrations and benchmarks."""
    from app.rag.search import retrieve
    with _retrieval_slots:
        try:
            r = retrieve(req.query, req.tenant_id, top_k=req.top_k, collections=req.collections,
                         file_types=req.file_types, rerank=req.rerank)
        except Exception as exc:
            log.exception("search failed")
            raise HTTPException(status_code=503, detail=f"search unavailable: {type(exc).__name__}") from exc
    return {"hits": [h.public() for h in r.hits], "query_identifiers": r.query_ids,
            "missing_identifiers": r.missing_ids, "timings": r.timings}


@app.get("/api/v1/related", dependencies=[Depends(require_key)])
def related(identifier: str = Query(..., min_length=2, max_length=80), tenant_id: str = Depends(tenant_param)):
    return relations.related_identifiers(tenant_id, [identifier.strip().upper()], limit=25)


# ---------------------------------------------------------------------------
# Documents & ingestion
# ---------------------------------------------------------------------------
def _doc_dict(d: Document) -> dict:
    return {k: getattr(d, k) for k in ("id", "name", "collection", "file_type", "status", "error", "pages",
                                       "chunk_count", "size_bytes", "created_at", "indexed_at", "path")}


@app.get("/api/v1/documents", dependencies=[Depends(require_key)])
def list_documents(tenant_id: str = Depends(tenant_param), q: str = Query("", max_length=200),
                   collection: str = Query("", max_length=128), status: str = Query("", max_length=16),
                   limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    with SessionLocal() as db:
        stmt = select(Document).where(Document.tenant_id == tenant_id)
        if q:
            stmt = stmt.where(Document.name.ilike(f"%{q}%"))
        if collection:
            stmt = stmt.where(Document.collection == collection)
        if status:
            stmt = stmt.where(Document.status == status)
        total = db.execute(select(func.count()).select_from(stmt.subquery())).scalar()
        docs = db.execute(stmt.order_by(Document.updated_at.desc()).limit(limit).offset(offset)).scalars().all()
    return {"total": total, "documents": [_doc_dict(d) for d in docs]}


@app.get("/api/v1/collections", dependencies=[Depends(require_key)])
def list_collections(tenant_id: str = Depends(tenant_param)):
    with SessionLocal() as db:
        rows = db.execute(sql("""SELECT collection, count(*) AS docs, coalesce(sum(chunk_count),0) AS chunks
                                 FROM documents WHERE tenant_id=:t GROUP BY collection ORDER BY collection"""),
                          {"t": tenant_id}).all()
        types = db.execute(sql("SELECT DISTINCT file_type FROM documents WHERE tenant_id=:t ORDER BY 1"), {"t": tenant_id}).scalars().all()
    return {"collections": [{"name": r.collection, "documents": r.docs, "chunks": int(r.chunks)} for r in rows],
            "file_types": [t for t in types if t]}


@app.get("/api/v1/chunks/{chunk_id}", dependencies=[Depends(require_key)])
def get_chunk(chunk_id: int, tenant_id: str = Depends(tenant_param), context: int = Query(2, ge=0, le=10)):
    """A chunk with its neighbours, for 'open source' in the UI."""
    with SessionLocal() as db:
        c = db.get(Chunk, chunk_id)
        if c is None or c.tenant_id != tenant_id:
            raise HTTPException(404, "chunk not found")
        doc = db.get(Document, c.document_id)
        rows = db.execute(select(Chunk).where(Chunk.document_id == c.document_id,
                                              Chunk.ord.between(c.ord - context, c.ord + context)).order_by(Chunk.ord)).scalars().all()
    return {"document": _doc_dict(doc) if doc else None, "chunk_id": chunk_id,
            "chunks": [{"id": r.id, "ord": r.ord, "page": r.page, "section": r.section_title, "content_type": r.content_type,
                        "text": r.text, "identifiers": r.identifiers} for r in rows]}


_MEDIA_TYPES = {".pdf": "application/pdf", ".txt": "text/plain; charset=utf-8", ".md": "text/plain; charset=utf-8",
                ".log": "text/plain; charset=utf-8", ".csv": "text/csv; charset=utf-8", ".png": "image/png",
                ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


@app.get("/api/v1/documents/{document_id}/file")
def document_file(document_id: str, request: Request, tenant_id: str = Depends(tenant_param),
                  key: str = Query("", max_length=200)):
    """The original file, so operators can check an answer word for word (PDFs open at #page=N).
    Links cannot carry headers, so the API key may also be passed as ?key=."""
    if settings.api_key and request.headers.get("x-api-key") != settings.api_key and key != settings.api_key:
        raise HTTPException(status_code=401, detail="Missing or invalid API key")
    with SessionLocal() as db:
        doc = db.get(Document, document_id)
    if doc is None or doc.tenant_id != tenant_id or not os.path.isfile(doc.path):
        raise HTTPException(404, "document file not found")
    ext = os.path.splitext(doc.path)[1].lower()
    media = _MEDIA_TYPES.get(ext, "application/octet-stream")
    disposition = "inline" if ext in _MEDIA_TYPES else "attachment"
    headers = {"X-Content-Type-Options": "nosniff"}
    if ext != ".pdf":  # the browser PDF viewer does not run inside a sandboxed document
        headers["Content-Security-Policy"] = "sandbox"
    return FileResponse(doc.path, media_type=media, filename=doc.name, content_disposition_type=disposition,
                        headers=headers)


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ ()-]+")


@app.post("/api/v1/upload", dependencies=[Depends(require_key)])
async def upload(file: UploadFile = File(...), tenant_id: str = Form("default", pattern=TENANT_RE),  # noqa: B008
                 collection: str = Form("uploads", max_length=64)):
    name = _SAFE_NAME.sub("_", os.path.basename(file.filename or "")).strip(" .")
    if not name or not is_supported_file(name):
        raise HTTPException(400, f"Unsupported file type. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    coll = _SAFE_NAME.sub("_", collection).strip(" .") or "uploads"
    target_dir = os.path.join(settings.media_path, "uploads", tenant_id, coll)
    os.makedirs(target_dir, exist_ok=True)
    target = os.path.join(target_dir, name)
    tmp = target + f".part-{uuid.uuid4().hex[:8]}"
    limit = settings.max_upload_mb * 1024 * 1024
    size = 0
    try:
        with open(tmp, "wb") as out:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"File larger than {settings.max_upload_mb} MB")
                out.write(chunk)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    job = jobs.enqueue(tenant_id, target, force=True)
    return {"message": f"Uploaded {name} ({size / 1048576:.1f} MB); indexing queued.", "job_id": job.id,
            "collection": coll}


@app.post("/api/v1/documents/{document_id}/reindex", dependencies=[Depends(require_key)])
def reindex(document_id: str, tenant_id: str = Depends(tenant_param)):
    with SessionLocal() as db:
        doc = db.get(Document, document_id)
    if doc is None or doc.tenant_id != tenant_id:
        raise HTTPException(404, "document not found")
    job = jobs.enqueue(tenant_id, doc.path, force=True)
    return {"job_id": job.id}


@app.delete("/api/v1/documents/{document_id}", dependencies=[Depends(require_key)])
def delete_document(document_id: str, tenant_id: str = Depends(tenant_param)):
    with SessionLocal() as db:
        doc = db.get(Document, document_id)
    if doc is None or doc.tenant_id != tenant_id:
        raise HTTPException(404, "document not found")
    uploads_root = os.path.realpath(os.path.join(settings.media_path, "uploads")) + os.sep
    if not os.path.realpath(doc.path).startswith(uploads_root) and os.path.exists(doc.path):
        raise HTTPException(409, "This file lives in a watched media folder; remove it there and it will be "
                                 "dropped from the index automatically.")
    from app.rag.ingestion import remove_document
    remove_document(document_id)
    if os.path.exists(doc.path):
        os.remove(doc.path)
    cache.bump_index_version(tenant_id)
    return {"deleted": document_id}


@app.post("/api/v1/ingest/scan", dependencies=[Depends(require_key)])
def rescan():
    from app.worker import scan_media
    return scan_media()


@app.get("/api/v1/ingest/jobs", dependencies=[Depends(require_key)])
def list_jobs(tenant_id: str = Depends(tenant_param), limit: int = Query(50, ge=1, le=200)):
    return [{k: getattr(j, k) for k in ("id", "source_name", "file_type", "status", "attempts", "chunks_total",
                                         "error", "created_at", "completed_at", "document_id")}
            for j in jobs.recent(tenant_id, limit)]


@app.get("/api/v1/formats")
def formats():
    return {"supported_extensions": sorted(SUPPORTED_EXTENSIONS)}


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------
@app.get("/api/v1/conversations", dependencies=[Depends(require_key)])
def list_conversations(tenant_id: str = Depends(tenant_param), limit: int = Query(50, ge=1, le=200)):
    with SessionLocal() as db:
        rows = db.execute(select(Conversation).where(Conversation.tenant_id == tenant_id)
                          .order_by(Conversation.updated_at.desc()).limit(limit)).scalars().all()
    return [{"id": c.id, "title": c.title, "updated_at": c.updated_at} for c in rows]


@app.post("/api/v1/conversations", dependencies=[Depends(require_key)])
def create_conversation(tenant_id: str = Depends(tenant_param)):
    conv = Conversation(id=str(uuid.uuid4()), tenant_id=tenant_id)
    with SessionLocal() as db:
        db.add(conv)
        db.commit()
    return {"id": conv.id, "title": conv.title}


@app.get("/api/v1/conversations/{conversation_id}", dependencies=[Depends(require_key)])
def get_conversation(conversation_id: str, tenant_id: str = Depends(tenant_param)):
    with SessionLocal() as db:
        conv = db.get(Conversation, conversation_id)
        if conv is None or conv.tenant_id != tenant_id:
            raise HTTPException(404, "conversation not found")
        msgs = db.execute(select(Message).where(Message.conversation_id == conversation_id).order_by(Message.id)).scalars().all()
    return {"id": conv.id, "title": conv.title,
            "messages": [{"role": m.role, "content": m.content, "payload": m.payload, "created_at": m.created_at} for m in msgs]}


class Rename(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


@app.patch("/api/v1/conversations/{conversation_id}", dependencies=[Depends(require_key)])
def rename_conversation(conversation_id: str, body: Rename, tenant_id: str = Depends(tenant_param)):
    with SessionLocal() as db:
        conv = db.get(Conversation, conversation_id)
        if conv is None or conv.tenant_id != tenant_id:
            raise HTTPException(404, "conversation not found")
        conv.title = body.title
        db.commit()
    return {"id": conversation_id, "title": body.title}


@app.delete("/api/v1/conversations/{conversation_id}", dependencies=[Depends(require_key)])
def delete_conversation(conversation_id: str, tenant_id: str = Depends(tenant_param)):
    with SessionLocal() as db:
        db.execute(delete(Conversation).where(Conversation.id == conversation_id, Conversation.tenant_id == tenant_id))
        db.commit()
    return {"deleted": conversation_id}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"), headers={"Cache-Control": "no-cache"})


@app.exception_handler(Exception)
async def unhandled(_request: Request, exc: Exception):
    log.exception("unhandled error")
    return JSONResponse({"detail": "Internal error. See server logs."}, status_code=500)
