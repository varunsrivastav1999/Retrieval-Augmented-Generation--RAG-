"""Document indexing: parse -> chunk -> embed (dense + BM25) -> Postgres + Qdrant.

* Change detection per file (size/mtime, then SHA-256): unchanged files are
  skipped without touching the chunk tables.
* Streaming batches keep memory flat for very large files (1M-row CSV exports).
* Chunk ids come from one Postgres sequence and double as Qdrant point ids.
* A failed file leaves no partial index behind.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
import uuid
from itertools import islice
from typing import Callable, Iterable, Iterator, List, Optional

from sqlalchemy import delete, insert, select, text as sql

from app.config import get_settings
from app.db import Chunk, Document, SessionLocal, utcnow
from app.rag import cache, models, vector_store
from app.rag.chunking import ChunkDraft, chunk_pages, record_chunk
from app.rag.parsers import extract_archive, get_file_type, is_record_file, iter_records, parse_file, extension, ARCHIVE_EXT

log = logging.getLogger(__name__)
ProgressFn = Callable[[int, float], None]


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def derive_collection(path: str, tenant_id: str) -> str:
    """First folder below the media root (or below uploads/<tenant>/) names the collection."""
    root = os.path.realpath(get_settings().media_path)
    rel = os.path.relpath(os.path.realpath(path), root).split(os.sep)
    if rel[:2] == ["uploads", tenant_id]:
        rel = rel[2:]
    return rel[0][:128] if len(rel) > 1 else "general"


def get_or_create_document(db, tenant_id: str, path: str) -> Document:
    doc = db.execute(select(Document).where(Document.tenant_id == tenant_id, Document.path == path)).scalar_one_or_none()
    if doc is None:
        doc = Document(id=str(uuid.uuid4()), tenant_id=tenant_id, path=path, name=os.path.basename(path),
                       collection=derive_collection(path, tenant_id), file_type=get_file_type(path), status="pending")
        db.add(doc)
        db.flush()
    return doc


def remove_document(document_id: str) -> None:
    vector_store.delete_document(document_id)
    with SessionLocal() as db:
        db.execute(delete(Chunk).where(Chunk.document_id == document_id))
        db.execute(delete(Document).where(Document.id == document_id))
        db.commit()


def _drafts(path: str, doc_name: str, section_prefix: str, stats: dict) -> Iterator[ChunkDraft]:
    s = get_settings()
    if is_record_file(path):
        for sheet, header, row, n in iter_records(path):
            stats["rows"] = stats.get("rows", 0) + 1
            draft = record_chunk(doc_name, sheet, header, row, n)
            if draft:
                yield draft
        return
    result = parse_file(path)
    if not result.success:
        raise ValueError(result.error or "parse failed")
    stats["pages"] = len(result.pages)
    stats["parser"] = result.metadata.get("parser")
    yield from chunk_pages(doc_name, result.pages, section_prefix, s.chunk_chars)


def _batched(it: Iterable, n: int) -> Iterator[List]:
    it = iter(it)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


def _write_batch(document: Document, batch: List[ChunkDraft], start_ord: int) -> None:
    texts = [d.embed_text for d in batch]
    dense = models.embed_documents(texts)
    sparse = models.sparse_documents(texts)
    with SessionLocal() as db:
        ids = [r[0] for r in db.execute(sql("SELECT nextval('chunks_id_seq') FROM generate_series(1, :n)"), {"n": len(batch)})]
        db.execute(insert(Chunk), [{
            "id": cid, "document_id": document.id, "tenant_id": document.tenant_id, "ord": start_ord + i,
            "page": d.page, "section_id": d.section_id, "section_title": d.section_title,
            "content_type": d.content_type, "text": d.text, "identifiers": d.identifiers,
        } for i, (cid, d) in enumerate(zip(ids, batch))])
        vector_store.upsert([vector_store.PointIn(id=cid, dense=dv, sparse=sv, payload={
            "tenant_id": document.tenant_id, "document_id": document.id, "collection": document.collection,
            "file_type": document.file_type, "content_type": d.content_type, "ids": d.identifiers,
        }) for cid, dv, sv, d in zip(ids, dense, sparse, batch)], wait=True)
        db.commit()


def index_file(path: str, tenant_id: str = "default", *, force: bool = False,
               progress: Optional[ProgressFn] = None) -> dict:
    """Index (or re-index) one file. Returns a summary dict; raises on failure."""
    s = get_settings()
    started = time.perf_counter()
    path = os.path.realpath(path)
    st = os.stat(path)

    with SessionLocal() as db:
        doc = get_or_create_document(db, tenant_id, path)
        unchanged_meta = doc.size_bytes == st.st_size and doc.mtime == st.st_mtime
        if not force and doc.status == "indexed" and unchanged_meta:
            db.commit()
            return {"document_id": doc.id, "status": "skipped", "reason": "unchanged"}
        sha = file_sha256(path)
        # An interrupted run of the same content (worker/database restart) resumes
        # after its last committed batch instead of starting over.
        resume = doc.status == "indexing" and doc.sha256 == sha
        if not force and doc.status == "indexed" and doc.sha256 == sha:
            doc.mtime, doc.size_bytes = st.st_mtime, st.st_size
            db.commit()
            return {"document_id": doc.id, "status": "skipped", "reason": "same content"}
        doc.status, doc.error, doc.sha256 = "indexing", None, sha
        doc.size_bytes, doc.mtime, doc.updated_at = st.st_size, st.st_mtime, utcnow()
        doc.collection = derive_collection(path, tenant_id)
        doc.file_type = get_file_type(path)
        db.commit()
        document = doc

    done = 0
    if resume:
        with SessionLocal() as db:
            done = db.execute(sql("SELECT coalesce(max(ord) + 1, 0) FROM chunks WHERE document_id=:id"),
                              {"id": document.id}).scalar()
        if done:
            log.info("resuming interrupted indexing", extra={"file": document.name, "from_chunk": done})
    if not done:
        # Drop the previous version of this file from both stores.
        vector_store.delete_document(document.id)
        with SessionLocal() as db:
            db.execute(delete(Chunk).where(Chunk.document_id == document.id))
            db.commit()

    if extension(path) in ARCHIVE_EXT:
        stem = os.path.basename(path).split(".")[0]
        files = extract_archive(path, os.path.join(os.path.dirname(path), f"_extracted_{stem}"))
        _finish(document.id, chunks=0, pages=len(files))
        return {"document_id": document.id, "status": "indexed", "extracted_files": files}

    stats: dict = {}
    total = done
    try:
        section_prefix = document.id[:8]
        drafts = islice(_drafts(path, document.name, section_prefix, stats), done, None)
        for batch in _batched(drafts, s.ingest_batch_size):
            _write_batch(document, batch, total)
            total += len(batch)
            if progress:
                progress(total, stats.get("rows", 0))
    except Exception as exc:
        log.exception("indexing failed: %s", path)
        permanent = isinstance(exc, (ValueError, FileNotFoundError))  # unparseable file
        try:
            if permanent:  # leave no partial index behind
                vector_store.delete_document(document.id)
            with SessionLocal() as db:
                if permanent:
                    db.execute(delete(Chunk).where(Chunk.document_id == document.id))
                # Transient failures keep committed batches; status stays "indexing" so the retry resumes.
                db.execute(sql("UPDATE documents SET status=:st, error=:e, updated_at=now() WHERE id=:id"),
                           {"st": "failed" if permanent else "indexing", "e": f"{type(exc).__name__}: {exc}"[:2000],
                            "id": document.id})
                db.commit()
        except Exception:
            log.exception("could not record the failure (database unavailable?)")
        raise

    _finish(document.id, chunks=total, pages=stats.get("pages") or stats.get("rows") or 0)
    cache.bump_index_version(tenant_id)
    seconds = time.perf_counter() - started
    log.info("indexed", extra={"file": document.name, "chunks": total, "seconds": round(seconds, 2),
                               "chunks_per_s": round(total / seconds, 1) if seconds else None, "parser": stats.get("parser")})
    return {"document_id": document.id, "status": "indexed", "chunks": total, "seconds": round(seconds, 2)}


def _finish(document_id: str, chunks: int, pages: int) -> None:
    with SessionLocal() as db:
        db.execute(sql("""UPDATE documents SET status='indexed', error=NULL, chunk_count=:c, pages=:p,
                          indexed_at=now(), updated_at=now() WHERE id=:id"""), {"c": chunks, "p": pages, "id": document_id})
        db.commit()
