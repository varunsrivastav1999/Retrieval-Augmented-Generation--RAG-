import hashlib
import re

import fitz # PyMuPDF
from sqlalchemy.exc import IntegrityError

from app.database import DocumentChunk, SessionLocal
from app.rag.jobs import complete_ingestion_job, fail_ingestion_job, update_ingestion_job
from app.rag.model_loader import encode_text, get_embedding_model_id


def hash_chunk(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def semantic_chunking(text: str) -> list:
    # Normalize accidental line breaks from PDF extraction
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    
    # Split by actual sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
            
        if len(current_chunk) + len(s) < 1200:
            current_chunk += s + " "
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = s + " "
            
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def ingest_pdf(pdf_path: str, tenant_id: str = "default", job_id: str = None) -> dict:
    """
    Parses PDF, chunks semantically, deduplicates using hash, embeds, and saves to pgvector.
    """
    db = SessionLocal()
    doc = None
    chunks_total = 0
    chunks_inserted = 0
    try:
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        embedding_model = get_embedding_model_id()
        section = 0

        for page_num, page in enumerate(doc, start=1):
            page_chunks = semantic_chunking(page.get_text())
            for chunk in page_chunks:
                chunks_total += 1
                chunk_hash = hash_chunk(chunk)

                existing = (
                    db.query(DocumentChunk)
                    .filter(
                        DocumentChunk.tenant_id == tenant_id,
                        DocumentChunk.doc_id == pdf_path,
                        DocumentChunk.chunk_hash == chunk_hash,
                        DocumentChunk.embedding_model == embedding_model,
                    )
                    .first()
                )
                if existing:
                    section += 1
                    continue

                vector = encode_text(chunk)

                doc_chunk = DocumentChunk(
                    tenant_id=tenant_id,
                    doc_id=pdf_path,
                    chunk_hash=chunk_hash,
                    text_content=chunk,
                    section=section,
                    doc_metadata={
                        "type": "pdf",
                        "page_count": page_count,
                        "page_num": page_num,
                        "embedding_model": embedding_model,
                    },
                    embedding_model=embedding_model,
                    embedding=vector
                )
                db.add(doc_chunk)
                try:
                    db.commit()
                    chunks_inserted += 1
                    print(f"[Ingest] Saved chunk {section} for {pdf_path} into pgvector")
                except IntegrityError:
                    db.rollback() # Handle race conditions

                section += 1
                if job_id and chunks_total % 10 == 0:
                    update_ingestion_job(
                        job_id,
                        chunks_total=chunks_total,
                        chunks_inserted=chunks_inserted,
                    )

        if job_id:
            complete_ingestion_job(job_id, chunks_total, chunks_inserted)
        return {"chunks_total": chunks_total, "chunks_inserted": chunks_inserted}
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        if job_id:
            fail_ingestion_job(job_id, str(e))
        raise
    finally:
        if doc is not None:
            doc.close()
        db.close()
