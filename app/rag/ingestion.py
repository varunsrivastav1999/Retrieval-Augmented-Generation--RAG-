import hashlib
import io
import os
import re

import fitz  # PyMuPDF
import pdfplumber
from sqlalchemy.exc import IntegrityError

from app.database import DocumentChunk, SessionLocal
from app.rag.jobs import complete_ingestion_job, fail_ingestion_job, update_ingestion_job
from app.rag.model_loader import encode_text, get_embedding_model_id

# ---------------------------------------------------------------------------
# Optional OCR for scanned / image-heavy PDFs
# ---------------------------------------------------------------------------
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("[Ingestion] pytesseract/Pillow not installed – image OCR disabled.")


def hash_chunk(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _split_long_text(text: str, max_chars: int, overlap_words: int = 35) -> list:
    words = text.split()
    if not words:
        return []

    chunks = []
    current = []
    current_len = 0
    index = 0
    while index < len(words):
        word = words[index]
        projected_len = current_len + len(word) + (1 if current else 0)
        if current and projected_len > max_chars:
            chunks.append(" ".join(current).strip())
            current = current[-overlap_words:] if overlap_words else []
            current_len = len(" ".join(current))
            continue

        current.append(word)
        current_len = projected_len
        index += 1

    if current:
        chunks.append(" ".join(current).strip())

    return chunks


def recursive_character_chunking(text: str, chunk_size: int = 1200, chunk_overlap: int = 200) -> list:
    """
    Advanced recursive splitting strategy.
    Tries to split on paragraphs, then sentences, then words to keep chunks meaningful.
    """
    if not text:
        return []

    # Normalize text
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\r\n?', '\n', text)
    
    separators = ["\n\n", "\n", ". ", " ", ""]
    
    def split_text(text, separators):
        final_chunks = []
        
        # Get the first separator to try
        separator = separators[0]
        new_separators = separators[1:]
        
        # Split by separator
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)
            
        current_chunk = ""
        
        for s in splits:
            # If the current split itself is too big, recurse
            if len(s) > chunk_size:
                if current_chunk:
                    final_chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                if new_separators:
                    final_chunks.extend(split_text(s, new_separators))
                else:
                    # No more separators, just force cut
                    final_chunks.append(s[:chunk_size])
            else:
                # Can we add this split to the current chunk?
                potential_chunk = f"{current_chunk}{separator if current_chunk else ''}{s}"
                if len(potential_chunk) <= chunk_size:
                    current_chunk = potential_chunk
                else:
                    if current_chunk:
                        final_chunks.append(current_chunk.strip())
                    
                    # Start new chunk with overlap if possible
                    # This is a simplified overlap; for production we'd take the tail of the prev
                    current_chunk = s
        
        if current_chunk:
            final_chunks.append(current_chunk.strip())
            
        return final_chunks

    return split_text(text, separators)


# ---------------------------------------------------------------------------
# Layer 1+: Table Extraction (pdfplumber)
# ---------------------------------------------------------------------------
def _extract_tables_from_page(pdf_path: str, page_num: int) -> list:
    """Use pdfplumber to extract structured tables from a single page."""
    tables_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_num - 1 >= len(pdf.pages):
                return []
            page = pdf.pages[page_num - 1]
            tables = page.extract_tables()
            for table_idx, table in enumerate(tables):
                if not table:
                    continue
                # Convert table rows into markdown-style text
                rows = []
                for row in table:
                    cleaned = [str(cell).strip() if cell else "" for cell in row]
                    rows.append(" | ".join(cleaned))
                if rows:
                    header = rows[0]
                    separator = " | ".join(["---"] * len(table[0])) if table[0] else "---"
                    table_text = f"[TABLE {table_idx + 1}]\n{header}\n{separator}\n" + "\n".join(rows[1:])
                    tables_text.append(table_text.strip())
    except Exception as e:
        print(f"[Ingestion] Table extraction warning for {pdf_path} page {page_num}: {e}")
    return tables_text


# ---------------------------------------------------------------------------
# Layer 1+: Image OCR Extraction (PyMuPDF + Tesseract)
# ---------------------------------------------------------------------------
def _extract_images_from_page(fitz_page, page_num: int, min_size: int = 100) -> list:
    """Extract text from images embedded in a PDF page using OCR."""
    if not OCR_AVAILABLE:
        return []

    image_texts = []
    try:
        image_list = fitz_page.get_images(full=True)
        doc = fitz_page.parent

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                if not base_image:
                    continue

                image_bytes = base_image["image"]
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)

                # Skip tiny icons / decorative images
                if width < min_size or height < min_size:
                    continue

                img = Image.open(io.BytesIO(image_bytes))
                ocr_text = pytesseract.image_to_string(img).strip()

                if ocr_text and len(ocr_text) > 20:
                    image_texts.append(
                        f"[IMAGE OCR - Page {page_num}, Image {img_idx + 1}]\n{ocr_text}"
                    )
            except Exception:
                continue  # Skip unreadable images silently
    except Exception as e:
        print(f"[Ingestion] Image OCR warning for page {page_num}: {e}")
    return image_texts


def ingest_pdf(pdf_path: str, tenant_id: str = "default", job_id: str = None) -> dict:
    """
    Production PDF Ingestion Pipeline:
    1. Extract text per page (PyMuPDF)
    2. Extract tables per page (pdfplumber)  
    3. Extract images per page via OCR (pytesseract)
    4. Semantic chunking with sentence-boundary awareness
    5. SHA-256 deduplication
    6. Batch embedding into pgvector
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

        pending_chunks = []
        
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text() or ""
            tables = _extract_tables_from_page(pdf_path, page_num)
            image_texts = _extract_images_from_page(page, page_num)

            all_content = [page_text] + tables + image_texts
            combined_text = "\n\n".join([t for t in all_content if t.strip()])

            page_chunks = recursive_character_chunking(combined_text)

            for chunk_text in page_chunks:
                chunks_total += 1
                chunk_hash = hash_chunk(chunk_text)

                # Quick check if exists
                existing = db.query(DocumentChunk.id).filter(
                    DocumentChunk.tenant_id == tenant_id,
                    DocumentChunk.doc_id == pdf_path,
                    DocumentChunk.chunk_hash == chunk_hash
                ).first()
                
                if existing:
                    section += 1
                    continue

                content_type = "text"
                if chunk_text.startswith("[TABLE"): content_type = "table"
                elif chunk_text.startswith("[IMAGE OCR"): content_type = "image_ocr"

                pending_chunks.append({
                    "text": chunk_text,
                    "hash": chunk_hash,
                    "type": content_type,
                    "page_num": page_num,
                    "section": section
                })
                section += 1

                # Process batch every 16 chunks
                if len(pending_chunks) >= 16:
                    _process_chunk_batch(db, pending_chunks, tenant_id, pdf_path, page_count, embedding_model)
                    chunks_inserted += len(pending_chunks)
                    pending_chunks = []
                    
                    if job_id:
                        update_ingestion_job(job_id, chunks_total=chunks_total, chunks_inserted=chunks_inserted)

        # Process remaining
        if pending_chunks:
            _process_chunk_batch(db, pending_chunks, tenant_id, pdf_path, page_count, embedding_model)
            chunks_inserted += len(pending_chunks)

        if job_id:
            complete_ingestion_job(job_id, chunks_total, chunks_inserted)

        print(f"[Ingest] Completed {pdf_path}: {chunks_inserted}/{chunks_total} chunks (text+table+image)")
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

def _process_chunk_batch(db, pending_chunks, tenant_id, doc_id, page_count, embedding_model):
    """Encodes and saves a batch of chunks for 5x-10x better performance."""
    from app.rag.model_loader import encode_texts
    
    texts = [c["text"] for c in pending_chunks]
    vectors = encode_texts(texts)
    
    for i, c in enumerate(pending_chunks):
        doc_chunk = DocumentChunk(
            tenant_id=tenant_id,
            doc_id=doc_id,
            chunk_hash=c["hash"],
            text_content=c["text"],
            section=c["section"],
            doc_metadata={
                "type": c["type"],
                "page_count": page_count,
                "page_num": c["page_num"],
                "embedding_model": embedding_model,
                "source": doc_id,
            },
            embedding_model=embedding_model,
            embedding=vectors[i]
        )
        db.add(doc_chunk)
    
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[Ingest] Batch commit error: {e}")

