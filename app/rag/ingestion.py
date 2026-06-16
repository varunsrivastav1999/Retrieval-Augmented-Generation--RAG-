"""
=============================================================================
 Enterprise Level RAG: Layers 1-4 — Universal File Ingestion Engine
=============================================================================
 Layer 1: Universal Document Parser (PDF, DOCX, XLSX, PPTX, CSV, TXT, IMG, VIDEO)
 Layer 2: Smart OCR & Table/Image Extraction
 Layer 3: Semantic Parent-Child Chunking
 Layer 4: Batch Embedding into pgvector (batch=32, parallel)

 Key Features:
 - Supports ALL file formats (no file limit)
 - Parent-child chunk hierarchy for broad + precise retrieval
 - Sentence-boundary aware splitting
 - SHA-256 deduplication
 - Batch embedding (32 at a time for speed)
 - 100% offline processing
=============================================================================
"""

import hashlib
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from sqlalchemy import func, cast, String
from sqlalchemy.exc import IntegrityError

from app.database import DocumentChunk, SessionLocal
from app.rag.jobs import complete_ingestion_job, fail_ingestion_job, update_ingestion_job
from app.rag.model_loader import encode_text, encode_texts, get_embedding_model_id, extract_entities, encode_image, get_ollama_generate_url, OLLAMA_MODEL, RAG_EMBEDDING_QUANTIZE
from app.rag.graph import graph_db
from app.rag.parsers import ParseResult, get_file_type, is_supported_file, parse_file
from app.rag.extraction import looks_like_extractable_page, extract_structured_data_from_page, format_structured_data_for_embedding
from PIL import Image
import io


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BATCH_SIZE = 32  # Chunks per embedding batch (up from 16)
PARENT_CHUNK_SIZE = 2400  # Parent chunks for broad retrieval
CHILD_CHUNK_SIZE = 600   # Child chunks for precise retrieval
CHUNK_OVERLAP = 150       # Overlap between chunks


def hash_chunk(text: str) -> str:
    """SHA-256 hash for deduplication."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


# ---------------------------------------------------------------------------
# Layer 3: Semantic Parent-Child Chunking
# ---------------------------------------------------------------------------
def recursive_character_chunking(
    text: str,
    chunk_size: int = CHILD_CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Advanced recursive splitting strategy.
    Tries to split on paragraphs → sentences → words to keep chunks meaningful.
    Sentence-boundary aware — never breaks mid-sentence.
    """
    if not text or not text.strip():
        return []

    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\r\n?', '\n', text)

    separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

    def split_text(text, separators):
        final_chunks = []
        separator = separators[0]
        new_separators = separators[1:]

        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        current_chunk = ""

        for s in splits:
            if len(s) > chunk_size:
                if current_chunk:
                    final_chunks.append(current_chunk.strip())
                    current_chunk = ""
                if new_separators:
                    final_chunks.extend(split_text(s, new_separators))
                else:
                    # Force cut — last resort
                    for i in range(0, len(s), chunk_size):
                        final_chunks.append(s[i:i + chunk_size])
            else:
                potential = f"{current_chunk}{separator if current_chunk else ''}{s}"
                if len(potential) <= chunk_size:
                    current_chunk = potential
                else:
                    if current_chunk:
                        final_chunks.append(current_chunk.strip())
                    # Overlap: keep tail of previous chunk
                    if chunk_overlap > 0 and current_chunk:
                        overlap_text = current_chunk[-chunk_overlap:]
                        current_chunk = overlap_text + separator + s
                    else:
                        current_chunk = s

        if current_chunk:
            final_chunks.append(current_chunk.strip())

        return final_chunks

    chunks = split_text(text, separators)
    return [c for c in chunks if c and len(c.strip()) > 10]


def create_parent_child_chunks(text: str) -> List[Dict]:
    """
    Create parent-child chunk hierarchy:
    - Parent chunks: 2400 chars (for broad context retrieval)
    - Child chunks: 600 chars (for precise answer extraction)
    
    Each child references its parent, enabling contextual window expansion.
    """
    if not text or not text.strip():
        return []

    # Create parent chunks
    parent_chunks = recursive_character_chunking(text, chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=200)

    result = []
    for parent_idx, parent_text in enumerate(parent_chunks):
        # Create child chunks from parent
        child_chunks = recursive_character_chunking(parent_text, chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

        if not child_chunks:
            # Parent is small enough to be a single chunk
            result.append({
                "text": parent_text,
                "is_parent": True,
                "parent_idx": parent_idx,
                "child_idx": None,
            })
        else:
            for child_idx, child_text in enumerate(child_chunks):
                result.append({
                    "text": child_text,
                    "is_parent": False,
                    "parent_idx": parent_idx,
                    "child_idx": child_idx,
                })

    return result


def chunk_table_per_row(table_md: str) -> List[str]:
    """
    Chunk a markdown table into INDIVIDUAL ROW chunks.
    Each chunk contains: [Table Title] + [Column Headers] + [Separator] + [1 Data Row]
    
    This is critical for lookup queries like "SNC2448L1125c door kit number".
    When a table has 20+ rows, naive splitting might put the target row in a
    different chunk from the headers, making the LLM unable to match columns.
    Per-row chunking guarantees every data row is paired with its column names.
    """
    lines = table_md.strip().split('\n')
    # Need at least: title, header, separator, 1 data row
    if len(lines) < 4:
        return [table_md]

    # Lines[0] = table title/info (e.g., "[TABLE 1 - Page 1]")
    # Lines[1] = column headers
    # Lines[2] = separator (---)
    header_block = "\n".join(lines[:3])
    data_rows = lines[3:]

    if not data_rows:
        return [table_md]

    # If the table is small (≤5 data rows), keep it as one chunk
    if len(data_rows) <= 5:
        return [table_md]

    # For large tables: create per-row chunks
    row_chunks = []
    for row in data_rows:
        row = row.strip()
        if not row:
            continue
        row_chunk = f"{header_block}\n{row}"
        row_chunks.append(row_chunk)

    return row_chunks


def _strip_table_text_from_raw(raw_text: str, tables: List[str]) -> str:
    """
    Remove table content from PyMuPDF raw text when pdfplumber already
    extracted the same data as structured markdown.
    
    PyMuPDF's get_text() garbles multi-column tables (columns get jumbled).
    If pdfplumber extracted the table cleanly, the raw text version is noise
    that dilutes search quality.
    """
    if not tables or not raw_text:
        return raw_text
    
    # Extract significant tokens from each table (model numbers, part numbers, etc.)
    import re
    table_tokens = set()
    for table_md in tables:
        # Extract tokens that look like product codes / model numbers
        tokens = re.findall(r'[A-Za-z0-9]{2,}[A-Za-z0-9-]*', table_md)
        table_tokens.update(t.lower() for t in tokens)
    
    if not table_tokens:
        return raw_text
    
    # Remove raw text lines that are mostly composed of table tokens
    cleaned_lines = []
    for line in raw_text.split('\n'):
        line_stripped = line.strip()
        if not line_stripped:
            cleaned_lines.append(line)
            continue
        
        line_tokens = re.findall(r'[A-Za-z0-9]{2,}[A-Za-z0-9-]*', line_stripped)
        if line_tokens:
            overlap = sum(1 for t in line_tokens if t.lower() in table_tokens)
            overlap_ratio = overlap / len(line_tokens)
            # If >60% of tokens in this line are from the table, it's a duplicate
            if overlap_ratio > 0.6 and len(line_tokens) >= 2:
                continue
        
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


# ---------------------------------------------------------------------------
# Universal File Ingestion
# ---------------------------------------------------------------------------
def ingest_file(
    file_path: str,
    tenant_id: str = "default",
    job_id: Optional[str] = None,
    force_reindex: bool = False,
) -> Dict:
    """
    Universal File Ingestion Pipeline:
    1. Parse any file format (Layer 1)
    2. Extract text, tables, images via OCR (Layer 2)
    3. Semantic parent-child chunking (Layer 3)
    4. Batch embedding into pgvector (Layer 4)
    
    Supports: PDF, DOCX, XLSX, PPTX, CSV, TXT, images, video subtitles
    No file size limit. 100% offline.
    """
    db = SessionLocal()
    chunks_total = 0
    chunks_inserted = 0

    try:
        # --- Layer 1: Universal Document Parser ---
        file_type = get_file_type(file_path)
        print(f"[Ingest] Starting {file_type.upper()} ingestion: {os.path.basename(file_path)}")

        parse_result = parse_file(file_path)

        if not parse_result.success:
            error_msg = f"Parse failed: {parse_result.error}"
            print(f"[Ingest] ❌ {error_msg}")
            if job_id:
                fail_ingestion_job(job_id, error_msg)
            return {"chunks_total": 0, "chunks_inserted": 0, "error": error_msg}

        if not parse_result.pages:
            msg = "No content extracted from file"
            if job_id:
                complete_ingestion_job(job_id, 0, 0)
            return {"chunks_total": 0, "chunks_inserted": 0, "message": msg}

        embedding_model = get_embedding_model_id()
        page_count = len(parse_result.pages)
        section = 0
        pending_chunks = []
        total_pages = len(parse_result.pages)
        doc_title = os.path.basename(file_path)

        # --- OPTIMIZED CONTEXTUAL RETRIEVAL ---
        # Generate a global document context from the first few pages to prepend to chunks
        # This solves the "lost in the middle" problem with O(1) LLM calls.
        intro_text = ""
        for p in parse_result.pages[:3]:
            intro_text += p.text + "\n"
        intro_text = intro_text[:3000].strip()
        
        doc_context_summary = ""
        if intro_text:
            prompt = f"Write a 1-sentence summary of what this document is about, focusing on main entities and topics. Document start:\n{intro_text}"
            try:
                import requests
                res = requests.post(get_ollama_generate_url(), json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0}
                }, timeout=15)
                if res.status_code == 200:
                    doc_context_summary = res.json().get("response", "").strip()
                    print(f"[Contextual Retrieval] Generated Context: {doc_context_summary}")
            except Exception as e:
                print(f"[Contextual Retrieval] Failed to generate context: {e}")

        for page_idx, page in enumerate(parse_result.pages):
            page_chunks = []
            
            # --- LLM Pre-processing Node (Enterprise Documents) ---
            if looks_like_extractable_page(page.text):
                print(f"[Ingest] LLM Pre-processing Node triggered for Page {page.page_num}...")
                extracted_root_obj = extract_structured_data_from_page(page.text)
                if extracted_root_obj:
                    formatted_chunks = format_structured_data_for_embedding(extracted_root_obj)
                    for chunk_str in formatted_chunks:
                        if chunk_str:
                            page_chunks.append({
                                "text": chunk_str,
                                "is_parent": True,
                                "parent_idx": None,
                                "child_idx": None,
                                "content_type": "llm_extracted_data"
                            })
                            
            # 1. Chunk normal text — but STRIP duplicate table content first
            #    PyMuPDF's get_text() garbles multi-column tables. If pdfplumber
            #    already extracted them cleanly, the raw text version is noise.
            cleaned_text = page.text
            if page.tables:
                cleaned_text = _strip_table_text_from_raw(page.text, page.tables)
            if cleaned_text and cleaned_text.strip():
                page_chunks.extend(create_parent_child_chunks(cleaned_text))
                
            # 2. Chunk tables into PER-ROW entries (each row gets column headers!)
            #    This is the KEY fix for lookup queries like "SNC2448L1125c door kit"
            for table_md in page.tables:
                if table_md and table_md.strip():
                    table_parts = chunk_table_per_row(table_md)
                    for t_part in table_parts:
                        page_chunks.append({
                            "text": t_part,
                            "is_parent": True,
                            "parent_idx": None,
                            "child_idx": None
                        })
                        
            # 3. Add Image OCR text as standalone chunks
            for img_txt in page.image_texts:
                if img_txt and img_txt.strip():
                    page_chunks.append({
                        "text": img_txt,
                        "is_parent": True,
                        "parent_idx": None,
                        "child_idx": None
                    })

            doc_title = os.path.basename(file_path)

            for chunk_info in page_chunks:
                raw_text = chunk_info["text"]
                if not raw_text.strip():
                    continue

                # Contextual Chunk Header: Prepend doc title, page, and Global Context
                header = f"[{doc_title} | Page {page.page_num}]"
                if doc_context_summary:
                    header += f"\n[GLOBAL CONTEXT: {doc_context_summary}]"
                chunk_text = f"{header}\n{raw_text}"

                chunks_total += 1
                chunk_hash = hash_chunk(chunk_text)

                # Deduplication check
                if not force_reindex:
                    existing = db.query(DocumentChunk.id).filter(
                        DocumentChunk.tenant_id == tenant_id,
                        DocumentChunk.doc_id == file_path,
                        DocumentChunk.chunk_hash == chunk_hash,
                        DocumentChunk.embedding_model == embedding_model,
                    ).first()

                    if existing:
                        section += 1
                        continue

                # Detect content type
                content_type = chunk_info.get("content_type", "text")
                if content_type == "text":
                    if chunk_text.startswith("[TABLE"):
                        content_type = "table"
                    elif chunk_text.startswith("[IMAGE OCR") or chunk_text.startswith("[FULL PAGE OCR"):
                        content_type = "image_ocr"
                    elif chunk_text.startswith("[VIDEO SUBTITLE") or chunk_text.startswith("[SUBTITLE"):
                        content_type = "subtitle"
                    elif chunk_text.startswith("[EXCEL SHEET"):
                        content_type = "table"
                    elif chunk_text.startswith("[CSV DATA"):
                        content_type = "table"

                pending_chunks.append({
                    "text": chunk_text,
                    "hash": chunk_hash,
                    "type": content_type,
                    "page_num": page.page_num,
                    "section": section,
                    "file_type": file_type,
                    "is_parent": chunk_info.get("is_parent", False),
                    "parent_idx": chunk_info.get("parent_idx"),
                    "entities": extract_entities(chunk_text),
                })
                section += 1
                
                # Graph DB: populate offline graph
                graph_db.populate_from_chunk(chunk_hash, chunk_text, tenant_id)

                # --- Layer 4: Batch Embedding (batch=32) ---
                if len(pending_chunks) >= BATCH_SIZE:
                    inserted = _process_chunk_batch(
                        db, pending_chunks, tenant_id, file_path,
                        page_count, embedding_model, file_type,
                    )
                    chunks_inserted += inserted
                    pending_chunks = []

                    if job_id:
                        progress = ((page_idx + 1) / total_pages) * 100
                        update_ingestion_job(
                            job_id,
                            chunks_total=chunks_total,
                            chunks_inserted=chunks_inserted,
                            progress_pct=round(progress, 1),
                        )

            # Process any raw image bytes for Vision embeddings
            if hasattr(page, "image_bytes") and page.image_bytes:
                for img_bytes in page.image_bytes:
                    try:
                        img = Image.open(io.BytesIO(img_bytes))
                        vector = encode_image(img)
                        if vector:
                            img_hash = hash_chunk(str(img_bytes[:100]))
                            img_chunk = DocumentChunk(
                                tenant_id=tenant_id,
                                doc_id=file_path,
                                chunk_hash=img_hash,
                                file_type=file_type,
                                embedding_model=embedding_model,
                                image_embedding=vector,
                                doc_metadata={
                                    "page_num": page.page_num, 
                                    "type": "image",
                                    "source": file_path,
                                    "embedding_model": embedding_model,
                                }
                            )
                            db.add(img_chunk)
                            db.commit()
                            chunks_inserted += 1
                            chunks_total += 1
                    except IntegrityError:
                        db.rollback()
                    except Exception as e:
                        db.rollback()
                        print(f"[Ingest] Vision embedding failed: {e}")

        # Process remaining chunks
        if pending_chunks:
            inserted = _process_chunk_batch(
                db, pending_chunks, tenant_id, file_path,
                page_count, embedding_model, file_type,
            )
            chunks_inserted += inserted
            
        # NOTE: RAPTOR tree build is NOT run per-file (too expensive).
        # Trigger via POST /raptor/build endpoint or on a schedule instead.

        if job_id:
            complete_ingestion_job(job_id, chunks_total, chunks_inserted)

        print(
            f"[Ingest] ✅ Completed {os.path.basename(file_path)}: "
            f"{chunks_inserted}/{chunks_total} chunks "
            f"({file_type}, {page_count} pages)"
        )
        return {"chunks_total": chunks_total, "chunks_inserted": chunks_inserted}

    except Exception as e:
        error_msg = f"Error processing {os.path.basename(file_path)}: {e}"
        print(f"[Ingest] ❌ {error_msg}")
        if job_id:
            fail_ingestion_job(job_id, str(e))
        raise
    finally:
        db.close()


def _process_chunk_batch(
    db,
    pending_chunks: List[Dict],
    tenant_id: str,
    doc_id: str,
    page_count: int,
    embedding_model: str,
    file_type: str,
) -> int:
    """Encode and save a batch of chunks. Returns count of inserted chunks."""
    texts = [c["text"] for c in pending_chunks]
    vectors = encode_texts(texts)
    inserted = 0

    quantized_batch = []
    if RAG_EMBEDDING_QUANTIZE == "int8":
        from app.rag.quantization import quantize_vector_batch
        q_bytes, q_scale, q_zp = quantize_vector_batch(vectors)
        import json, base64
        for qb in q_bytes:
            b64 = base64.b64encode(qb).decode("ascii")
            quantized_batch.append(json.dumps({"data": b64, "scale": q_scale, "zp": q_zp}))

    for i, c in enumerate(pending_chunks):
        kw = dict(
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
                "file_type": file_type,
                "is_parent": c.get("is_parent", False),
                "entities": c.get("entities", []),
            },
            embedding_model=embedding_model,
            embedding=vectors[i],
            file_type=file_type,
        )
        if quantized_batch:
            kw["quantized_embedding"] = quantized_batch[i]
        db.add(DocumentChunk(**kw))

    try:
        db.commit()
        inserted = len(pending_chunks)
    except Exception:
        db.rollback()
        for i, c in enumerate(pending_chunks):
            try:
                single_kw = dict(
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
                        "file_type": file_type,
                        "is_parent": c.get("is_parent", False),
                        "entities": c.get("entities", []),
                    },
                    embedding_model=embedding_model,
                    embedding=vectors[i],
                    file_type=file_type,
                )
                if quantized_batch:
                    single_kw["quantized_embedding"] = quantized_batch[i]
                db.add(DocumentChunk(**single_kw))
                db.commit()
                inserted += 1
            except IntegrityError:
                db.rollback()
            except Exception as e:
                db.rollback()
                print(f"[Ingest] Single chunk error: {e}")

    return inserted

# ---------------------------------------------------------------------------
# Legacy compatibility — redirect PDF ingestion to universal pipeline
# ---------------------------------------------------------------------------
def ingest_pdf(
    pdf_path: str,
    tenant_id: str = "default",
    job_id: Optional[str] = None,
    force_reindex: bool = False,
) -> Dict:
    """Legacy function — redirects to universal ingest_file()."""
    return ingest_file(pdf_path, tenant_id=tenant_id, job_id=job_id, force_reindex=force_reindex)
