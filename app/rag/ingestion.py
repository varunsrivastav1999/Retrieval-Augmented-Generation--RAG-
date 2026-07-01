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
from typing import Dict, List, Optional

from sqlalchemy.exc import IntegrityError

from app.database import DocumentChunk, SessionLocal
from app.rag.jobs import complete_ingestion_job, fail_ingestion_job, update_ingestion_job
from app.rag.model_loader import encode_texts, get_embedding_model_id, get_ollama_generate_url, OLLAMA_MODEL
from app.rag.parsers import get_file_type, parse_file
from app.rag.vision_extractor import looks_like_extractable_page, format_structured_data_for_embedding
try:
    from app.rag.table_engine import chunk_rich_table
    TABLE_ENGINE_AVAILABLE = True
except ImportError:
    TABLE_ENGINE_AVAILABLE = False
try:
    from app.rag.canonical_table_store import (
        rich_table_to_canonical_rows,
        upsert_canonical_rows,
    )
    CANONICAL_STORE_AVAILABLE = True
except ImportError:
    CANONICAL_STORE_AVAILABLE = False
try:
    from app.rag.doc_classifier import (
        classify_and_enrich_text_block,
        ContentType,
    )
    DOC_CLASSIFIER_AVAILABLE = True
except ImportError:
    DOC_CLASSIFIER_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BATCH_SIZE = 16  # Chunks per embedding batch. Reduced to strictly protect VRAM limits
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10MB per image for CLIP vision pipeline
PARENT_CHUNK_SIZE = 2400  # Parent chunks for broad retrieval
CHILD_CHUNK_SIZE = 600   # Child chunks for precise retrieval
CHUNK_OVERLAP = 150       # Overlap between chunks
TABLE_ROW_ADJACENT = 1   # ±N adjacent rows included in each table row chunk


def hash_chunk(text: str) -> str:
    """SHA-256 hash for deduplication."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


# ---------------------------------------------------------------------------
# Layer 3: Semantic Chunking (Layout-Aware)
# ---------------------------------------------------------------------------
def semantic_chunking(text: str, max_chunk_size: int = PARENT_CHUNK_SIZE, child_chunk_size: int = CHILD_CHUNK_SIZE) -> List[Dict]:
    """
    Semantic chunking strategy that respects document layout boundaries.
    Since Docling extracts text into semantic blocks (paragraphs/sections)
    separated by double newlines (\n\n), we split strictly on these natural
    boundaries instead of breaking context with arbitrary character counts.
    """
    if not text or not text.strip():
        return []

    # Normalize carriage returns but preserve semantic double newlines
    text = text.replace('\r\n', '\n')
    
    # Split natively by semantic layout blocks (paragraphs, sections)
    blocks = [b.strip() for b in text.split('\n\n') if len(b.strip()) > 10]
    
    result = []
    parent_idx = 0
    child_idx = 0

    def _append_parent_and_children(parent_text: str):
        nonlocal parent_idx, child_idx
        if not parent_text:
            return
        
        start = 0
        while start < len(parent_text):
            end = min(start + child_chunk_size, len(parent_text))
            child_text = parent_text[start:end].strip()
            if child_text:
                result.append({
                    "text": child_text,
                    "parent_text": parent_text,  # NeMo-style context
                    "is_parent": False,
                    "parent_idx": parent_idx,
                    "child_idx": child_idx,
                })
                child_idx += 1
            start = end - CHUNK_OVERLAP if end < len(parent_text) else end
        parent_idx += 1

    for block in blocks:
        # Only split a semantic block if it severely exceeds the embedding token limit
        if len(block) > max_chunk_size:
            sentences = [s.strip() + "." for s in block.split('. ') if s.strip()]
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                    _append_parent_and_children(current_chunk.strip())
                    current_chunk = sentence
                else:
                    current_chunk += " " + sentence if current_chunk else sentence
            if current_chunk:
                _append_parent_and_children(current_chunk.strip())
        else:
            _append_parent_and_children(block)

    return result


def chunk_table_per_row(table_md: str, table_group_id: str = "") -> List[Dict]:
    """
    Chunk a markdown table into INDIVIDUAL ROW chunks.
    Each chunk contains: [Table Title] + [Column Headers] + [Separator] + [Data Rows]
    
    Includes bi-directional empty cell inheritance to reconstruct spanning/merged cells
    that were lost during Docling's markdown export.
    
    Returns list of dicts with "text" and "table_group" keys for cross-chunk linking.
    """
    import re
    lines = table_md.strip().split('\n')
    if len(lines) < 3:
        return [{"text": table_md, "table_group": table_group_id}]

    # Robustly find the markdown table separator line (e.g. |---|---|)
    separator_idx = -1
    for i, line in enumerate(lines):
        if '|' in line and re.match(r'^[\s\|\-\:]+$', line):
            separator_idx = i
            break
            
    if separator_idx == -1:
        # Fallback if it's not a standard markdown table
        return [{"text": table_md, "table_group": table_group_id}]

    header_block = "\n".join(lines[:separator_idx + 1])
    raw_data_rows = lines[separator_idx + 1:]

    if not raw_data_rows:
        return [{"text": table_md, "table_group": table_group_id}]

    # --- FIX BROKEN MARKDOWN ROWS (Newlines inside cells) ---
    fixed_data_rows = []
    buffer = ""
    for row in raw_data_rows:
        stripped_row = row.strip()
        if not stripped_row:
            continue
        if buffer:
            buffer += " " + stripped_row
        else:
            buffer = stripped_row
            
        # A valid complete markdown row usually ends with '|'
        # If it doesn't, Docling probably leaked a newline into a cell.
        if buffer.endswith('|'):
            fixed_data_rows.append(buffer)
            buffer = ""
            
    if buffer:
        fixed_data_rows.append(buffer)
        
    # --- BI-DIRECTIONAL CELL INHERITANCE FOR MERGED CELLS ---
    # Parse rows into cells
    table_grid = []
    for row in fixed_data_rows:
        if not row.strip() or '|' not in row:
            table_grid.append({"is_data": False, "raw": row})
            continue
            
        row_content = row.strip()
        if row_content.startswith('|'): row_content = row_content[1:]
        if row_content.endswith('|'): row_content = row_content[:-1]
        
        cells = [c.strip() for c in row_content.split('|')]
        table_grid.append({"is_data": True, "cells": cells})
        
    # Forward Fill (Top to Bottom)
    for i in range(1, len(table_grid)):
        if not table_grid[i]["is_data"] or not table_grid[i-1]["is_data"]:
            continue
        prev_cells = table_grid[i-1]["cells"]
        curr_cells = table_grid[i]["cells"]
        for c_idx in range(len(curr_cells)):
            if not curr_cells[c_idx] and c_idx < len(prev_cells) and prev_cells[c_idx]:
                curr_cells[c_idx] = prev_cells[c_idx]
                
    # Backward Fill (Bottom to Top)
    for i in range(len(table_grid) - 2, -1, -1):
        if not table_grid[i]["is_data"] or not table_grid[i+1]["is_data"]:
            continue
        next_cells = table_grid[i+1]["cells"]
        curr_cells = table_grid[i]["cells"]
        for c_idx in range(len(curr_cells)):
            if not curr_cells[c_idx] and c_idx < len(next_cells) and next_cells[c_idx]:
                curr_cells[c_idx] = next_cells[c_idx]
                
    # Reconstruct data rows
    data_rows = []
    for row_obj in table_grid:
        if not row_obj["is_data"]:
            data_rows.append(row_obj["raw"])
        else:
            data_rows.append("| " + " | ".join(row_obj["cells"]) + " |")
    # --------------------------------------------------------

    row_chunks = []
    chunk_size = 5

    for i in range(0, len(data_rows), chunk_size):
        block = "\n".join(data_rows[i:i+chunk_size])
        row_chunk = f"{header_block}\n{block}"
        row_chunks.append({"text": row_chunk.strip(), "table_group": table_group_id})

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

    import re
    table_tokens = set()
    table_lines = set()
    for table_md in tables:
        tokens = re.findall(r'[A-Za-z0-9]{2,}[A-Za-z0-9-]*', table_md)
        table_tokens.update(t.lower() for t in tokens)
        for line in table_md.split('\n'):
            cleaned = line.strip()
            if cleaned and not cleaned.startswith('[') and not cleaned.startswith('|---'):
                table_lines.add(cleaned.lower())

    if not table_tokens:
        return raw_text

    cleaned_lines = []
    for line in raw_text.split('\n'):
        line_stripped = line.strip()
        if not line_stripped:
            cleaned_lines.append(line)
            continue

        # Check exact match against a table line
        if line_stripped.lower() in table_lines:
            continue

        # Fuzzy: if >40% of tokens appear in table tokens, it's duplicate
        line_tokens = re.findall(r'[A-Za-z0-9]{2,}[A-Za-z0-9-]*', line_stripped)
        if line_tokens:
            overlap = sum(1 for t in line_tokens if t.lower() in table_tokens)
            overlap_ratio = overlap / len(line_tokens)
            # Short lines (≤3 tokens) need higher confidence to avoid false positives
            threshold = 0.6 if len(line_tokens) <= 3 else 0.4
            if overlap_ratio > threshold:
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

        # ─── v6.0: Build Hierarchical Document Structure ─────────────────
        doc_tree = None
        section_lookup = {}  # page_num → best matching section node
        try:
            from app.rag.document_structure import build_document_tree, DocumentStructureBuilder
            from app.database import DocumentSection

            doc_tree = build_document_tree(
                parse_result.pages,
                document_name=doc_title,
                document_id=file_path,
            )
            builder = DocumentStructureBuilder()
            flat_sections = builder.flatten_sections(doc_tree)

            # Store DocumentSection records in PostgreSQL
            if flat_sections:
                # Clean old sections for this document
                db.query(DocumentSection).filter(
                    DocumentSection.document_id == file_path,
                    DocumentSection.tenant_id == tenant_id,
                ).delete(synchronize_session=False)
                db.commit()

                section_embeddings_batch = []  # For Qdrant upsert
                for sec_node in flat_sections:
                    section_summary = builder.get_section_summary(sec_node, max_chars=500)
                    doc_section = DocumentSection(
                        id=sec_node.node_id,
                        document_id=file_path,
                        tenant_id=tenant_id,
                        level=sec_node.level,
                        title=sec_node.title,
                        content_summary=section_summary[:500],
                        full_text=sec_node.content[:50000] if sec_node.content else "",
                        page_start=sec_node.page_start,
                        page_end=sec_node.page_end,
                        parent_id=sec_node.parent_id,
                        heading_path=sec_node.heading_path,
                        chunk_count=0,
                        embedding_model=embedding_model,
                    )
                    db.add(doc_section)

                    # Build page→section lookup for chunk enrichment
                    for pg in range(sec_node.page_start, sec_node.page_end + 1):
                        # Keep the most specific (deepest) section for each page
                        if pg not in section_lookup or _level_depth(sec_node.level) > _level_depth(section_lookup[pg].level):
                            section_lookup[pg] = sec_node

                    # Prepare section embedding
                    if section_summary.strip():
                        section_embeddings_batch.append({
                            "id": sec_node.node_id,
                            "text": section_summary,
                            "payload": {
                                "tenant_id": tenant_id,
                                "embedding_model": embedding_model,
                                "section_id": sec_node.node_id,
                                "document_id": file_path,
                                "document_name": doc_title,
                                "title": sec_node.title,
                                "level": sec_node.level,
                                "heading_path": sec_node.heading_path,
                                "page_start": sec_node.page_start,
                                "page_end": sec_node.page_end,
                                "parent_id": sec_node.parent_id,
                            },
                        })

                db.commit()

                # Generate and store section embeddings in Qdrant
                if section_embeddings_batch:
                    _upsert_section_embeddings(section_embeddings_batch)

                print(f"[Ingest/v6.0] Built hierarchy: {len(flat_sections)} sections, "
                      f"{len(section_lookup)} page mappings")

        except Exception as e:
            print(f"[Ingest/v6.0] Document structure building failed (non-fatal): {e}")
            import traceback
            traceback.print_exc()
        # ─── End v6.0 ────────────────────────────────────────────────────

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
                    "options": {
                        "temperature": 0.0,
                        "num_ctx": int(os.getenv("OLLAMA_CONTEXT_LENGTH", "32768"))
                    }
                }, timeout=30)
                if res.status_code == 200:
                    doc_context_summary = res.json().get("response", "").strip()
                    print(f"[Contextual Retrieval] Generated Context: {doc_context_summary}")
            except Exception as e:
                print(f"[Contextual Retrieval] Failed to generate context: {e}")

        # force_reindex: clean Qdrant + PG before re-ingesting
        if force_reindex:
            from app.rag.qdrant_client import delete_qdrant_points_by_source
            delete_qdrant_points_by_source(tenant_id, [file_path])
            deleted = db.query(DocumentChunk).filter(
                DocumentChunk.tenant_id == tenant_id,
                DocumentChunk.doc_id == file_path,
            ).delete(synchronize_session=False)
            if deleted:
                print(f"[Ingest] force_reindex: deleted {deleted} existing chunks for {file_path}")
            db.commit()

        for page_idx, page in enumerate(parse_result.pages):
            page_chunks = []
            
            # --- LLM Pre-processing Node (Enterprise Documents) ---
            if looks_like_extractable_page(page.text):
                if file_path.lower().endswith(".pdf"):
                    from app.rag.vision_extractor import extract_table_with_vision
                    print(f"[Ingest] Vision Pre-processing Node triggered for Page {page.page_num}...")
                    extracted_root_obj = extract_table_with_vision(file_path, page_idx)
                    
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
                page_chunks.extend(semantic_chunking(cleaned_text))
                
            # 2. Chunk tables — prefer RichTable objects (1 row per chunk with full context)
            #    Also upsert into canonical_table_rows for 0-token SQL exact lookup
            if TABLE_ENGINE_AVAILABLE and getattr(page, 'rich_tables', None):
                for rich_table in page.rich_tables:
                    rich_table.section_title = rich_table.section_title or getattr(page, 'section_title', '')

                    # a) Canonical store (JSONB rows for exact SQL lookup)
                    if CANONICAL_STORE_AVAILABLE:
                        try:
                            can_rows = rich_table_to_canonical_rows(
                                rich_table, tenant_id, file_path, os.path.basename(file_path)
                            )
                            upsert_canonical_rows(db, can_rows)
                        except Exception as e:
                            print(f"[Ingestion] Canonical store upsert failed: {e}")

                    # b) Vector chunks (1 row per chunk for embedding)
                    row_chunks = chunk_rich_table(rich_table, doc_title, max_adjacent_rows=TABLE_ROW_ADJACENT)
                    for rc in row_chunks:
                        page_chunks.append(rc)
            else:
                # Legacy path: markdown-based 1-row chunking
                for table_idx, table_md in enumerate(page.tables):
                    if table_md and table_md.strip():
                        unique_table_group = f"{os.path.basename(file_path)}_p{page.page_num}_t{table_idx}"
                        sec_title = getattr(page, 'section_title', '')
                        table_parts = chunk_table_per_row(table_md, unique_table_group)
                        for t_part in table_parts:
                            pc = {
                                "text": t_part["text"],
                                "is_parent": True,
                                "parent_idx": None,
                                "child_idx": None,
                                "content_type": "table",
                                "table_group": t_part.get("table_group"),
                                "section_title": sec_title,
                            }
                            page_chunks.append(pc)

                        
            # 3. Add Image OCR text as standalone chunks
            for img_txt in page.image_texts:
                if img_txt and img_txt.strip():
                    page_chunks.append({
                        "text": img_txt,
                        "is_parent": True,
                        "parent_idx": None,
                        "child_idx": None,
                        "content_type": "image_ocr",
                    })

            doc_title = os.path.basename(file_path)

            for chunk_info in page_chunks:
                raw_text = chunk_info["text"]
                if not raw_text.strip():
                    continue

                # Contextual Chunk Header: Prepend doc title and page
                header = f"[{doc_title}, Page {page.page_num}]"
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

                # Detect content type — use doc_classifier for rich format detection
                content_type = chunk_info.get("content_type", "text")
                detected_content = None
                if content_type in ("text", "paragraph") and DOC_CLASSIFIER_AVAILABLE:
                    try:
                        detected_content = classify_and_enrich_text_block(
                            raw_text,
                            page_num=page.page_num,
                            section_title=chunk_info.get("section_title", getattr(page, 'section_title', '')),
                        )
                        if detected_content.content_type != ContentType.PARAGRAPH:
                            content_type = detected_content.content_type
                            # Use the enriched chunk text (adds [MCQ], [FILL_BLANK], etc. tags)
                            chunk_text = f"{header}\n{detected_content.chunk_text}"
                    except Exception as e:
                        print(f"[Ingestion] Doc classifier failed: {e}")

                # Fallback legacy type detection
                if content_type == "text":
                    if raw_text.startswith("[TABLE"):
                        content_type = "table"
                    elif raw_text.startswith("[IMAGE OCR") or raw_text.startswith("[FULL PAGE OCR"):
                        content_type = "image_ocr"
                    elif raw_text.startswith("[VIDEO SUBTITLE") or raw_text.startswith("[SUBTITLE"):
                        content_type = "subtitle"
                    elif raw_text.startswith("[EXCEL SHEET"):
                        content_type = "table"
                    elif raw_text.startswith("[CSV DATA"):
                        content_type = "table"

                pending_chunks.append({
                    "text": chunk_text,
                    "nl_text": (detected_content.nl_sentence if detected_content else None) or chunk_info.get("nl_text", chunk_text),
                    "hash": chunk_hash,
                    "type": content_type,
                    "page_num": page.page_num,
                    "section": section,
                    "file_type": file_type,
                    "is_parent": chunk_info.get("is_parent", False),
                    "parent_idx": chunk_info.get("parent_idx"),
                    "child_idx": chunk_info.get("child_idx"),
                    "parent_text": chunk_info.get("parent_text"),
                    "entities": [], # GraphRAG extracted entities removed for VRAM
                    "table_group": chunk_info.get("table_group") or chunk_info.get("table_id"),
                    "table_id": chunk_info.get("table_id") or chunk_info.get("table_group"),
                    "section_title": chunk_info.get("section_title", getattr(page, 'section_title', '')),
                    "cell_values": chunk_info.get("json_cells"),
                    "header_path": chunk_info.get("header_path", []),
                    "row_index": chunk_info.get("row_index"),
                    "global_context": doc_context_summary,
                    # --- Doc format classifier fields (v5.1) ---
                    "structured_data": detected_content.structured_data if detected_content else chunk_info.get("structured_data"),
                    "search_tags": detected_content.search_tags if detected_content else chunk_info.get("search_tags", []),
                    "format_confidence": detected_content.confidence if detected_content else 1.0,
                    # --- v6.0 Section-aware fields ---
                    "section_id": None,
                    "heading_path": [],
                    "chunk_index_in_section": None,
                    "total_chunks_in_section": None,
                })

                # v6.0: Enrich chunk with section metadata from hierarchy
                if section_lookup and page.page_num in section_lookup:
                    sec_node = section_lookup[page.page_num]
                    pending_chunks[-1]["section_id"] = sec_node.node_id
                    pending_chunks[-1]["heading_path"] = sec_node.heading_path

                section += 1
                
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

            # --- Vision Pipeline (DISABLED — CLIP removed to save VRAM) ---
            # Image bytes are extracted by parsers.py and OCR text is already
            # embedded as text chunks above via image_texts.  When a vision
            # model is re-enabled, re-add CLIP encoding + Qdrant image_chunks
            # insertion here.


        # Process remaining chunks
        if pending_chunks:
            inserted = _process_chunk_batch(
                db, pending_chunks, tenant_id, file_path,
                page_count, embedding_model, file_type,
            )
            chunks_inserted += inserted

        # v6.0: Update section chunk counts
        if section_lookup:
            _update_section_chunk_counts(db, tenant_id, file_path, embedding_model)

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

    from app.rag.qdrant_client import insert_qdrant_points
    from qdrant_client.http import models

    for i, c in enumerate(pending_chunks):
        doc_metadata = {
            "type": c["type"],
            "page_count": page_count,
            "page_num": c["page_num"],
            "embedding_model": embedding_model,
            "source": doc_id,
            "file_type": file_type,
            "is_parent": c.get("is_parent", False),
            "parent_idx": c.get("parent_idx"),
            "child_idx": c.get("child_idx"),
            "parent_text": c.get("parent_text"),
            "entities": c.get("entities", []),
            "table_group": c.get("table_group"),
            # --- Table-aware fields ---
            "table_id": c.get("table_id") or c.get("table_group"),
            "section_title": c.get("section_title", ""),
            "cell_values": c.get("cell_values"),
            "header_path": c.get("header_path", []),
            "row_index": c.get("row_index"),
            # --- Document format classifier fields (v5.1) ---
            "content_type": c.get("type", "text"),   # mcq, fill_blank, form_field, specification, ...
            "structured_data": c.get("structured_data"),  # MCQ options, form fields, spec values
            "search_tags": c.get("search_tags", []),     # Boosted BM25 terms
            "format_confidence": c.get("format_confidence", 1.0),
            "nl_sentence": c.get("nl_text"),             # Embeddable NL sentence
            # --- v6.0 Section-aware fields ---
            "section_id": c.get("section_id"),
            "heading_path": c.get("heading_path", []),
        }
        
        chunk_obj = DocumentChunk(
            tenant_id=tenant_id,
            doc_id=doc_id,
            chunk_hash=c["hash"],
            text_content=c["text"],
            section=c["section"],
            doc_metadata=doc_metadata,
            embedding_model=embedding_model,
            file_type=file_type,
            # v6.0 section-aware columns
            section_id=c.get("section_id"),
            heading_path_json=c.get("heading_path"),
            chunk_index_in_section=c.get("chunk_index_in_section"),
            total_chunks_in_section=c.get("total_chunks_in_section"),
        )
        
        try:
            db.add(chunk_obj)
            db.flush()

            insert_qdrant_points(
                "document_chunks",
                [
                    models.PointStruct(
                        id=chunk_obj.id,
                        vector=vectors[i],
                        payload={
                            "tenant_id": tenant_id,
                            "doc_id": doc_id,
                            "file_type": file_type,
                            "text_content": c["text"],
                            "section": c["section"],
                            "table_group": c.get("table_group"),
                            # NEW table-aware payload fields (enables Qdrant metadata filters)
                            "table_id": c.get("table_id") or c.get("table_group"),
                            "section_title": c.get("section_title", ""),
                            "cell_values": c.get("cell_values"),
                            "header_path": c.get("header_path", []),
                            "row_index": c.get("row_index"),
                            "metadata": doc_metadata
                        }
                    )
                ]
            )
            db.commit()
            inserted += 1
        except IntegrityError:
            qdrant_point_id = str(chunk_obj.id)
            try:
                from app.rag.qdrant_client import get_qdrant_client
                get_qdrant_client().delete(
                    collection_name="document_chunks",
                    points_selector=models.PointIdsList(points=[qdrant_point_id]),
                )
            except Exception:
                pass
            db.rollback()
            print(f"[Ingest] Duplicate chunk skipped (hash exists)")
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


# ---------------------------------------------------------------------------
# v6.0: Section-aware helpers
# ---------------------------------------------------------------------------

def _level_depth(level: str) -> int:
    """Convert section level name to numeric depth for comparison."""
    return {"document": 0, "chapter": 1, "section": 2, "subsection": 3, "paragraph": 4}.get(level, 5)


def _upsert_section_embeddings(section_batch: List[Dict]) -> None:
    """Generate embeddings for sections and upsert into Qdrant section_embeddings collection."""
    try:
        from app.rag.qdrant_client import insert_qdrant_points
        from qdrant_client.http import models

        texts = [s["text"] for s in section_batch]
        vectors = encode_texts(texts)

        points = []
        for i, sec in enumerate(section_batch):
            points.append(
                models.PointStruct(
                    id=sec["id"],
                    vector=vectors[i],
                    payload=sec["payload"],
                )
            )

        # Batch upsert
        insert_qdrant_points("section_embeddings", points)
        print(f"[Ingest/v6.0] Upserted {len(points)} section embeddings into Qdrant")

    except Exception as e:
        print(f"[Ingest/v6.0] Section embedding upsert failed (non-fatal): {e}")


def _update_section_chunk_counts(
    db,
    tenant_id: str,
    document_id: str,
    embedding_model: str,
) -> None:
    """Update chunk_count on DocumentSection records after all chunks are ingested."""
    try:
        from app.database import DocumentSection
        from sqlalchemy import func

        # Count chunks per section_id
        counts = (
            db.query(
                DocumentChunk.section_id,
                func.count(DocumentChunk.id).label("cnt"),
            )
            .filter(
                DocumentChunk.tenant_id == tenant_id,
                DocumentChunk.doc_id == document_id,
                DocumentChunk.embedding_model == embedding_model,
                DocumentChunk.section_id.isnot(None),
            )
            .group_by(DocumentChunk.section_id)
            .all()
        )

        for section_id, cnt in counts:
            db.query(DocumentSection).filter(
                DocumentSection.id == section_id
            ).update({"chunk_count": cnt})

            # Also update total_chunks_in_section on each chunk
            db.query(DocumentChunk).filter(
                DocumentChunk.section_id == section_id,
                DocumentChunk.tenant_id == tenant_id,
                DocumentChunk.doc_id == document_id,
            ).update({"total_chunks_in_section": cnt})

        db.commit()
        if counts:
            print(f"[Ingest/v6.0] Updated chunk counts for {len(counts)} sections")

    except Exception as e:
        print(f"[Ingest/v6.0] Section chunk count update failed (non-fatal): {e}")

