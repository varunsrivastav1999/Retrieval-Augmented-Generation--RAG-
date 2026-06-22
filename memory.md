# Enterprise Level RAG: 18-Layer Architecture — System Memory

> This file documents the complete architecture, APIs, configuration, and operational knowledge
> of the Enterprise Level RAG microservice. Updated: 2026-06-22 (v5.2 — 10GB VRAM Optimized Architecture).

---

## 🏗️ Architecture Overview

### 18-Layer Intelligence Pipeline (v5.0)

```
┌─────────────────────────────────────────────────────────────────────────┐
│      Enterprise Level RAG Engine v5.2 (10GB VRAM Optimized)             │
│                                                                         │
│  ANY FILE ──► Layer 0:  Table Reconstruction Engine (pdfplumber)        │
│              Layer 1:  Universal Document Parser                        │
│              Layer 2:  CPU OCR (Tesseract via Docling)                  │
│              Layer 3:  Table-Aware 1-Row-Per-Chunk Ingestion            │
│              Layer 4:  Text Embedding (BAAI/bge-small-en-v1.5)          │
│              Layer 5:  RAPTOR Hierarchical Summarization                │
│  QUERY   ──► Layer 6:  Hybrid Search + Exact Lookups                    │
│                         ① Exact SQL ILIKE (catalogue numbers)           │
│                         ② Dense Vector (Qdrant HNSW)                   │
│                         ③ BM25 Postgres (table-noise cleaned)           │
│              Layer 7:  Base Reranking (BAAI/bge-reranker-base)          │
│              Layer 8:  MMR + Table Group Expansion                      │
│              Layer 9:  Table-Aware Context Assembly (HTML tables)       │
│              Layer 10: 4-Tier Semantic Router                           │
│              Layer 11: Query Intelligence (Spelling, Expand)            │
│              Layer 12: 🛡️ Hallucination Guard                            │
│              Layer 13: Extractive Fast-Path (< 15ms)                    │
│              Layer 14: Semantic Query Cache (Redis)                     │
│              Layer 15: Active RAG (FLARE self-reflection)               │
│              Layer 16: Real-Time Token Streaming                        │
│                                                                         │
└────┬──────────┬──────────┬──────────┬───────────────────────────────────┘
     │          │          │          │
┌────▼───┐ ┌───▼────┐ ┌──▼──────┐ ┌─▼─────────┐ ┌──────────┐
│Postgres│ │ Redis  │ │ Ollama  │ │ File Store │ │  Qdrant  │
│pgvector│ │ Cache  │ │ LLM     │ │ /media     │ │ VectorDB │
└────────┘ └────────┘ └─────────┘ └────────────┘ └──────────┘
```

---

## 🆕 v5.0 Changes — Table Reconstruction Engine

### What Was Broken (Before v5.0)
| Problem | Location | Impact |
|---------|----------|--------|
| Docling exports table as flat markdown — rowspan/colspan silently discarded | `parsers.py:323` | Merged cells become empty strings |
| Multi-level headers collapse to one row | `parsers.py:322-327` | "EQL > Single Phase > W" becomes just "W" |
| Tables on page N and page N+1 treated as unrelated | `parsers.py:308-350` | Continuation rows have no headers |
| 5-row chunking: a specific row may not be in the retrieved window | `ingestion.py:218` | Wrong row returned for catalogue queries |
| Section title NOT stored with table | `ingestion.py:146` | Can't group "EQL Single Phase" tables |
| BM25 tokenizes `|` and `---` as tokens | `retrieval.py:71` | Spurious table-markdown matches |
| No exact lookup for catalogue numbers (ECL2412SD) | `retrieval.py` | Vector search not designed for exact string match |

### What Was Fixed (v5.0)
| Fix | File | Description |
|-----|------|-------------|
| `table_engine.py` created | `app/rag/table_engine.py` | Full table reconstruction module |
| Section header tracking | `parsers.py` | `last_section_header` flows into every `[TABLE]` label |
| Multi-page stitching | `parsers.py` | `stitch_continuation_tables()` merges continuation tables |
| 1-row-per-chunk | `ingestion.py` | Each row = own chunk with resolved headers |
| NL row serialization | `ingestion.py` | Rows embedded as natural-language sentences |
| Table metadata stored | `ingestion.py` + `database.py` | `table_id`, `section_title`, `cell_values`, `header_path`, `row_index` |
| BM25 noise stripping | `retrieval.py` | `re.sub(r'[\|\-]{2,}', ' ', query)` |
| Exact catalogue lookup | `retrieval.py` | `exact_catalogue_lookup()` → SQL `ILIKE %pattern%` |
| HTML table context | `context.py` | `build_table_html_context()` → structured HTML for LLM |

---

## 📁 Project Structure (v5.0)

```
enterprise-level-rag/
├── app/
│   ├── main.py                         # FastAPI app, APIs, Active RAG
│   ├── database.py                      # SQLAlchemy models, schema migrations (v5.0+v5.1)
│   └── rag/
│       ├── table_engine.py              # 🆕 Layer 0: Table Reconstruction Engine (v5.0)
│       ├── query_router.py              # 🆕 4-tier Query Router: exact/comparison/aggregation/narrative
│       ├── canonical_table_store.py     # 🆕 Canonical Table Store: JSONB rows, 0-token SQL lookup
│       ├── doc_classifier.py            # 🆕 Document Format Classifier: MCQ/fill-blank/form/spec
│       ├── parsers.py                   # Layer 1-2: Universal parser + section tracking + MCQ enrich
│       ├── ingestion.py                 # Layers 3-4: 1-row chunking + canonical store upsert + format classifier
│       ├── raptor.py                    # Layer 5: RAPTOR hierarchical summarization
│       ├── retrieval.py                 # Layer 6: 5-signal hybrid search + exact lookup
│       ├── reranker.py                  # Layer 7: Base reranking
│       ├── context.py                   # Layers 8-9: MMR + HTML table context assembly
│       ├── query_router.py              # Layer 10 NEW: 4-tier semantic routing with RouteResult
│       ├── query_intelligence.py        # Layer 11: Spelling, expansion, decomposition
│       ├── grounding.py                 # Layer 12: Hallucination guard
│       ├── model_loader.py              # Model management, device detection
│       ├── qdrant_client.py             # Qdrant vector DB interface
│       ├── extraction.py                # Structured data extraction helpers
│       └── jobs.py                      # Background worker, auto-scanner
├── Dockerfile
├── local.yml                            # Docker Compose (local)
├── production.yml                       # Docker Compose (production)
├── requirements.txt                     # Python dependencies (v5.2: 10GB VRAM optimized)
├── memory.md                            # This file
└── README.md
```

---

## 📁 Universal File Support

| Category | Extensions | Parser | Notes |
|----------|-----------|--------|-------|
| **Documents** | `.pdf` | PyMuPDF + pdfplumber | Text, tables (table_engine.py v5.0), images (OCR) |
| **Word/PPT** | `.docx`, `.doc`, `.pptx` | Docling | Headings tracked → section_title on tables |
| **Excel** | `.xlsx`, `.xls`, `.csv` | openpyxl / csv | All sheets → RichTable objects |
| **Code** | `.py`, `.js`, `.java`, etc. | Universal | Strict indentation preservation |
| **Email** | `.eml`, `.msg` | `email` module | Subject, From, To, Date, body |
| **Web Links** | `.url`, `.webloc` | urllib + bs4 | Offline scraper |
| **Images** | `.png`, `.jpg`, etc. | Pillow + pytesseract | Auto-rotate + full OCR |
| **Video** | `.mp4`, `.avi`, etc. | ffmpeg | Subtitle extraction |
| **Archives** | `.zip`, `.tar.gz`, `.rar` | zipfile | Auto-extract → recursive ingest |
| **Text** | `.txt`, `.md`, `.json`, etc. | Python stdlib | Auto-encoding via chardet |

---

## 🔌 API Reference

### Ingestion APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/upload` | Upload any file, queue background ingestion |
| `POST` | `/api/v1/ingest` | Scan `/media` volume and queue all files |
| `GET` | `/api/v1/ingest/jobs` | List ingestion jobs (with progress %) |
| `GET` | `/api/v1/ingest/jobs/{id}` | Get specific job status |
| `GET` | `/api/v1/formats` | List supported file formats |

### Query API

```json
POST /api/v1/query
{
  "query": "What is the catalogue number for a 24-circuit EQL loadcentre?",
  "tenant_id": "default",
  "top_k": 12,
  "stream": true,
  "auto": true
}
```

**Response includes:**
- `answer` — Strictly grounded in document content
- `sources` — Exact document citations `[filename.pdf, Page X]`
- `grounding` — Pre-generation grounding score
- `verification` — Post-generation confidence (high/medium/low)
- `latency_ms` — Response time

### Health APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health/live` | Liveness probe |
| `GET` | `/health/ready` | Readiness + stats (chunks, docs, file types, active jobs) |

---

## ⚙️ Configuration (Environment Variables)

### Core
| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://rag_user:rag_password@postgres:5432/rag_db` | PostgreSQL |
| `REDIS_URL` | `redis://redis:6379/0` | Redis for semantic caching |
| `OLLAMA_URL` | `http://ollama:11434/api/generate` | Ollama LLM endpoint |
| `OLLAMA_MODEL` | `llama3.1:8b` | LLM model name |
| `MEDIA_PATH` | `/media` | Shared volume for auto-scan |

### Models
| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model (384d) |
| `RAG_RERANKER_MODEL` | `BAAI/bge-reranker-base` | Base reranker (1.1GB VRAM) |
| `RAG_EMBEDDING_DIM` | `1024` | Vector dimension |
| `RAG_MODEL_DEVICE` | auto-detect | Force: `mps`, `cuda`, `cpu` |

### Performance
| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_DEFAULT_TOP_K` | `12` | Default retrieval count |
| `RAG_MAX_TOP_K` | `50` | Maximum retrieval count |
| `OLLAMA_NUM_PREDICT` | `1024` | Max tokens per response |
| `OLLAMA_CONTEXT_LENGTH` | `32768` | Context window (Llama 3.1 full utilization) |

### Table Engine (v5.0)
| Variable | Default | Description |
|----------|---------|-------------|
| `TABLE_ROW_ADJACENT` | `1` | ±N adjacent rows in each row chunk |
| `RAG_ENABLE_CONTEXT_EXPANSION` | `true` | Fetch neighboring chunks |
| `GROUNDING_THRESHOLD` | `0.35` | Pre-generation guard (0.35→0.25→0.15 FLARE) |

---

## 📊 Database Schema (v5.0)

### `document_chunks`
| Column | Type | Description |
|--------|------|-------------|
| `id` | integer (PK) | Auto-increment |
| `tenant_id` | string | Multi-tenant isolation |
| `doc_id` | string | Source file path |
| `chunk_hash` | string | SHA-256 for deduplication |
| `text_content` | text | Chunk text / table row markdown |
| `section` | integer | Chunk ordering within document |
| `doc_metadata` | JSON | Includes: type, page, source, file_type, entities, table_group, **table_id**, **section_title**, **cell_values**, **header_path**, **row_index** |
| `embedding_model` | string | Model used for embedding |
| `file_type` | string | pdf/docx/xlsx/pptx/csv/text/image/video |
| `parent_chunk_id` | integer | Reference to parent chunk |
| `confidence_score` | float | Grounding confidence |
| `table_id` | string | 🆕 Stable table identifier (e.g., `p3_t1`) |
| `section_title` | string | 🆕 Owning document section heading |
| `nl_representation` | text | 🆕 Natural-language sentence for the row |
| `quantized_embedding` | text | INT8 quantized vector (Qdrant mirrored) |
| `raptor_level` | integer | RAPTOR hierarchy level (0 = leaf) |
| `created_at` | timestamp | Ingestion timestamp |

### Indexes
- HNSW on Qdrant vectors (cosine_ops, m=16, ef_construction=64)
- GIN on `doc_metadata` (for `cell_values` JSON queries)
- B-tree on `tenant_id`, `embedding_model`, `file_type`, `table_id`, `section_title`

---

## 🧠 Key Design Decisions (v5.0)

### Table-First Retrieval Strategy
1. **Query arrives** → `classify_query()` determines `table_lookup` / `table_compare` / `text`
2. **Exact lookup first** → `extract_catalogue_patterns()` detects `ECL2412SD` etc. → SQL ILIKE (score 2.0)
3. **Vector search second** → NL-serialized row sentences embedded via bge-large
4. **BM25 third** → markdown noise stripped before tokenization
5. **Table group expansion** → All sibling rows fetched once any row from a table is retrieved
6. **HTML assembly** → `build_table_html_context()` renders rows as HTML table for LLM

### 1-Row-Per-Chunk Strategy
- **Before v5.0**: 5 rows per chunk → one embedding diluted across 5 rows → wrong row retrieved
- **After v5.0**: 1 row per chunk with ±1 adjacent rows for context → precise retrieval
- Each chunk stores: full `header_path`, `section_title`, `cell_values` JSON, `row_index`

### Multi-Page Table Stitching
- Detects: same column count (±1), table B has no unique headers, first row of B has numeric content
- Merges: `table_a.data_rows.extend(table_b.data_rows)`
- Result: continuation tables become one `RichTable` with correct headers throughout

### Zero Hallucination Policy
- Layer 12 (Grounding Guard): Score < 0.35 → refuse to answer
- FLARE Layer 15: Adaptive retries (0.35 → 0.25 → 0.15)
- Strict LLM prompt: "ONLY use the DATABASE RECORDS below"
- Post-generation sentence verification

### Parent-Child Chunking (text content)
- **Parent chunks**: 2400 chars — broad context for LLM synthesis
- **Child chunks**: 600 chars — precise retrieval matching
- Table rows: always 1-row chunks (no parent-child split needed)

---

## 🐳 Deployment

### Docker (Full Stack)
```bash
./start.sh production up
```
Services: `rag_api`, `postgres` (pgvector), `redis`, `qdrant`, `ollama`, `neo4j`

### Local Development
```bash
./start.sh local up
# Hot-reload at http://localhost:1000
```

### After v5.0 Upgrade (Existing Deployment)
```bash
# Schema migrations run automatically on startup via _run_schema_migrations()
# New columns: table_id, section_title, nl_representation added via IF NOT EXISTS
# Re-ingest existing documents to populate new metadata:
curl -X POST http://localhost:1000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"force_reindex": true}'
```

---

## 📚 `table_engine.py` Public API Reference

```python
from app.rag.table_engine import (
    # Data classes
    RichTable,          # Fully reconstructed table with resolved spans
    TableCell,          # Single cell with text, row, col, header_path

    # Extraction
    extract_tables_pdfplumber,   # pdfplumber geometry-based extraction
    markdown_to_rich_table,      # Docling markdown → RichTable

    # Reconstruction
    stitch_continuation_tables,  # Multi-page table merging
    annotate_section_title,      # Extract section heading from surrounding text

    # Chunking
    chunk_rich_table,            # RichTable → list of {text, nl_text, json_cells, ...}

    # Context assembly
    assemble_html_table_from_chunks,  # Chunk dicts → HTML <table>

    # Query classification
    classify_query,              # Returns QueryType constant
    extract_catalogue_patterns,  # Returns list of model number strings
    QueryType,                   # TEXT_RETRIEVAL / TABLE_LOOKUP / TABLE_COMPARE / ...
)
```

---

## ⚡ Performance Benchmarks

| Mode | Latency | Scenario |
|------|---------|----------|
| **Cache hit** | **<1ms** | Repeated queries |
| **Exact catalogue lookup** | **<5ms** | `ECL2412SD door kit number` |
| **Extractive auto** | **6–15ms** | Simple text fact |
| **Full LLM table QA** | **500ms–3s** | Multi-row table synthesis |
| **Streaming first token** | **<200ms** | Real-time UX |

---

## 🔧 Operational Notes

- **Force re-ingest** after v5.0 upgrade to populate `table_id`, `section_title`, `nl_representation` on all chunks
- **pdfplumber** is used for geometry-aware table extraction; Docling markdown used as fallback
- **Table stitching** requires documents to be processed page-by-page (default behavior)
- **BM25 clean query**: markdown pipe and dash tokens stripped before PostgreSQL FTS
- **GIN index** on `doc_metadata` enables fast JSON `cell_values` queries without full table scans
