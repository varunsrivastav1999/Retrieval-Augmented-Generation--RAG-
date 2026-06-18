# Enterprise Level RAG: 17-Layer Architecture — System Memory

> This file documents the complete architecture, APIs, configuration, and operational knowledge
> of the Enterprise Level RAG microservice. Updated: 2026-05-22.

---

## 🏗️ Architecture Overview

### 17-Layer Intelligence Pipeline

```
┌────────────────────────────────────────────────────────────────────┐
│      Enterprise Level RAG 17-Layer Engine v4.0              │
│                                                            │
│  ANY FILE ──► Layer 1:  Universal Document Parser            │
│              Layer 2:  Smart OCR & Table/Image Extraction    │
│              Layer 3:  Semantic Parent-Child Chunking         │
│              Layer 4:  Batch Embedding (32/batch, GPU)        │
│              Layer 5:  RAPTOR Hierarchical Summarization      │
│  QUERY   ──► Layer 6:  Hybrid Search (HNSW + BM25 + Trigram) │
│              Layer 7:  ColBERT Late-Interaction Reranking     │
│              Layer 8:  Max Marginal Relevance (MMR)           │
│              Layer 9:  Contextual Window Expansion            │
│              Layer 10: Agentic Router (Multi-tool)            │
│              Layer 11: Query Intelligence (Spelling, Expand)  │
│              Layer 12: 🛡️ Hallucination Guard                  │
│              Layer 13: Extractive Fast-Path (< 5ms)           │
│              Layer 14: Semantic Query Cache (Redis)           │
│              Layer 15: Active RAG (FLARE self-reflection)     │
│              Layer 16: GraphRAG (Neo4j)                       │
│              Layer 17: Real-Time Token Streaming              │
│                                                            │
└────┬──────────┬──────────┬──────────┬──────────────────────────────┘
     │          │          │          │
┌────▼───┐ ┌───▼────┐ ┌──▼──────┐ ┌─▼─────────┐
│Postgres│ │ Redis  │ │ Ollama  │ │ File Store │
│pgvector│ │ Cache  │ │ LLM     │ │ /media     │
└────────┘ └────────┘ └─────────┘ └────────────┘
```

---

## 📁 Universal File Support (Omni-Ingestion)

| Category | Extensions | Parser | Notes |
|----------|-----------|--------|-------|
| **Documents** | `.pdf` | PyMuPDF + pdfplumber | Text, tables, images (OCR), **Anti-Watermark**, MCQ Ticks |
| **Word/PPT** | `.docx`, `.doc`, `.pptx` | python-docx / pptx | Headings, paragraphs, tables, images, speaker notes |
| **Excel** | `.xlsx`, `.xls`, `.csv` | openpyxl / csv | All sheets → markdown tables, auto-pagination |
| **Code** | `.py`, `.js`, `.java`, etc. | Universal | **Strict Indentation Preservation**, markdown code blocks |
| **Email** | `.eml`, `.msg` | `email` module | Extracts Subject, From, To, Date, and pure text body |
| **Web Links**| `.url`, `.webloc` | `urllib` + `bs4` | **100% Offline Auto-Scraper** (Removes ads/nav, extracts article) |
| **Images** | `.png`, `.jpg`, `.jpeg`, etc. | Pillow + pytesseract | Auto-Rotates flipped scans (OSD), full OCR |
| **Video** | `.mp4`, `.avi`, `.mkv`, etc. | ffmpeg | Embedded subtitle extraction |
| **Subtitles**| `.srt`, `.ass`, `.vtt` | pysubs2 | Clean text extraction, HTML/tag removal |
| **Archives** | `.zip`, `.tar.gz`, `.rar` | `zipfile` | **Auto-Extracts** to hidden dir and recursive multi-ingest |
| **Text** | `.txt`, `.md`, `.json`, etc. | Python stdlib | Auto-encoding detection via chardet |

---

## 🔌 API Reference (Microservice)

### Ingestion APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/upload` | Upload any file (no size limit), auto-detect format, queue background ingestion |
| `POST` | `/api/v1/ingest` | Scan `/media` volume for all supported files and queue ingestion |
| `GET` | `/api/v1/ingest/jobs` | List ingestion jobs (with progress %) |
| `GET` | `/api/v1/ingest/jobs/{id}` | Get specific job status |
| `GET` | `/api/v1/formats` | List all supported file formats |

### Query API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/query` | Query knowledge base with 17-layer pipeline |

**Query Request Body:**
```json
{
    "query": "What is a DC Sensor?",
    "tenant_id": "default",
    "top_k": 12,
    "stream": true,
    "parent": "Robot Zone",
    "child": "Sensors"
}
```

**Query Response includes:**
- `answer` — Strictly grounded in document content
- `sources` — Exact document citations
- `grounding` — Pre-generation grounding score
- `verification` — Post-generation confidence (high/medium/low)
- `latency_ms` — Response time

### Health APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health/live` | Liveness probe (always returns `ok`) |
| `GET` | `/health/ready` | Readiness + stats (chunks, docs, file types, active jobs) |

---

## ⚙️ Configuration (Environment Variables)

### Core
| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://rag_user:rag_password@postgres:5432/rag_db` | PostgreSQL with pgvector |
| `REDIS_URL` | `redis://redis:6379/0` | Redis for semantic caching |
| `OLLAMA_URL` | `http://ollama:11434/api/generate` | Ollama LLM endpoint |
| `OLLAMA_MODEL` | `llama3.1:8b` | LLM model name |
| `MEDIA_PATH` | `/media` | Shared volume for auto-scan |

### Models
| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | Embedding model (1024d, SOTA open-source) |
| `RAG_RERANKER_MODEL` | `colbert-ir/colbertv2.0` | ColBERT late-interaction reranker |
| `RAG_EMBEDDING_DIM` | `1024` | Embedding vector dimension |
| `RAG_MODEL_DEVICE` | auto-detect | Force device: `mps`, `cuda`, `cpu` |

### Performance
| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_DEFAULT_TOP_K` | `12` | Default retrieval count |
| `RAG_MAX_TOP_K` | `50` | Maximum retrieval count |
| `RAG_BROAD_QUERY_TOP_K` | `16` | Top-K for broad queries |
| `RAG_INGESTION_WORKER_POLL_SECONDS` | `5` | Worker poll interval |
| `OLLAMA_NUM_PREDICT` | `1024` | Max tokens per response |

### Feature Flags
| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_ENABLE_INGESTION_WORKER` | `true` | Enable background ingestion worker |
| `RAG_PRELOAD_MODELS_ON_STARTUP` | env-dependent | Pre-download models on startup |
| `RAG_ALLOW_HASH_FALLBACK` | `false` | Allow hashing fallback if model unavailable |
| `RAG_HF_OFFLINE` | `false` | Force offline-only HuggingFace mode |

---

## 🐳 Deployment

### Docker (Full Stack)
```bash
docker-compose -f local.yml up --build
```
Services: `rag_api`, `postgres` (pgvector), `redis`, `ollama`

### Native (Mac GPU)
```bash
pip install -r requirements.txt
python app/main.py
```
Auto-detects Apple Silicon MPS for GPU acceleration.

### Shared Media Volume
Place files in the Docker-mapped `/media` path. On startup, the auto-scanner detects
all supported files and queues background ingestion automatically.

```yaml
# local.yml
volumes:
  - /path/to/your/documents:/media
```

---

## 🧠 Key Design Decisions

### Zero Hallucination Policy
- **Layer 9 (Hallucination Guard)**: Computes grounding score BEFORE calling LLM
  - Score < 0.25 → refuses to answer ("not available in documents")
  - Skips LLM call entirely → extremely fast rejection
- **Layer 10 (Answer Verification)**: After LLM generates answer, verifies claims against sources
  - Confidence: high (>70% grounded), medium (40-70%), low (<40%)
- **Strict Prompt**: LLM is explicitly instructed to NEVER use general knowledge

### Parent-Child Chunking
- **Parent chunks**: 2400 chars — broad context for LLM
- **Child chunks**: 600 chars — precise retrieval matching
- Each child references its parent for contextual expansion

### Omni-Ingestion Capabilities
- **Anti-Watermark**: Dynamically detects and deletes repeating corporate footers/headers across 3+ PDF pages.
- **Archive Extraction**: Auto-unzips `.zip`/`.tar.gz` and queues all internal files.
- **Code & Emails**: Explicitly preserves indentation and structure for complex queries.
- **Web Scraping**: Downloading internet `.url` shortcuts into offline markdown text instantly.
- **PDF Flip & MCQ Ticks**: Auto-rotates inverted pages using OSD and explicitly injects `[CORRECT ANSWER]` tags next to unicode checkmarks.

---

## 📊 Database Schema

### `document_chunks`
| Column | Type | Description |
|--------|------|-------------|
| `id` | integer (PK) | Auto-increment |
| `tenant_id` | string | Multi-tenant isolation |
| `doc_id` | string | Source file path |
| `chunk_hash` | string | SHA-256 for deduplication |
| `text_content` | text | Chunk text content |
| `section` | integer | Section ordering |
| `doc_metadata` | JSON | Type, page, source, etc. |
| `embedding_model` | string | Model used for embedding |
| `embedding` | vector(384) | pgvector embedding |
| `file_type` | string | pdf/docx/xlsx/pptx/csv/text/image/video |
| `parent_chunk_id` | integer | Reference to parent chunk |
| `confidence_score` | float | Grounding confidence |
| `created_at` | timestamp | Ingestion timestamp |

### Indexes
- HNSW on `embedding` (vector_cosine_ops, m=16, ef_construction=64)
- GIN on `text_content` (tsvector + trigram)
- B-tree on `tenant_id`, `embedding_model`, `file_type`, `chunk_hash`

---

## 🔧 Project Structure

```
enterprise-level-rag/
├── app/
│   ├── main.py              # FastAPI app, APIs, dashboard, Active RAG
│   ├── database.py           # SQLAlchemy models, pgvector, migrations
│   └── rag/
│       ├── parsers.py        # Layer 1-2: Universal document parser & OCR
│       ├── ingestion.py      # Layers 3-4: Chunking + batch embedding
│       ├── raptor.py         # Layer 5: RAPTOR hierarchical summarization
│       ├── retrieval.py      # Layer 6: Hybrid search (HNSW + BM25)
│       ├── reranker.py       # Layer 7: ColBERT late-interaction reranking
│       ├── context.py        # Layers 8-9: MMR + context expansion
│       ├── router.py         # Layer 10: Agentic multi-tool routing
│       ├── query_intelligence.py  # Layer 11: Spelling, expansion, decomposition
│       ├── grounding.py           # Layer 12: Hallucination guard
│       ├── graph.py               # Layer 16: GraphRAG (Neo4j)
│       ├── model_loader.py        # Model management, device detection
│       └── jobs.py                # Background worker, auto-scanner
├── Dockerfile
├── local.yml                 # Docker Compose (local)
├── production.yml            # Docker Compose (production)
├── Modelfile                 # Ollama custom LLM brain
├── requirements.txt          # Python dependencies
├── memory.md                 # This file
└── README.md
```
