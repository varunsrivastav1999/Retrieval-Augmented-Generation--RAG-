# i-Tips RAG: World-Class 12-Layer Production Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)](https://www.docker.com/)

An advanced, **12-Layer Retrieval-Augmented Generation (RAG)** microservice with **zero hallucination**, **universal file support**, and **millisecond search latency**. 100% offline, enterprise-grade.

---

## 🧠 The 12-Layer Intelligence Pipeline

```
┌────────────────────────────────────────────────────────────────────┐
│                  i-Tips RAG 12-Layer Engine v2.0                    │
│                                                                    │
│  ANY FILE ──► Layer 1:  Universal Document Parser                  │
│              Layer 2:  Smart OCR & Table/Image Extraction          │
│              Layer 3:  Semantic Parent-Child Chunking               │
│              Layer 4:  Batch Embedding (32/batch, GPU-accelerated)  │
│  QUERY   ──► Layer 5:  Hybrid Search (HNSW + BM25 + Trigram)       │
│              Layer 6:  Cross-Encoder Reranking                      │
│              Layer 7:  Max Marginal Relevance (MMR)                 │
│              Layer 8:  Contextual Window Expansion                  │
│              Layer 9:  🛡️ Hallucination Guard (ZERO general answers) │
│              Layer 10: ✅ Answer Verification & Grounding            │
│              Layer 11: Semantic Query Cache (Redis SHA-256)         │
│              Layer 12: Real-Time Token Streaming                    │
│                                                                    │
└────┬──────────┬──────────┬──────────┬──────────────────────────────┘
     │          │          │          │
┌────▼───┐ ┌───▼────┐ ┌──▼──────┐ ┌─▼─────────┐
│Postgres│ │ Redis  │ │ Ollama  │ │ File Store │
│pgvector│ │ Cache  │ │ LLM     │ │ /media     │
└────────┘ └────────┘ └─────────┘ └────────────┘
```

---

## 📁 Universal File Support

| Category | Formats | Extraction |
|----------|---------|------------|
| **Documents** | PDF, DOCX, DOC | Text + Tables + Images (OCR) |
| **Spreadsheets** | XLSX, XLS, CSV | All sheets → Markdown tables |
| **Presentations** | PPTX, PPT | Slides + Notes + Tables |
| **Text** | TXT, MD, LOG, JSON, XML | Auto-encoding detection |
| **Images** | PNG, JPG, JPEG, BMP, TIFF, GIF, WEBP | Full OCR text extraction |
| **Video** | MP4, AVI, MKV, MOV, WMV, FLV | Embedded subtitle extraction |
| **Subtitles** | SRT, ASS, SSA, VTT | Clean text parsing |

**No file size limit.** Drop files into the shared Docker volume — background ingestion starts automatically.

---

## 🛡️ Zero Hallucination

The system **NEVER gives general answers**. Every response is strictly grounded in your uploaded documents:

- **Layer 9 (Hallucination Guard)**: Computes a grounding score BEFORE calling the LLM. If no relevant content exists → refuses to answer instantly.
- **Layer 10 (Answer Verification)**: After generation, verifies every claim maps back to a source chunk. Adds confidence scoring (high/medium/low).
- **Strict Prompt**: The LLM is explicitly forbidden from using general knowledge.

---

## 🚀 Quick Start

### Docker (Full Stack)
```bash
git clone https://github.com/varunsrivastav1999/Retrieval-Augmented-Generation--RAG-.git
cd Retrieval-Augmented-Generation--RAG-

# Start all services
docker-compose -f local.yml up --build
```

### Native Mode (Mac GPU)
```bash
pip install -r requirements.txt
python app/main.py
```
Auto-detects Apple Silicon MPS / NVIDIA CUDA for GPU acceleration.

### Auto-Ingestion
Place any supported file in your mapped `/media` volume. The system auto-detects and begins background chunking + embedding immediately.

---

## 🔌 Microservice API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/upload` | Upload any file (unlimited size) |
| `POST` | `/api/v1/ingest` | Scan /media and ingest all files |
| `POST` | `/api/v1/query` | Query knowledge base (12-layer pipeline) |
| `GET` | `/api/v1/ingest/jobs` | List ingestion jobs + progress |
| `GET` | `/api/v1/formats` | List supported formats |
| `GET` | `/health/ready` | Readiness + system stats |
| `GET` | `/` | Production dashboard UI |

---

## 📖 Documentation

See [`memory.md`](memory.md) for complete system documentation including:
- All 12 layers explained
- Database schema
- Configuration reference (all env vars)
- Deployment guide

---

## Roadmap
- [ ] **Multi-Modal Support**: Retrieval of images and diagrams via CLIP.
- [ ] **Agentic Re-Ranking**: Use a small LLM to decide which chunks are "actually" useful.
- [ ] **Evaluations Suite**: Integrated RAGAS benchmarks for accuracy tracking.
- [ ] **Plugin System**: Connect to SharePoint, Google Drive, and Slack.
- [ ] **Whisper Integration**: Speech-to-text for video files without subtitles.

---

## Contributing
1. Fork the repo.
2. Create a feature branch (`git checkout -b feature/AmazingNewLayer`).
3. Commit your changes (`git commit -m 'Add some AmazingNewLayer'`).
4. Push to the branch (`git push origin feature/AmazingNewLayer`).
5. Open a Pull Request.

---

## License
Distributed under the MIT License. See `LICENSE` for more information.

---
**Created by the community for high-performance, private AI.**
