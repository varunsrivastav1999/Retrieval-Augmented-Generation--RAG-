<div align="center">
  <h1>Industrial Maintenance RAG</h1>
  <p><strong>Developed &amp; Owned by: Varun Srivastav</strong></p>
  <p>Offline question answering over robot-painting maintenance knowledge — manuals, alarm lists, PLC/SCADA tag lists and maintenance history — with the evidence for every answer.</p>
</div>

---

## What it does

* **Word-for-word answers.** The answer is the exact text from the manual — the alarm row, the "Condition:" line, the torque step, the complete numbered procedure — with document, section and page, plus **Open page N** (the original PDF opens at that page) and **Copy**. Measured: 206/206 questions answered word for word on a PDF/DOCX test library.
* **Optional AI summary.** A local LLM can add a short cited summary below the exact text; it is labelled as a summary and checked sentence by sentence, code by code and number by number against the sources.
* **Exact codes win.** Alarm codes (`SRVO-050`, `PNT1-128`), tags (`OVN2_Z3_TEMP_PV`), PLC addresses (`DB120.DBD24`) and work orders are matched exactly, never "close enough". Unknown codes are reported as unknown instead of answered.
* **Fast on CPU.** Documented evidence comes back in well under a second on a 4-core PC; the local LLM then streams a written answer.
* **Says "not in the documents"** instead of guessing, or "closest match — please check" naming the words it could not find (e.g. a machine that is not in any manual).
* **Fully offline.** No cloud APIs, no telemetry, no runtime downloads. All models are baked into the images.
* **Built for large libraries:** int8-quantized vector index (~0.4 GB RAM per 1M passages), streamed and resumable indexing (a restart continues where it stopped).

## Quick start

```bash
docker compose up --build
```

Open **http://localhost:1000**. The first build needs internet (Python packages, ~200 MB of retrieval models and the ~2 GB LLM). After that the stack runs with no network at all.

Add documents by dragging them into the **Documents** page, or copy them into the `media` volume:

```bash
docker compose cp ./my-manuals worker:/media/manuals
```

The worker indexes new, changed and deleted files automatically (every 60 s). The first folder level becomes the *collection* (e.g. `manuals`, `alarm_lists`, `work_orders`) that you can filter on.

### Air-gapped sites

On a machine with internet:

```bash
scripts/offline-bundle.sh        # Windows: scripts\offline-bundle.ps1
```

Copy the repository folder and `industrial-rag-images.tar.gz` to the site, then:

```bash
docker load -i industrial-rag-images.tar.gz
docker compose up -d
```

### Linux, macOS, Windows

The stack is plain Docker Compose with multi-architecture images (x86-64 and ARM64), so it runs the same on Linux, macOS (Docker Desktop) and Windows (Docker Desktop / WSL2). On Windows use the `.ps1` script variants. GPU acceleration for the LLM is optional (`docker-compose.gpu.yml`, NVIDIA on Linux/WSL2).

## Architecture

```
            ┌──────────── browser UI (static, same origin) ────────────┐
            │                                                           │
   ┌────────▼─────────┐  hybrid search (1 request)   ┌──────────────────┐
   │ api  (FastAPI)   │ ───────────────────────────▶ │ qdrant           │ dense int8 + BM25 sparse
   │ ONNX: embed,     │ ◀─────────────────────────── │ vector index     │ + identifier payload index
   │ BM25, rerank     │       chunk text, sections    └──────────────────┘
   │ pipeline, verify │ ───────────────────────────▶ ┌──────────────────┐
   │                  │                               │ postgres         │ documents, chunks, jobs,
   │                  │ ── answer cache ──▶ redis     │                  │ conversations, audit log
   │                  │ ── generation ────▶ ollama    └────────▲─────────┘
   └──────────────────┘   (qwen2.5:3b, CPU)                    │
                                                    ┌──────────┴───────┐
   /media (volume) ───────── watched / uploads ───▶ │ worker           │ parse → chunk → embed → store
                                                    │ pdfplumber, OCR  │ (separate process: never slows queries)
                                                    └──────────────────┘
```

**Query path** (`app/rag/pipeline.py`) — no hidden LLM calls before the answer:

1. Rule-based intent (alarm, troubleshoot, procedure, parameter, history, …) — < 1 ms.
2. Exact answer cache (Redis), invalidated whenever the index changes.
3. One batched Qdrant request: dense vector + BM25 sparse + exact-identifier filter.
4. Weighted reciprocal-rank fusion → chunk text from Postgres → cross-encoder rerank, with a bonus for chunks containing the exact codes asked about.
5. For procedures/alarms: the whole section is reassembled in order (complete steps, not fragments).
6. "Not in documents" decision (unknown code, or nothing relevant enough).
7. The **exact answer** (verbatim lines/rows/steps with page) is returned immediately; with "Exact + AI summary" the LLM then writes a cited summary from that evidence only.
8. Deterministic verification: unsupported sentences, codes and numbers not present in the sources are flagged in the UI.

**Relationships / root cause.** Every chunk stores the identifiers it mentions. Codes that are documented together (alarm "related alarms", work orders, tag lists) form a lightweight relationship graph shown as "Related" chips, with the number of historical records — without running a separate graph database.

## Choosing the LLM

The LLM only writes the final answer, so retrieval quality does not depend on it.

| Hardware | `RAG_LLM_MODEL` | Notes |
|---|---|---|
| 4 cores / 8 GB, CPU only | `qwen2.5:3b` (default) | Evidence < 1 s, written answer streams after |
| Very small PCs | `qwen2.5:1.5b` | Faster, less fluent |
| 16 GB RAM or a GPU | `qwen2.5:7b` | Best written answers |
| No LLM wanted | any + mode "Evidence only" | Sub-second, purely extractive |

Change it in `.env`, then `docker compose build ollama && docker compose up -d`.

## Configuration

All settings have defaults; put overrides in `.env` (see `.env.example`). Most useful:

| Variable | Default | Purpose |
|---|---|---|
| `RAG_PORT` | `1000` | Web/API port on the host |
| `RAG_API_KEY` | *(empty)* | Require `X-API-Key` on all API calls (UI asks once) |
| `POSTGRES_PASSWORD` | `change-me-rag` | Database password — change it |
| `RAG_LLM_MODEL` | `qwen2.5:3b` | Local LLM baked into the ollama image |
| `RAG_TOP_K` | `6` | Evidence passages per answer |
| `RAG_MIN_RELEVANCE` | `-4.0` | Reranker score below which the answer is "not in documents" |
| `RAG_AUTO_SCAN_SECONDS` | `60` | Media folder scan interval (0 = off) |
| `WITH_DOCLING` | `false` | Heavier layout-aware PDF parser in the worker image |
| `API_MEMORY` … `OLLAMA_MEMORY` | see `.env.example` | Container memory limits |

## API

Interactive docs: `http://localhost:1000/api/docs`.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/query` | Ask (`mode`: `auto`/`evidence`/`answer`, `stream`: SSE events `retrieval`, `token`, `done`) |
| `POST` | `/api/v1/search` | Retrieval only (for integrations) |
| `GET` | `/api/v1/related?identifier=SRVO-050` | Co-documented codes and history count |
| `POST` | `/api/v1/upload` | Upload a file (multipart, optional `collection`) |
| `GET` | `/api/v1/documents` · `/collections` · `/chunks/{id}` | Browse the index |
| `POST`/`DELETE` | `/api/v1/documents/{id}/reindex` · `/api/v1/documents/{id}` | Manage documents |
| `GET` | `/api/v1/ingest/jobs` · `POST /api/v1/ingest/scan` | Ingestion queue |
| `GET/POST/PATCH/DELETE` | `/api/v1/conversations…` | Conversation history |
| `GET` | `/api/v1/status` · `/health/ready` · `/health/live` · `/metrics` | Status, health, Prometheus metrics |

## Supported files

PDF (text + OCR for scanned pages), DOCX, PPTX, HTML, TXT/MD/LOG/JSON/XML/YAML, PLC/source code (`.st`, `.scl`, …), images (OCR), e-mail (`.eml`), subtitles, CSV/TSV/XLSX (streamed row by row — alarm logs, work-order exports, tag lists), ZIP/TAR (extracted and indexed). Legacy binary `.doc/.xls/.ppt` must be saved as the modern format first.

## Operations

* **Status:** the *System* page, `GET /api/v1/status`, or `/metrics` (Prometheus).
* **Logs:** `docker compose logs -f api worker` (JSON lines).
* **Backup / restore:** `scripts/backup.sh` → `backups/<timestamp>/`; `scripts/restore.sh backups/<timestamp>`.
* **Failure behaviour:** Redis down → no caching; LLM down or busy for more than `RAG_LLM_QUEUE_WAIT_SECONDS` (20 s) → evidence-only answer with a notice; Postgres/Qdrant down → readiness fails and queries return 503 until they are back (connections recover automatically, no restart needed). An interrupted ingestion (worker/database restart) is re-queued and **resumes after the last committed batch** instead of starting over.
* **Indexing never starves questions:** the worker runs at low CPU priority (`nice`) and is capped at `WORKER_CPUS` (default 2). For a one-off bulk load, start it with more: `WORKER_CPUS=8 docker compose up -d worker`.
* **Changing the embedding model** requires re-indexing: set a new `RAG_QDRANT_COLLECTION` and the new model's build args, rebuild, then *Rescan*.

## Benchmarks

See [`bench/RESULTS.md`](bench/RESULTS.md) for the measured comparison between the previous implementation and this one (accuracy, word-for-word PDF answers, latency, ingestion speed, concurrency, offline and failure tests). Reproduce with:

```bash
python bench/corpus.py --out bench/data
python bench/e2e_bench.py --url http://localhost:1000 --upload --llm-sample 10 --concurrency 1,4,8
python bench/pdf_corpus.py --out bench/data_pdf && python bench/pdf_bench.py --upload   # word-for-word PDF test
```

## License

MIT — see [LICENSE](LICENSE).
