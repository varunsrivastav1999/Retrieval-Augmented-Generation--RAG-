# Contributing to i-Tips RAG

Thank you for your interest in contributing to the **i-Tips RAG 13-Layer Engine** — the world's most accurate offline RAG system! 🚀

## Getting Started

### Prerequisites
- **Docker & Docker Compose** (recommended)
- **Python 3.10+** (for native development)
- **Git**

### Local Development Setup

```bash
# 1. Fork and clone
git clone https://github.com/<your-username>/Retrieval-Augmented-Generation--RAG-.git
cd Retrieval-Augmented-Generation--RAG-

# 2. Copy environment files
cp .env.example .envs/.local/.rag

# 3. Start with Docker
docker-compose -f local.yml up --build

# 4. Open the dashboard
open http://localhost:1000
```

### Native Development (Mac/Linux)

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start PostgreSQL + Redis (via Docker or local)
docker-compose -f local.yml up postgres redis -d

# 4. Run natively (uses GPU if available)
python app/main.py
```

## Architecture

The project follows a **13-Layer Pipeline** architecture. See [`memory.md`](memory.md) for complete documentation.

### Key Modules

| Module | Layer(s) | Purpose |
|--------|----------|---------|
| `app/rag/parsers.py` | 1-2 | Universal document parsing (30+ formats) |
| `app/rag/ingestion.py` | 3-4 | Chunking + batch embedding |
| `app/rag/query_intelligence.py` | 13 | Spelling correction, query expansion |
| `app/rag/retrieval.py` | 5 | Hybrid search (HNSW + BM25) |
| `app/rag/reranker.py` | 6 | Cross-encoder reranking |
| `app/rag/context.py` | 7-8 | MMR diversity + context window expansion |
| `app/rag/grounding.py` | 9-10 | Zero hallucination guard |
| `app/main.py` | 11-12 | Cache, streaming, API, dashboard |
| `app/rag/jobs.py` | — | Background worker + auto-scanner |

## How to Contribute

### Reporting Bugs

1. Check existing [Issues](https://github.com/varunsrivastav1999/Retrieval-Augmented-Generation--RAG-/issues) first.
2. Create a new issue with:
   - **Title**: Clear, one-line summary
   - **Steps to reproduce**
   - **Expected vs actual behavior**
   - **Environment**: OS, Docker version, Python version

### Adding a New File Format

1. Add your parser function in `app/rag/parsers.py`
2. Register the extension in `SUPPORTED_EXTENSIONS`
3. Add a case in `parse_file()` that routes to your parser
4. Test with a sample file
5. Update the format table in `README.md` and `memory.md`

### Adding a New Layer

1. Create a new module in `app/rag/` (e.g., `app/rag/your_layer.py`)
2. Import and integrate into the pipeline in `app/main.py`
3. Add documentation in `memory.md`
4. Update the architecture diagram in `README.md`

### Code Style

- **Python**: Follow PEP 8
- **Docstrings**: Use triple-quote docstrings for all public functions
- **Print logs**: Use `print(f"[ModuleName] message")` format for consistency
- **Error handling**: Always use try/except with meaningful messages — never let the server crash
- **Comments**: Preserve existing comments unless your change directly affects them

## Pull Request Process

1. **Fork** the repository
2. Create a **feature branch**: `git checkout -b feature/my-new-feature`
3. Make your changes with clear commit messages
4. Ensure Docker builds successfully: `docker-compose -f local.yml up --build`
5. Test the key endpoints:
   - `GET /health/ready` — must return all green
   - `POST /api/v1/upload` — upload a test file
   - `POST /api/v1/query` — query with a test question
6. Open a **Pull Request** with:
   - Summary of changes
   - Testing evidence (screenshots/logs appreciated)
   - Any breaking changes noted

## Code of Conduct

- Be respectful and constructive
- Help newcomers — this project welcomes all skill levels
- Focus on quality — we aim for zero errors in production

---

**Thank you for making i-Tips RAG better for everyone!** 🙏
