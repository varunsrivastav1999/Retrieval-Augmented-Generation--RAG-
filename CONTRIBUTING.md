# Contributing

## Run the stack

```bash
docker compose up --build        # http://localhost:1000
```

## Develop and test without Docker

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements/dev.txt
pytest -q                         # unit tests (no services needed)
ruff check app bench scripts tests --select F,E9,B
```

To run the API against the compose databases, publish their ports in a local
`docker-compose.override.yml` and point `DATABASE_URL`, `QDRANT_URL`,
`REDIS_URL` and `OLLAMA_BASE_URL` at `localhost`, then
`uvicorn app.main:app --reload` and `python -m app.worker`.

## Benchmarks (run before and after any retrieval/model change)

```bash
python bench/corpus.py --out bench/data                     # deterministic test corpus + labelled questions
python bench/e2e_bench.py --url http://localhost:1000 --upload --llm-sample 10 --concurrency 1,4,8
```

Results are written to `bench/results/`. A change that lowers `answer@5` or the
correct-refusal rate needs a good reason.

## Layout

```
app/main.py            HTTP API + UI routes
app/worker.py          ingestion worker (job queue + media folder watcher)
app/config.py          all settings (environment variables)
app/db.py              PostgreSQL schema
app/rag/pipeline.py    query pipeline (intent -> cache -> retrieval -> evidence -> LLM -> verification)
app/rag/search.py      hybrid retrieval (dense + BM25 + exact identifiers, RRF, rerank)
app/rag/answer.py      prompt, evidence answer, deterministic verification
app/rag/ingestion.py   parse -> chunk -> embed -> store
app/rag/parsers.py     offline parsers (PDF/DOCX/PPTX/HTML/text/images/CSV/XLSX/archives)
app/static/            UI (vanilla JS, vendored marked + DOMPurify)
bench/                 corpus generator and benchmarks
```

## Guidelines

* Keep the query path free of extra LLM calls; latency on CPU-only sites depends on it.
* Never add a runtime network dependency; models are baked into images at build time.
* Prefer deterministic logic for accuracy checks (identifiers, numbers, citations).
