# syntax=docker/dockerfile:1.7
# =============================================================================
# Industrial Maintenance RAG — application images (CPU-first, offline at runtime)
#
#   target "api"    : FastAPI query service + UI  (ONNX models, no PyTorch)
#   target "worker" : api + document parsers + Tesseract OCR (ingestion)
#
# Internet is needed only while building (pip + model download). The built
# images contain every model they use and run with HF_HUB_OFFLINE=1.
# =============================================================================
ARG PYTHON_IMAGE=python:3.11-slim-bookworm

FROM ${PYTHON_IMAGE} AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

# ---- python dependencies (installed into /install, copied into final stages) --
FROM base AS deps-api
COPY requirements/api.txt /tmp/requirements/api.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install -r /tmp/requirements/api.txt

FROM base AS deps-worker
COPY requirements/ /tmp/requirements/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install -r /tmp/requirements/worker.txt

# ---- models: downloaded once at build time, verified, then copied ----------
FROM base AS models
COPY --from=deps-api /install /usr/local
COPY scripts/download_models.py /tmp/download_models.py
ARG RAG_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ARG RAG_RERANKER_MODEL=Xenova/ms-marco-MiniLM-L-6-v2
ARG RAG_SPARSE_MODEL=Qdrant/bm25
# Classic HTTP downloads are more reliable behind proxies than the Xet backend.
ENV HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60
RUN python /tmp/download_models.py --cache /models/fastembed \
      --embedding "${RAG_EMBEDDING_MODEL}" --reranker "${RAG_RERANKER_MODEL}" --sparse "${RAG_SPARSE_MODEL}"

# ---- API ---------------------------------------------------------------------
FROM base AS api
COPY --from=deps-api /install /usr/local
COPY --from=models /models /models
RUN groupadd -r -g 10001 app && useradd -r -u 10001 -g app -d /app app \
 && mkdir -p /media/uploads && chown -R app:app /media
COPY --chown=app:app app /app/app
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    RAG_MODEL_CACHE=/models \
    MEDIA_PATH=/media
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=4)" || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log", "--timeout-keep-alive", "30"]

# ---- worker ------------------------------------------------------------------
FROM base AS worker
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
 && rm -rf /var/lib/apt/lists/*
COPY --from=deps-worker /install /usr/local
COPY --from=models /models /models
# Optional heavy layout/table parser. Build with: --build-arg WITH_DOCLING=true
ARG WITH_DOCLING=false
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$WITH_DOCLING" = "true" ]; then \
      pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 && pip install docling==2.15.1 ; \
    fi
RUN groupadd -r -g 10001 app && useradd -r -u 10001 -g app -d /app app \
 && mkdir -p /media/uploads && chown -R app:app /media
COPY --chown=app:app app /app/app
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    RAG_MODEL_CACHE=/models \
    MEDIA_PATH=/media \
    RAG_USE_DOCLING=${WITH_DOCLING}
USER app
# Healthy while the worker loop (or a running ingestion) refreshed its heartbeat recently.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import os,sys,time; sys.exit(0 if time.time()-os.path.getmtime('/tmp/worker.heartbeat')<600 else 1)" || exit 1
CMD ["nice", "-n", "10", "python", "-m", "app.worker"]
