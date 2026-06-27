# ==============================================================================
# Enterprise Level RAG — Multi-Stage Production Dockerfile
# ==============================================================================
# Stage 1: Builder - Install system deps, compile wheels, download models
# Stage 2: Runtime - Minimal image with only runtime dependencies
# ==============================================================================

# ---- Stage 1: Builder --------------------------------------------------------
FROM python:3.10-slim AS builder

# Labels
LABEL maintainer="Varun Srivastava <varunsrivastav1999>"
LABEL description="Enterprise Level RAG 17-Layer Engine — Zero-Hallucination RAG"
LABEL version="3.0.0"
LABEL org.opencontainers.image.source="https://github.com/varunsrivastav1999/Retrieval-Augmented-Generation--RAG-"
LABEL org.opencontainers.image.licenses="MIT"

# System dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    gcc \
    tesseract-ocr \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set up model cache directory
ENV HF_HOME=/models/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/models/huggingface/sentence-transformers
ENV HF_HUB_CACHE=/models/huggingface/hub
ENV DOCLING_HOME=/models/docling
ENV EASYOCR_MODULE_PATH=/models/easyocr

# Install Python dependencies with pip cache mount
# The --mount=type=cache persists /root/.cache/pip across builds so that
# unchanged packages are NOT re-downloaded. Only new/updated packages install.
WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Pre-download models in separate steps to prevent Docker OOM crashes
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/clip-ViT-L-14')"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-large')"
RUN python -c "from docling.document_converter import DocumentConverter, PdfFormatOption; \
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions; \
    from docling.datamodel.base_models import InputFormat; \
    opts = PdfPipelineOptions(); \
    opts.do_ocr = True; \
    opts.do_table_structure = True; \
    opts.table_structure_options = TableStructureOptions(mode='accurate'); \
    DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})"

# NOTE: spaCy model en_core_web_sm-3.8.0 is already installed via requirements.txt
# No duplicate install needed here.

# ---- Stage 2: Runtime --------------------------------------------------------
FROM python:3.10-slim AS runtime

# Runtime system dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    tesseract-ocr \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    libgomp1 \
    curl \
    zstd \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy model cache from builder
COPY --from=builder /models /models

# Install Ollama binary (official install script — handles OS/arch detection)
RUN curl -fsSL https://ollama.com/install.sh | sh

# Pre-download Ollama LLM model (baked into image — zero runtime/download at startup)
RUN ollama serve &>/tmp/ollama-srv.log & \
    for i in $(seq 1 30); do ollama list 2>/dev/null && break; sleep 1; done && \
    ollama pull llama3.1:8b && \
    pkill ollama 2>/dev/null; true && \
    mkdir -p /models/ollama/blobs /models/ollama/manifests && \
    cp -r /root/.ollama/models/blobs/* /models/ollama/blobs/ && \
    cp -r /root/.ollama/models/manifests/* /models/ollama/manifests/

# Set model cache environment
ENV HF_HOME=/models/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/models/huggingface/sentence-transformers
ENV HF_HUB_CACHE=/models/huggingface/hub
ENV DOCLING_HOME=/models/docling
ENV EASYOCR_MODULE_PATH=/models/easyocr

# Ensure NVIDIA runtime passes GPU libraries
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /bin/bash appuser

# Working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Create required directories and fix model cache permissions
RUN mkdir -p /media /var/log/rag && chown -R appuser:appuser /media /var/log/rag /models

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 1000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:1000/health/ready || exit 1

# Default command (overridden in docker-compose)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "1000", "--workers", "1"]