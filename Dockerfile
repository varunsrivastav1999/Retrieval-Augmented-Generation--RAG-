# ==============================================================================
# i-Tips RAG — Multi-Stage Production Dockerfile
# ==============================================================================
# Stage 1: Builder - Install system deps, compile wheels, download models
# Stage 2: Runtime - Minimal image with only runtime dependencies
# ==============================================================================

# ---- Stage 1: Builder --------------------------------------------------------
FROM python:3.11-slim AS builder

# Labels
LABEL maintainer="Varun Srivastava <varunsrivastav1999>"
LABEL description="i-Tips RAG 13-Layer Engine — Zero-Hallucination RAG"
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
    libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set up model cache directory
ENV HF_HOME=/models/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/models/huggingface/sentence-transformers
ENV HF_HUB_CACHE=/models/huggingface/hub

# Install Python dependencies with pip cache
WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download models (baked into builder layer)
RUN --mount=type=cache,target=/root/.cache/huggingface \
    python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); SentenceTransformer('sentence-transformers/clip-ViT-B-32'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# ---- Stage 2: Runtime --------------------------------------------------------
FROM python:3.11-slim AS runtime

# Runtime system dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    tesseract-ocr \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1-mesa-dri \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy model cache from builder
COPY --from=builder /models /models

# Set model cache environment
ENV HF_HOME=/models/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/models/huggingface/sentence-transformers
ENV HF_HUB_CACHE=/models/huggingface/hub
ENV TRANSFORMERS_CACHE=/models/huggingface/hub

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /bin/bash appuser

# Working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Create required directories
RUN mkdir -p /media /var/log/rag && chown -R appuser:appuser /media /var/log/rag

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 1000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:1000/health/ready || exit 1

# Default command (overridden in docker-compose)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "1000", "--workers", "4"]