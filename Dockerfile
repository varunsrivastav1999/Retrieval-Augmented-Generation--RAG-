FROM python:3.10-slim

# Labels for open source
LABEL maintainer="Varun Srivastava <varunsrivastav1999>"
LABEL description="i-Tips RAG 13-Layer Engine — Zero-Hallucination RAG"
LABEL version="3.0.0"
LABEL org.opencontainers.image.source="https://github.com/varunsrivastav1999/Retrieval-Augmented-Generation--RAG-"
LABEL org.opencontainers.image.licenses="MIT"

# System dependencies (all needed for document parsing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    gcc \
    tesseract-ocr \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Install Python dependencies (cache layer optimization)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Pre-download models so they are baked into the Docker image
ENV HF_HOME=/models/huggingface
RUN --mount=type=cache,target=/root/.cache/huggingface \
    python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
    SentenceTransformer('sentence-transformers/clip-ViT-B-32'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy application code
COPY . .

# Create media directory for volume mount
RUN mkdir -p /media

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:1000/health/live || exit 1

# Expose port
EXPOSE 1000

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "1000"]
