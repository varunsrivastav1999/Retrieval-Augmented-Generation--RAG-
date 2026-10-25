# CPU-only benchmark image. Contains BOTH the original model stack
# (sentence-transformers / torch: bge-large, bge-reranker-v2-m3) and the
# candidate ONNX stack (fastembed) so they can be compared on equal hardware.
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 HF_HOME=/models/hf FASTEMBED_CACHE_PATH=/models/fastembed
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2 \
 && pip install "numpy<2" sentence-transformers==2.7.0 "transformers==4.44.2" \
    fastembed==0.5.1 psycopg2-binary==2.9.10 sqlalchemy==2.0.36 requests==2.32.3 \
    pdfplumber==0.11.4 chardet==5.2.0 httpx==0.27.2
WORKDIR /work
