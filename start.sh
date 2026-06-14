#!/bin/bash

# Default to local if not specified
ENV=${1:-local}

if [[ "$ENV" != "local" && "$ENV" != "production" ]]; then
    echo "Usage: ./start.sh [local|production]"
    exit 1
fi

COMPOSE_FILE="${ENV}.yml"

echo "============================================================"
echo "🔍 Auto-Detecting Hardware..."

# 1. Detect CUDA (NVIDIA)
if command -v nvidia-smi &> /dev/null; then
    echo "✅ CUDA (NVIDIA GPU) detected!"
    echo "🚀 Starting $ENV environment with Docker GPU Passthrough..."
    docker compose -f $COMPOSE_FILE -f docker-gpu.yml up --build -d

# 2. Detect MPS (Apple Silicon)
elif [[ $(uname -m) == 'arm64' && $(uname -s) == 'Darwin' ]]; then
    echo "✅ Apple Silicon (MPS) detected!"
    echo "⚠️  Note: Docker on Mac does not support native MPS passthrough yet."
    echo "   The containers will run on CPU. To utilize MPS fully, run 'python app/main.py' natively outside Docker."
    echo "🚀 Starting $ENV environment..."
    docker compose -f $COMPOSE_FILE up --build -d

# 3. Fallback to CPU
else
    echo "⚠️  No GPU detected."
    echo "🐢 Starting $ENV environment in CPU mode..."
    docker compose -f $COMPOSE_FILE up --build -d
fi
echo "============================================================"
