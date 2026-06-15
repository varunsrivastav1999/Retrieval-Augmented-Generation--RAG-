#!/bin/bash

# Default to local if not specified
ENV=${1:-local}
CMD=${2:-up}

if [[ "$ENV" != "local" && "$ENV" != "production" ]]; then
    echo "Usage: ./start.sh [local|production] [up|down|build|logs...]"
    exit 1
fi

COMPOSE_FILE="${ENV}.yml"
COMPOSE_ARGS="-f $COMPOSE_FILE"

echo "============================================================"
echo "🔍 Auto-Detecting Hardware..."

# 1. Detect CUDA (NVIDIA)
if command -v nvidia-smi &> /dev/null || [ -e /dev/nvidia0 ] || grep -qi nvidia /proc/driver/nvidia/version 2>/dev/null; then
    echo "✅ CUDA (NVIDIA GPU) detected!"
    COMPOSE_ARGS="$COMPOSE_ARGS -f docker-gpu.yml"

# 2. Detect MPS (Apple Silicon)
elif [[ $(uname -m) == 'arm64' && $(uname -s) == 'Darwin' ]]; then
    echo "✅ Apple Silicon (MPS) detected!"
    echo "⚠️  Note: Docker on Mac does not support native MPS passthrough yet."
    echo "   The containers will run on CPU. To utilize MPS fully, run 'python app/main.py' natively outside Docker."

# 3. Fallback to CPU
else
    echo "⚠️  No GPU detected. Using CPU."
fi

echo "🚀 Executing '$CMD' for $ENV environment..."
echo "============================================================"

# Execute the requested docker compose command
if [ "$CMD" = "up" ]; then
    docker compose $COMPOSE_ARGS up --build -d --remove-orphans
else
    # For down, logs, restart, build, etc.
    docker compose $COMPOSE_ARGS $CMD
fi
