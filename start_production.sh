#!/bin/bash
# =============================================================================
# i-Tips RAG — Production Start Script with Auto-Detect GPU
# =============================================================================

set -e

echo "Checking for NVIDIA GPU..."

if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected! Starting containers with GPU acceleration..."
    COMPOSE_ARGS="-f production.yml -f production.gpu.yml"
else
    echo "⚠️ No NVIDIA GPU detected. Starting containers in CPU-only mode..."
    COMPOSE_ARGS="-f production.yml"
fi

docker compose $COMPOSE_ARGS up -d "$@"

echo "Waiting for RAG API readiness..."
for i in $(seq 1 60); do
    if curl -fsS http://localhost:1000/health/ready >/dev/null 2>&1; then
        echo "✅ RAG API is ready."
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "❌ RAG API did not become ready within 5 minutes."
        echo "Run: docker compose $COMPOSE_ARGS logs --tail=200 rag_api"
        exit 1
    fi
    sleep 5
done

echo "========================================================================"
echo "Startup complete. Run 'docker compose -f production.yml logs -f' to view logs."
