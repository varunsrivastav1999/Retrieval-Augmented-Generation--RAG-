#!/bin/bash
# =============================================================================
# i-Tips RAG — Production Start Script with Auto-Detect GPU
# =============================================================================

echo "Checking for NVIDIA GPU..."

if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected! Starting containers with GPU acceleration..."
    docker compose -f production.yml -f production.gpu.yml up -d "$@"
else
    echo "⚠️ No NVIDIA GPU detected. Starting containers in CPU-only mode..."
    docker compose -f production.yml up -d "$@"
fi

echo "========================================================================"
echo "Startup complete. Run 'docker compose -f production.yml logs -f' to view logs."
