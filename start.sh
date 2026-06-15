#!/bin/bash

# ==============================================================================
# Enterprise RAG — Smart Start Script
# ==============================================================================
# Usage:
#   ./start.sh [local|production] [up|down|build|logs|restart|update...]
#
# SMART BUILD LOGIC:
#   - "up"     → Only rebuilds if Dockerfile/requirements.txt changed
#   - "update" → Fast code-only update: pulls git, restarts containers (NO rebuild)
#   - "build"  → Full image rebuild (use only when deps change)
#
# AUTO-CLEANUP:
#   - Prunes dangling images/build cache BEFORE building to prevent disk full
# ==============================================================================

set -e

ENV=${1:-local}
CMD=${2:-up}

if [[ "$ENV" != "local" && "$ENV" != "production" ]]; then
    echo "Usage: ./start.sh [local|production] [up|down|build|logs|restart|update...]"
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

echo "============================================================"

# ==============================================================================
# Helper: Auto-cleanup Docker to prevent disk full
# ==============================================================================
auto_cleanup() {
    echo "🧹 Auto-cleaning Docker disk space..."
    # Remove dangling images (old layers from previous builds)
    docker image prune -f 2>/dev/null || true
    # Remove unused build cache
    docker builder prune -f --filter "until=72h" 2>/dev/null || true
    # Remove stopped containers
    docker container prune -f 2>/dev/null || true
    echo "✅ Cleanup done."
}

# ==============================================================================
# Helper: Check if image rebuild is actually needed
# ==============================================================================
needs_rebuild() {
    local IMAGE_NAME="itips_rag_prod:latest"

    # If image doesn't exist at all, we need to build
    if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
        echo "🔨 Image not found. Full build required."
        return 0
    fi

    # Get image creation timestamp
    local IMAGE_DATE
    IMAGE_DATE=$(docker image inspect "$IMAGE_NAME" --format '{{.Created}}' 2>/dev/null || echo "")
    if [ -z "$IMAGE_DATE" ]; then
        return 0
    fi

    # Check if Dockerfile or requirements.txt changed since image was built
    # (These are the only files that require a full rebuild with model downloads)
    local IMAGE_EPOCH
    IMAGE_EPOCH=$(date -d "$IMAGE_DATE" +%s 2>/dev/null || date -j -f "%Y-%m-%dT%H:%M:%S" "${IMAGE_DATE%%.*}" +%s 2>/dev/null || echo "0")

    for check_file in Dockerfile requirements.txt; do
        if [ -f "$check_file" ]; then
            local FILE_EPOCH
            FILE_EPOCH=$(stat -c %Y "$check_file" 2>/dev/null || stat -f %m "$check_file" 2>/dev/null || echo "0")
            if [ "$FILE_EPOCH" -gt "$IMAGE_EPOCH" ]; then
                echo "🔨 $check_file changed since last build. Rebuild required."
                return 0
            fi
        fi
    done

    echo "✅ Image is up-to-date. Skipping rebuild (code changes use bind mount)."
    return 1
}

# ==============================================================================
# Execute the requested command
# ==============================================================================
echo "🚀 Executing '$CMD' for $ENV environment..."
echo "============================================================"

case "$CMD" in
    # --------------------------------------------------------------------------
    # UP: Smart start — only rebuild if Dockerfile/requirements changed
    # --------------------------------------------------------------------------
    up)
        if needs_rebuild; then
            auto_cleanup
            docker compose $COMPOSE_ARGS up --build -d --remove-orphans
        else
            # No rebuild needed — just start/restart containers
            # Code changes are picked up via bind mount in production.yml
            docker compose $COMPOSE_ARGS up -d --remove-orphans
        fi
        ;;

    # --------------------------------------------------------------------------
    # UPDATE: Fast code-only update — pull git + restart (ZERO rebuild)
    # --------------------------------------------------------------------------
    update)
        echo "📥 Pulling latest code..."
        git pull --ff-only 2>/dev/null || git pull
        echo "🔄 Restarting containers with new code..."
        docker compose $COMPOSE_ARGS restart rag_api
        echo "✅ Update complete! Code changes are live."
        ;;

    # --------------------------------------------------------------------------
    # BUILD: Explicit full rebuild (when you change Dockerfile/requirements.txt)
    # --------------------------------------------------------------------------
    build)
        auto_cleanup
        docker compose $COMPOSE_ARGS build --no-cache
        echo "✅ Full rebuild complete."
        ;;

    # --------------------------------------------------------------------------
    # CLEAN: Nuclear cleanup — remove ALL old images and build cache
    # --------------------------------------------------------------------------
    clean)
        echo "🗑️  Deep cleaning Docker..."
        docker compose $COMPOSE_ARGS down --remove-orphans 2>/dev/null || true
        docker image prune -af 2>/dev/null || true
        docker builder prune -af 2>/dev/null || true
        docker volume prune -f 2>/dev/null || true
        echo "✅ Deep clean complete. Next 'up' will do a full rebuild."
        ;;

    # --------------------------------------------------------------------------
    # DOWN, LOGS, RESTART, etc. — pass through to docker compose
    # --------------------------------------------------------------------------
    down|logs|restart|ps|exec|stop|start)
        docker compose $COMPOSE_ARGS $CMD ${@:3}
        ;;

    *)
        echo "❌ Unknown command: $CMD"
        echo "Available: up, update, build, clean, down, logs, restart, ps, stop, start"
        exit 1
        ;;
esac
