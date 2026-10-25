#!/usr/bin/env sh
# Back up the index and history of a running stack into ./backups/<timestamp>/
#   - PostgreSQL (documents, chunks, conversations, audit log): pg_dump custom format
#   - Qdrant (vector index): collection snapshot
# Media files live in the `media` volume (or your bind-mounted folder) — back that up with your normal file backup.
set -eu
cd "$(dirname "$0")/.."
TS=$(date +%Y%m%d_%H%M%S)
OUT="backups/$TS"
mkdir -p "$OUT"
DC="docker compose"
COLLECTION="${RAG_QDRANT_COLLECTION:-rag_chunks}"

echo "PostgreSQL dump -> $OUT/postgres.dump"
$DC exec -T postgres sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc' > "$OUT/postgres.dump"

echo "Qdrant snapshot -> $OUT/qdrant.snapshot"
SNAP=$($DC exec -T api python -c "
import json, urllib.request
r = urllib.request.urlopen(urllib.request.Request('http://qdrant:6333/collections/$COLLECTION/snapshots', method='POST'), timeout=600)
print(json.load(r)['result']['name'])")
$DC exec -T api python -c "
import shutil, sys, urllib.request
with urllib.request.urlopen('http://qdrant:6333/collections/$COLLECTION/snapshots/$SNAP', timeout=600) as r:
    shutil.copyfileobj(r, sys.stdout.buffer)" > "$OUT/qdrant.snapshot"
$DC exec -T api python -c "
import urllib.request
urllib.request.urlopen(urllib.request.Request('http://qdrant:6333/collections/$COLLECTION/snapshots/$SNAP', method='DELETE'), timeout=60)"
ls -lh "$OUT"
echo "Backup complete: $OUT"
