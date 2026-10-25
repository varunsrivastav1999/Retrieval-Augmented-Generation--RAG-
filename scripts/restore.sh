#!/usr/bin/env sh
# Restore a backup made by scripts/backup.sh:   scripts/restore.sh backups/<timestamp>
set -eu
cd "$(dirname "$0")/.."
DIR="${1:?usage: scripts/restore.sh backups/<timestamp>}"
DC="docker compose"
COLLECTION="${RAG_QDRANT_COLLECTION:-rag_chunks}"
[ -f "$DIR/postgres.dump" ] && [ -f "$DIR/qdrant.snapshot" ] || { echo "missing files in $DIR"; exit 1; }

echo "Stopping api/worker…"
$DC stop api worker
echo "Restoring PostgreSQL…"
$DC exec -T postgres sh -c 'pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" --clean --if-exists --no-owner' < "$DIR/postgres.dump"
echo "Restoring Qdrant collection $COLLECTION…"
$DC cp "$DIR/qdrant.snapshot" qdrant:/qdrant/storage/restore.snapshot
$DC run --rm --no-deps -T --entrypoint python api -c "
import json, urllib.request
req = urllib.request.Request('http://qdrant:6333/collections/$COLLECTION/snapshots/recover',
    data=json.dumps({'location': 'file:///qdrant/storage/restore.snapshot', 'priority': 'snapshot'}).encode(),
    headers={'Content-Type': 'application/json'}, method='PUT')
print(urllib.request.urlopen(req, timeout=1800).read().decode())"
$DC exec -T qdrant rm -f /qdrant/storage/restore.snapshot || true
$DC start api worker
echo "Restore complete."
