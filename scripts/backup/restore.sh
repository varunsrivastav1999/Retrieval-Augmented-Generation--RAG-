#!/bin/sh
# =============================================================================
# Enterprise Level RAG — Restore from Backup
# =============================================================================
# Usage:
#   ./scripts/backup/restore.sh [pg|neo4j|all] <backup_file>
#
# Examples:
#   ./scripts/backup/restore.sh pg /backup/postgres/rag_pg_20250101_020000.dump
#   ./scripts/backup/restore.sh neo4j /backup/neo4j/rag_neo4j_20250101_020000.dump
#   ./scripts/backup/restore.sh all
# =============================================================================

set -e

BACKUP_DIR="${BACKUP_DIR:-/backup}"
PGHOST="${PGHOST:-postgres}"
PGUSER="${PGUSER:-rag_user}"
PGDATABASE="${PGDATABASE:-rag_db}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

restore_postgres() {
  local file="$1"
  if [ -z "$file" ]; then
    file=$(ls -t "${BACKUP_DIR}/postgres"/rag_pg_*.dump 2>/dev/null | head -1)
  fi
  if [ -z "$file" ] || [ ! -f "$file" ]; then
    log "ERROR: No PostgreSQL backup file found at ${file:-$BACKUP_DIR/postgres/}"
    exit 1
  fi

  log "Restoring PostgreSQL from: $file"
  PGPASSWORD="${PGPASSWORD}" pg_restore \
    -h "${PGHOST}" \
    -U "${PGUSER}" \
    -d "${PGDATABASE}" \
    --clean \
    --if-exists \
    --jobs=4 \
    -v "$file" 2>&1 | tail -20
  log "PostgreSQL restore complete."
}

restore_neo4j() {
  local file="$1"
  if [ -z "$file" ]; then
    file=$(ls -t "${BACKUP_DIR}/neo4j"/rag_neo4j_*.dump 2>/dev/null | head -1)
  fi
  if [ -z "$file" ] || [ ! -f "$file" ]; then
    log "ERROR: No Neo4j backup file found at ${file:-$BACKUP_DIR/neo4j/}"
    exit 1
  fi

  local dbname=$(basename "$file" .dump)
  log "Restoring Neo4j from: $file"
  neo4j-admin database load --from-path="$file" --database=neo4j
  log "Neo4j restore complete. Restart Neo4j to apply."
}

case "${1:-all}" in
  pg|postgres)
    restore_postgres "$2"
    ;;
  neo4j)
    restore_neo4j "$2"
    ;;
  all|*)
    restore_postgres "$2"
    restore_neo4j "$3"
    ;;
esac

log "=== Restore Complete ==="
