#!/bin/sh
# =============================================================================
# Enterprise Level RAG — Restore from Backup
# =============================================================================
# Usage:
#   ./scripts/backup/restore.sh [sqlite|neo4j|all] <backup_file>
#
# Examples:
#   ./scripts/backup/restore.sh sqlite /backup/sqlite/rag_sqlite_20250101_020000.db
#   ./scripts/backup/restore.sh neo4j /backup/neo4j/rag_neo4j_20250101_020000.dump
#   ./scripts/backup/restore.sh all
# =============================================================================

set -e

BACKUP_DIR="${BACKUP_DIR:-/backup}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

restore_sqlite() {
  local file="$1"
  if [ -z "$file" ]; then
    file=$(ls -t "${BACKUP_DIR}/sqlite"/rag_sqlite_*.db 2>/dev/null | head -1)
  fi
  if [ -z "$file" ] || [ ! -f "$file" ]; then
    log "ERROR: No SQLite backup file found at ${file:-$BACKUP_DIR/sqlite/}"
    exit 1
  fi

  log "Restoring SQLite from: $file"
  cp "$file" "/app/rag_db.sqlite3"
  log "SQLite restore complete."
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
  sqlite)
    restore_sqlite "$2"
    ;;
  neo4j)
    restore_neo4j "$2"
    ;;
  all|*)
    restore_sqlite "$2"
    restore_neo4j "$3"
    ;;
esac

log "=== Restore Complete ==="
