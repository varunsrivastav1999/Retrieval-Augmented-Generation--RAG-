#!/bin/sh
# =============================================================================
# Enterprise Level RAG — Automated Backup Script
# =============================================================================
# Backups:
#   1. SQLite backup (copy file)
#   2. Neo4j database dump (neo4j-admin)
#   3. Cleans backups older than RETENTION_DAYS
# =============================================================================

set -e

BACKUP_DIR="${BACKUP_DIR:-/backup}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "${BACKUP_DIR}/sqlite" "${BACKUP_DIR}/neo4j" "${BACKUP_DIR}/logs"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${BACKUP_DIR}/logs/backup.log"
}

# --- SQLite Backup --------------------------------------------------------
backup_sqlite() {
  local filename="rag_sqlite_${TIMESTAMP}.db"
  log "Starting SQLite backup: ${filename}"
  
  if [ -f "/app/rag_db.sqlite3" ]; then
    cp "/app/rag_db.sqlite3" "${BACKUP_DIR}/sqlite/${filename}"
    local size=$(du -h "${BACKUP_DIR}/sqlite/${filename}" | cut -f1)
    log "SQLite backup complete: ${filename} (${size})"
  else
    log "SQLite database file not found at /app/rag_db.sqlite3. Skipping."
  fi
}

# --- Neo4j Backup ------------------------------------------------------------
backup_neo4j() {
  local filename="rag_neo4j_${TIMESTAMP}"
  log "Starting Neo4j backup: ${filename}"

  neo4j-admin database dump \
    --to-path="${BACKUP_DIR}/neo4j" \
    --database=neo4j \
    "${filename}" 2>> "${BACKUP_DIR}/logs/neo4j_dump.log"

  local size=$(du -h "${BACKUP_DIR}/neo4j/${filename}.dump" | cut -f1)
  log "Neo4j backup complete: ${filename}.dump (${size})"
}

# --- Clean old backups --------------------------------------------------------
cleanup_old_backups() {
  log "Cleaning backups older than ${RETENTION_DAYS} days..."

  local sqlite_count=$(find "${BACKUP_DIR}/sqlite" -name "rag_sqlite_*.db" -type f -mtime +${RETENTION_DAYS} | wc -l)
  find "${BACKUP_DIR}/sqlite" -name "rag_sqlite_*.db" -type f -mtime +${RETENTION_DAYS} -delete
  log "Deleted ${sqlite_count} old SQLite backup(s)"

  local neo4j_count=$(find "${BACKUP_DIR}/neo4j" -name "rag_neo4j_*.dump" -type f -mtime +${RETENTION_DAYS} | wc -l)
  find "${BACKUP_DIR}/neo4j" -name "rag_neo4j_*.dump" -type f -mtime +${RETENTION_DAYS} -delete
  log "Deleted ${neo4j_count} old Neo4j backup(s)"
}

# --- Main ---------------------------------------------------------------------
log "=== Backup Run Started ==="

backup_sqlite
backup_neo4j
cleanup_old_backups

log "=== Backup Run Complete ==="
echo ""
