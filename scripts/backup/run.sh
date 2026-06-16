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
PGHOST="${PGHOST:-}"
PGUSER="${PGUSER:-rag_user}"
PGDATABASE="${PGDATABASE:-rag_db}"

mkdir -p "${BACKUP_DIR}/postgres" "${BACKUP_DIR}/neo4j" "${BACKUP_DIR}/logs"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${BACKUP_DIR}/logs/backup.log"
}

# --- PostgreSQL Backup --------------------------------------------------------
backup_postgres() {
  if [ -z "$PGHOST" ] || [ "$PGHOST" = "" ]; then
    log "PGHOST not set. Skipping PostgreSQL backup."
    return
  fi

  local filename="rag_pg_${TIMESTAMP}.dump"
  log "Starting PostgreSQL backup: ${filename}"

  PGPASSWORD="${PGPASSWORD}" pg_dump \
    -h "${PGHOST}" \
    -U "${PGUSER}" \
    -d "${PGDATABASE}" \
    -F c \
    -Z 9 \
    -f "${BACKUP_DIR}/postgres/${filename}" \
    -v 2>> "${BACKUP_DIR}/logs/pg_dump.log" || true

  if [ -f "${BACKUP_DIR}/postgres/${filename}" ]; then
    local size=$(du -h "${BACKUP_DIR}/postgres/${filename}" | cut -f1)
    log "PostgreSQL backup complete: ${filename} (${size})"
  else
    log "PostgreSQL backup failed."
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

  local pg_count=$(find "${BACKUP_DIR}/postgres" -name "rag_pg_*.dump" -type f -mtime +${RETENTION_DAYS} 2>/dev/null | wc -l)
  find "${BACKUP_DIR}/postgres" -name "rag_pg_*.dump" -type f -mtime +${RETENTION_DAYS} -delete 2>/dev/null || true
  log "Deleted ${pg_count} old PostgreSQL backup(s)"

  local neo4j_count=$(find "${BACKUP_DIR}/neo4j" -name "rag_neo4j_*.dump" -type f -mtime +${RETENTION_DAYS} | wc -l)
  find "${BACKUP_DIR}/neo4j" -name "rag_neo4j_*.dump" -type f -mtime +${RETENTION_DAYS} -delete
  log "Deleted ${neo4j_count} old Neo4j backup(s)"
}

# --- Main ---------------------------------------------------------------------
log "=== Backup Run Started ==="

backup_postgres
backup_neo4j
cleanup_old_backups

log "=== Backup Run Complete ==="
echo ""
