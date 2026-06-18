#!/usr/bin/env bash
# telemetry-rotate.sh - Weekly rotation on gateway-40: VACUUM, archive, delete old, checkpoint.
# Run via cron weekly: 0 3 * * 0 /usr/local/bin/telemetry-rotate.sh
set -euo pipefail

DB="${TELEMETRY_DB:-/dev/shm/telemetry.db}"
ARCHIVE_DIR="${TELEMETRY_ARCHIVE_DIR:-/tmp/telemetry-archives}"
RETENTION_DAYS=30

mkdir -p "$ARCHIVE_DIR"

TIMESTAMP=$(date +%Y%m%d)
ARCHIVE_FILE="${ARCHIVE_DIR}/telemetry-${TIMESTAMP}.sql.gz"

echo "[$(date -Iseconds)] starting weekly rotation"

# Archive
echo "[$(date -Iseconds)] archiving to $ARCHIVE_FILE"
sqlite3 "$DB" ".dump" | gzip > "$ARCHIVE_FILE"

# Delete old rows
echo "[$(date -Iseconds)] deleting rows older than ${RETENTION_DAYS} days"
CUTOFF=$(date -d "-${RETENTION_DAYS} days" +%s)
sqlite3 "$DB" "DELETE FROM llm_events WHERE ts_start < ${CUTOFF};"

# VACUUM + checkpoint
echo "[$(date -Iseconds)] VACUUM"
sqlite3 "$DB" "VACUUM;"

echo "[$(date -Iseconds)] wal_checkpoint TRUNCATE"
sqlite3 "$DB" "PRAGMA wal_checkpoint(TRUNCATE);"

# Rotate archives: keep last 4
ls -1t "${ARCHIVE_DIR}/telemetry-"*.sql.gz 2>/dev/null | tail -n +5 | while read -r old; do
  rm -f "$old"
  echo "[$(date -Iseconds)] rotated archive: $old"
done

# Size check
SIZE=$(stat -c%s "$DB" 2>/dev/null || stat -f%z "$DB" 2>/dev/null || echo 0)
SIZE_MB=$((SIZE / 1048576))
echo "[$(date -Iseconds)] done. DB size: ${SIZE_MB}MB"

# Alert if over 100MB
if [ "$SIZE_MB" -gt 100 ]; then
  echo "[$(date -Iseconds)] WARN DB over 100MB cap" >&2
fi
