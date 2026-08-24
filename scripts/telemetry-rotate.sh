#!/usr/bin/env bash
# telemetry-rotate.sh - Weekly rotation on gateway-40: archive, delete old, VACUUM, checkpoint.
# Run via cron weekly: 0 3 * * 0 TELEMETRY_DB=/home/ubuntu/gateway-data/telemetry.db scripts/telemetry-rotate.sh
#
# 2026-08-24 (#761): sqlite3 CLI is not installed on gateway-40 and system
# installs need Owen. All DB ops moved to the python3 stdlib sqlite3 module -
# no system packages required. Dump format matches `sqlite3 .dump`.
set -euo pipefail

DB="${TELEMETRY_DB:-/dev/shm/telemetry.db}"
ARCHIVE_DIR="${TELEMETRY_ARCHIVE_DIR:-/tmp/telemetry-archives}"
RETENTION_DAYS=30

mkdir -p "$ARCHIVE_DIR"

TIMESTAMP=$(date +%Y%m%d)
ARCHIVE_FILE="${ARCHIVE_DIR}/telemetry-${TIMESTAMP}.sql.gz"

echo "[$(date -Iseconds)] starting weekly rotation"

echo "[$(date -Iseconds)] archiving to $ARCHIVE_FILE"
python3 - "$DB" "$ARCHIVE_FILE" "$RETENTION_DAYS" <<'PY'
import gzip
import sqlite3
import sys
import time

db_path, archive_path, retention_days = sys.argv[1], sys.argv[2], int(sys.argv[3])
conn = sqlite3.connect(db_path, timeout=30)
try:
    cutoff = int(time.time()) - retention_days * 86400
    with gzip.open(archive_path, "wt", encoding="utf-8") as gz:
        for line in conn.iterdump():
            gz.write(line + "\n")
    print(f"archived {archive_path}")
    cur = conn.execute("DELETE FROM llm_events WHERE ts_start < ?", (cutoff,))
    conn.commit()
    print(f"deleted {cur.rowcount} rows older than {retention_days} days")
    conn.execute("VACUUM")
    res = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    print(f"wal_checkpoint(TRUNCATE): {res}")
finally:
    conn.close()
PY

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
