#!/bin/sh
set -eu

PREFIX=${1:-/opt/ai-workload-router}
systemctl stop ai-workload-router.service || true
if [ -d "$PREFIX.previous" ]; then
  rm -rf "$PREFIX.rollback-current"
  mv "$PREFIX" "$PREFIX.rollback-current"
  mv "$PREFIX.previous" "$PREFIX"
fi
systemctl start ai-workload-router.service
systemctl --no-pager --full status ai-workload-router.service
