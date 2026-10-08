#!/bin/sh
set -eu

PREFIX=${1:-/opt/ai-workload-router}
SERVICE_DIR=/etc/systemd/system
ENV_DIR=/etc/ai-workload-router
DATA_DIR=/var/lib/ai-workload-router

mkdir -p "$PREFIX" "$ENV_DIR" "$DATA_DIR"
cp -R . "$PREFIX/"
install -m 0644 deploy/ai-workload-router.service "$SERVICE_DIR/ai-workload-router.service"
if [ ! -e "$ENV_DIR/router.env" ]; then
  install -m 0600 deploy/router.env.example "$ENV_DIR/router.env"
fi
systemctl daemon-reload
systemctl enable ai-workload-router.service
systemctl restart ai-workload-router.service
systemctl --no-pager --full status ai-workload-router.service
