#!/bin/bash
set -e

REPO_DIR="/home/ubuntu/LLM-API-Key-Proxy"
LOG_FILE="/home/ubuntu/sync_daemon.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

cd "$REPO_DIR" || exit 1

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

git fetch origin "$CURRENT_BRANCH"

CHANGED=$(git diff --name-only "origin/$CURRENT_BRANCH")

if echo "$CHANGED" | grep -q "config/"; then
    log "Config changes detected, pulling from $CURRENT_BRANCH..."
    git pull origin "$CURRENT_BRANCH" --no-rebase

    log "Gracefully reloading gateway..."
    pkill -f 'main.py' -HUP 2>/dev/null || log "No running gateway process found"

    sleep 2

    if pgrep -f 'main.py' > /dev/null; then
        log "Gateway reloaded successfully"
    else
        log "Gateway not running, starting..."
        cd "$REPO_DIR"
        source venv/bin/activate
        nohup python src/proxy_app/main.py --host 0.0.0.0 --port 8000 > ~/llm_proxy.log 2>&1 &
        log "Gateway started"
    fi
else
    log "No config changes detected"
fi
