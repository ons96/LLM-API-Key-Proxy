#!/usr/bin/env bash
set -euo pipefail

# VPS auto-deploy script with rollback on failure
# Usage: ./scripts/deploy.sh [tag]
#   tag: optional git tag/branch to deploy (default: main)

DEPLOY_TARGET="${1:-main}"
APP_DIR="/home/ubuntu/LLM-API-Key-Proxy"
VENV_DIR="$APP_DIR/venv"
BACKUP_DIR="/home/ubuntu/backups"
HEALTH_URL="http://127.0.0.1:8000/health"
HEALTH_TIMEOUT=30
HEALTH_RETRIES=6
LOG_FILE="/home/ubuntu/llm_proxy.log"
PID_FILE="/tmp/llm_proxy.pid"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die() { log "FATAL: $*"; exit 1; }

get_pid() {
    pgrep -f "python.*src/proxy_app/main.py" 2>/dev/null || true
}

stop_server() {
    local pid
    pid=$(get_pid)
    if [ -n "$pid" ]; then
        log "Stopping server (PID: $pid)..."
        kill "$pid" 2>/dev/null || true
        for i in $(seq 1 10); do
            if ! kill -0 "$pid" 2>/dev/null; then
                log "Server stopped."
                return 0
            fi
            sleep 1
        done
        log "Force killing..."
        kill -9 "$pid" 2>/dev/null || true
        sleep 1
    fi
}

start_server() {
    log "Starting server..."
    cd "$APP_DIR"
    source "$VENV_DIR/bin/activate"
    nohup python src/proxy_app/main.py --host 0.0.0.0 --port 8000 > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    log "Server started (PID: $!)"
}

wait_for_health() {
    log "Waiting for health check..."
    for i in $(seq 1 $HEALTH_RETRIES); do
        sleep 5
        if curl -sf --max-time "$HEALTH_TIMEOUT" "$HEALTH_URL" > /dev/null 2>&1; then
            log "Health check passed!"
            return 0
        fi
        log "Health check attempt $i/$HEALTH_RETRIES failed, retrying..."
    done
    log "Health check failed after $HEALTH_RETRIES attempts"
    return 1
}

create_backup() {
    mkdir -p "$BACKUP_DIR"
    local backup_name="backup-$(date '+%Y%m%d-%H%M%S')"
    local backup_path="$BACKUP_DIR/$backup_name"

    log "Creating backup at $backup_path..."
    cd "$APP_DIR"
    local current_commit
    current_commit=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
    echo "$current_commit" > "$backup_path.commit"

    pip freeze > "$backup_path.requirements" 2>/dev/null || true

    echo "$backup_path"
}

rollback() {
    local commit_file
    commit_file=$(ls -t "$BACKUP_DIR"/*.commit 2>/dev/null | head -1)

    if [ -z "$commit_file" ]; then
        die "No backup found to rollback to!"
    fi

    local old_commit
    old_commit=$(cat "$commit_file")
    log "Rolling back to commit $old_commit..."

    cd "$APP_DIR"
    git checkout "$old_commit" -- .
    source "$VENV_DIR/bin/activate"
    pip install -r requirements.txt --quiet 2>/dev/null || true

    stop_server
    start_server

    if wait_for_health; then
        log "Rollback successful!"
    else
        die "Rollback failed! Manual intervention required."
    fi
}

deploy() {
    log "=== Starting deployment: $DEPLOY_TARGET ==="

    cd "$APP_DIR" || die "App directory not found: $APP_DIR"

    create_backup

    log "Fetching latest changes..."
    git fetch origin || die "git fetch failed"

    if [[ "$DEPLOY_TARGET" == v* ]]; then
        git checkout "tags/$DEPLOY_TARGET" || die "Failed to checkout tag $DEPLOY_TARGET"
    else
        git checkout "$DEPLOY_TARGET" || die "Failed to checkout $DEPLOY_TARGET"
        git pull origin "$DEPLOY_TARGET" || die "git pull failed"
    fi

    log "Installing dependencies..."
    source "$VENV_DIR/bin/activate"
    pip install -r requirements.txt --quiet 2>/dev/null || log "pip install had warnings (non-fatal)"

    stop_server
    start_server

    if wait_for_health; then
        log "=== Deployment successful! ==="
    else
        log "=== Deployment failed, initiating rollback ==="
        rollback
        exit 1
    fi
}

case "${1:-deploy}" in
    rollback)
        rollback
        ;;
    *)
        deploy
        ;;
esac
