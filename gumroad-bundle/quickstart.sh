#!/usr/bin/env bash
# Free LLM Gateway -- loopback-only installer
#
# Run from a source checkout with this bundle at gumroad-bundle/. The service
# binds only to 127.0.0.1. It never opens a firewall or public listener.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SERVICE_NAME="llm-gateway"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
PORT="${GATEWAY_PORT:-8000}"
HOST="127.0.0.1"
VENV_DIR="${REPO_DIR}/venv"
MEMORY_MAX_MB="${GATEWAY_MEMORY_MAX_MB:-400}"
ENV_FILE="${REPO_DIR}/.env"

log() { printf '[setup] %s\n' "$*"; }
die() { printf '[setup] ERROR: %s\n' "$*" >&2; exit 1; }
step() { printf '\n=== %s ===\n' "$*"; }

env_value() {
    local key="$1" line value=""
    while IFS= read -r line || [ -n "${line}" ]; do
        if [[ "${line}" == "${key}="* ]]; then
            value="${line#*=}"
        fi
    done < "${ENV_FILE}"
    value="${value#\"}"
    value="${value%\"}"
    printf '%s' "${value}"
}

step "Preflight"
[ -d "${REPO_DIR}/src/proxy_app" ] || die "Run from an LLM-API-Key-Proxy source checkout."
[ -f "${REPO_DIR}/requirements.txt" ] || die "requirements.txt is missing."
[[ "${PORT}" =~ ^[0-9]+$ ]] && [ "${PORT}" -gt 0 ] && [ "${PORT}" -lt 65536 ] \
    || die "GATEWAY_PORT must be an integer from 1 through 65535."

if [ -n "${SUDO_USER:-}" ] && [ "${SUDO_USER}" != "root" ]; then
    RUN_USER="${SUDO_USER}"
else
    RUN_USER="$(id -un)"
fi
[ "${RUN_USER}" != "root" ] || die "Run as a normal user; the script uses sudo when needed."
RUN_HOME="$(getent passwd "${RUN_USER}" | cut -d: -f6)"
sudo -v

step "Check configuration"
if [ ! -f "${ENV_FILE}" ]; then
    [ -f "${REPO_DIR}/gumroad-bundle/.env.starter" ] || die "Missing .env and starter template."
    cp "${REPO_DIR}/gumroad-bundle/.env.starter" "${ENV_FILE}"
    chmod 600 "${ENV_FILE}"
    die "Created ${ENV_FILE}. Generate a key with: openssl rand -hex 32; set PROXY_API_KEY, add provider credentials, then rerun."
fi
chmod 600 "${ENV_FILE}"
PROXY_KEY="$(env_value PROXY_API_KEY)"
case "${PROXY_KEY}" in
    ''|change-this*|YOUR_*|*TODO*) die "Set PROXY_API_KEY to a generated, non-placeholder value before starting." ;;
esac
[ "${#PROXY_KEY}" -ge 32 ] || die "PROXY_API_KEY must be at least 32 characters. Generate one with: openssl rand -hex 32"

step "Install uv and Python"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    source "${RUN_HOME}/.local/bin/env" 2>/dev/null || source "${RUN_HOME}/.cargo/env" 2>/dev/null || true
    export PATH="${RUN_HOME}/.local/bin:${PATH}"
fi
command -v uv >/dev/null 2>&1 || die "uv installation failed; install it manually and rerun."
uv python find 3.12 >/dev/null 2>&1 || uv python install 3.12

step "Install dependencies"
cd "${REPO_DIR}"
[ -d "${VENV_DIR}" ] || uv venv "${VENV_DIR}" --python 3.12
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
uv pip install -r requirements.txt

step "Install loopback-only service"
cat > "/tmp/${SERVICE_NAME}.service" <<SERVICEEOF
[Unit]
Description=LLM API Gateway Proxy
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${REPO_DIR}
EnvironmentFile=${ENV_FILE}
ExecStart=${VENV_DIR}/bin/python src/proxy_app/main.py --host ${HOST} --port ${PORT}
Restart=on-failure
RestartSec=10
NoNewPrivileges=true
MemoryMax=${MEMORY_MAX_MB}M

[Install]
WantedBy=multi-user.target
SERVICEEOF
sudo install -o root -g root -m 0644 "/tmp/${SERVICE_NAME}.service" "${SERVICE_FILE}"
rm -f "/tmp/${SERVICE_NAME}.service"
sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"
sudo systemctl restart "${SERVICE_NAME}"

step "Verify loopback service"
HEALTH_OK=false
for i in $(seq 1 12); do
    sleep 3
    if curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/health" >/dev/null; then
        HEALTH_OK=true
        break
    fi
done
[ "${HEALTH_OK}" = true ] || {
    sudo journalctl -u "${SERVICE_NAME}" -n 30 --no-pager || true
    die "Gateway did not become healthy."
}
LISTENERS="$(sudo ss -ltnH "sport = :${PORT}" || true)"
printf '%s\n' "${LISTENERS}" | grep -Fq "127.0.0.1:${PORT}" || die "Gateway is not listening on loopback only. Check the service definition."
case "${LISTENERS}" in
    *"0.0.0.0:${PORT}"*|*"[::]:${PORT}"*) die "Gateway has a non-loopback listener. Check the service definition." ;;
esac

step "Authenticated smoke test"
MODELS_RESP="$(curl -fsS --max-time 10 "http://127.0.0.1:${PORT}/v1/models" -H "Authorization: Bearer ${PROXY_KEY}" || true)"
if [ -n "${MODELS_RESP}" ]; then
    MODEL_COUNT="$(printf '%s' "${MODELS_RESP}" | grep -o '"id"' | wc -l | tr -d ' ')"
    log "Models endpoint responded; approximately ${MODEL_COUNT} models returned."
else
    log "Gateway is healthy but /v1/models did not return data. Check provider credentials and journalctl."
fi

cat <<DONEEOF

Gateway is running only at http://127.0.0.1:${PORT}/v1.

For personal remote access, use an SSH tunnel from your client:
  ssh -N -L ${PORT}:127.0.0.1:${PORT} ${RUN_USER}@YOUR_SERVER

Then point the client to http://127.0.0.1:${PORT}/v1.
For internet-facing access, configure an HTTPS reverse proxy on ports 80/443
that forwards to 127.0.0.1:${PORT}. Never expose TCP/${PORT} directly.

Logs:    sudo journalctl -u ${SERVICE_NAME} -f
Restart: sudo systemctl restart ${SERVICE_NAME}
Status:  sudo systemctl status ${SERVICE_NAME}
DONEEOF
