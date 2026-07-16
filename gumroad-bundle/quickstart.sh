#!/usr/bin/env bash
# =============================================================================
# Free LLM Gateway -- one-click setup script
# Bundle: gumroad-bundle/quickstart.sh
#
# Run this ON your fresh Oracle Cloud (or any Ubuntu/Debian) VPS, as a normal
# user with sudo. It:
#   1. Installs uv + Python 3.12 if missing
#   2. Creates a venv and installs gateway deps
#   3. Writes a systemd service (MemoryMax=400M, auto-restart, port 8000)
#   4. Starts the service
#   5. Runs a smoke test (curl /v1/models)
#   6. Prints the gateway URL and next steps
#
# Idempotent: safe to re-run. Does not overwrite an existing .env.
#
# Usage:  bash gumroad-bundle/quickstart.sh
# =============================================================================
set -euo pipefail

# ---- config -----------------------------------------------------------------
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SERVICE_NAME="llm-gateway"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
PORT="${GATEWAY_PORT:-8000}"
HOST="0.0.0.0"
LOG_FILE="/var/log/${SERVICE_NAME}.log"
VENV_DIR="${REPO_DIR}/venv"
MEMORY_MAX_MB="${GATEWAY_MEMORY_MAX_MB:-400}"

# ---- helpers ----------------------------------------------------------------
log()  { printf '[setup] %s\n' "$*"; }
err()  { printf '[setup] ERROR: %s\n' "$*" >&2; }
die()  { err "$*"; exit 1; }
step() { printf '\n=== %s ===\n' "$*"; }

# ---- preflight --------------------------------------------------------------
step "Preflight checks"
[ -d "${REPO_DIR}/src/proxy_app" ] || die "Repo not found at ${REPO_DIR}. Run from inside the cloned LLM-API-Key-Proxy directory."
[ -f "${REPO_DIR}/requirements.txt" ] || die "requirements.txt missing. Is this the right repo?"

# Determine the service user. On Oracle Ubuntu images it's 'ubuntu'; on
# others it may be the current user. We use SUDO_USER if invoked via sudo,
# otherwise the current user.
if [ -n "${SUDO_USER:-}" ] && [ "${SUDO_USER}" != "root" ]; then
    RUN_USER="${SUDO_USER}"
else
    RUN_USER="$(id -un)"
fi
[ "${RUN_USER}" = "root" ] && die "Don't run this as root directly. Run as a normal user (the script will sudo when needed)."
RUN_HOME="$(getent passwd "${RUN_USER}" | cut -d: -f6)"
log "Repo:      ${REPO_DIR}"
log "Service:   ${SERVICE_NAME} (port ${PORT}, user ${RUN_USER})"
log "Memory cap: ${MEMORY_MAX_MB} MB"

# Require sudo access for systemd install
if ! sudo -n true 2>/dev/null && [ ! -w /etc/systemd/system ]; then
    die "This script needs sudo to install a systemd service. Re-run with: sudo -E bash gumroad-bundle/quickstart.sh  (or just provide your password when prompted)."
fi

# ---- .env check -------------------------------------------------------------
step "Check .env"
if [ ! -f "${REPO_DIR}/.env" ]; then
    if [ -f "${REPO_DIR}/gumroad-bundle/.env.starter" ]; then
        log "No .env found. Copying starter template..."
        cp "${REPO_DIR}/gumroad-bundle/.env.starter" "${REPO_DIR}/.env"
        chmod 600 "${REPO_DIR}/.env"
        log "Copied. Now edit it and fill in your free keys:"
        log "  nano ${REPO_DIR}/.env"
        log "Then re-run this script. Aborting so you can fill in keys first."
        exit 0
    else
        die "No .env and no gumroad-bundle/.env.starter found. Create .env manually (see SETUP_GUIDE.md)."
    fi
fi

# Warn if .env still has TODO placeholders for the two essential providers
if grep -qE 'GROQ_API_KEY_1="(# TODO|YOUR_)' "${REPO_DIR}/.env" 2>/dev/null \
   && grep -qE 'GEMINI_API_KEY_1="(# TODO|YOUR_)' "${REPO_DIR}/.env" 2>/dev/null; then
    log "WARNING: GROQ_API_KEY_1 and GEMINI_API_KEY_1 still look like placeholders."
    log "The gateway will start but may have no working providers. Edit ${REPO_DIR}/.env"
    log "and fill in at least one real key. Continuing anyway (you can restart later)..."
fi

# ---- install uv + python ----------------------------------------------------
step "Install uv and Python"
if ! command -v uv >/dev/null 2>&1; then
    log "Installing uv (fast Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    source "${RUN_HOME}/.local/bin/env" 2>/dev/null || source "${RUN_HOME}/.cargo/env" 2>/dev/null || true
    export PATH="${RUN_HOME}/.local/bin:${PATH}"
fi
if ! command -v uv >/dev/null 2>&1; then
    die "uv install failed. Install manually: https://docs.astral.sh/uv/"
fi
log "uv: $(uv --version)"

# Ensure python 3.12 via uv (does not touch system python)
if ! uv python find 3.12 >/dev/null 2>&1; then
    log "Installing Python 3.12 via uv..."
    uv python install 3.12
fi

# ---- venv + deps ------------------------------------------------------------
step "Create venv and install dependencies"
cd "${REPO_DIR}"
if [ ! -d "${VENV_DIR}" ]; then
    log "Creating venv at ${VENV_DIR}..."
    uv venv "${VENV_DIR}" --python 3.12
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
log "Installing requirements (this takes a few minutes on first run)..."
uv pip install -r requirements.txt

# ---- systemd service --------------------------------------------------------
step "Install systemd service"
# ponytail: hardcoded user/home paths because systemd does not expand shell
# vars in ExecStart reliably; we template them in below.
cat > "/tmp/${SERVICE_NAME}.service" <<SERVICEEOF
[Unit]
Description=LLM API Gateway Proxy (free multi-provider)
After=network.target

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${REPO_DIR}
EnvironmentFile=${REPO_DIR}/.env
ExecStart=${VENV_DIR}/bin/python src/proxy_app/main.py --host ${HOST} --port ${PORT}
Restart=on-failure
RestartSec=10
StandardOutput=append:${LOG_FILE}
StandardError=append:${LOG_FILE}
# Memory cap: keeps the gateway from OOMing a 1 GB VPS. Raise to 800M on the
# 6 GB ARM shape if you have headroom.
MemoryMax=${MEMORY_MAX_MB}M

[Install]
WantedBy=multi-user.target
SERVICEEOF

sudo cp "/tmp/${SERVICE_NAME}.service" "${SERVICE_FILE}"
sudo chown root:root "${SERVICE_FILE}"
sudo chmod 644 "${SERVICE_FILE}"
rm -f "/tmp/${SERVICE_NAME}.service"

# Make sure the log file is writable by the service user
sudo touch "${LOG_FILE}"
sudo chown "${RUN_USER}":"$(id -gn "${RUN_USER}")" "${LOG_FILE}"

sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"

# ---- start / restart --------------------------------------------------------
step "Start the gateway"
sudo systemctl restart "${SERVICE_NAME}"

# ---- wait for health --------------------------------------------------------
step "Wait for health"
HEALTH_OK=false
for i in $(seq 1 12); do
    sleep 3
    if curl -sf --max-time 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        HEALTH_OK=true
        log "Health check passed (attempt ${i})."
        break
    fi
    log "Waiting for gateway... (attempt ${i}/12)"
done

if [ "${HEALTH_OK}" != "true" ]; then
    err "Gateway did not become healthy in time. Recent logs:"
    sudo journalctl -u "${SERVICE_NAME}" -n 30 --no-pager || true
    die "See SETUP_GUIDE.md -> Troubleshooting. Common causes: bad provider key (still starts, check /v1/models), port in use, OOM (lower MemoryMax)."
fi

# ---- smoke test -------------------------------------------------------------
step "Smoke test: list models"
PROXY_KEY="$(grep -E '^PROXY_API_KEY=' "${REPO_DIR}/.env" | head -1 | sed -E 's/^PROXY_API_KEY="?([^"]*)"?$/\1/')"
MODELS_RESP=""
if [ -n "${PROXY_KEY}" ]; then
    MODELS_RESP="$(curl -sf --max-time 10 "http://127.0.0.1:${PORT}/v1/models" -H "Authorization: Bearer ${PROXY_KEY}" 2>/dev/null || true)"
else
    MODELS_RESP="$(curl -sf --max-time 10 "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null || true)"
fi

if [ -n "${MODELS_RESP}" ]; then
    MODEL_COUNT="$(printf '%s' "${MODELS_RESP}" | grep -o '"id"' | wc -l | tr -d ' ')"
    log "Models endpoint responded. ~${MODEL_COUNT} models available."
else
    log "WARNING: /v1/models returned empty or failed. The service is up (/health ok) but"
    log "no providers may be configured. Check your .env keys and restart:"
    log "  sudo systemctl restart ${SERVICE_NAME}"
fi

# ---- firewall reminder ------------------------------------------------------
step "Firewall reminder"
PUBLIC_IP="$(curl -sf --max-time 5 https://ifconfig.me 2>/dev/null || true)"
if [ -n "${PUBLIC_IP}" ]; then
    log "Your VPS public IP appears to be: ${PUBLIC_IP}"
fi
log "Port ${PORT} must be open in the Oracle Cloud security list (Ingress, TCP,"
log "source = your IP/32 or a Tailscale range). See SETUP_GUIDE.md Step 1.5."
log "Do NOT open it to 0.0.0.0/0 unless you accept that the gateway is reachable"
log "by anyone (even with a PROXY_API_KEY, that is not recommended)."

# ---- done -------------------------------------------------------------------
step "Done"
cat <<DONEEOF

============================================================
  Your free LLM gateway is live.

  Local URL:   http://127.0.0.1:${PORT}/v1
  Public URL:  http://${PUBLIC_IP:-YOUR_VPS_IP}:${PORT}/v1
  API key:     the PROXY_API_KEY you set in ${REPO_DIR}/.env
  Logs:        sudo journalctl -u ${SERVICE_NAME} -f
               (or: tail -f ${LOG_FILE})
  Restart:     sudo systemctl restart ${SERVICE_NAME}
  Status:      sudo systemctl status ${SERVICE_NAME}

  Next steps:
    1. Open port ${PORT} in the Oracle security list (see SETUP_GUIDE.md 1.5)
    2. Point your tools at the Public URL above (see SETUP_GUIDE.md Step 5)
    3. Test a chat completion:
       curl -X POST http://${PUBLIC_IP:-YOUR_VPS_IP}:${PORT}/v1/chat/completions \\
         -H "Content-Type: application/json" \\
         -H "Authorization: Bearer YOUR_PROXY_API_KEY" \\
         -d '{"model":"coding-elite","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
============================================================
DONEEOF
