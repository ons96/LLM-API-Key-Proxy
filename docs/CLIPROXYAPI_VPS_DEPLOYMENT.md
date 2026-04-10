# CLIProxyAPI VPS Deployment Guide

## Quick Start

You've successfully merged and pushed the CLIProxyAPI integration. Now deploy to your Oracle Micro VPS.

---

## Step 1: SSH into VPS

```bash
ssh ubuntu@your-vps-ip
```

---

## Step 2: Pull Latest Code

```bash
cd /opt/LLM-API-Key-Proxy  # or wherever your gateway is
git pull origin main
```

---

## Step 3: Install CLIProxyAPI

```bash
# Download installation script
wget https://raw.githubusercontent.com/ons96/LLM-API-Key-Proxy/main/deploy/cliproxyapi/install_cliproxyapi.sh
chmod +x install_cliproxyapi.sh

# Run installer (creates user, directories, systemd service)
sudo ./install_cliproxyapi.sh
```

---

## Step 4: Set Up Providers

**IMPORTANT: Use cookie method for iFlow!**

```bash
# Gemini CLI (OAuth - auto-refresh works)
sudo -u cliproxyapi /opt/cliproxyapi/cliproxyapi -gemini-login

# iFlow - USE COOKIE METHOD (NOT OAuth!)
# OAuth expires in 7 days, cookie lasts months
sudo -u cliproxyapi /opt/cliproxyapi/cliproxyapi -iflow-cookie

# Antigravity (OAuth - auto-refresh works)
sudo -u cliproxyapi /opt/cliproxyapi/cliproxyapi -antigravity-login

# Qwen Code (OAuth - auto-refresh works)
sudo -u cliproxyapi /opt/cliproxyapi/cliproxyapi -qwen-login
```

**Why cookie for iFlow?**
- OAuth: Expires every 7 days with NO auto-refresh
- Cookie: Lasts months, auto-refreshes automatically

---

## Step 5: Start Service

```bash
sudo systemctl start cliproxyapi
sudo systemctl status cliproxyapi
```

---

## Step 6: Verify Installation

```bash
# Check health
curl http://127.0.0.1:8317/health

# Check available models
curl http://127.0.0.1:8317/v1/models

# Test chat completion
curl http://127.0.0.1:8317/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemini/gemini-2.5-flash","messages":[{"role":"user","content":"Hello"}]}'
```

---

## Step 7: Configure Gateway

Add to your gateway's `.env`:

```bash
# CLIProxyAPI Configuration
CLIPROXYAPI_ENABLED=true
CLIPROXYAPI_BASE_URL=http://127.0.0.1:8317
CLIPROXYAPI_TIMEOUT=120
PROVIDER_PRIORITY_CLIPROXYAPI=1
```

Restart your gateway to pick up the new provider.

---

## Step 8: Test Integration

From your local machine:

```bash
# Copy test script to VPS
scp deploy/cliproxyapi/test_integration.sh ubuntu@your-vps-ip:/tmp/

# Run tests
ssh ubuntu@your-vps-ip "chmod +x /tmp/test_integration.sh && /tmp/test_integration.sh"
```

---

## Available Models

| Backend | Prefix | Models |
|---------|--------|--------|
| Gemini | `gemini/` | gemini-2.5-pro, gemini-2.5-flash |
| iFlow | `iflow/` | glm-4-plus, glm-4-flash, glm-4-air |
| Antigravity | `antigravity/` | gemini-2.0-flash-exp, gemini-3-flash |
| Qwen | `qwen/` | qwen3-coder-plus, qwen3-coder-lite |

---

## Troubleshooting

### Service won't start

```bash
# Check logs
sudo journalctl -u cliproxyapi -n 50

# Run manually for debugging
sudo -u cliproxyapi /opt/cliproxyapi/cliproxyapi --config /opt/cliproxyapi/config/config.yaml
```

### No models available

```bash
# Check auth files
ls -la /opt/cliproxyapi/auth/

# Re-authenticate
sudo -u cliproxyapi /opt/cliproxyapi/cliproxyapi -gemini-login
sudo -u cliproxyapi /opt/cliproxyapi/cliproxyapi -iflow-cookie
```

### iFlow token expired

```bash
# Re-auth with cookie method
sudo -u cliproxyapi /opt/cliproxyapi/cliproxyapi -iflow-cookie

# Restart service
sudo systemctl restart cliproxyapi
```

---

## Memory Usage

| Service | Memory |
|---------|--------|
| CLIProxyAPI | ~50MB |
| Gateway | ~100MB |
| **Total** | ~150MB |

Fits comfortably on 1GB VPS with ZeroClaw and other services.

---

## API Endpoints (Gateway)

After integration, these endpoints are available:

```
GET /api/cliproxyapi/health      - Sidecar health
GET /api/cliproxyapi/status      - Detailed status
GET /api/cliproxyapi/models      - Available models
GET /api/cliproxyapi/backend/{n} - Backend status
GET /api/cliproxyapi/config      - Configuration
```

---

## Success!

You now have:
- ✅ CLIProxyAPI sidecar running on port 8317
- ✅ Auto token refresh for Gemini, Antigravity, Qwen
- ✅ Cookie-based auth for iFlow (auto-refresh)
- ✅ Integration with your LLM gateway
- ✅ All 4 backends available through unified endpoint

**Remember**: Always use `./cliproxyapi -iflow-cookie` for iFlow!
