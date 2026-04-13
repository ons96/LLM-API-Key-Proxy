# CLIProxyAPI Provider

## Overview

CLIProxyAPI is a Go-based OAuth router that acts as a sidecar service for the LLM gateway. It handles:

- **OAuth token refresh** for Gemini CLI, Antigravity, Qwen Code
- **Cookie-based authentication** for iFlow (auto-refresh)
- **Multi-account load balancing**
- **Format translation** (OpenAI ↔ Claude ↔ Gemini)

## Why CLIProxyAPI?

### Problem
- OAuth tokens expire and need refresh
- iFlow OAuth tokens expire every 7 days with NO auto-refresh
- Multiple accounts per provider need load balancing
- Different providers use different API formats

### Solution
CLIProxyAPI runs as a sidecar and handles all OAuth/token management, providing a single OpenAI-compatible endpoint.

## Architecture

```
┌─────────────────┐      ┌─────────────────┐
│  LLM Gateway    │      │   CLIProxyAPI   │
│    (Python)     │──────│   (Go binary)   │
│                 │ HTTP │                 │
│  ~50-100MB RAM  │      │   ~30-50MB RAM  │
└─────────────────┘      └─────────────────┘
                                  │
                                  ▼
         ┌──────────────────────────────────────┐
         │         OAuth Providers              │
         │  Gemini │ iFlow │ Antigravity │ Qwen │
         └──────────────────────────────────────┘
```

## Installation

### On VPS (Oracle Micro - 1GB RAM)

```bash
# Download and run installer
cd /tmp
wget https://raw.githubusercontent.com/ons96/LLM-API-Key-Proxy/main/deploy/cliproxyapi/install_cliproxyapi.sh
chmod +x install_cliproxyapi.sh
sudo ./install_cliproxyapi.sh
```

### Manual Installation

```bash
# Download binary
wget https://github.com/router-for-me/CLIProxyAPI/releases/download/v6.9.19/cli-proxy-api-linux-amd64
chmod +x cli-proxy-api-linux-amd64
mv cli-proxy-api-linux-amd64 /opt/cliproxyapi/cliproxyapi

# Create directories
mkdir -p /opt/cliproxyapi/{config,auth}

# Create config
cat > /opt/cliproxyapi/config/config.yaml << 'EOF'
server:
  host: "127.0.0.1"
  port: 8317

auth-dir: "/opt/cliproxyapi/auth"

routing:
  strategy: round-robin
  fallback-enabled: true
EOF
```

## Provider Authentication

### Gemini CLI

```bash
/opt/cliproxyapi/cliproxyapi -gemini-login
```

Opens browser for Google OAuth. Token auto-refreshes.

### iFlow (IMPORTANT: Use Cookie Method!)

```bash
# ✅ CORRECT - Cookie method (auto-refresh works)
/opt/cliproxyapi/cliproxyapi -iflow-cookie

# ❌ WRONG - OAuth method (expires in 7 days, NO auto-refresh)
/opt/cliproxyapi/cliproxyapi -iflow-login
```

**Why cookie method?**
- OAuth tokens expire every 7 days with no refresh capability
- Cookie sessions last months and auto-refresh automatically

### Antigravity

```bash
/opt/cliproxyapi/cliproxyapi -antigravity-login
```

### Qwen Code

```bash
/opt/cliproxyapi/cliproxyapi -qwen-login
```

## Gateway Configuration

Add to your `.env`:

```bash
CLIPROXYAPI_ENABLED=true
CLIPROXYAPI_BASE_URL=http://127.0.0.1:8317
CLIPROXYAPI_TIMEOUT=120
PROVIDER_PRIORITY_CLIPROXYAPI=1
```

## Supported Models

### Gemini CLI (OAuth)
- `gemini/gemini-2.5-pro`
- `gemini/gemini-2.5-flash`
- `gemini/gemini-2.0-flash-exp`

### iFlow (Cookie)
- `iflow/glm-4-plus`
- `iflow/glm-4-flash`
- `iflow/glm-4-air`

### Antigravity (OAuth)
- `antigravity/gemini-2.0-flash-exp`
- `antigravity/gemini-3-flash`

### Qwen Code (OAuth)
- `qwen/qwen3-coder-plus`
- `qwen/qwen3-coder-lite`

## API Endpoints

### CLIProxyAPI Sidecar

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /v1/models` | List available models |
| `POST /v1/chat/completions` | Chat completion |

### Gateway Integration

| Endpoint | Description |
|----------|-------------|
| `GET /api/cliproxyapi/health` | Sidecar health |
| `GET /api/cliproxyapi/status` | Detailed status |
| `GET /api/cliproxyapi/models` | Available models |
| `GET /api/cliproxyapi/backend/{name}` | Backend status |
| `GET /api/cliproxyapi/config` | Configuration |

## Usage Examples

### Direct CLIProxyAPI Request

```bash
curl http://127.0.0.1:8317/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini/gemini-2.5-flash",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Through Gateway

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini/gemini-2.5-flash",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Streaming

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "iflow/glm-4-flash",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

## Systemd Service

### Start

```bash
sudo systemctl start cliproxyapi
```

### Check Status

```bash
sudo systemctl status cliproxyapi
```

### View Logs

```bash
sudo journalctl -u cliproxyapi -f
```

### Restart

```bash
sudo systemctl restart cliproxyapi
```

## Troubleshooting

### CLIProxyAPI Won't Start

```bash
# Check if port is in use
ss -tlnp | grep 8317

# Check logs
sudo journalctl -u cliproxyapi -n 50

# Run manually to debug
/opt/cliproxyapi/cliproxyapi --config /opt/cliproxyapi/config/config.yaml
```

### iFlow Token Expired

```bash
# Check auth files
ls -la /opt/cliproxyapi/auth/

# Re-authenticate with cookie method
/opt/cliproxyapi/cliproxyapi -iflow-cookie
```

### No Models Available

```bash
# Check if providers are authenticated
curl http://127.0.0.1:8317/v1/models

# Re-authenticate providers
/opt/cliproxyapi/cliproxyapi -gemini-login
/opt/cliproxyapi/cliproxyapi -iflow-cookie
```

### Gateway Can't Connect

```bash
# Verify CLIProxyAPI is running
curl http://127.0.0.1:8317/health

# Check gateway env vars
env | grep CLIPROXYAPI

# Restart gateway
sudo systemctl restart llm-gateway
```

## Resource Requirements

| Service | Memory | Notes |
|---------|--------|-------|
| CLIProxyAPI | ~30-50MB | Go binary |
| Gateway (Python) | ~50-100MB | With CLIProxyAPI provider |
| **Total** | ~80-150MB | Fits 1GB VPS with margin |

## Security Considerations

1. **Auth Directory**: Should be `0700` (owner-only)
2. **Service User**: Runs as `cliproxyapi` user, not root
3. **Local Binding**: Binds to `127.0.0.1` by default
4. **No Credentials in Config**: Auth stored separately in auth directory

## Comparison with Other Routers

| Feature | CLIProxyAPI | 9Router | OmniRoute |
|---------|-------------|---------|-----------|
| Language | Go | Node.js | Node.js |
| Memory | ~50MB | ~200MB | ~200MB |
| Database | None | SQLite | SQLite |
| iFlow Cookie | ✅ | ✅ | ✅ |
| Auto-refresh | ✅ | ✅ | ✅ |
| Dashboard | ❌ | ✅ | ✅ |
| MCP Server | ❌ | ❌ | ✅ |

CLIProxyAPI is the best choice for low-memory VPS deployments.

## Terms of Service Notes

### ⚠️ Important Warnings

| Provider | ToS Risk | Notes |
|----------|----------|-------|
| **Claude Pro/Max** | 🔴 HIGH | Anthropic explicitly bans OAuth token extraction (Feb 2026) |
| **Gemini CLI** | 🟡 MEDIUM | Free 180K/month, less explicit prohibition |
| **iFlow** | 🟢 LOW | Free unlimited via cookie method |
| **Qwen Code** | 🟢 LOW | Free unlimited |

### Safe Approach

- Use free providers (iFlow, Qwen, Gemini CLI free tier)
- Use API key-based providers (OpenRouter, DeepSeek, etc.)
- Avoid routing paid subscriptions through third-party tools
