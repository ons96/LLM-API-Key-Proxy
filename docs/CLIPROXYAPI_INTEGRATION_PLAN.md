# CLIProxyAPI Integration Plan for LLM-API-Key-Proxy

**Created:** April 9, 2026  
**Project:** Integrate CLIProxyAPI with your existing LLM gateway  
**Target VPS:** Oracle Micro (1GB RAM + ZRAM + 4GB swap)

---

## Executive Summary

This plan outlines how to integrate CLIProxyAPI (a Go-based OAuth/router proxy) with your existing Python-based LLM gateway (`LLM-API-Key-Proxy`). Since your gateway is Python and CLIProxyAPI is Go, we'll run CLIProxyAPI as a **sidecar service** rather than embedding it directly.

### Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Integration approach** | Sidecar (separate process) | Your gateway is Python, CLIProxyAPI is Go |
| **iFlow auth method** | Cookie-based | Auto-refresh works, OAuth doesn't |
| **Memory budget** | ~50-70MB | Fits your 1GB VPS with margin |
| **Monitoring** | Custom endpoint | Add to your gateway, not CLIProxyAPI |

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                        Your VPS                               │
│                                                               │
│  ┌─────────────────────┐      ┌─────────────────────────┐   │
│  │  LLM-API-Key-Proxy  │      │     CLIProxyAPI         │   │
│  │     (Python)        │      │      (Go binary)        │   │
│  │                     │      │                         │   │
│  │  • Main routing     │      │  • OAuth handling       │   │
│  │  • Fallback logic   │◄─────│  • Token refresh        │   │
│  │  • Provider mgmt    │ HTTP │  • Cookie-based auth    │   │
│  │  • API endpoints    │      │  • Multi-account LB     │   │
│  │                     │      │                         │   │
│  │  ~10-50MB RAM       │      │  ~30-50MB RAM           │   │
│  └─────────────────────┘      └─────────────────────────┘   │
│           │                              │                   │
│           ▼                              ▼                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Providers (OAuth + API Key)              │   │
│  │                                                       │   │
│  │  Gemini CLI │ iFlow │ Antigravity │ Qwen │ Claude   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## Phase 1: VPS Setup & CLIProxyAPI Installation

### 1.1 Prerequisites Check

Run on your VPS:

```bash
# Check available resources
free -h
df -h /

# Check existing services
systemctl status zeroclaw 2>/dev/null || echo "ZeroClaw not found"
ps aux | grep -E "(gateway|proxy)" | grep -v grep

# Check ports
ss -tlnp | grep -E "(8000|8317|8080)"
```

### 1.2 Download CLIProxyAPI

```bash
# SSH into VPS
cd /opt  # or your preferred directory

# Download latest release
wget https://github.com/router-for-me/CLIProxyAPI/releases/download/v6.9.19/cli-proxy-api-linux-amd64

# Make executable
chmod +x cli-proxy-api-linux-amd64
mv cli-proxy-api-linux-amd64 cliproxyapi

# Verify
./cliproxyapi --version
```

### 1.3 Create Directory Structure

```bash
# Create config directories
mkdir -p /opt/cliproxyapi/{config,auth,logs}

# Set permissions
chmod 700 /opt/cliproxyapi/auth
```

---

## Phase 2: Provider Configuration

### 2.1 Initial Configuration File

Create `/opt/cliproxyapi/config/config.yaml`:

```yaml
# CLIProxyAPI Configuration
# Generated for LLM-API-Key-Proxy integration

server:
  host: "127.0.0.1"
  port: 8317

# Authentication directory
auth-dir: "/opt/cliproxyapi/auth"

# Logging
log-level: "info"
log-file: "/opt/cliproxyapi/api.log"

# Providers configuration
providers:
  # Gemini CLI (OAuth)
  gemini-cli:
    type: oauth
    enabled: true
    models:
      - "gemini-2.5-pro"
      - "gemini-2.5-flash"

  # iFlow (Cookie-based - IMPORTANT: Use cookie, not OAuth!)
  iflow:
    type: cookie
    enabled: true
    models:
      - "glm-4-plus"
      - "glm-4-flash"

  # Antigravity (OAuth)
  antigravity:
    type: oauth
    enabled: true
    models:
      - "gemini-2.0-flash-exp"

  # Qwen Code (OAuth)
  qwen:
    type: oauth
    enabled: true
    models:
      - "qwen3-coder-plus"

# Routing configuration
routing:
  strategy: round-robin
  fallback-enabled: true
  retry-attempts: 3
  timeout-seconds: 120

# Health check
health:
  enabled: true
  interval-seconds: 30
```

### 2.2 Provider Authentication Setup

#### Gemini CLI (OAuth)

```bash
# Run on local machine with browser, then copy auth file to VPS
./cliproxyapi -gemini-login

# Follow browser OAuth flow
# Auth file saved to: ~/.cli-proxy-api/gemini_oauth_*.json

# Copy to VPS (if setting up locally first)
scp ~/.cli-proxy-api/gemini_oauth_*.json user@vps:/opt/cliproxyapi/auth/
```

#### iFlow (Cookie-based - CRITICAL)

```bash
# Step 1: Log into iFlow in your browser
# Go to: https://iflow.cn (or iFlow platform)
# Complete login

# Step 2: Get cookies from browser
# In DevTools (F12) → Application → Cookies
# Copy the full cookie string

# Step 3: Run cookie authentication
./cliproxyapi -iflow-cookie

# Paste cookies when prompted
# CLIProxyAPI will:
#   1. Extract API key from cookies
#   2. Store auth with expiration
#   3. Start auto-refresh daemon
```

**Why Cookie Method?**
- OAuth tokens expire every 7 days with NO auto-refresh
- Cookie sessions last much longer (months)
- CLIProxyAPI auto-refreshes cookie-based auth

#### Antigravity (OAuth)

```bash
./cliproxyapi -antigravity-login
# Complete OAuth flow in browser
```

#### Qwen Code (OAuth)

```bash
./cliproxyapi -qwen-login
# Complete OAuth flow in browser
```

---

## Phase 3: Systemd Service Setup

### 3.1 Create Service File

Create `/etc/systemd/system/cliproxyapi.service`:

```ini
[Unit]
Description=CLIProxyAPI - OAuth Router for LLM Providers
Documentation=https://help.router-for.me/
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=root
Group=root
WorkingDirectory=/opt/cliproxyapi
ExecStart=/opt/cliproxyapi/cliproxyapi --config /opt/cliproxyapi/config/config.yaml
ExecReload=/bin/kill -HUP $MAINPID

# Resource limits (critical for 1GB VPS)
MemoryMax=100M
CPUQuota=50%

# Restart policy
Restart=on-failure
RestartSec=10

# Security
NoNewPrivileges=true
PrivateTmp=true

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=cliproxyapi

[Install]
WantedBy=multi-user.target
```

### 3.2 Enable and Start Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable on boot
sudo systemctl enable cliproxyapi

# Start service
sudo systemctl start cliproxyapi

# Check status
sudo systemctl status cliproxyapi

# View logs
sudo journalctl -u cliproxyapi -f
```

---

## Phase 4: Gateway Integration

### 4.1 Add CLIProxyAPI as Provider in Your Gateway

In your `LLM-API-Key-Proxy`, add a new provider file:

**File:** `src/rotator_library/providers/cliproxyapi_provider.py`

```python
"""
CLIProxyAPI Provider - Routes through CLIProxyAPI sidecar
Handles OAuth providers via CLIProxyAPI's token management
"""

import aiohttp
from typing import Optional, Dict, Any, AsyncIterator
from .provider_interface import BaseProvider


class CLIProxyAPIProvider(BaseProvider):
    """
    Provider that routes requests through CLIProxyAPI sidecar.
    
    CLIProxyAPI handles:
    - OAuth token refresh (Gemini, Antigravity, Qwen)
    - Cookie-based auth (iFlow)
    - Multi-account load balancing
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.base_url = config.get("base_url", "http://127.0.0.1:8317")
        self.timeout = config.get("timeout", 120)
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self.session
    
    async def chat_completion(
        self,
        messages: list,
        model: str,
        stream: bool = False,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Send chat completion request through CLIProxyAPI.
        
        Model prefixes route to different OAuth providers:
        - gemini/* → Gemini CLI OAuth
        - iflow/* → iFlow (cookie-based)
        - antigravity/* → Antigravity OAuth
        - qwen/* → Qwen Code OAuth
        """
        session = await self._get_session()
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        
        url = f"{self.base_url}/v1/chat/completions"
        
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                error = await response.text()
                raise Exception(f"CLIProxyAPI error: {response.status} - {error}")
            
            if stream:
                async for line in response.content:
                    if line:
                        yield self._parse_sse_line(line)
            else:
                data = await response.json()
                yield data
    
    async def models(self) -> list:
        """Get available models from CLIProxyAPI."""
        session = await self._get_session()
        
        async with session.get(f"{self.base_url}/v1/models") as response:
            if response.status == 200:
                data = await response.json()
                return data.get("data", [])
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """Check CLIProxyAPI health status."""
        session = await self._get_session()
        
        try:
            async with session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    return {"status": "healthy", "response": await response.json()}
                return {"status": "unhealthy", "code": response.status}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def close(self):
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    def _parse_sse_line(self, line: bytes) -> Dict[str, Any]:
        """Parse SSE line from CLIProxyAPI stream."""
        import json
        
        line_str = line.decode("utf-8").strip()
        
        if not line_str or line_str == "data: [DONE]":
            return {"done": True}
        
        if line_str.startswith("data: "):
            data_str = line_str[6:]  # Remove "data: " prefix
            try:
                return json.loads(data_str)
            except json.JSONDecodeError:
                return {"raw": data_str}
        
        return {"raw": line_str}


# Provider registration
PROVIDER_CLASS = CLIProxyAPIProvider
PROVIDER_NAME = "cliproxyapi"
SUPPORTED_MODELS = [
    # Gemini CLI
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    # iFlow
    "glm-4-plus",
    "glm-4-flash",
    # Antigravity
    "gemini-2.0-flash-exp",
    # Qwen
    "qwen3-coder-plus",
]
```

### 4.2 Register Provider in Gateway

**File:** `src/rotator_library/provider_factory.py` (modify existing)

```python
# Add to imports
from .providers.cliproxyapi_provider import CLIProxyAPIProvider

# Add to provider registry
PROVIDER_REGISTRY = {
    # ... existing providers ...
    "cliproxyapi": CLIProxyAPIProvider,
}
```

### 4.3 Add Configuration

**File:** `.env` or config

```bash
# CLIProxyAPI Sidecar Configuration
CLIPROXYAPI_ENABLED=true
CLIPROXYAPI_BASE_URL=http://127.0.0.1:8317
CLIPROXYAPI_TIMEOUT=120

# Model routing prefixes (optional, for clarity)
# gemini/* → Gemini CLI OAuth via CLIProxyAPI
# iflow/* → iFlow cookie auth via CLIProxyAPI
# antigravity/* → Antigravity OAuth via CLIProxyAPI
# qwen/* → Qwen Code OAuth via CLIProxyAPI
```

---

## Phase 5: Monitoring & Observability

### 5.1 Add Health Check to Gateway

**File:** `src/proxy_app/routes/health.py` (or equivalent)

```python
from fastapi import APIRouter
import aiohttp

router = APIRouter()

@router.get("/health/cliproxyapi")
async def cliproxyapi_health():
    """Check CLIProxyAPI sidecar health."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://127.0.0.1:8317/health") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {
                        "status": "healthy",
                        "cliproxyapi": data
                    }
                return {
                    "status": "unhealthy",
                    "code": resp.status
                }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@router.get("/health/providers")
async def providers_health():
    """Check all provider health including CLIProxyAPI."""
    health_status = {
        "gateway": "healthy",
        "providers": {}
    }
    
    # Check CLIProxyAPI
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://127.0.0.1:8317/v1/models") as resp:
                if resp.status == 200:
                    models = await resp.json()
                    health_status["providers"]["cliproxyapi"] = {
                        "status": "healthy",
                        "models_count": len(models.get("data", []))
                    }
                else:
                    health_status["providers"]["cliproxyapi"] = {
                        "status": "unhealthy",
                        "code": resp.status
                    }
    except Exception as e:
        health_status["providers"]["cliproxyapi"] = {
            "status": "error",
            "error": str(e)
        }
    
    return health_status
```

### 5.2 Prometheus Metrics (Optional)

Add to your gateway's metrics:

```python
# In your metrics module
from prometheus_client import Counter, Gauge, Histogram

cliproxyapi_requests_total = Counter(
    'cliproxyapi_requests_total',
    'Total requests to CLIProxyAPI',
    ['provider', 'model', 'status']
)

cliproxyapi_latency_seconds = Histogram(
    'cliproxyapi_latency_seconds',
    'CLIProxyAPI request latency',
    ['provider', 'model']
)

cliproxyapi_healthy = Gauge(
    'cliproxyapi_healthy',
    'Whether CLIProxyAPI sidecar is healthy'
)
```

---

## Phase 6: Testing & Validation

### 6.1 Unit Tests

**File:** `tests/test_cliproxyapi_provider.py`

```python
import pytest
from unittest.mock import AsyncMock, patch
from src.rotator_library.providers.cliproxyapi_provider import CLIProxyAPIProvider


@pytest.fixture
def provider():
    return CLIProxyAPIProvider({"base_url": "http://127.0.0.1:8317"})


@pytest.mark.asyncio
async def test_health_check_healthy(provider):
    """Test health check when CLIProxyAPI is running."""
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "ok"})
        mock_get.return_value.__aenter__.return_value = mock_response
        
        result = await provider.health_check()
        assert result["status"] == "healthy"


@pytest.mark.asyncio
async def test_health_check_unhealthy(provider):
    """Test health check when CLIProxyAPI is not responding."""
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_get.side_effect = Exception("Connection refused")
        
        result = await provider.health_check()
        assert result["status"] == "error"


@pytest.mark.asyncio
async def test_chat_completion(provider):
    """Test chat completion through CLIProxyAPI."""
    messages = [{"role": "user", "content": "Hello"}]
    
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "id": "test",
            "choices": [{"message": {"content": "Hi there!"}}]
        })
        mock_post.return_value.__aenter__.return_value = mock_response
        
        results = []
        async for chunk in provider.chat_completion(messages, "gemini-2.5-pro"):
            results.append(chunk)
        
        assert len(results) > 0
```

### 6.2 Integration Tests

```bash
# On VPS, test CLIProxyAPI directly
curl http://127.0.0.1:8317/health
curl http://127.0.0.1:8317/v1/models

# Test chat completion
curl http://127.0.0.1:8317/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.5-flash",
    "messages": [{"role": "user", "content": "Hello"}]
  }'

# Test through your gateway
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_KEY" \
  -d '{
    "model": "gemini-2.5-flash",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

---

## Phase 7: Git Worktree Setup (For Parallel Development)

### 7.1 Why Git Worktrees?

If you have another OpenCode session working on the same repository, you need to avoid conflicts. Git worktrees create separate working directories on different branches.

### 7.2 Setup Worktree for This Feature

```bash
# From your gateway repo
cd /home/osees/CodingProjects/LLM-API-Key-Proxy

# Check current branch and status
git status

# Create feature branch worktree
git worktree add ../LLM-API-Key-Proxy-cliproxy feature/cliproxy-integration

# This creates:
# /home/osees/CodingProjects/LLM-API-Key-Proxy-cliproxy (new worktree)
# - On branch: feature/cliproxy-integration
# - Separate from main worktree
```

### 7.3 Work on Feature in Worktree

```bash
# OpenCode session for CLIProxyAPI integration
cd /home/osees/CodingProjects/LLM-API-Key-Proxy-cliproxy

# Work happens here, isolated from your other session
# Make changes, test, commit

# When ready to merge
cd /home/osees/CodingProjects/LLM-API-Key-Proxy
git merge feature/cliproxy-integration

# Clean up worktree
git worktree remove ../LLM-API-Key-Proxy-cliproxy
```

### 7.4 Worktree Best Practices

| Scenario | Approach |
|----------|----------|
| Other session editing different files | Worktrees not needed |
| Other session might edit same files | **Use worktree** |
| Other session on different branch | Can work in same repo |
| Need to test both branches | Use worktree |

---

## Troubleshooting Guide

### Issue: CLIProxyAPI won't start

```bash
# Check logs
sudo journalctl -u cliproxyapi -n 50

# Check port availability
ss -tlnp | grep 8317

# Check memory
free -h

# Run manually to debug
/opt/cliproxyapi/cliproxyapi --config /opt/cliproxyapi/config/config.yaml
```

### Issue: iFlow auth expires

```bash
# Check auth file
ls -la /opt/cliproxyapi/auth/

# Check token expiration
cat /opt/cliproxyapi/auth/iflow-*.json | jq '.expires_at'

# Re-auth with cookie method
./cliproxyapi -iflow-cookie
```

### Issue: Token refresh fails

```bash
# Check refresh logs
grep "refresh" /opt/cliproxyapi/api.log

# For Gemini/OAuth providers: re-auth
./cliproxyapi -gemini-login

# For iFlow: use cookie method (not OAuth)
./cliproxyapi -iflow-cookie
```

### Issue: Gateway can't connect to CLIProxyAPI

```bash
# Verify CLIProxyAPI is running
systemctl status cliproxyapi

# Check network
curl -v http://127.0.0.1:8317/health

# Check firewall (if any)
sudo iptables -L -n | grep 8317
```

---

## Resource Monitoring

### Memory Usage Check

```bash
# Check CLIProxyAPI memory
ps aux | grep cliproxyapi | awk '{print $6/1024 " MB"}'

# Check total VPS usage
free -h

# Monitor in real-time
watch -n 5 'free -h && ps aux | grep -E "(cliproxyapi|gateway)" | grep -v grep'
```

### Expected Resource Allocation

| Service | Memory | Notes |
|---------|--------|-------|
| ZeroClaw | ~10MB | Your existing service |
| LLM Gateway | ~50-100MB | Your Python gateway |
| CLIProxyAPI | ~30-50MB | Go binary |
| System + ZRAM | ~200-300MB | OS overhead |
| **Total** | ~300-450MB | Leaves 550MB+ margin |

---

## Rollback Plan

If something goes wrong:

```bash
# Stop CLIProxyAPI
sudo systemctl stop cliproxyapi
sudo systemctl disable cliproxyapi

# Remove service
sudo rm /etc/systemd/system/cliproxyapi.service
sudo systemctl daemon-reload

# Remove CLIProxyAPI
rm -rf /opt/cliproxyapi

# Gateway continues to work with direct providers
# (Your gateway should gracefully handle CLIProxyAPI unavailability)
```

---

## Next Steps After Implementation

1. **Monitor for 24-48 hours** — Watch for token refresh issues
2. **Add more providers** — Add Kimi, Codex if needed
3. **Integrate monitoring** — Connect to your existing monitoring
4. **Optimize routing** — Tune fallback priorities based on usage
5. **Consider embedding** — If you migrate gateway to Go, embed CLIProxyAPI SDK

---

## Files Changed Checklist

### VPS (New Files)

- [ ] `/opt/cliproxyapi/cliproxyapi` — CLIProxyAPI binary
- [ ] `/opt/cliproxyapi/config/config.yaml` — Configuration
- [ ] `/opt/cliproxyapi/auth/*.json` — OAuth/cookie tokens
- [ ] `/etc/systemd/system/cliproxyapi.service` — Systemd service

### Gateway (New Files)

- [ ] `src/rotator_library/providers/cliproxyapi_provider.py` — Provider
- [ ] `tests/test_cliproxyapi_provider.py` — Tests

### Gateway (Modified Files)

- [ ] `src/rotator_library/provider_factory.py` — Register provider
- [ ] `.env` or config — Add CLIProxyAPI settings
- [ ] `src/proxy_app/routes/health.py` — Add health endpoint (optional)

---

## Summary

This integration adds CLIProxyAPI as a sidecar service to handle OAuth token management and cookie-based authentication for your LLM gateway. The key benefits:

1. **Auto token refresh** — No more manual 7-day iFlow re-auth
2. **Multi-account load balancing** — Distribute across accounts
3. **Lightweight** — ~50MB RAM, fits your 1GB VPS
4. **Clean separation** — Sidecar pattern, easy to debug
5. **Future-proof** — Can embed if you migrate to Go

The cookie method for iFlow is critical — OAuth expires every 7 days without refresh capability.
