# LLM-API-Key-Proxy Codebase

**Last Updated:** 2026-02-24
**Commit:** a55806d
**Branch:** main

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT REQUEST                            │
│                   POST /v1/chat/completions                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Gateway                              │
│                   (src/proxy_app/main.py)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ API Key     │  │ Virtual     │  │ Responses API           │  │
│  │ Validation  │  │ Model Router│  │ (/v1/responses)         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Router Core                                    │
│              (src/proxy_app/router_core.py)                      │
│  • Fallback chain execution                                      │
│  • Provider selection & health tracking                          │
│  • Rate limit handling                                           │
│  • Request/response transformation                               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   RotatingClient                                 │
│            (src/rotator_library/client.py)                       │
│  • HTTP connection pooling                                       │
│  • Retry logic with exponential backoff                          │
│  • Credential rotation                                           │
│  • Usage tracking                                                │
└─────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
┌──────────────────────────┐   ┌──────────────────────────┐
│    Provider Adapters     │   │   Provider Plugins       │
│  (rotator_library/)      │   │   (providers/)           │
│  • G4F (g4f library)     │   │   • antigravity          │
│  • Groq (direct API)     │   │   • gemini_cli           │
│  • Gemini (direct API)   │   │   • (extensible)         │
│  • Together (direct API) │   │                          │
│  • + 10 more providers   │   │                          │
└──────────────────────────┘   └──────────────────────────┘
```

---

## What's Done

### Core Infrastructure
- [x] FastAPI gateway with `/v1/chat/completions` endpoint
- [x] `/v1/responses` endpoint for OpenAI Responses API
- [x] `/v1/models` endpoint for model listing
- [x] `/stats` endpoint for usage statistics
- [x] Virtual model routing with automatic fallback chains
- [x] API key authentication (`Authorization: Bearer <key>`)
- [x] Request logging (optional, via `--enable-request-logging`)

### Virtual Models (7 models)
| Model | Description | Primary Provider | Timeout |
|-------|-------------|------------------|---------|
| `coding-elite` | Best agentic coding | G4F claude-opus-4-5 | 180s |
| `coding-smart` | Balanced coding | G4F gemini-3-flash | 120s |
| `coding-fast` | Fastest coding | Groq llama-3.1-8b-instant | 30s |
| `chat-elite` | Most intelligent | G4F gemini-3-pro | 300s |
| `chat-smart` | Best ratio | G4F gemini-3-pro | 120s |
| `chat-fast` | Fastest chat | Groq llama-3.1-8b-instant | 15s |
| `chat-rp` | Roleplay | G4F mn-violet-lotus-12b | 30s |

### Providers (16+ configured)
- **Free Tier (no API key needed):** G4F, Groq (free tier), Gemini (free tier)
- **API Key Required:** OpenAI, Anthropic, DeepSeek, Qwen, Grok, Cerebras, NVIDIA, Mistral, Together, OpenRouter, SambaNova, GitHub Models, Cloudflare
- **Search Providers:** Tavily, Brave, DuckDuckGo, Exa, Jina

### Deployment
- [x] VPS deployed at `http://40.233.101.233:8000`
- [x] OpenCode integration configured
- [x] Docker support (`Dockerfile`, `docker-compose.yml`)
- [x] GitHub Actions CI/CD (`.github/workflows/`)
- [x] HTTPS setup scripts

### Developer Experience
- [x] Terminal UI launcher (`python src/proxy_app/main.py`)
- [x] OAuth credential tool (`--add-credential`)
- [x] Settings management tool (`settings_tool.py`)

---

## What's Missing / TODO

### High Priority
- [ ] **Automated testing** - No comprehensive test suite
- [ ] **Monitoring/alerting** - No uptime monitoring or alerts
- [ ] **Provider health dashboard** - Manual status checks only

### Medium Priority
- [ ] **Dynamic model rankings** - VIRTUAL_MODELS_PLAN.md has formulas, not implemented
- [ ] **Rate limit optimization** - Static limits, no adaptive throttling
- [ ] **Response caching** - No caching layer for repeated requests
- [ ] **Usage analytics** - Basic tracking, no dashboards

### Low Priority
- [ ] **WebSocket support** - Real-time streaming improvements
- [ ] **Multi-tenancy** - Single API key per deployment
- [ ] **Load balancing** - Single instance deployment

---

## Known Issues

### From BUGS.md
1. **Dormant router logic** - Some routing code in `main.py` bypasses `RouterCore`
2. **Complex G4F model IDs** - Some models don't work (stick to simple names)

### From Codebase Analysis
- No automated test suite (tests exist but not integrated)
- Manual deployment workflow (no CI/CD pipeline for deployment)
- `.env` file handling could be more robust

---

## Tech Stack

### Backend
- **Framework:** FastAPI
- **Server:** Uvicorn (ASGI)
- **HTTP Client:** httpx (async)
- **LLM Library:** g4f (GPT4Free), LiteLLM
- **CLI/UI:** Rich (terminal formatting)

### Configuration
- **Config Format:** YAML (`config/*.yaml`)
- **Environment:** python-dotenv (`.env` files)
- **Logging:** Python logging module

### Data
- **Database:** SQLite (`provider_status.db`)
- **Storage:** JSON for credentials, YAML for config

### Deployment
- **Container:** Docker
- **CI/CD:** GitHub Actions
- **VPS:** Ubuntu on Oracle Cloud (40.233.101.233)

---

## How to Run

### Development (Local)
```bash
# Clone and setup
git clone https://github.com/ons96/LLM-API-Key-Proxy
cd LLM-API-Key-Proxy
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Copy and configure .env
cp .env.example .env
# Edit .env with your API keys

# Run with TUI launcher
python src/proxy_app/main.py

# Or run directly
python src/proxy_app/main.py --host 0.0.0.0 --port 8000
```

### Production (VPS)
```bash
ssh -i ~/.ssh/oracle.key ubuntu@40.233.101.233
cd ~/LLM-API-Key-Proxy && git pull
pkill -f 'main.py'
source venv/bin/activate
nohup python src/proxy_app/main.py --host 0.0.0.0 --port 8000 > ~/llm_proxy.log 2>&1 &
```

### Docker
```bash
docker build -t llm-proxy .
docker run -p 8000:8000 --env-file .env llm-proxy
```

---

## How to Test

### Quick Tests
```bash
# Test gateway is running
curl http://localhost:8000/v1/models -H "Authorization: Bearer YOUR_KEY"

# Test virtual model
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_KEY" \
  -d '{"model": "coding-fast", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Test Suite
```bash
# Run pytest (if configured)
pytest tests/

# Or individual test files
python test_api_endpoints.py
python test_provider_status.py
```

---

## How to Build

```bash
# No build step required - Python application
# Just install dependencies
pip install -r requirements.txt

# For Docker
docker build -t llm-proxy .
```

---

## File Structure

```
LLM-API-Key-Proxy/
├── config/                    # YAML configurations
│   ├── router_config.yaml     # Provider settings, virtual models
│   └── virtual_models.yaml    # Fallback chain definitions
├── src/
│   ├── proxy_app/             # Main gateway application
│   │   ├── main.py            # FastAPI app entry point (1476 lines)
│   │   ├── router_core.py     # Fallback/routing logic (1683 lines)
│   │   ├── settings_tool.py   # Settings management (2450 lines)
│   │   ├── launcher_tui.py    # Terminal UI launcher (1003 lines)
│   │   └── provider_urls.py   # URL construction helpers
│   └── rotator_library/       # Core resilience library
│       ├── client.py          # RotatingClient (2674 lines)
│       ├── credential_tool.py # OAuth credential management (2255 lines)
│       ├── usage_manager.py   # Usage tracking (1792 lines)
│       ├── model_info_service.py # Model metadata (1352 lines)
│       ├── error_handler.py   # Error handling (976 lines)
│       └── providers/         # Provider adapters
├── tests/                     # Test files
├── scripts/                   # Utility scripts
├── docs/                      # Documentation
├── .env                       # API keys (NOT committed)
├── .env.example               # Template for .env
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker build
├── docker-compose.yml         # Docker compose
└── pytest.ini                 # Test configuration
```

---

## Related Documentation

| Document | Purpose |
|----------|---------|
| [AGENTS.md](./AGENTS.md) | AI agent instructions and conventions |
| [README.md](./README.md) | User-facing documentation |
| [VIRTUAL_MODELS_PLAN.md](./VIRTUAL_MODELS_PLAN.md) | Dynamic model ranking plan |
| [BUGS.md](./BUGS.md) | Known issues |
| [PROJECT_STATUS.md](./PROJECT_STATUS.md) | Detailed feature status |
| [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) | Full deployment instructions |
