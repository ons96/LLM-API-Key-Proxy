# LLM-API-Key-Proxy Knowledge Base

**Generated:** 2026-02-05
**Commit:** 0342491
**Branch:** (check with `git branch`)

## OVERVIEW

FastAPI-based LLM API proxy with virtual models and automatic fallback. Routes requests through 10+ free LLM providers (Groq, Gemini, G4F, etc.) with intelligent fallback chains.

## STRUCTURE

```
LLM-API-Key-Proxy/
├── config/              # YAML configs (router, virtual models, aliases)
├── src/
│   ├── proxy_app/       # Main gateway (entry: main.py, 1357 lines)
│   │   ├── main.py           # FastAPI app, /v1/chat/completions endpoint
│   │   ├── router_core.py    # Router logic (1683 lines)
│   │   ├── settings_tool.py  # Settings management (2450 lines)
│   │   ├── launcher_tui.py   # Terminal UI launcher (1003 lines)
│   │   └── provider_urls.py  # Provider URL construction
│   └── rotator_library/  # Core library (49k lines total)
│       ├── client.py           # RotatingClient (2674 lines)
│       ├── credential_tool.py  # OAuth credential management (2255 lines)
│       ├── usage_manager.py    # Usage tracking (1792 lines)
│       ├── model_info_service.py # Model metadata (1352 lines)
│       ├── error_handler.py    # Error handling (976 lines)
│       └── providers/          # Provider adapters (antigravity, gemini_cli, etc.)
├── scripts/             # Utility scripts
├── tests/               # Test fixtures
├── docs/                # Documentation
└── .env                 # API keys (NOT committed)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add `/v1/responses` endpoint | `src/proxy_app/main.py:882` | Add new route for OpenCode Responses API |
| Virtual models | `config/virtual_models.yaml` | coding-elite, coding-fast, etc. |
| Provider routing | `src/proxy_app/router_core.py` | Fallback logic |
| Provider adapters | `src/rotator_library/providers/` | Individual provider implementations |
| Router configuration | `config/router_config.yaml` | Provider enable/disable |
| API key auth | `src/proxy_app/` | Search for `verify_api_key` |
| TUI launcher | `src/proxy_app/launcher_tui.py` | Interactive launcher |

## KEY CONCEPTS

### Virtual Models

Pre-configured model aliases with automatic fallback chains:
- **coding-elite**: Groq llama-3.3-70b → Gemini 1.5-pro → G4F gpt-4
- **coding-fast**: Groq llama-3.1-8b-instant → Gemini 1.5-flash → G4F
- **chat-smart**: Groq llama-3.3-70b → Gemini 1.5-pro
- **chat-fast**: Groq llama-3.1-8b-instant → Gemini 1.5-flash → G4F

### Fallback Chain

Router automatically tries next provider if current one fails (rate limit, error, timeout). No manual intervention.

### FREE_ONLY_MODE

Gateway runs in free-only mode by default (`FREE_ONLY_MODE=true`). No paid API keys needed.

## CONVENTIONS (Deviations from Standard)

### Python
- **Formatting:** Black, 88-char line limit
- **Type hints:** Required for function signatures
- **Async:** Heavy use of async/await for I/O-bound operations
- **Imports:** `from pathlib import Path` (not `os.path`)
- **Logging:** Use `logging.getLogger(__name__)`

### FastAPI
- **Dependencies:** Use `Depends()` for injection (get_rotating_client, verify_api_key)
- **Exceptions:** Raise `HTTPException(status_code=..., detail=...)`
- **Request parsing:** `await request.json()` for body

### Configuration
- **YAML files** for configs (router_config.yaml, virtual_models.yaml)
- **Environment variables** for secrets (.env file loaded via python-dotenv)
- **Virtual models** in `config/virtual_models.yaml`

## ANTI-PATTERNS (THIS PROJECT)

❌ **DO NOT** modify `.env` file - credentials handled via OAuth credential tool
❌ **DO NOT** hardcode API keys - use `os.getenv()` or credential tool
❌ **DO NOT** block on async operations - use `await`
❌ **DO NOT** make synchronous HTTP calls - use httpx with async
❌ **DO NOT** disable rate limit checks - respect provider limits

## COMMANDS

```bash
# Development
python src/proxy_app/main.py --host 0.0.0.0 --port 8000

# With custom config
python src/proxy_app/main.py --host 0.0.0.0 --port 8000 --enable-request-logging

# OAuth credential tool
python src/proxy_app/main.py --add-credential

# Terminal UI launcher (interactive)
python src/proxy_app/main.py
```

## CURRENT ISSUE (Priority)

**OpenCode Integration Broken**

OpenCode uses OpenAI **Responses API** (`POST /v1/responses`), but gateway only implements **Chat Completions API** (`POST /v1/chat/completions`).

**Error:** `404 Not Found` when OpenCode calls `/v1/responses`

**Solution needed:**
1. Add `/v1/responses` endpoint to `src/proxy_app/main.py`
2. Translate Responses API format → Chat Completions format
3. Call existing `chat_completions()` function
4. Translate response back to Responses API format
5. Commit → push to GitHub → VPS pulls automatically

**Test command:**
```bash
curl -X POST http://40.233.101.233:8000/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer CHANGE_ME_TO_A_STRONG_SECRET_KEY" \
  -d '{"model": "coding-elite", "input": [{"type": "message", "role": "user", "content": [{"type": "text", "text": "Hello"}]}]}'
```

## TESTING

```bash
# Test gateway is running
curl http://40.233.101.233:8000/v1/models \
  -H "Authorization: Bearer CHANGE_ME_TO_A_STRONG_SECRET_KEY"

# Test virtual model
curl -X POST http://40.233.101.233:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer CHANGE_ME_TO_A_STRONG_SECRET_KEY" \
  -d '{"model": "coding-elite", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10}'
```

## GIT WORKFLOW

**Remote:** https://github.com/ons96/LLM-API-Key-Proxy
**VPS pulls:** Changes auto-pull on VPS (verify with SSH)

```bash
# On VPS
ssh -i ~/.ssh/oracle.key ubuntu@40.233.101.233
cd ~/LLM-API-Key-Proxy
git pull

# Restart gateway
pkill -f 'main.py'
source venv/bin/activate
nohup python src/proxy_app/main.py --host 0.0.0.0 --port 8000 > ~/llm_proxy.log 2>&1 &
```

## OPENCODE CONFIGURATION

**File:** `/home/owens/.config/opencode/opencode.json`

Configured to use VPS gateway:
```json
{
  "model": "openai/coding-elite",
  "provider": {
    "openai": {
      "name": "My VPS LLM Gateway",
      "options": {
        "baseURL": "http://40.233.101.233:8000/v1",
        "apiKey": "CHANGE_ME_TO_A_STRONG_SECRET_KEY"
      }
    }
  }
}
```

## GOTCHAS

⚠️ **VPS uses different port than local:** Local testing uses 8000, ensure correct
⚠️ **API key in header:** Use `Authorization: Bearer <key>` (not `X-API-Key`)
⚠️ **Free-only mode:** Gateway rejects non-free providers unless configured
⚠️ **G4F models:** Some complex IDs don't work (stick to simple names like `g4f/gpt-4`)
