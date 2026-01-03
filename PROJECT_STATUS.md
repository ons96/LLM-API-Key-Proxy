# Project Status: LLM-API-Key-Proxy

## Overview

LLM-API-Key-Proxy is a universal OpenAI-compatible LLM proxy server that provides a single API endpoint for multiple LLM providers. The project enables seamless failover, key rotation, and intelligent routing without requiring code changes in client applications.

### Core Components

| Component | Description | Location |
|-----------|-------------|----------|
| **API Proxy** | FastAPI application providing universal `/v1/chat/completions` endpoint | `src/proxy_app/` |
| **Resilience Library** | Python library for API key management, rotation, and failover | `src/rotator_library/` |
| **Provider Plugins** | Provider-specific implementations for authentication and routing | `src/rotator_library/providers/` |

---

## Architecture Overview

### FastAPI Server (`src/proxy_app/`)

The proxy server handles incoming requests and routes them to appropriate providers:

- **main.py**: Entry point and FastAPI application setup
- **TUI Launcher**: Rich-based text UI for interactive configuration
- **Request Batching**: Efficient embedding request aggregation
- **Logging**: Detailed per-request logging for debugging

### RotatingClient (`src/rotator_library/client.py`)

The core resilience component provides:

- **Async-native architecture**: Built on `asyncio` and `httpx`
- **Intelligent key selection**: Tiered, model-aware credential locking
- **Deadline-driven requests**: Configurable global timeouts with retries
- **Automatic failover**: Seamless rotation between keys on errors
- **OAuth support**: Gemini CLI, Antigravity, Qwen, iFlow providers

### Provider Plugin System (`src/rotator_library/providers/`)

Extensible provider implementations:

| Provider | Auth Type | Features |
|----------|-----------|----------|
| Gemini | API Key / OAuth | Standard + CLI OAuth |
| OpenAI | API Key | Standard, Azure |
| Anthropic | API Key | Claude models |
| OpenRouter | API Key | Multi-provider routing |
| Groq | API Key | Fast inference |
| Mistral | API Key | Mistral models |
| NVIDIA | API Key | NIM endpoints |
| Cohere | API Key | Command models |
| Chutes | API Key | Custom models |
| Antigravity | OAuth | Gemini 3, Claude 4.5 |
| Qwen Code | API Key + OAuth | Dual auth, reasoning content |
| iFlow | API Key + OAuth | Hybrid auth |

---

## Phase 1 Completion Status: G4F Fallback Providers Integration


### Verification Results

| Item | Status | Details |
|------|--------|---------|
| G4F provider config in `.env.example` | ✅ COMPLETE | Added G4F_API_KEY, G4F_MAIN_API_BASE, and PROVIDER_PRIORITY_* variables |
| PROVIDER_PRIORITY_* variables | ✅ COMPLETE | Priority tier variables added to `.env.example` |
| G4F Fallback section in `README.md` | ✅ COMPLETE | "G4F Fallback Providers" section added |

### Current Status: PHASE 1 & 2 COMPLETE

G4F integration is fully implemented. The fallback system:
1.  **Priorities**: Respects `PROVIDER_PRIORITY_*` tiers.
2.  **Logic**: `client.py` loops through the `get_fallback_chain` upon provider failure.
3.  **Tests**: `test_g4f_integration.py` verifies both provider instantiation and priority logic.

---

## Key Features Implemented

### Core Capabilities

- ✅ Universal OpenAI-compatible endpoint (`/v1/chat/completions`)
- ✅ Multi-provider support via LiteLLM integration
- ✅ Automatic key rotation and load balancing
- ✅ Interactive TUI for configuration
- ✅ Detailed request logging

### Resilience & High Availability

- ✅ Global timeout with deadline-driven retries
- ✅ Escalating cooldowns per model (10s → 30s → 60s → 120s)
- ✅ Key-level lockouts for consistently failing keys
- ✅ Stream error detection and graceful recovery
- ✅ Batch embedding aggregation
- ✅ Automatic daily resets for cooldowns and usage stats

### Credential Management

- ✅ Auto-discovery of API keys from environment variables
- ✅ OAuth discovery from standard paths
- ✅ Duplicate detection
- ✅ Credential prioritization (paid vs free tier)
- ✅ Stateless deployment support
- ✅ Local-first credential storage (`oauth_creds/`)

### Advanced Configuration

- ✅ Model whitelists/blacklists with wildcard support
- ✅ Per-provider concurrency limits
- ✅ Rotation modes (balanced, sequential)
- ✅ Priority multipliers for concurrency
- ✅ Model quota groups
- ✅ Temperature override controls

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Status check |
| `/v1/chat/completions` | POST | Chat completions (main) |
| `/v1/embeddings` | POST | Text embeddings |
| `/v1/models` | GET | List available models with pricing |
| `/v1/models/{model_id}` | GET | Get specific model details |
| `/v1/providers` | GET | List configured providers |
| `/v1/token-count` | POST | Calculate token count |
| `/v1/cost-estimate` | POST | Estimate cost based on tokens |

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Web Framework | FastAPI |
| Server | Uvicorn |
| HTTP Client | httpx (async) |
| Provider Integration | LiteLLM |
| CLI/TUI | Rich |
| Configuration | python-dotenv |

---

## Configuration Files

| File | Purpose |
|------|---------|
| `.env` | Credentials and environment variables |
| `.env.example` | Template for environment configuration |
| `launcher_config.json` | TUI-specific settings |

---

## Model Format

All requests must use the `provider/model_name` format:

```
gemini/gemini-2.5-flash
openai/gpt-4o
anthropic/claude-3-5-sonnet
openrouter/anthropic/claude-3-opus
antigravity/claude-sonnet-4-5
```

---

## Next Steps

See [INTEGRATION_ROADMAP.md](INTEGRATION_ROADMAP.md) for detailed Phase 2 implementation requirements.
