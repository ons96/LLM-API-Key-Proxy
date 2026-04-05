# AGENTS.md

## Universal LLM Gateway Router with Auto-Selection

---

## 1. Role/Mission

**Mission:** Build an OpenAI-compatible LLM gateway that provides intelligent model auto-selection, automatic fallbacks, rate limit management, and free tier access across multiple LLM providers.

### Key Objectives

- **Unified API Surface:** Expose a single OpenAI-compatible REST API (`/v1/chat/completions`, `/v1/models`) that abstracts away provider differences
- **Intelligent Auto-Selection:** Automatically route requests to the optimal model based on task type (coding-planning, coding-generation, chat-fast, chat-smartest, MoE)
- **Resilience via Fallbacks:** If a provider fails or hits rate limits, automatically fall back to the next available provider/model
- **Free Tier Access:** Utilize free tier quotas from multiple providers (Anthropic, Google, Mistral, Ollama local, etc.) to minimize costs
- **Virtual Model Abstraction:** Present logical "virtual models" (e.g., `gateway-coding-fast`, `gateway-planning-smart`) that map to real provider endpoints
- **MoE (Mixture of Experts):** Implement a routing layer that dispatches requests across multiple models and aggregates responses where appropriate

---

## 2. Technical Stack

| Component | Technology | Rationale |
|-----------|------------|------------|
| **API Framework** | FastAPI | High-performance, async, OpenAPI docs built-in, Python native |
| **HTTP Client** | httpx | Async HTTP, connection pooling, timeouts |
| **Configuration** | Pydantic Settings | Type-safe config, environment variable support |
| **Data Validation** | Pydantic | Request/response models, serialization |
| **Rate Limiting** | SlowAPI (or custom) | Token bucket algorithm per provider |
| **Caching** | redis (optional) / in-memory | Response caching for identical requests |
| **Logging** | structlog + uvicorn logs | Structured JSON logging |
| **Testing** | pytest + pytest-asyncio + httpx TestClient | Async test support |
| **Mocking** | respx | Mock httpx responses |
| **Documentation** | MkDocs + mike | Versioned API docs |

### Providers to Support (Free Tiers)

1. **Ollama** - Local LLaMA/Mistral models (run locally, free)
2. **OpenAI** - GPT-4 (free tier credits or paid)
3. **Anthropic** - Claude (free tier available)
4. **Google AI** - Gemini (free tier)
5. **Mistral AI** - Mistral (free tier)
6. **Groq** - Fast inference (free tier)

---

## 3. Requirements (Numbered)

### Core Functionality

1. **OpenAI-Compatible Endpoints**
   - Implement `/v1/chat/completions` (POST)
   - Implement `/v1/models` (GET)
   - Implement `/health` (GET)
   - Match OpenAI request/response schemas (`messages`, `model`, `temperature`, `max_tokens`, etc.)

2. **Model Auto-Selection Logic**
   - Define task categories: `coding-planning`, `coding-generation`, `chat-fast`, `chat-smartest`, `moe`
   - Map incoming `model` parameter to provider-specific model IDs
   - Implement selection rules (e.g., coding → DeepSeek-Coder or Claude-Sonnet, fast → Gemini-Flash)

3. **Fallback System**
   - Define fallback chains per model (primary → secondary → tertiary)
   - Detect provider errors: 429 (rate limit), 5xx, timeout, invalid API key
   - Automatic retry with exponential backoff (max 3 retries)
   - Track provider health status

4. **Rate Limit Management**
   - Implement token bucket or sliding window per provider API key
   - Queue requests when limit approached (optional: simple rejection)
   - Return `429` to client with `Retry-After` header if hard limit

5. **Virtual Models**
   - Define virtual model names: `gateway-fast`, `gateway-smart`, `gateway-coding`, `gateway-planning`
   - Map virtual → real provider model at runtime
   - Expose virtual models in `/v1/models` response

6. **MoE Routing**
   - For `moe` task type, dispatch to multiple models in parallel
   - Aggregate responses (e.g., majority vote or composite)
   - Return merged response with metadata

7. **Configuration Management**
   - Support `.env` file for API keys
   - Configure provider endpoints, model mappings, fallback chains
   - Support hot-reload of config (optional)

8. **Request/Response Transformations**
   - Convert OpenAI format → provider-specific format
   - Convert provider response → OpenAI format
   - Handle differences in function calling, tool-use

### Observability

9. **Logging**
   - Log all requests: `{method, model, provider, latency, status}`
   - Log fallback events: `{model, fallback_chain, error}`
   - Use structured JSON logs

10. **Metrics (Optional)**
    - Track request counts per provider
    - Track latency percentiles
    - Track fallback frequency

### Security

11. **Authentication**
    - Support `Authorization: Bearer