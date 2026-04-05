# AGENTS.md

## 1. Role/Mission

**Role:** Senior Software Architect & Full-Stack Developer

**Mission:** Build a Dynamic LLM API Gateway that provides intelligent routing between multiple LLM providers using virtual model names. The gateway must:

- Expose consistent virtual model identifiers (`coding-best`, `coding-fast`, `chat-smart`, `chat-fast`) that map to actual provider models
- Automatically detect provider failures (rate limits, outages, configuration errors) and route requests to fallback models
- Track provider health metrics in Redis for intelligent load balancing
- Implement configurable priority files so model rankings can be modified without code changes
- Operate as a lightweight proxy server that transforms virtual model names to real provider-specific model names
- Use only free/zero-cost resources for API calls (OpenRouter free tier, g4f, etc.)

**Autonomy Level:** Make all technical decisions independently. Only escalate to QUESTIONS.md if architectural direction is unclear or if required external resources are unavailable.

---

## 2. Technical Stack

| Component | Technology | Notes |
|-----------|------------|-------|
| **API Gateway** | FastAPI (Python) | Lightweight, async-capable HTTP proxy |
| **LLM Providers** | OpenRouter API, g4f | Free tier aggregation |
| **Health Tracking** | Redis (via fakeredis for local dev) | Provider failure tracking |
| **Configuration** | YAML priority files | Model rankings, fallback chains |
| **Request Routing** | Custom async router | Failure detection, fallback logic |
| **Testing** | pytest, httpx TestClient | Unit + integration tests |
| **Deployment** | Python uvicorn | Local dev server |

**External Free Resources:**
- OpenRouter API (free models: llama-3.2, qwen, etc.)
- g4f library (free local providers)
- Redis cloud free tier (optional, with fakeredis fallback)

---

## 3. Requirements

### Core Gateway

1. **Virtual Model Name System**
   - Define virtual model names: `coding-best`, `coding-fast`, `chat-smart`, `chat-fast`
   - Create mapping from virtual names to provider-specific model names
   - Support config-file-based mapping changes without code modification

2. **Fallback Routing Logic**
   - When primary model fails, auto-route to next available model in priority list
   - Track failure reason: `rate_limit` (temporary, retry after cooldown), `outage` (provider down), `config_error` (invalid setup), `invalid_response` (malformed output)
   - Implement exponential backoff for rate-limited requests

3. **Provider Health Tracking**
   - Store provider health data in Redis with TTL
   - Track: failure count, last failure timestamp, failure type, cooldown timer
   - Implement "de-prioritization" - temporarily reduce priority of failing providers
   - Auto-recover providers after cooldown period

4. **Configurable Model Priorities**
   - YAML file defining each virtual model's provider priority list
   - Include weight/score for each provider to influence selection
   - Allow per-request provider exclusion

5. **Rate Limit Detection**
   - Parse rate limit headers from provider responses
   - Implement per-provider rate limit tracking
   - Queue requests during rate limit cooldown if needed

### API Endpoints

6. **POST /v1/chat/completions**
   - Accept virtual model name in request body
   - Transform to real provider model, execute LLM call
   - Return OpenAI-compatible response format

7. **GET /health**
   - Return gateway status and all provider health summaries

8. **POST /admin/reset-provider/{provider}**
   - Manual endpoint to reset provider health state (for testing)

### Infrastructure

9. **Logging & Monitoring**
   - Log all requests with virtual model, actual model, provider, outcome
   - Log failure reasons for debugging
   - Include request latency tracking

10. **Error Handling**
    - Graceful degradation when all providers fail
    - Return meaningful error messages to clients
    - Never expose internal API keys in errors

---

## 4. File Structure

```
llm-gateway/
├── AGENTS.md                    # This file
├── QUESTIONS.md                # Agent questions for human review
├── pyproject.toml              # Poetry/PEP 517 config
├── requirements.txt            # pip requirements
├── .env.example               # Environment template
│
├── config/
│   ├── model_priorities.yaml  # Virtual model → provider mappings
│   └── provider_configs.yaml  # API keys, endpoints, settings
│
├── src/
│   ├── __init__.py
│   ├── main.py                # FastAPI application entry point
│   ├── config.py             # Configuration loader
│   ├── models.py             # Pydantic request/response models
│   │
│   ├── router/
│   │   ├── __init__.py
│   │   ├── gateway.py        # Main routing logic
│   │   ├── fallback.py      # Fallback chain executor
│   │   └── selection.py     # Model selection algorithm
│   │
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py          # Abstract provider class
│   │   ├── openrouter.py    # OpenRouter implementation
│   │   ├── g4f_provider.py # g4f implementation
│   │   └── types.py         # Provider-specific types
│   │
│   ├── health/
│   │   ├── __init__.py
│   │   ├── manager.py       # Provider health tracking
│   │   ├── redis_client.py # Redis connection + fakeredis fallback
│   │   └── strategies.py   # Deprioritization strategies
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py        # Structured logging
│       └── errors.py        # Custom exception classes
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py         # pytest fixtures
│   ├── test_router/
│   │   ├── __init__.py
│   │   ├── test_gateway.py
│   │   ├── test_fallback.py
│   │   └── test_selection.py
│   │
│   ├── test_providers/
│   │   ├── __init__.py
│   │   ├── test_openrouter.py
│   │   └── test_g4f.py
│   │
│   ├── test_health/
│   │   ├── __init__.py
│   │   └── test_manager.py
│   │
│   └── test_integration/
│       ├── __init__.py
│       └── test_end_to_end.py
│
└── scripts/
    ├── run_gateway.py      # Start gateway server
    └── generate_config.py # Generate sample config files
```

---

## 5. Testing Requirements

### Unit Tests (Minimum 80% coverage)

- **Router Tests**
  - Test virtual → real model name mapping
  - Test fallback chain progression
  - Test provider selection with weighted scores
  - Test failure reason classification

- **Provider Tests**
  - Test OpenRouter request formation
  - Test g4f request formation
  - Test rate limit header parsing
  - Test error response handling

- **Health Manager Tests**
  - Test failure tracking storage
  - Test de-prioritization calculation
  - Test cooldown expiration logic
  - Test provider recovery detection

### Integration Tests

- **End-to-End Tests**
  - Test full request flow with mock providers
  - Test fallback on provider failure
  - Test rate limit retry behavior
  - Test health endpoint response

- **Testing Strategy**
  - Use `responses` library or unittest.mock for HTTP calls
  - Use fakeredis for Redis dependencies
  - Mock external API calls entirely (no real network requests in CI)

### Test Execution

```bash
# Run all tests with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run only unit tests (fast)
pytest tests/unit/ -v

# Run integration tests
pytest tests/integration/ -v
```

---

## 6. Git Protocol

### Commit Message Format

Use conventional commits with prefixes:

```
