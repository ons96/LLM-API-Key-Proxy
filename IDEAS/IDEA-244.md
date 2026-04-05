# AGENTS.md: Intelligent LLM Gateway with Robust Model Fallback Chains

## 1. Role/Mission

**Mission:** Design, implement, and deploy an Intelligent LLM Gateway API that serves as an intelligent intermediary layer between client applications and multiple LLM providers. The gateway must autonomously route requests to virtual models, manage fallback chains when models fail, handle rate limiting and errors gracefully, and dynamically reorder models based on real-time performance metrics—all without exposing internal errors to external systems like "Omo."

**Key Objectives:**
- Create a unified gateway API that abstracts away the complexity of multi-model LLM routing
- Implement robust fallback chains that activate automatically on any failure
- Ensure zero error propagation to Omo (all errors handled internally)
- Dynamically optimize model ordering based on real-time performance data
- Manage rate limits proactively to prevent token exhaustion or service degradation
- Enable virtual model configuration where one model name can represent multiple underlying models with automatic switching

---

## 2. Technical Stack

**Core Technologies:**
- **Language:** Python 3.11+ (async-first for concurrent request handling)
- **Database:** SQLite (tracking model status, performance metrics, fallback history)
- **HTTP Server:** FastAPI (built-in async support, easy middleware, OpenAPI docs)
- **HTTP Client:** httpx (async HTTP calls to LLM providers)
- **Retry Logic:** Tenacity or custom async retry decorator
- **Caching:** In-memory TTL cache for rate limit tracking

**Key Libraries:**
```
fastapi>=0.109.0
httpx>=0.26.0
tenacity>=8.2.0
aiosqlite>=0.19.0
pydantic>=2.5.0
python-dotenv>=1.0.0
```

**Free Resource Constraints:**
- Use only free-tier LLM APIs (e.g., Ollama local, Cerebras free tier, Groq free tier, Together AI free credits)
- No paid external services unless absolutely necessary
- Self-hosted options preferred (local LLM via Ollama)

---

## 3. Requirements

### 3.1 Core Requirements

1. **Virtual Model Abstraction**
   - Define virtual model names that map to one or more underlying LLM provider endpoints
   - Support different reasoning effort levels per virtual model (e.g., "fast", "balanced", "deep")
   - Allow flexible mapping where one virtual model can auto-select among multiple providers

2. **Automatic Fallback Chains**
   - Implement a configurable chain: `Primary Model → Fallback Model 1 → Fallback Model 2 → ... → Default Model`
   - Trigger fallbacks on ANY error: API errors, rate limits, timeout, malformed response, invalid API key
   - Never expose fallback attempts or errors to Omo (gateway returns final successful response or fails silently with cached response)
   - Log all fallback events internally for debugging

3. **Error Suppression Layer**
   - Catch ALL exceptions internally
   - Return a sanitized response to Omo (success with data, or graceful failure with cached/stub response)
   - Never propagate: HTTP errors, JSON parse errors, timeout errors, rate limit errors, authentication errors
   - Ensure Omo's fallback system is never activated (gateway handles everything)

4. **Rate Limit Management**
   - Track rate limits per provider in SQLite
   - Implement token bucket or sliding window algorithm
   - Preemptively switch models when approaching rate limits
   - Monitor usage and log warnings when approaching limits

5. **Dynamic Model Ordering**
   - Track success rate, latency, and cost per model in SQLite
   - Implement scoring algorithm: `score = (success_rate * weight) - (latency * weight) - (cost * weight)`
   - Automatically promote better-performing models in the fallback chain
   - Periodic re-evaluation (every N requests or time-based)

6. **Retry Logic with Max Attempts**
   - Implement exponential backoff between fallback attempts (1s, 2s, 4s, ...)
   - Hard cap: Maximum 3 fallback attempts per request
   - Timeout handling: 30-second default timeout per model
   - Maximum total retries across all models: 10 attempts total per request

### 3.2 System Requirements

7. **Configuration Management**
   - All configuration via `config.yaml` or environment variables
   - Virtual model definitions, API keys, fallback chains all configurable
   - No hardcoded values in source code

8. **Health Checks**
   - Endpoint `/health` returning gateway status
   - Endpoint `/models` returning available virtual models and their status
   - Internal health check that verifies all configured providers are reachable

9. **Metrics and Logging**
   - Log all requests: timestamp, virtual model, actual model used, success/failure, latency
   - Store metrics in SQLite for analysis
   - Use structured logging (JSON) for parseability

10. **Graceful Shutdown**
    - Handle SIGTERM/SIGINT to complete in-flight requests
    - Persist state to SQLite before shutdown

---

## 4. File Structure

```
llm_gateway/
├── AGENTS.md                    # This file
├── README.md                    # Project overview
├── config.yaml                  # Configuration file
├── .env.example                 # Environment variables template
├── requirements.txt             # Python dependencies
├── pyproject.toml               # Project metadata
├── src/
│   ├── __init__.py
│   ├── main.py                  # FastAPI application entry point
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py             # Configuration loader
│   │   ├── router.py             # API route handlers
│   │   └── models.py             # Pydantic models/schemas
│   ├── gateway/
│   │   ├── __init__.py
│   │   ├── client.py             # LLM client abstraction
│   │   ├── fallback.py           # Fallback chain logic
│   │   ├── rate_limiter.py       # Rate limit management
│   │   ├── router.py            # Model routing logic
│   │   └── scorer.py            # Dynamic model scoring
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── database.py           # SQLite connection and init
│   │   ├── metrics.py            # Metrics storage/retrieval
│   │   └── config_store.py       # Model config storage
│   └── utils/
│       ├── __init__.py
│       ├── logging.py           # Structured logging setup
│       └── retry.py              # Retry decorators
└── tests/
    ├── __init__.py
    ├── test_gateway/
    │   ├── __init__.py
    │   ├── test_fallback.py     # Fallback chain tests
    │   ├── test_rate_limiter.py # Rate limit tests
    │   └── test_router.py       # Routing tests
    ├── test_integration/
    │   │   ├── __init__.py
    │   │   └── test_end_to_end.py # E2E tests
    └── conftest.py               # Pytest fixtures
```

---

## 5. Testing Requirements

### 5.1 Unit Tests

- **Test Fallback Chain**: Verify fallback triggers on each error type (400, 401, 429, 500, timeout)
- **Test Rate Limiter**: Verify rate limiting blocks requests when limit exceeded
- **Test Router**: Verify virtual model correctly resolves to actual provider
- **Test Scorer**: Verify model scoring produces expected rankings
- **Test Retry Logic**: Verify exponential backoff and max attempt limits

### 5.2 Integration Tests

- **End-to-End Flow**: Send a request through gateway, verify fallback works when primary fails
- **Error Suppression**: Verify no errors leak to mock Omo client
- **Health Check**: Verify `/health` returns accurate status when models up/down
- **Configuration Reload**: Verify config changes apply without restart

### 5.3 Test Coverage Goals

- Minimum 80% code coverage
- All public functions must have tests
- Use free LLM endpoints for integration testing (mock responses allowed)

---

## 6. Git Protocol

### 6.1 Branch Strategy

- **Main branch**: Production-ready code only
- **Feature branches**: `feature