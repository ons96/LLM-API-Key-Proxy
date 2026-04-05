# AGENTS.md: Smart AI Model API Gateway with Cost-Performance Routing

---

## 1. Role/Mission

**Role:** Autonomous Coding Agent  
**Mission:** Build an intelligent API gateway that acts as a smart proxy layer between clients and multiple AI language model providers. The gateway must automatically select the optimal model for each request by analyzing task complexity, evaluating cost-performance tradeoffs, tracking token consumption against free quotas, and dynamically routing requests to maintain quality of service while minimizing expenses.

**Objective:** Create a production-ready system that reduces AI API costs by 60%+ through intelligent model routing while maintaining or improving response quality and availability.

---

## 2. Technical Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Core Framework** | Python 3.11+ | Required for litellm compatibility |
| **API Gateway** | FastAPI | High-performance async web framework |
| **LLM Abstraction** | litellm | Unified interface to 100+ LLM providers |
| **Database** | SQLite (via SQLAlchemy) | Free, zero-config, local tracking |
| **Caching** | In-memory dict + pickle | Free, simple token/quota caching |
| **Configuration** | Pydantic + .env files | Type-safe configuration management |
| **Task Analysis** | Heuristic complexity scoring | Free alternative to embeddings |
| **Testing** | pytest + pytest-asyncio | Industry-standard async testing |
| **HTTP Client** | httpx | Async HTTP for upstream calls |
| **Metrics** | prometheus-client | Free metrics export |

**Free External Resources (to use):**
- LiteLLM OpenAI-compatible proxy (self-hosted or free tier)
- Free tier API keys from: OpenAI, Anthropic, Google (configure via environment)
- Local SQLite database (no external service required)

---

## 3. Requirements

### 3.1 Core Functionality

1. **Model Routing Engine**
   - Implement a `ModelRouter` class that selects the optimal model based on: task type classification, estimated token count, latency requirements, and cost sensitivity score
   - Support at least 3 providers (OpenAI, Anthropic, Google) with unified API via litellm
   - Default routing logic: simple tasks → cheap models, complex tasks → capable models

2. **Token Tracking System**
   - Build a `TokenTracker` that records daily token usage per model/provider
   - Store: request_id, timestamp, model, input_tokens, output_tokens, cost_usd
   - Aggregate daily totals and persist to SQLite
   - Calculate running costs in real-time

3. **Quota Management**
   - Implement a `QuotaManager` that defines free tier limits per provider
   - Track daily/remaining quotas: input tokens, output tokens, requests
   - Raise alerts when quotas fall below 20% threshold
   - Implement automatic fallback to backup provider

4. **Cost Multiplier Handling**
   - Handle variable pricing: different models have different cost-per-token
   - Apply multipliers for: peak hours, batch processing, streaming
   - Calculate total cost per request and running totals

5. **Performance Benchmarking**
   - Cache latency measurements per model per task type
   - Track: first_token_latency, total_latency, tokens_per_second
   - Build a simple performance profile per model

6. **Load Balancing**
   - Distribute requests across multiple API keys for the same provider
   - Implement round-robin or least-loaded selection
   - Handle key rotation on rate limit errors

### 3.2 API Endpoints

7. **Gateway HTTP Interface**
   - POST `/v1/chat/completions` - Main chat completion endpoint (OpenAI-compatible)
   - POST `/v1/completions` - Legacy completion endpoint
   - GET `/health` - Health check endpoint
   - GET `/stats` - Return token usage statistics
   - POST `/admin/reset-quotas` - Reset quota counters (admin only)

8. **Webhook/Configuration Interface**
   - POST `/admin/config` - Update routing weights
   - POST `/admin/config` - Update quota limits

### 3.3 Observability

9. **Logging & Metrics**
   - Implement structured logging with request IDs
   - Track: requests_processed, tokens_used, costs_accrued, routing_decisions
   - Expose Prometheus metrics endpoint

### 3.4 Safety & Resilience

10. **Error Handling**
    - Graceful degradation when primary provider fails
    - Retry logic with exponential backoff
    - Circuit breaker pattern for failing providers

11. **Configuration Management**
    - All settings via config.yaml or environment variables
    - No hardcoded API keys (use environment)
    - Routing weights configurable at runtime

---

## 4. File Structure

```
ai-gateway/
├── .env.example                 # Example environment variables
├── .gitignore                  # Git ignore rules
├── CODEOWNERS                   # Code ownership
├── CONTRIBUTING.md             # Contribution guidelines
├── README.md                    # Project readme
├── pytest.ini                  # Pytest configuration
├── pyproject.toml               # Python project configuration
├── config.yaml                 # Gateway configuration
│
├── src/
│   └── gateway/
│       ├── __init__.py
│       ├── main.py             # FastAPI application entry point
│       ├── config.py           # Configuration management
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── router.py       # Model routing logic
│       │   ├── tracker.py      # Token tracking
│       │   ├── quota.py        # Quota management
│       │   └── cost.py         # Cost calculation
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── requests.py     # Pydantic request models
│       │   ├── responses.py     # Pydantic response models
│       │   └── tasks.py        # Task type classification
│       │
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── base.py         # Base provider interface
│       │   ├── litellm_.py     # LiteLLM provider wrapper
│       │   └── pool.py         # API key pool management
│       │
│       ├── storage/
│       │   ├── __init__.py
│       │   ├── database.py     # SQLite database setup
│       │   ├── token_repo.py   # Token usage repository
│       │   └── quota_repo.py   # Quota repository
│       │
│       ├── api/
│       │   ├── __init__.py
│       │   ├── routes.py       # API routes
│       │   ├── middleware.py   # Request/response middleware
│       │   └── errors.py       # Error handling
│       │
│       └── metrics/
│           ├── __init__.py
│           ├── collector.py     # Metrics collection
│           └── prometheus.py    # Prometheus export
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # Pytest fixtures
│   │
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_router.py
│   │   ├── test_tracker.py
│   │   ├── test_quota.py
│   │   └── test_cost.py
│   │
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_api.py
│   │   └── test_providers.py
│   │
│   └── mocks/
│       ├── __init__.py
│       ├── mock_litellm.py
│       └── mock_providers.py
│
├── scripts/
│   ├── init_db.py              # Initialize database
│   ├── seed_config.py           # Seed default configuration
│   └── run_gateway.py          # Run the gateway
│
├── docs/
│   ├── ARCHITECTURE.md         # System architecture
│   ├── API_SPEC.md             # API specification
│   └── ROUTING_LOGIC.md        # Routing decision explanation
│
└── QUESTIONS.md                # Questions for human review
```

---

## 5. Testing Requirements

### 5.1 Unit Tests (Minimum: 90% Coverage)

| Module | Tests Required |
|--------|----------------|
| `router.py` | Test model selection, task classification, weight application |
| `tracker.py` | Test token recording, aggregation, cost calculation |
| `quota.py` | Test quota checking, threshold alerts, reset logic |
| `cost.py` | Test cost multipliers, currency conversion |

### 5.2 Integration Tests

- Test end-to-end request flow through the gateway
- Test provider failover when primary is unavailable
- Test quota enforcement triggers fallback routing
- Test health endpoint returns correct status

### 5.3 Test Fixtures

- Mock litellm responses to avoid hitting real APIs
- Use test database (in-memory SQLite)
- Mock environment variables for API keys

### 5.4 Performance Tests (Optional)

- Load test with 100 concurrent requests
- Benchmark routing decision latency

### 5.5 Test Execution

```bash
# Run all tests with coverage
pytest --cov=src/gateway --cov-report=html

# Run only unit tests
pytest tests/unit/ -v

# Run with verbose output
pytest -vv
```

---

## 6. Git Protocol

### 6.1 Branch Strategy

- **Main branch**: `main` - Production-ready code only
- **Development branch**: `develop` - Integration branch
- **Feature branches**: `feature