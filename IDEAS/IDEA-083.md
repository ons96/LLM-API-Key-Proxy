# AGENTS.md - LLM API Proxy Gateway with Multi-Provider Fallback

## 1. Role/Mission

**Mission:** Build an autonomous LLM API proxy gateway that seamlessly routes requests across multiple LLM API providers with intelligent automatic fallback, ensuring high availability and redundancy for all supported models including virtual model aliases.

**Core Objectives:**
- Create a unified API gateway that abstracts away provider differences
- Implement automatic failover when providers fail or hit rate limits
- Support virtual model aliases (e.g., `coding-smart`, `coding-fast`) that map to optimal provider models
- Allow manual model/provider selection with provider chain traversal
- Provide health monitoring for all configured providers
- Use only free/trial resources to minimize costs during development

---

## 2. Technical Stack

| Component | Technology | Notes |
|-----------|------------|-------|
| **Language** | Python 3.11+ | Async-first design |
| **Web Framework** | FastAPI | High-performance, easy async |
| **HTTP Client** | httpx | Async HTTP for provider calls |
| **Rate Limiting** | slowapi | Token bucket implementation |
| **Caching** | In-memory dict or redis | For health check results |
| **Configuration** | Pydantic Settings | Environment-based config |
| **Testing** | pytest + pytest-asyncio | Async test support |
| **Mocking** | respx | HTTP mocking for tests |
| **API Documentation** | OpenAPI (auto) | Built-in FastAPI docs |

**Provider Support (Free Tier Targets):**
- OpenAI (free tier availability)
- Anthropic (free tier availability)
- Google Gemini (free tier)
- Cohere (free tier)
- Mistral (free tier)
- xAI Grok (free tier when available)

---

## 3. Requirements

### 3.1 Core Functionality

1. **Unified API Endpoint** - Single `/v1/chat/completions` endpoint mimicking OpenAI format
2. **Multi-Provider Routing** - Route requests to available providers based on model mapping
3. **Automatic Fallback** - Detect provider failures (5xx, rate limits) and automatically try next provider
4. **Virtual Model Aliases** - Support logical model names that map to specific provider models:
   - `coding-smart` → Best coding model available (e.g., Claude-3.5-Opus, GPT-4)
   - `coding-fast` → Fast coding model (e.g., GPT-4o-mini, Claude-3-Haiku)
   - `general-smart` → General purpose smart model
   - `general-fast` → General purpose fast model
5. **Manual Provider Selection** - Allow users to specify exact provider preference
6. **Provider Chain** - Try multiple providers in priority order until success or exhaust all
7. **Error Aggregation** - When all providers fail, return detailed error with each provider's failure reason

### 3.2 Provider Management

8. **Provider Health Checks** - Periodic background health checks for all configured providers
9. **Health Status API** - Endpoint to query current provider health status
10. **Dynamic Provider Weights** - Adjust provider priority based on recent success rates
11. **Configurable Retry Logic** - Per-provider retry count, timeout, and backoff settings

### 3.3 Rate Limiting & Controls

12. **Global Rate Limiting** - Per-minute request limits at gateway level
13. **Provider-Specific Rate Limits** - Respect individual provider rate limits
14. **Rate Limit Headers** - Return X-RateLimit-* headers to clients
15. **Request Queuing** - Queue requests when approaching rate limits (optional)

### 3.4 Monitoring & Observability

16. **Request Logging** - Log all requests with timing, provider, status
17. **Metrics Endpoint** - Basic metrics (requests by provider, success rate, latency)
18. **Health Check Endpoint** - `GET /health` returning overall system status

### 3.5 Developer Experience

19. **API Documentation** - Auto-generated OpenAPI docs at `/docs`
20. **Test Client** - Ready-to-use examples for testing integrations
21. **Configuration via Environment** - All settings via `.env` file

---

## 4. File Structure

```
llm-gateway/
├── .env.example                      # Example environment variables
├── .github/
│   └── workflows/
│       ├── test.yml                 # GitHub Actions test workflow
│       └── deploy.yml               # Optional deployment workflow
├── AGENTS.md                        # This file
├── QUESTIONS.md                     # Questions for human review
├── pyproject.toml                   # Project metadata & dependencies
├── uv.lock                          # Locked dependencies
├── README.md                        # Project documentation
├── src/
│   └── llm_gateway/
│       ├── __init__.py              # Package initialization
│       ├── main.py                  # FastAPI application entry point
│       ├── config.py                # Configuration management
│       ├── models/
│       │   ├── __init__.py
│       │   ├── requests.py           # Request models (Pydantic)
│       │   ├── responses.py         # Response models (Pydantic)
│       │   └── providers.py         # Provider configuration models
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── base.py              # Base provider abstract class
│       │   ├── openai.py           # OpenAI provider implementation
│       │   ├── anthropic.py       # Anthropic provider implementation
│       │   ├── google.py           # Google Gemini provider
│       │   ├── cohere.py           # Cohere provider
│       │   ├── mistral.py          # Mistral provider
│       │   └── registry.py         # Provider registry/mapping
│       ├── routing/
│       │   ├── __init__.py
│       │   ├── router.py           # Main routing logic
│       │   ├── fallback.py         # Fallback sequence logic
│       │   └── mapping.py          # Model alias mappings
│       ├── middleware/
│       │   ├── __init__.py
│       │   ├── rate_limit.py        # Rate limiting middleware
│       │   └── health_check.py     # Health check scheduler
│       ├── monitoring/
│       │   ├── __init__.py
│       │   ├── logger.py           # Request logging
│       │   └── metrics.py         # Metrics collection
│       └── utils/
│           ├── __init__.py
│           └── exceptions.py       # Custom exceptions
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Pytest fixtures
│   ├── test_providers/
│   │   ├── __init__.py
│   │   ├── test_openai.py         # OpenAI provider tests
│   │   ├── test_anthropic.py      # Anthropic provider tests
│   │   └── test_registry.py       # Provider registry tests
│   ├── test_routing/
│   │   ├── __init__.py
│   │   ├── test_router.py         # Routing logic tests
│   │   └── test_fallback.py      # Fallback logic tests
│   ├── test_integration/
│   │   ├── __init__.py
│   │   └── test_end_to_end.py    # End-to-end tests
│   └── test_main/
│       ├── __init__.py
│       └── test_health.py         # Health endpoint tests
├── scripts/
│   ├── generate_env.py           # Generate .env from template
│   └── run_health_checks.py      # Manual health check runner
└── docs/
    └── api-reference.md          # API documentation
```

---

## 5. Testing Requirements

### 5.1 Test Coverage Goals

| Category | Target Coverage |
|----------|-----------------|
| Core routing logic | 90%+ |
| Provider implementations | 85%+ |
| Fallback mechanisms | 90%+ |
| Overall project | 80%+ |

### 5.2 Test Types Required

1. **Unit Tests** - Test individual functions and classes in isolation
2. **Provider Tests** - Test each provider implementation with mocked responses
3. **Routing Tests** - Test model mapping and provider selection logic
4. **Fallback Tests** - Test automatic failover with simulated failures
5. **Integration Tests** - Test full request/response flow with mocked providers
6. **Health Check Tests** - Test health check scheduler and status reporting

### 5.3 Test Fixtures

```python
# tests/conftest.py key fixtures
@pytest.fixture
def mock_providers():
    """Mock all provider HTTP calls"""
    
@pytest.fixture  
def sample_request():
    """Sample chat completion request"""
    
@pytest.fixture
def provider_config():
    """Test provider configurations"""
```

### 5.4 Running Tests

```bash
# Run all tests with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test files
uv run pytest tests/test_routing/

# Run with verbose output
uv run pytest -v
```

---

## 6. Git Protocol

### 6.1 Branch Strategy

| Branch | Purpose | Protected |
|--------|---------|-----------|
| `main` | Production code | Yes |
| `develop` | Integration branch | Yes |
| `feature/*` | New features | No |
| `bugfix/*` | Bug fixes | No |
| `test/*` | Experiments | No |

### 6.2 Commit Convention

Format: