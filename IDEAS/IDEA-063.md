# LLM Gateway with Fallback Providers

## AGENTS.md

---

## 1. Role/Mission

**Role:** Autonomous LLM Gateway Agent

**Mission:** Build and maintain an intelligent LLM (Large Language Model) gateway system that automatically scrapes leaderboard data from multiple sources (UGI, models.dev) to create dynamic fallback model orders. The system must integrate free API providers, route requests intelligently based on model performance/d availability, and handle provider failures gracefully—all while operating exclusively on free resources.

**Primary Objectives:**
- Continuously discover and catalog free LLM API providers from leaderboard sources
- Build dynamic fallback chains ordered by model capability and availability
- Route LLM requests to optimal available providers with automatic failover
- Operate the gateway using only free resources (free hosts, combined free services, or cost-effective hardware)
- Ensure 99%+ uptime through intelligent fallback routing

---

## 2. Technical Stack

### Core Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| Language | Python 3.11+ | Primary implementation language |
| HTTP Server | FastAPI | Lightweight API gateway |
| Async Runtime | asyncio | Concurrent request handling |
| HTTP Client | httpx | Async HTTP requests to providers |
| Caching | in-memory (dict/LRU) | Fast model/provider metadata cache |
| Scheduler | APScheduler | Periodic leaderboard refresh |
| Data Storage | JSON/YAML files | Configuration and provider registry |
| Logging | structlog | Structured logging |

### Leaderboard Sources

| Source | API Type | Data Retrieved |
|--------|---------|----------------|
| UGI Leaderboard | Web scraping/API | Model rankings, performance metrics |
| models.dev | Web scraping/API | Free provider endpoints, rate limits |

### Free Provider Integration

- **Potential Free Providers:** (to be discovered via scraping)
  - OpenRouter
  - Lyzr
  - Infolox
  - Other free tier APIs from models.dev
  - Open-source model endpoints

### Hosting Options (Priority Order)

1. **Free Tier Services:** Render Free, Railway, Fly.io free tier, Glitch, Replit (if applicable)
2. **Combined Free Hosts:** Aggregate multiple free tier services for redundancy
3. **Self-Hosted:** Raspberry Pi or cheap VPS if more cost-effective
4. **Pay-Per-Usage:** Consider if usage-based pricing beats fixed free tier limits

### Development Tools

| Tool | Purpose |
|------|----------|
| uv | Package management |
| ruff | Linting |
| mypy | Type checking |
| pytest | Testing framework |
| pytest-asyncio | Async testing |
| requests-mock | HTTP mocking for tests |

---

## 3. Requirements

### Phase 1: Core Infrastructure

1. **Initialize Project:** Set up FastAPI project with proper structure
2. **Configuration System:** Create YAML-based config for all settings
3. **Logging System:** Implement structured logging with appropriate log levels
4. **Health Check Endpoint:** Create `/health` endpoint for gateway status
5. **Basic LLM Proxy:** Implement `/v1/chat/completions` proxy endpoint

### Phase 2: Leaderboard Integration

6. **UGI Leaderboard Scraper:** Create scraper for UGI leaderboard data
7. **models.dev Scraper:** Create scraper for free provider endpoints
8. **Data Normalization:** Normalize leaderboard data into unified format
9. **Periodic Refresh:** Implement scheduler for periodic leaderboard updates
10. **Provider Registry:** Store discovered providers in registry file

### Phase 3: Fallback Routing

11. **Provider Interface:** Define abstract provider interface
12. **Provider Implementations:** Implement client for each free provider type
13. **Fallback Chain Builder:** Build ordered fallback chains from leaderboard data
14. **Request Router:** Route requests through fallback chain on failure
15. **Rate Limit Handling:** Implement rate limit detection and provider rotation

### Phase 4: Intelligence & Optimization

16. **Performance Tracking:** Track success/failure rates per provider
17. **Dynamic Reordering:** Reorder fallback chains based on performance
18. **Cost Tracking:** Monitor token usage and costs (even free tiers have limits)
19. **Circuit Breaker:** Implement circuit breaker pattern for failing providers

### Phase 5: Reliability & Security

20. **Error Handling:** Comprehensive error handling with appropriate HTTP codes
21. **Request Validation:** Validate all incoming request payloads
22. **Secure API Keys:** Secure storage for provider API keys
23. **Input Sanitization:** Sanitize inputs before sending to providers
24. **Timeout Configuration:** Configurable timeouts for all provider requests

### Phase 6: Documentation

25. **API Documentation:** OpenAPI docs for all endpoints
26. **Provider Setup Guide:** Document how to add new providers
27. **Architecture Docs:** Document system design and flow

---

## 4. File Structure

```
llm-gateway/
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI workflow
├── src/
│   └── llm_gateway/
│       ├── __init__.py
│       ├── main.py                 # FastAPI application entry point
│       ├── config.py               # Configuration management
│       ├── logging_config.py       # Logging setup
│       ├── models/
│       │   ├── __init__.py
│       │   ├── requests.py         # Pydantic request models
│       │   └── responses.py        # Pydantic response models
│       ├── scrapers/
│       │   ├── __init__.py
│       │   ├── base.py              # Base scraper class
│       │   ├── ugi.py               # UGI leaderboard scraper
│       │   └── models_dev.py        # models.dev scraper
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── base.py              # Base provider interface
│       │   ├── openrouter.py        # OpenRouter provider
│       │   ├── generic.py           # Generic HTTP provider
│       │   └── registry.py          # Provider registry
│       ├── router/
│       │   ├── __init__.py
│       │   ├── fallback.py          # Fallback routing logic
│       │   └── request_handler.py   # Main request handler
│       ├── services/
│       │   ├── __init__.py
│       │   └── leaderboard.py      # Leaderboard data service
│       └── utils/
│           ├── __init__.py
│           └── http.py             # HTTP client utilities
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_config.py
│   │   ├── test_scrapers/
│   │   │   ├── __init__.py
│   │   │   ├── test_ugi.py
│   │   │   └── test_models_dev.py
│   │   ├── test_providers/
│   │   │   ├── __init__.py
│   │   │   └── test_registry.py
│   │   └── test_router/
│   │       ├── __init__.py
│   │       └── test_fallback.py
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_endpoints.py
│   │   └── test_full_flow.py
│   └── fixtures/
│       ├── __init__.py
│       └── mock_leaderboard.json
├── config/
│   ├── defaults.yaml              # Default configuration
│   ├── providers.yaml              # Provider-specific config
│   └── .env.example                # Environment variables template
├── pyproject.toml                  # Project metadata and dependencies
├── uv.lock                         # Locked dependencies
├── ruff.toml                       # Linter configuration
├── mypy.ini                        # Type checker configuration
├── Dockerfile                      # Container build (optional)
├── docker-compose.yml              # Local development setup
├── README.md                       # Project overview
├── AGENTS.md                       # This file
├── QUESTIONS.md                   # Questions for human review
└── LICENSE                        # MIT License
```

---

## 5. Testing Requirements

### Testing Philosophy

- **Test Behavior, Not Implementation:** Focus on verifying correct behavior rather than internal implementation details
- **Fail Fast:** Tests should fail quickly with clear error messages
- **Independence:** Each test should be independent and not rely on external services

### Test Coverage Requirements

| Category | Coverage Target | Notes |
|----------|-----------------|-------|
| Unit Tests | 80%+ | Core logic, scrapers, providers |
| Integration Tests | All endpoints | HTTP status codes, response formats |
| Mocked External Calls | 100% | No real HTTP calls in CI |

### Testing Guidelines

1. **Use Fixtures:** Create fixtures for common test data (mock leaderboard responses, provider configs)
2. **Mock HTTP:** Use `requests-mock` or similar for HTTP mocking
3. **Async Testing:** Use `pytest-asyncio` for async test functions
4. **Environment:** Tests should not require environment variables; use config overrides
5. **Temporary Files:** Use `tempfile` for any file-based operations in tests

### Required Test Cases

- **Scrapers:**
  - Test successful data extraction from mock HTML/JSON
  - Test handling of malformed data
  - Test handling of network errors

- **Providers:**
  - Test successful requests
  - Test rate limit handling
  - Test error propagation

- **Fallback Router:**
  - Test successful fallback on provider failure
  - Test exhausted fallback chain
  - Test provider reordering

- **Endpoints:**
  - Test health check
  - Test chat completions with valid request
  - Test error handling for invalid requests

### CI Testing

- Run on every push to `main` and PRs to `main`
- Run linting (ruff) before tests
- Run type checking (mypy) before tests
- Run tests with coverage reporting
- Fail if coverage drops below threshold

---

## 6. Git Protocol

### Branch Strategy

- **Main Branch:** `main` - Production-ready code only
- **Development Branches:** `feature/*`, `fix/*` - Working branches
- **Release Branches:** `release/*` - Release preparation

### Commit Messages

Follow conventional commits format:

```
