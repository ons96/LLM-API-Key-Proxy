# Free LLM API Router & Smart Fallback Manager

## Project Overview

This project implements an intelligent proxy layer that automatically manages a pool of free/tiered LLM providers. It continuously monitors provider health, benchmarks performance, handles rate limits, and dynamically routes requests to optimal fallback chains to maintain high availability.

---

## 1. Role/Mission

### Primary Mission

Build an autonomous routing system that ensures maximum uptime and performance for free LLM API calls by automatically:

1. **Discovering and tracking** free LLM providers and models
2. **Continuously benchmarking** provider speed, accuracy, and availability
3. **Dynamically routing** requests to optimal providers based on real-time metrics
4. **Managing fallbacks** with intelligent chains when primary providers fail
5. **Enforcing rate limits** to prevent quota exhaustion and automatic provider disabling
6. **Auto-healing** by detecting degraded/offline providers and removing them from the pool

### Agent Responsibilities

- Make all architectural decisions independently based on the requirements
- Implement all components iteratively with working builds at each step
- Write comprehensive tests before considering features complete
- Document all decisions in the codebase
- Flag any blockers or clarifications needed in `QUESTIONS.md`

---

## 2. Technical Stack

### Core Framework
- **Proxy Server**: FastAPI (Python) for high-performance async handling
- **Language**: Python 3.11+

### Data & State Management
- **Redis**: Rate limiting, provider state, metrics caching
- **In-Memory**: Fast access hot path data

### Monitoring & Observability
- **Prometheus**: Metrics collection and querying
- **Grafana**: Visualization dashboards (via free tier or local)

### External Integrations
- **Free LLM Providers**: OpenRouter, other free tier APIs
- **MCP Tools**: For extensible tool integration

### Testing
- **pytest**: Unit and integration testing
- **pytest-asyncio**: Async test support

### Deployment (Free Resources Only)
- **GitHub Actions**: CI/CD pipeline
- **Railway/Render (Optional)**: Free tier hosting if needed

---

## 3. Requirements

### 3.1 Provider Discovery & Management

1. **Provider Registry**
   - Maintain a configurable list of free LLM providers with endpoint URLs, API key requirements, model lists
   - Support adding providers via configuration file (JSON/YAML)
   - Auto-discover available models by querying providers' `/models` endpoints

2. **Provider Health Monitor**
   - Ping each provider every 60 seconds to detect availability
   - Track consecutive failures and auto-disable providers after 3 failures
   - Auto-re-enable providers after 2 successful health checks

3. **Virtual Model Abstraction**
   - Define "virtual models" that map to one or more underlying provider models
   - Example: `gpt-3.5-turbo-fast` maps to `[openai-gpt3.5, anthropic-haiku, mistral]`
   - Support priority-ordered fallback chains per virtual model

### 3.2 Request Routing Engine

4. **Intelligent Routing**
   - Route requests to the fastest available provider for the requested virtual model
   - Consider provider latency (rolling average of last 50 requests)
   - Prioritize providers with available rate limit quota

5. **Fallback Chain Execution**
   - Automatically try next provider in chain on failure
   - Implement timeout handling (30 second default)
   - Track which provider succeeded for metrics

6. **Rate Limit Management**
   - Per-provider rate limit tracking (requests/minute, tokens/minute)
   - Redis-based distributed rate limiter using sliding window
   - Respect provider-specific limits and auto-pause when approaching threshold

### 3.3 Benchmarking System

7. **Automated Benchmarks**
   - Run lightweight benchmark every 5 minutes against all healthy providers
   - Measure: latency, successful response, JSON validity
   - Store benchmark results in Redis with timestamps

8. **Performance Metrics**
   - Track request latency (p50, p95, p99)
   - Track success rate per provider
   - Calculate "provider score" = (success_rate × 0.5) + (1/latency × 0.5)
   - Use scores to influence routing priority

### 3.4 API Implementation

9. **REST API Endpoints**
   - `POST /v1/chat/completions` - OpenAI-compatible chat completion endpoint
   - `GET /providers` - List all providers with status
   - `GET /models` - List available virtual models
   - `GET /health` - Router health check
   - `GET /metrics` - Prometheus-compatible metrics

10. **Request Transformation**
    - Accept OpenAI-format payloads
    - Transform to appropriate provider format based on routing decision
    - Transform provider responses back to OpenAI format

### 3.5 Observability

11. **Prometheus Metrics**
    - `llm_router_requests_total` - Counter by provider, status, model
    - `llm_router_latency_seconds` - Histogram by provider, model
    - `llm_router_provider_status` - Gauge (1=healthy, 0=degraded)
    - `llm_router_rate_limit_remaining` - Gauge per provider

12. **Logging**
    - Structured JSON logging with request IDs
    - Log routing decisions with reason codes
    - Log all errors with stack traces

---

## 4. File Structure

```
free-llm-router/
├──agents.md                    # This file
├──questions.md                # Agent questions for human review
├──requirements.txt             # Python dependencies
├──config.example.yaml         # Example configuration
├──.env.example                # Example environment variables
├──docker-compose.yml          # Local development stack
├──docker/
│   └── Dockerfile
├──src/
│   ├── __init__.py
│   ├── main.py                # FastAPI application entry point
│   ├── config.py              # Configuration management
│   ├── logging.py             # Logging setup
│   ├── models/
│   │   ├── __init__.py
│   │   ├── schemas.py         # Pydantic schemas
│   │   ├── provider.py        # Provider data models
│   │   └── virtual_model.py   # Virtual model definitions
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py            # Base provider class
│   │   ├── openrouter.py     # OpenRouter implementation
│   │   ├── anthropic.py      # Anthropic provider
│   │   ├── openai.py         # OpenAI provider
│   │   └── registry.py       # Provider registry
│   ├── routing/
│   │   ├── __init__.py
│   │   ├── engine.py         # Main routing engine
│   │   ├── fallback.py       # Fallback chain logic
│   │   └── scorer.py         # Provider scoring
│   ├── rate_limiting/
│   │   ├── __init__.py
│   │   ├── manager.py        # Rate limit manager
│   │   ├── redis_store.py   # Redis-based storage
│   │   └── sliding_window.py # Sliding window algorithm
│   ├── benchmarking/
│   │   ├── __init__.py
│   │   ├── runner.py         # Benchmark runner
│   │   └── tasks.py         # Benchmark task definitions
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── metrics.py       # Prometheus metrics
│   │   ├── health.py        # Health checks
│   │   └── collector.py     # Metrics collection
│   ├── api/
│   │   ├── __init__.py
│   │   ├── chat.py          # Chat completions endpoint
│   │   ├── providers.py     # Provider status endpoint
│   │   └── models.py        # Models listing endpoint
│   └── utils/
│       ├── __init__.py
│       ├── retry.py          # Retry logic
│       ├── timeout.py       # Timeout handling
│       └── validation.py     # Response validation
├──tests/
│   ├── __init__.py
│   ├── conftest.py          # Test fixtures
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_providers.py
│   │   ├── test_routing.py
│   │   └── test_rate_limiting.py
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_api.py
│   │   └── test_full_chain.py
│   └── fixtures/
│       ├── providers.json   # Test provider configs
│       └── requests.json   # Test requests
├──scripts/
│   ├── run_benchmark.py    # Manual benchmark runner
│   ├── check_providers.py  # Provider status checker
│   └── seed_config.py     # Configuration seeder
└──docs/
    ├── architecture.md
    ├── providers.md
    └── routing.md
```

---

## 5. Testing Requirements

### 5.1 Unit Tests

**All modules must have unit tests with >80% coverage:**

1. **Provider Tests**
   - Test provider initialization from config
   - Test health check timeout handling
   - Test model parsing

2. **Routing Tests**
   - Test fallback chain ordering
   - Test scoring algorithm
   - Test timeout handling

3. **Rate Limiting Tests**
   - Test sliding window calculation
   - Test rate limit detection
   - Test redis integration (mocked)

4. **Benchmarking Tests**
   - Test benchmark task execution
   - Test metrics storage

### 5.2 Integration Tests

5. **API Tests**
   - Test `/v1/chat/completions` with mock providers
   - Test fallback on provider failure
   - Test rate limit responses

6. **End-to-End Tests**
   - Test full request flow with mocked HTTP responses
   - Test graceful degradation

### 5.3 Test Execution

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/unit/test_routing.py

# Run integration only
pytest tests/integration/
```

---

## 6. Git Protocol

### Branch Strategy

- **`main`** - Production-ready code
- **`develop`** - Integration branch
- **`feature/*`** - New features
- **`fix/*`** - Bug fixes
- **`refactor/*`** - Code improvements

### Commit Conventions

Format: