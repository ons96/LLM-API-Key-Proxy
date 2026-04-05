# AGENTS.md - Intelligent Model Fallback & Rate-Limit Blacklist System

## 1. Role/Mission

### Purpose
Build an intelligent gateway service that automatically manages multi-model AI interactions by implementing smart fallback chains and rate-limit awareness to maximize成功率 while minimizing wasted API calls and costs.

### Mission
Create a robust, self-healing model routing system that:
- **Detects failures** in real-time (rate limits, payment errors, provider downtime, unavailable models)
- **Intelligently chains fallbacks** through a configurable list of models until success
- **Blocks/blacklists** problematic models temporarily based on rate-limit tracking
- **Optimizes API usage** by avoiding known-bad models until they're healthy again
- **Tracks temporal patterns** to learn when providers typically have capacity

### Success Criteria
The system achieves >95% successful request completion by automatically routing around failures using fallback models, while reducing unnecessary API calls by at least 40% through intelligent blocking.

---

## 2. Technical Stack

### Language & Runtime
- **Language**: Python 3.10+
- **Runtime**: Standard library with minimal external dependencies

### Core Dependencies
- `httpx` - Modern async HTTP client (for API calls)
- `asyncio` - Async/await concurrency
- `pydantic` - Data validation and settings management (optional, can use dataclasses)
- `pytest` - Testing framework
- `pytest-asyncio` - Async test support

### Storage
- **In-memory caching** with JSON file persistence for:
  - Model status/integrity records
  - Rate-limit tracking data
  - Blocklist with timestamps
- **Optional**: SQLite for persistent storage if needed

### Free Resources Only
- GitHub Actions (free tier) for CI/CD
- No paid external services
- localhost testing where possible
- Mock external API responses for testing

---

## 3. Requirements

### 3.1 Model Registry & Configuration
1. Define a flexible model configuration system that supports multiple providers (OpenAI, Anthropic, Google, local, etc.)
2. Support ordered fallback chains (list of models to try in priority order)
3. Allow per-model settings: timeout, max retries, rate-limit window, block duration
4. Support provider aliases (e.g., "gpt-4" → specific deployment)

### 3.2 Failure Detection Engine
5. Implement comprehensive failure detection for:
   - HTTP 429 (rate limit)
   - HTTP 402 (payment required)
   - HTTP 503/504 (service unavailable)
   - HTTP 500+ (server errors)
   - Timeout detection
   - Invalid API key errors
   - Model not found/unavailable errors
6. Extract retry-after headers when available
7. Parse error messages to identify specific failure types

### 3.3 Rate-Limit Tracking System
8. Track rate limit occurrences per model with timestamps
9. Implement time-window-based analysis (configurable window, default 60 seconds)
10. Calculate rate-limit probability based on historical patterns
11. Store historical rate-limit data for pattern learning

### 3.4 Intelligent Blocking/Blacklist System
12. Implement automatic temporary blacklisting when rate limits detected
13. Support configurable block durations (default: exponential backoff starting at 30s)
14. Implement automatic unblocking when block duration expires
15. Support manual unblock/force-block operations
16. Track blocking reason and timestamps for audit

### 3.5 Fallback Chain Logic
17. Implement sequential fallback traversal through model list
18. Skip blocked models during fallback traversal
19. Stop fallback chain on first successful response
20. Return detailed fallback history (which models tried, why failed)

### 3.6 Request Orchestration
21. Implement async HTTP request handling with configurable timeouts
22. Support request queuing and concurrency limits
23. Implement circuit breaker pattern (stop calling failed provider after N consecutive failures)
24. Support request context/correlation IDs for tracing

### 3.7 Monitoring & Observability
25. Log all fallback events with timestamps and reasons
26. Log all blocking/unblocking events
27. Track success/failure rates per model
28. Provide health check endpoint showing current model statuses

### 3.8 Configuration Management
29. Support YAML/JSON configuration files
30. Support environment variable overrides
31. Support dynamic configuration reloading

---

## 4. File Structure

```
intelligent-fallback-gateway/
├── .github/
│   └── workflows/
│       └── continuous-integration.yml    # GitHub Actions CI
├── src/
│   └── fallback_gateway/
│       ├── __init__.py
│       ├── main.py                       # Entry point
│       ├── config/
│       │   ├── __init__.py
│       │   ├── settings.py               # Configuration management
│       │   └── models.py                 # Model config schemas
│       ├── core/
│       │   ├── __init__.py
│       │   ├── failure_detector.py       # Failure detection logic
│       │   ├── rate_limiter.py           # Rate-limit tracking
│       │   ├── blacklist.py              # Blocking system
│       │   ├── circuit_breaker.py        # Circuit breaker pattern
│       │   └── fallback_chain.py          # Fallback orchestration
│       ├── http/
│       │   ├── __init__.py
│       │   ├── client.py                  # HTTP client wrapper
│       │   └── middleware.py             # Request/response middleware
│       ├── storage/
│       │   ├── __init__.py
│       │   ├── memory_store.py           # In-memory storage
│       │   └── persistence.py            # JSON file persistence
│       ├── api/
│       │   ├── __init__.py
│       │   ├── routes.py                  # API routes if needed
│       │   └── schemas.py                # Request/response schemas
│       └── utils/
│           ├── __init__.py
│           ├── logger.py                 # Logging utilities
│           └── datetime.py               # Date/time helpers
├── tests/
│   ├── __init__.py
│   ├── conftest.py                       # Pytest fixtures
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_failure_detector.py
│   │   ├── test_rate_limiter.py
│   │   ├── test_blacklist.py
│   │   ├── test_fallback_chain.py
│   │   └── test_circuit_breaker.py
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_end_to_end.py
│   │   └── test_fallback_scenarios.py
│   └── mocks/
│       ├── __init__.py
│       ├── mock_responses.py             # Mock API responses
│       └── server.py                    # Mock HTTP server
├── config/
│   ├── default.yaml                     # Default configuration
│   └── models.yaml                    # Model configurations
├── data/
│   └── .gitkeep                        # Data directory marker
├── scripts/
│   └── run_tests.py                   # Test runner script
├── requirements.txt                    # Python dependencies
├── requirements-dev.txt                 # Development dependencies
├── pyproject.toml                      # Project metadata
├── README.md                           # Project documentation
├── AGENTS.md                           # This file
└── QUESTIONS.md                       # Questions for human review
```

---

## 5. Testing Requirements

### 5.1 Unit Tests (Minimum: 90% Coverage)
- `test_failure_detector.py`: Test detection of each failure type
- `test_rate_limiter.py`: Test time-window tracking and probability calculation
- `test_blacklist.py`: Test blocking, unblocking, and expiration
- `test_fallback_chain.py`: Test sequential traversal and skipping
- `test_circuit_breaker.py`: Test open/closed/half-open states

### 5.2 Integration Tests
- `test_end_to_end.py`: Complete request flow with real HTTP mocks
- `test_fallback_scenarios.py`: Test various fallback scenarios:
  - Primary model rate-limited, fallback succeeds
  - All models fail, return last error with history
  - Model becomes unblocked after duration
  - Concurrent requests handled correctly

### 5.3 Mock Infrastructure
- Create mock HTTP server simulating various provider behaviors
- Implement response delay simulation
- Implement failure injection (simulate 429, 500, timeouts)

### 5.4 Test Fixtures
- Provide reusable fixtures for:
  - Clean state (empty blacklist, fresh rate tracking)
  - Pre-blocked models
  - Pre-filled rate limit history
  - Mock configuration

### 5.5 Test Execution
- All tests must run with `pytest` 
- Async tests with `pytest-asyncio`
- No external network calls in unit tests
- Integration tests use mocks/local servers only

---

## 6. Git Protocol

### 6.1 Branch Strategy
- **Main branch**: `main` - Production-ready code only
- **Development**: `develop` - Integration branch
- **Feature branches**: `feature