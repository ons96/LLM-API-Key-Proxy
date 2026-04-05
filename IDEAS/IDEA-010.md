# AGENTS.md - All-in-One LLM Gateway/Router

## 1. Role/Mission

**Mission:** Design and implement a comprehensive LLM (Large Language Model) gateway and router system that intelligently manages requests across multiple LLM providers. The gateway must optimize for speed and cost efficiency through automatic model selection while providing robust reliability through intelligent fallbacks, rate limit handling, and provider rotation.

**Core Philosophy:** Build a unified interface that abstracts away the complexity of managing multiple LLM providers, allowing clients to make simple requests while the system handles the intelligent routing, fallback logic, and optimization behind the scenes.

**Target Users:** Developers and applications that need reliable access to LLM capabilities without worrying about provider-specific implementation details, rate limits, or model selection optimization.

---

## 2. Technical Stack

### Language & Framework
- **Language:** Python 3.11+ (excellent ecosystem for LLM integration, async support)
- **API Framework:** FastAPI (modern, async-first, built-in OpenAPI documentation)
- **Async Runtime:** Uvicorn (ASGI server)

### Key Dependencies
- `fastapi` >= 0.109.0 - API framework
- `uvicorn` >= 0.27.0 - ASGI server
- `httpx` >= 0.26.0 - Async HTTP client for provider calls
- `pydantic` >= 2.5.0 - Data validation
- `python-dotenv` >= 1.0.0 - Environment configuration
- `aiohttp` >= 3.9.0 - Additional async support
- `tenacity` >= 8.2.0 - Retry logic
- `prometheus-client` >= 0.19.0 - Metrics collection

### LLM Providers (Free Tier Compatible)
- **OpenAI** - GPT models (free tier available)
- **Anthropic** - Claude models (free tier available)
- **Google AI** - Gemini models (free tier available)
- **Ollama** - Local models (完全免费)
- **Groq** - Fast inference (free tier available)

### Infrastructure (Free Resources Only)
- **CI/CD:** GitHub Actions (free for open source)
- **Code Quality:** Ruff, MyPy (free Python tools)
- **Testing:** pytest with free account services

---

## 3. Requirements (Numbered)

### 3.1 Core Gateway Functionality
1. **Unified API Endpoint** - Single REST API endpoint that accepts LLM requests with provider-agnostic input format
2. **Provider Abstraction Layer** - Clean interface for adding new LLM providers without changing core logic
3. **Request Normalization** - Convert gateway requests to provider-specific formats
4. **Response Normalization** - Convert provider responses to unified gateway format

### 3.2 Automatic Model Selection
5. **Sufficiency Scoring** - Algorithm to determine if a model's output meets quality requirements
6. **Speed-Based Selection** - Select fastest model that meets sufficiency threshold
7. **Multi-Tier Classification** - Classify requests by complexity (simple/medium/complex) to match appropriate models
8. **Performance Metrics** - Track latency, quality, and cost per model for selection decisions
9. **Adaptive Learning** - Improve model selection based on historical performance data

### 3.3 Manual Model Selection
10. **Explicit Model Override** - Allow clients to specify exact model to use
11. **Provider Override** - Allow clients to specify provider while letting gateway choose model
12. **Selection Preferences** - Accept parameters for preferred provider, max latency, budget constraints

### 3.4 Intelligent Fallback System
13. **Provider Failure Detection** - Automatically detect provider errors (5xx, timeouts, service unavailable)
14. **Sequential Fallbacks** - Try备用 providers in priority order when primary fails
15. **Quality Preservation** - Ensure fallback model provides equivalent or better reasoning capability
16. **Fallback History** - Log fallback occurrences for analysis and optimization
17. **Graceful Degradation** - Return partial results or clear error messages when all providers fail

### 3.5 Rate Limit Detection & Handling
18. **Rate Limit Monitoring** - Track 429 responses and rate limit headers from all providers
19. **Dynamic Throttling** - Implement client-side rate limiting to avoid provider limits
20. **Retry with Backoff** - Implement exponential backoff for rate-limited requests
21. **Rate Limit Prediction** - Estimate remaining capacity based on usage patterns
22. **Provider-Specific Handling** - Parse rate limit headers from each provider (OpenAI, Anthropic, etc.)

### 3.6 Provider Rotation
23. **Load Distribution** - Distribute requests across providers to avoid overloading single provider
24. **Round-Robin Selection** - Simple rotation when all providers have capacity
25. **Weighted Distribution** - Distribute based on cost, speed, or quality weights
26. **Health-Based Exclusion** - Temporarily exclude providers that are experiencing issues
27. **Cost Optimization** - Prefer lower-cost providers when quality is acceptable

### 3.7 Consistent Reasoning Effort
28. **Effort Calibration** - Define reasoning effort levels (minimal, standard, high)
29. **Cross-Provider Effort Mapping** - Map effort levels to appropriate provider settings
30. **Prompt Enhancement** - Add reasoning instructions based on effort requirements
31. **Token Budget Allocation** - Allocate tokens for reasoning based on effort level

### 3.8 Observability & Monitoring
32. **Request Logging** - Log all requests with timing, provider, model, and outcome
33. **Metrics Dashboard** - Expose Prometheus metrics for gateway performance
34. **Health Check Endpoint** - Provide /health endpoint for each provider status
35. **Cost Tracking** - Track usage costs per provider and model

---

## 4. File Structure

```
llm-gateway/
├── AGENTS.md
├── QUESTIONS.md
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml
├── src/
│   └── llm_gateway/
│       ├── __init__.py
│       ├── main.py                    # FastAPI application entry point
│       ├── config.py                  # Configuration management
│       ├── models/
│       │   ├── __init__.py
│       │   ├── request.py             # Gateway request models
│       │   ├── response.py            # Gateway response models
│       │   └── provider_config.py     # Provider configuration models
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── base.py               # Base provider abstract class
│       │   ├── openai.py             # OpenAI provider implementation
│       │   ├── anthropic.py          # Anthropic provider implementation
│       │   ├── google.py             # Google AI provider implementation
│       │   ├── ollama.py              # Ollama provider implementation
│       │   └── groq.py                # Groq provider implementation
│       ├── router/
│       │   ├── __init__.py
│       │   ├── selector.py            # Model selection logic
│       │   ├── fallback.py            # Fallback orchestration
│       │   ├── rotation.py           # Provider rotation logic
│       │   └── effort.py              # Reasoning effort management
│       ├── circuit_breaker/
│       │   ├── __init__.py
│       │   └── breaker.py            # Circuit breaker pattern
│       ├── rate_limiter/
│       │   ├── __init__.py
│       │   ├── tracker.py            # Rate limit tracking
│       │   └── controller.py        # Rate limit control
│       ├── metrics/
│       │   ├── __init__.py
│       │   └── prometheus.py         # Prometheus metrics
│       └── utils/
│           ├── __init__.py
│           ├── logger.py             # Logging configuration
│           └── headers.py           # HTTP headers utilities
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # Test fixtures
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_selector.py         # Model selector tests
│   │   ├── test_fallback.py        # Fallback logic tests
│   │   ├── test_rotation.py        # Provider rotation tests
│   │   └── test_rate_limiter.py    # Rate limiter tests
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_providers.py       # Provider integration tests
│   │   └── test_gateway.py         # Gateway end-to-end tests
│   └── mocks/
│       ├── __init__.py
│       └── providerMocks.py        # Provider mocking utilities
├── docker-compose.yml               # Local development environment
├── Dockerfile                        # Container definition
├── Makefile                         # Common tasks
└── requirements.txt                 # Pip dependencies (if not using uv)
```

---

## 5. Testing Requirements

### 5.1 Test Organization
1. **Unit Tests** - Test individual components in isolation with mocked dependencies
2. **Integration Tests** - Test provider interactions with mocked HTTP responses
3. **Contract Tests** - Verify API request/response formats match specifications
4. **E2E Tests** - Full gateway flow tests with real or mocked providers

### 5.2 Coverage Requirements
5. Minimum **80% code coverage** required for core router logic
6. **100% coverage** required for fallback and rate limiting logic
7. All public API endpoints must have corresponding tests

### 5.3 Test Strategy
8. **Mock External Calls** - Use mocks for all LLM provider API calls (no real API keys in tests)
9. **Provider Test Matrix** - Test each provider implementation independently
10. **Failure Injection** - Test system behavior under provider failures, rate limits, timeouts
11. **Concurrency Tests** - Verify thread-safety and async behavior

### 5.4 Testing Tools
- **pytest** - Test framework
- **pytest-asyncio** - Async test support
- **pytest-mock** - Mocking utilities
- **coverage** - Code coverage measurement
- **httpx** - Async HTTP client for test requests

---

## 6. Git Protocol

### 6.1 Branch Strategy
1. **main** - Production-ready code, protected branch
2. **develop** - Integration branch for features
3. **feature/* - Feature development branches
4. **bugfix/* - Bug fix branches
5. **hotfix/* - Emergency production fixes

### 6.2 Commit Conventions
6. Use **Conventional Commits** format:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `refactor:` for code refactoring
   - `test:` for test additions/changes
   - `docs:` for documentation
   - `chore:` for maintenance tasks

### 6.3 Pull Request Workflow
7. Create PR from feature branch to develop
8. Require at least **one reviewer** approval
9. All CI checks must pass (linting, type checking, tests)
10. **Squash and merge** into main

### 6.4 CI Requirements (GitHub Actions)
11. **Lint** - Run Ruff on all Python files
12. **Type Check** - Run MyPy type checking
13. **Test** - Run pytest with coverage
14. **Security** - Run basic security checks (bandit)

---

## 7. Completion Criteria

### 7.1 Core Functionality (Must Have)
- [ ] Gateway accepts requests via REST API POST /v1/chat/completions
- [ ] Successfully routes requests to at least 3 different LLM providers
- [ ] Automatic model selection chooses appropriate model based on task complexity
- [ ] Manual model override works correctly
- [ ] Fallback triggers when primary provider fails
- [ ] Rate limit detection triggers fallback or retry
- [ ] Provider rotation distributes load across providers

### 7.2 Quality Attributes (Must Have)
- [ ] Response format matches OpenAI-compatible schema
- [ ] Latency under 5 seconds for simple requests (excluding provider time)
- [ ] Graceful error handling - no unhandled exceptions leak to clients
- [ ] Proper logging of all requests with correlation IDs

### 7.3 Testing (Must Have)
- [ ] Unit tests pass for model selector
- [ ] Unit tests pass for fallback logic
- [ ] Unit tests pass for rate limiter
- [ ] Integration tests verify provider interfaces
- [ ] Code coverage above 80%

### 7.4 Documentation (Must Have)
- [ ] README.md with setup instructions
- [ ] API documentation via FastAPI auto-generated OpenAPI spec
- [ ] Provider configuration documented in .env.example
- [ ] Inline code comments for complex logic

### 7.5 Observability (Should Have)
- [ ] Health endpoint /health returns provider statuses
- [ ] Metrics endpoint /metrics returns Prometheus format data
- [ ] Structured logging with JSON format

### 7.6 Future Enhancements (Nice to Have - Document for Later)
- [ ] WebSocket support for streaming responses
- [ ] Caching layer for repeated requests
- [ ] User authentication and API key management
- [ ] Request queuing for async processing
- [ ] Dashboard for visualization of metrics

---

## Critical Decision Log

This section will be populated as the autonomous agent makes architectural decisions:

| Date | Decision | Rationale |
|------|----------|-----------|
| TBD | | |

---

**End of AGENTS.md**