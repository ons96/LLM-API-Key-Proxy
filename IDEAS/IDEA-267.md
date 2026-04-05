# Multi-Provider AI API Gateway & Manager

## 1. Role/Mission

**Project Role:** You are an autonomous AI coding agent tasked with building an advanced AI API gateway that intelligently manages multiple LLM providers, models, and API keys while providing seamless fallback mechanisms and intelligent retry logic.

**Core Mission:** Create a production-ready API gateway that serves as a unified interface for numerous AI providers (OpenAI, Anthropic, Google, Cohere, Mistral, Ollama, and third-party providers), handling key rotation, virtual model chains with automatic failover, tool execution, and error recovery—all without service interruption.

**Success Definition:** A deployable gateway that can route AI requests across multiple providers with automatic failover, handle tool calls correctly, recover from malformed responses, and dynamically adapt to provider/changes without requiring restarts.

---

## 2. Technical Stack

**Primary Language:** Python 3.11+ (FastAPI framework)

**Core Dependencies:**
- **Framework:** FastAPI + Uvicorn (async server)
- **State/Rate Limiting:** Redis (via redis-py async)
- **HTTP Client:** httpx (async, OpenAI-compatible)
- **Configuration:** PyYAML + Python dotenv
- **Logging:** structlog + logging
- **Validation:** Pydantic v2
- **Testing:** pytest + pytest-asyncio + httpx test client

**Provider Support (Initial):**
- OpenAI (GPT-4, GPT-4 Turbo, GPT-3.5 Turbo)
- Anthropic (Claude 3 Opus, Claude 3 Sonnet, Claude 3 Haiku)
- Google AI (Gemini Pro, Gemini Ultra)
- Cohere (Command R, Command)
- Ollama (local models)
- Third-party proxies (OpenRouter, etc.)

**Infrastructure (Free Tiers):**
- Redis Cloud (free tier) for state management
- Local file-based config fallback (JSON/YAML)
- GitHub Actions for CI/CD

---

## 3. Requirements

### 3.1 Multi-Provider Management
1. Support at minimum 5 major AI providers with standardized request/response handling
2. Normalize different provider API schemas into a unified internal format
3. Extract provider-specific metadata (model, usage tokens, latency) from responses
4. Handle provider-specific error codes and transform them to consistent error types

### 3.2 Multi-Key Management & Rotation
5. Support multiple API keys per provider (minimum 3 keys per provider recommended)
6. Implement intelligent key rotation based on:
   - Round-robin distribution
   - Key-specific rate limits (requests/minute, tokens/minute)
   - Key health/status (automatically skip keys returning auth errors)
7. Persist key rotation state to Redis with local memory fallback
8. Track per-key usage statistics (request count, token count, error count)

### 3.3 Virtual Model Chains & Fallback
9. Define virtual "model chains" that link multiple models/providers in priority order
10. Implement automatic fallback: if primary model fails, automatically try next in chain
11. Support mixed-provider chains (e.g., OpenAI GPT-4 → Anthropic Claude → Google Gemini)
12. Preserve conversation context continuity across fallback attempts
13. Track fallback success/failure metrics for chain optimization

### 3.4 Dynamic Configuration
14. Load configuration from YAML/JSON files with environment variable overrides
15. Implement hot-reload: detect config changes without application restart
16. Use Redis pub/sub for cross-instance config synchronization
17. Support runtime API to:
    - Add/remove keys
    - Reorder fallback chains
    - Enable/disable providers
    - Adjust rate limits

### 3.5 OpenAI-Compatible API
18. Implement `/v1/chat/completions` endpoint compatible with OpenAI API spec
19. Implement `/v1/models` endpoint listing all available virtual models
20. Support OpenAI request format (messages, tools, temperature, max_tokens, etc.)
21. Return responses in OpenAI-compatible format
22. Support Streaming (`text/event-stream`) responses

### 3.6 Tool Calling Support
23. Parse and execute tool calls from provider response payloads
24. Support multiple concurrent tool executions
25. Implement tool result insertion back into conversation
26. Handle tool timeouts and errors gracefully
27. Cache tool definitions for repeated calls

### 3.7 Auto-Retry & Error Recovery
28. Implement automatic retry for:
    - HTTP 5xx errors (server-side failures)
    - Rate limit errors (429) with exponential backoff
    - Timeout errors (request took too long)
    - Malformed/incomplete responses (truncated JSON, partial content)
29. Configure retry limits per-request type
30. Detect "agentic" incomplete responses: sudden termination, code cuts, unclosed brackets
31. Preserve partial response context when retrying
32. Log all retry attempts with correlation IDs

### 3.8 Rate Limiting
33. Implement token bucket algorithm for rate limiting
34. Support two rate limit scopes: per-key and per-provider
35. Use Redis for distributed rate limit state across instances
36. Return proper `Retry-After` headers on 429 responses
37. Implement burst allowance with gradual refill

### 3.9 Logging & Observability
38. Structured JSON logging (structlog) for all operations
39. Log request/response correlation IDs
40. Include timing metrics for each provider call
41. Log fallback attempts and outcomes
42. Health check endpoint returning system status

---

## 4. File Structure

```
ai-gateway/
├── AGENTS.md
├── README.md
├── QUESTIONS.md
├── .env.example
├── pyproject.toml
├── requirements.txt
├── docker-compose.yml
├── .github/
│   └── workflows/
│       └── ci.yml
├── config/
│   ├── default.yaml
│   ├── providers/
│   │   ├── openai.yaml
│   │   ├── anthropic.yaml
│   │   ├── google.yaml
│   │   ├── cohere.yaml
│   │   └── ollama.yaml
│   └── chains/
│       ├── gpt-primary.yaml
│       └── claude-balanced.yaml
├── src/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry
│   ├── app.py                 # App factory, configuration
│   ├── config.py              # Configuration loader
│   ├── types.py               # Pydantic models
│   ├── errors.py              # Custom exceptions
│   ├── logging.py             # Logging setup
│   ├── health.py              # Health check endpoints
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py          # Main API routes
│   │   ├── chat.py            # /v1/chat/completions
│   │   ├── models.py          # /v1/models
│   │   └── admin.py           # Admin/runtime config
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py            # Base provider interface
│   │   ├── openai.py         # OpenAI provider
│   │   ├── anthropic.py     # Anthropic provider
│   │   ├── google.py         # Google provider
│   │   ├── cohere.py         # Cohere provider
│   │   └── ollama.py        # Ollama provider
│   ├── gateway/
│   │   ├── __init__.py
│   │   ├── router.py         # Request routing logic
│   │   ├── key_manager.py  # Key rotation
│   │   ├── chainer.py       # Fallback chain logic
│   │   ├── retry.py         # Retry logic
│   │   └── rate_limiter.py  # Token bucket rate limit
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── executor.py      # Tool execution
│   │   ├── registry.py     # Tool definitions
│   │   └── types.py        # Tool schemas
│   └── utils/
│       ├── __init__.py
│       ├── redis_client.py  # Redis management
│       ├── http.py         # HTTP utilities
│       └── metrics.py      # Metrics collection
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_providers/
│   │   ├── __init__.py
│   │   └── test_openai.py
│   ├── test_gateway/
│   │   ├── __init__.py
│   │   ├── test_router.py
│   │   ├── test_key_rotation.py
│   │   ├── test_fallback.py
│   │   └── test_retry.py
│   ├── test_tools/
│   │   ├── __init__.py
│   │   └── test_executor.py
│   ├── test_api/
│   │   ├── __init__.py
│   │   ├── test_chat.py
│   │   └── test_models.py
│   └── fixtures/
│       ├── config/
│       │   └── test.yaml
│       └── responses/
│           └── openai.json
├── scripts/
│   ├── load_test.py
│   └── benchmark.py
└── docs/
    ├── architecture.md
    ├── api-spec.md
    └── deployment.md
```

---

## 5. Testing Requirements

### 5.1 Unit Tests
1. **Configuration Tests:** Verify config loading, validation, hot-reload detection
2. **Provider Tests:** Test each provider adapter request/response transformation
3. **Key Manager Tests:** Test key rotation algorithms, health tracking
4. **Chainer Tests:** Test fallback logic, priority ordering, context preservation
5. **Retry Tests:** Test retry conditions, backoff timing, correlation preservation
6. **Rate Limiter Tests:** Test token bucket refill, burst handling, distributed state

### 5.2 Integration Tests
7. **API Contract Tests:** Verify OpenAI-compatible endpoints match spec
8. **Streaming Tests:** Verify SSE streaming works correctly
9. **Tool Execution Tests:** Test tool calls, results, context insertion
10. **Fallback Tests:** End-to-end fallback across providers with mock responses

### 5.3 Mock Strategy
11. Use `httpx` mock transport for provider API calls
12. Mock Redis with `fakeredis` for state tests
13. Create realistic mock response fixtures for each provider
14. Simulate rate limit errors, timeouts, malformed responses

### 5.4 Coverage Goals
15. Minimum 80% line coverage for core gateway logic
16. 100% coverage for error handling paths
17. All retry conditions must have explicit test cases

---

## 6. Git Protocol

### 6.1 Branch Strategy
- **Main branch:** `main` (production-ready code)
- **Development:** `develop` (integration branch)
- **Feature branches:** `feature/description` - e.g., `feature/dynamic-fallback`
- **Bugfix branches:** `bugfix/description` - e.g., `bugfix/retry-loop`
- **Hotfix branches:** `hotfix/description` - e.g., `hotfix/redis-timeout`

### 6.2 Commit Convention
Use conventional commits format:
- `feat: add key rotation for multiple OpenAI keys`
- `fix: handle timeout in tool executor`
- `docs: update API specification`
- `test: add fallback chain integration tests`
- `refactor: extract provider interface to base class`
- `chore: update dependencies`

### 6.3 Workflow
1. Create feature branch from `develop`
2. Write tests first (TDD approach for critical paths)
3. Implement feature with structured logging
4. Run full test suite before commit
5. Push and create pull request to `develop`
6. Require CI passing before merge
7. Merge to `main` for releases

### 6.4 CI/CD (GitHub Actions)
- Run tests on Python 3.11, 3.12
- Run linting (ruff) and type checking (mypy)
- Run security audit (bandit)
- Build Docker image on release tags

---

## 7. Completion Criteria

### 7.1 Minimum Viable Product
- [ ] **API Endpoints:** `/v1/chat/completions` and `/v1/models` working
- [ ] **Provider Support:** At least 2 providers (OpenAI + one other) integrated
- [ ] **Key Rotation:** Automatic rotation between 2+ keys per provider
- [ ] **Fallback Chains:** Virtual model chain with automatic failover
- [ ] **OpenAI Compatibility:** Works with OpenAI SDK (chat.completions client)
- [ ] **Tool Calls:** Basic tool calling support with at least 1 built-in tool
- [ ] **Configuration:** YAML config with runtime reload via endpoint

### 7.2 Production Requirements
- [ ] **Rate Limiting:** Token bucket rate limiting with Redis backend
- [ ] **Retry Logic:** Auto-retry on 429, 5xx, timeout with exponential backoff
- [ ] **Error Recovery:** Detect and retry malformed/truncated responses
- [ ] **Dynamic Reordering:** Runtime API to reorder fallback chains
- [ ] **Logging:** Structured JSON logs with correlation IDs
- [ ] **Health Check:** `/health` and `/ready` endpoints

### 7.3 Quality Gates
- [ ] **Test Coverage:** ≥80% line coverage on gateway logic
- [ ] **Linting:** Pass ruff, mypy strict mode
- [ ] **Security:** Pass bandit security scan
- [ ] **Docker:** Successfully build and run container locally
- [ ] **CI Pipeline:** All GitHub Actions jobs passing

### 7.4 Documentation
- [ ] **README.md:** Project overview, setup instructions, quick start guide
- [ ] **API Spec:** Document all endpoints, request/response formats
- [ ] **Architecture:** Document design decisions, data flows
- [ ] **CHANGELOG:** Track version history from initial release

---

## Important Notes for Execution

1. **Free Resources Only:** Use free tiers of Redis Cloud, no paid services
2. **Independent Decisions:** Make architectural choices without asking—document in ARCHITECTURE.md
3. **Save Questions:** Any unclear requirements → QUESTIONS.md
4. **Iterative Approach:** Implement core routing first, add features incrementally
5. **Test-Driven:** Write tests before implementation for critical paths
6. **Graceful Degradation:** If Redis unavailable, fall back to in-memory state

---

*This AGENTS.md was generated for an autonomous coding agent. The agent should proceed with implementation following the commit protocol and marking completion criteria as achieved.*