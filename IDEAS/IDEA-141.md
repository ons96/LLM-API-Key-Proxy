# AGENTS.md - Smart LLM Gateway with Multi-Model Fallback Routing

## 1. Role/Mission

**Role:** Autonomous Software Engineer

**Mission:** Build a production-ready Smart LLM Gateway system that intelligently routes API calls to the best available LLM model with automatic fallback capabilities. The system must optimize for reliability, cost-efficiency (using free tiers), and performance through parallel request handling.

**Core Objectives:**
- Create a gateway that acts as a unified interface for multiple LLM providers
- Implement intelligent model ranking based on performance metrics
- Build automatic failover logic that switches models when rate limits or errors occur
- Track usage limits per model and dynamically skip unavailable models
- Handle requests asynchronously with parallel fallback execution for minimal latency
- Use only free resources throughout (no paid APIs unless explicitly approved)

**Decision-Making Authority:**
- Make independent architectural decisions within the defined stack
- Choose implementation approaches unless explicitly constrained
- Decide how to structure code and tests
- Determine when to use mocks vs real free-tier APIs

**Escalation Protocol:**
- Save any clarifying questions to `QUESTIONS.md` in the root directory
- Do not wait indefinitely - proceed with reasonable assumptions and document them
- Flag any blockers or assumptions in commit messages

---

## 2. Technical Stack

**Primary Language:** Python 3.11+

**Core Dependencies:**
- `httpx` - Async HTTP client for parallel model requests
- `fastapi` - Lightweight API framework for the gateway
- `pydantic` - Data validation and settings management
- `asyncio` - Native async/await support for concurrent operations
- `tenacity` - Retry logic with exponential backoff

**Testing Stack:**
- `pytest` - Test framework
- `pytest-asyncio` - Async test support
- `pytest-cov` - Coverage reporting

**Monitoring/Logging (Free Tier):**
- `structlog` - Structured logging
- Built-in Python `logging` module

**Data Storage (Free Resources):**
- In-memory tracking (Python `dict`/`asyncio.Lock`) for usage tracking
- Optional: Free Redis tier (Upstash/Redis Cloud) if needed for distributed setups
- Local JSON/CSV files for persistence of usage stats

**LLM Providers (Free Tiers Only):**
- Hugging Face Inference API (free tier: 600 requests/month)
- Ollama (local, free) - Self-hosted option
- Mock providers for testing/development

---

## 3. Requirements

### 3.1 Core Gateway Functionality

1. **Gateway API Server**
   - Expose a POST `/v1/chat/completions` endpoint compatible with OpenAI format
   - Accept model name, messages, and standard parameters
   - Return responses in standard OpenAI format

2. **Model Registry**
   - Maintain a list of registered models with metadata (provider, endpoint, capabilities)
   - Support dynamic model addition/removal via configuration
   - Store model ranking/priority order

3. **Request Routing**
   - Route incoming requests to the highest-priority available model
   - Maintain sorted model list by performance score
   - Allow model-specific parameter mapping

### 3.2 Fallback & Failover System

4. **Automatic Fallback Logic**
   - On rate limit (429) or unavailable error (503), automatically try next model
   - Implement configurable max retry attempts per request (default: 3)
   - Preserve request parameters when falling back

5. **Parallel Fallback Execution**
   - Send parallel requests to multiple models simultaneously when latency is critical
   - Use configurable timeout per model request (default: 10 seconds)
   - Return first successful response and cancel remaining requests

6. **Health Checking**
   - Implement periodic health checks for registered models
   - Mark models as unavailable after consecutive failures (configurable threshold: 3)
   - Auto-re-enable models after recovery period (configurable: 60 seconds)

### 3.3 Usage Tracking

7. **Usage Metrics Tracking**
   - Track request count per model
   - Track token usage where available
   - Monitor rate limit proximity per model
   - Store usage data in memory with session persistence

8. **Rate Limit Management**
   - Track remaining requests per model/window
   - Preemptively skip models approaching rate limits
   - Implement configurable safety margins (default: 10% headroom)

### 3.4 Model Ranking Algorithm

9. **Dynamic Model Ranking**
   - Prioritize models based on: success rate, latency, rate limit headroom
   - Implement weighted scoring: `score = (success_rate * 0.5) + (1/latency * 0.3) + (rate_headroom * 0.2)`
   - Update rankings periodically or after each request batch

10. **Fallback Order Calculation**
    - Calculate fallback chain based on current rankings
    - Exclude unhealthy models from fallback chain
    - Maintain minimum 2-model redundancy

### 3.5 Configuration & Observability

11. **Configuration Management**
    - Support YAML-based configuration file
    - Allow environment variable overrides
    - Default to safe/production-ready values

12. **Logging & Error Handling**
    - Structured logging with request IDs
    - Log all fallbacks with reasons
    - Capture and report aggregated metrics

---

## 4. File Structure

```
smart-llm-gateway/
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI workflow
├── src/
│   └── gateway/
│       ├── __init__.py
│       ├── main.py                 # FastAPI application entry point
│       ├── config.py               # Configuration management
│       ├── models.py               # Pydantic data models
│       ├── registry.py             # Model registry and ranking
│       ├── router.py               # Core routing logic
│       ├── fallback.py             # Fallback/failover execution
│       ├── tracker.py              # Usage tracking
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── base.py             # Base provider abstract class
│       │   ├── mock.py             # Mock provider for testing
│       │   ├── hf_inference.py     # Hugging Face provider
│       │   └── ollama.py            # Ollama local provider
│       └── utils/
│           ├── __init__.py
│           └── logging.py           # Logging configuration
├── tests/
│   ├── __init__.py
│   ├── test_models.py              # Data model tests
│   ├── test_registry.py            # Registry and ranking tests
│   ├── test_router.py              # Routing logic tests
│   ├── test_fallback.py            # Fallback execution tests
│   ├── test_tracker.py             # Usage tracking tests
│   ├── test_providers/
│   │   ├── __init__.py
│   │   ├── test_base.py            # Provider base tests
│   │   └── test_mock.py            # Mock provider tests
│   └── integration/
│       ├── __init__.py
│       ├── test_gateway.py         # Gateway integration tests
│       └── test_end_to_end.py      # E2E tests
├── scripts/
│   └── test_providers.py           # Test script for provider health
├── config/
│   └── default.yaml                # Default configuration
├── pyproject.toml                 # Project configuration
├── uv.lock                        # Locked dependencies
├── QUESTIONS.md                   # Save questions here
└── README.md                      # Project documentation
```

---

## 5. Testing Requirements

### 5.1 Test Coverage Goals

- **Minimum 80% code coverage** across all modules
- Critical path (routing + fallback) must have **95%+ coverage**
- All public APIs must have integration tests

### 5.2 Test Categories

**Unit Tests:**
- Configuration loading and validation
- Model registry operations and ranking algorithm
- Usage tracking accuracy
- Data model serialization/deserialization

**Provider Tests:**
- Base provider interface compliance
- Mock provider response handling
- Error handling across provider implementations

**Routing Tests:**
- Primary route selection based on rankings
- Fallback chain generation
- Parallel execution with timeout handling
- Health check integration

**Integration Tests:**
- Full gateway request flow
- End-to-end fallback behavior
- Health check endpoint
- Metrics endpoint

### 5.3 Test Execution

```bash
# Run all tests with coverage
pytest --cov=src --cov-report=html

# Run only unit tests
pytest tests/ -m "not integration"

# Run with parallel execution
pytest -n auto

# Run specific test file
pytest tests/test_router.py -v
```

### 5.4 CI Requirements

- All tests must pass on every push
- Coverage check must pass (no decrease below threshold)
- Linting must pass (`ruff` or `flake8`)
- Type checking must pass (`mypy`)

---

## 6. Git Protocol

### 6.1 Branch Strategy

- `main` - Production-ready code only
- `develop` - Integration branch for features
- `feature/*` - Individual feature branches
- `fix/*` - Bug fix branches
- `refactor/*` - Code improvement branches

### 6.2 Commit Rules

- Use conventional commits format:
  - `feat:` for new features
  - `fix:` for bug fixes
  - `refactor:` for code improvements
  - `test:` for test additions/changes
  - `docs:` for documentation
  - `chore:` for maintenance tasks

- Example: `feat: add parallel fallback execution with timeout`

### 6.3 Pull Request Process

1. Create feature branch from `develop`
2. Implement changes with passing tests
3. Update documentation if needed
4. Push and create PR to `develop`
5. Require minimum 1 review approval
6. Squash and merge to `develop`
7. `develop` merges to `main` on release

### 6.4 Handling Questions/Blockers

- Save questions to `QUESTIONS.md` in the root
- Include context: what you're trying to do, what you tried, what you assume
- Continue with other tasks while waiting

---

## 7. Completion Criteria

### 7.1 Functional Criteria

| # | Criterion | Verification |
|---|-----------|--------------|
| 1 | Gateway accepts POST requests to `/v1/chat/completions` | Send test request, receive valid response |
| 2 | Requests route to highest-priority model by default | Check routing logs, verify model selection |
| 3 | Automatic fallback on 429/503 errors | Simulate rate limit, verify fallback |
| 4 | Parallel fallback execution works | Time multiple parallel requests |
| 5 | Usage tracking records correctly | Check metrics after test requests |
| 6 | Models marked unhealthy after failures | Inject failures, verify health state |
| 7 | Model ranking updates dynamically | Check ranking changes over time |
| 8 | Configuration loads from YAML | Modify config, verify changes |

### 7.2 Non-Functional Criteria

| # | Criterion | Verification |
|---|-----------|--------------|
| 9 | Test coverage >= 80% | Run `pytest --cov` |
| 10 | All tests pass | CI pipeline green |
| 11 | No linting errors | Run `ruff check` |
| 12 | No type errors | Run `mypy src` |
| 13 | Gateway starts without errors | Run and check logs |
| 14 | Graceful shutdown works | SIGTERM handling |

### 7.3 Demonstration Requirements

For completion sign-off, demonstrate:

1. **Basic Flow:** Show gateway handling a simple request
2. **Fallback:** Force a failure and show automatic fallback
3. **Metrics:** Show usage tracking output
4. **Tests:** Run full test suite showing 80%+ coverage

### 7.4 Free Resource Compliance

- All LLM calls use only free-tier providers or mocks
- No paid API keys required for testing
- All dependencies are open-source/free

---

## Summary

This file serves as the complete directive for the autonomous agent. The agent should:

1. **Start by understanding the codebase structure** outlined in Section 4
2. **Implement requirements systematically** per Section 3
3. **Write tests alongside code** to achieve coverage goals in Section 5
4. **Follow git protocol** in Section 6 for all changes
5. **Verify completion** against criteria in Section 7
6. **Save questions** to QUESTIONS.md if blocked

The system should be production-ready, using only free resources, with comprehensive test coverage. Proceed with implementation.