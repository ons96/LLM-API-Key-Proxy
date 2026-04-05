# AGENTS.md

## Universal Multi-API Gateway Framework

---

## 1. Role/Mission

**Role:** Senior Software Architect & Autonomous Implementation Lead

**Mission:** Design and implement a Universal Multi-API Gateway Framework that serves as a unified abstraction layer for multiple LLM API providers. The framework must:

- Automatically implement newly added APIs with minimal configuration
- Intelligently manage rate limits, pricing, speed, and free usage quotas
- Perform load balancing across available API endpoints
- Auto-select the optimal model based on prompt content for both coding and chat use cases
- Provide seamless switching/routing between providers (investigate new-api vs litellm)
- Operate entirely on free resources throughout development and testing

The agent shall investigate the trade-offs between `new-api` (https://github.com/QuantumNous/new-api) and `litellm` as routing solutions, make a justified recommendation, and implement the chosen architecture.

---

## 2. Technical Stack

### Core Technologies

| Component | Technology | Version/Notes |
|-----------|------------|---------------|
| **Language** | Python | 3.10+ |
| **Routing/Proxy** | `litellm` | Primary candidate; investigate `new-api` as alternative |
| **API Management** | Custom Python class hierarchy | YAML/JSON configuration files |
| **Rate Limiting** | `pyrqlite` or in-memory token bucket | For distributed/multi-instance support |
| **Configuration** | YAML | Human-readable, version-controllable |
| **Testing** | `pytest`, `pytest-asyncio` | Async test support |
| **HTTP Client** | `httpx` | Async HTTP client |
| **Caching** | In-memory dict / LRUCache | For response caching where appropriate |
| **CI/CD** | GitHub Actions | Free tier usage |

### Research Dependencies (To Install During Investigation)

- `litellm` - Primary routing solution
- `new-api` - Alternative to evaluate
- `openai` - For compatibility testing
- `anthropic` - For compatibility testing (if free tier available)

---

## 3. Requirements

### Phase 1: Investigation & Architecture

1. **Investigate Routing Solutions**
   - Conduct thorough review of `litellm` architecture and capabilities
   - Conduct thorough review of `new-api` (https://github.com/QuantumNous/new-api) architecture
   - Compare feature sets: rate limiting, pricing, load balancing, free tier support
   - Document findings in ROUTING_COMPARISON.md with recommendation

2. **Design Core Architecture**
   - Define unified API configuration schema (YAML-based)
   - Design plugin/extension system for adding new API providers
   - Design routing logic for auto-selecting models based on prompt analysis
   - Design rate limiting architecture (per-provider, per-key, global)

### Phase 2: Core Implementation

3. **API Configuration System**
   - Create `config/schemas/api_config.schema.yaml` for validated configuration
   - Implement `ConfigLoader` class to load and validate API configurations
   - Support for multiple API keys per provider with key rotation
   - Support for free tier tracking and quota management

4. **Rate Limiting Engine**
   - Implement token bucket algorithm for rate limiting
   - Support per-provider, per-key, and per-endpoint limits
   - Implement backoff/retry logic with exponential jitter
   - Track rate limit violations and implement circuit breaker pattern

5. **Model Router**
   - Implement prompt analyzer to detect intent (coding vs chat)
   - Implement model selector based on intent, speed, cost, accuracy
   - Implement load balancer for distributing requests across similar models
   - Support fallback chains (primary → secondary → tertiary)

6. **Free Tier Management**
   - Track free usage quotas per provider
   - Implement usage logging and reporting
   - Implement automatic switching when free quotas exhausted
   - Alert when free tier usage reaches thresholds

### Phase 3: Provider Integration

7. **Universal Provider Interface**
   - Define abstract `LLMProvider` base class
   - Implement standard interface for: chat completion, embedding, model listing
   - Implement connection pooling and keep-alive management

8. **Initial Provider Implementations**
   - Implement OpenAI-compatible provider (for OpenAI, Azure, local models)
   - Implement Anthropic-compatible provider (if free tier available)
   - Implement Ollama provider (for local LLM inference)
   - Document provider implementation guide in PROVIDER_GUIDE.md

### Phase 4: Intelligence & Optimization

9. **Smart Routing Logic**
   - Implement heuristic-based model selection:
     - Coding prompts → target coding-optimized models
     - Chat prompts → target chat-optimized models
     - Long context → allocate to providers with larger context windows
     - Complex reasoning → route to providers with better reasoning benchmarks
   - Implement latency-based adaptive routing

10. **Monitoring & Reporting**
    - Implement request logging with timing, cost, success/failure
    - Implement cost aggregation per provider, per model
    - Implement performance dashboards (text-based for CLI)
    - Implement health checks for all providers

### Phase 5: Documentation & Testing

11. **Comprehensive Documentation**
    - Create README.md with setup and usage instructions
    - Create CONFIGURATION_GUIDE.md with examples
    - Create PROVIDER_IMPLEMENTATION_GUIDE.md for adding new providers

12. **Test Suite**
    - Implement unit tests for core components
    - Implement integration tests with mock providers
    - Implement router tests with simulated prompts
    - Achieve >80% code coverage

---

## 4. File Structure

```
universal-api-gateway/
├── .github/
│   └── workflows/
│       └── agents.yml          # GitHub Actions workflow (AUTONOMOUS_AGENT)
├── src/
│   └── gateway/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py       # Configuration loading and validation
│       │   ├── provider.py     # Abstract provider interface
│       │   └── router.py       # Core routing logic
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── base.py         # Base provider implementation
│       │   ├── openai.py       # OpenAI-compatible provider
│       │   ├── anthropic.py   # Anthropic-compatible provider
│       │   └── ollama.py       # Ollama provider
│       ├── limits/
│       │   ├── __init__.py
│       │   ├── rate_limiter.py # Token bucket rate limiter
│       │   └── circuit_breaker.py # Circuit breaker pattern
│       ├── intelligence/
│       │   ├── __init__.py
│       │   ├── prompt_analyzer.py # Intent detection
│       │   └── model_selector.py # Model selection logic
│       └── monitoring/
│           ├── __init__.py
│           ├── logger.py      # Request/response logging
│           └── metrics.py    # Performance metrics
├── config/
│   ├── schemas/
│   │   └── api_config.schema.yaml  # JSON Schema for validation
│   ├── providers/
│   │   └── example.yaml     # Example provider configurations
│   └── gateway.yaml        # Main gateway configuration
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_config.py
│   │   ├── test_rate_limiter.py
│   │   └── test_router.py
│   ├── integration/
│   │   ├── __init__.py
│   │   └── test_providers.py
│   └── router/
│       ├── __init__.py
│       └── test_model_selection.py
├── scripts/
│   ├── validate_config.py   # Configuration validation script
│   └── run_tests.py        # Test runner script
├── docs/
│   ├── ROUTING_COMPARISON.md
│   ├── CONFIGURATION_GUIDE.md
│   ├── PROVIDER_IMPLEMENTATION_GUIDE.md
│   └── USAGE.md
├── pyproject.toml
├── uv.lock                 # UV lockfile (if using uv)
├── requirements.txt        # Fallback requirements
├── README.md
├── AGENTS.md               # This file
└── QUESTIONS.md            # Questions for human clarification
```

---

## 5. Testing Requirements

### Unit Testing

| Component | Test Coverage | Notes |
|-----------|--------------|-------|
| Configuration Loading | 100% | Test valid/invalid configs, edge cases |
| Rate Limiter | 100% | Test token bucket algorithm, overflow handling |
| Circuit Breaker | 100% | Test state transitions |
| Prompt Analyzer | >90% | Test intent detection for various prompt types |
| Model Selector | >90% | Test selection based on criteria |

### Integration Testing

| Test | Environment | Notes |
|------|-------------|-------|
| Provider Connection | Mock/Local | Test OpenAI-compatible, Ollama (local) |
| End-to-End Routing | Mock | Full routing chain with mock responses |
| Rate Limit Handling | Simulation | Simulate rate limits and verify backoff |
| Failover Handling | Simulation | Simulate provider failures and verify failover |

### Test Execution

```bash
# Run all tests with coverage
pytest --cov=src --cov-report=term-missing tests/

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/router/
```

### CI/CD Integration

- All tests must pass on GitHub Actions
- Coverage must not decrease below 80%
- Linting must pass (using `ruff` or `flake8`)

---

## 6. Git Protocol

### Branch Strategy

| Branch | Purpose | Merge Target |
|--------|---------|--------------|
| `main` | Stable release | - |
| `develop` | Integration | `main` (via PR) |
| `feature/*` | New features | `develop` (via PR) |
| `fix/*` | Bug fixes | `develop` (via PR) |
| `investigation/*` | Research tasks | `develop` (via PR) |

### Commit Messages

Follow Conventional Commits:

```
