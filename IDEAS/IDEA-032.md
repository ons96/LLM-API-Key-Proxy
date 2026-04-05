# AGENTS.md - Dynamic Multi-Provider LLM Router with g4f

---

## 1. Role/Mission

**Mission**: Build an intelligent LLM (Large Language Model) routing system that automatically discovers available models from multiple providers using the g4f library, determines pricing/token multipliers, and provides a unified API for dynamic model selection.

**Agent Responsibilities**:
- Implement provider discovery by querying each provider's `/v1/models` endpoint (or equivalent)
- Build a dynamic routing layer that can select the best available model based on availability, pricing, and capabilities
- Create extensible provider adapters to support any API provider (including obscure/custom ones from Discord servers, community sites like linux.do, doi.top, etc.)
- Cache and refresh model lists periodically to handle provider availability changes
- Handle authentication, rate limiting, and error recovery gracefully
- Make independent decisions when encountering issues, preferring working solutions over stalling
- Document all decisions and save unresolved questions to QUESTIONS.md

**Success Criteria**: The system should be able to query multiple g4f providers, list their available models with metadata, and route requests to appropriate providers based on user preferences (cost, speed, model capability).

---

## 2. Technical Stack

### Core Dependencies
- **Python 3.10+** - Primary language
- **g4f** - GPT4Free library for multi-provider LLM access (`pip install g4f`)
- **aiohttp** or **httpx** - Async HTTP client for provider requests
- **pydantic** - Data validation and settings management
- **tenacity** - Retry logic for transient failures
- **redis** (optional) - Model list caching if available, else in-memory TTL cache
- **loguru** or **structlog** - Structured logging

### Development/Tooling
- **pytest** with **pytest-asyncio** - Testing framework
- **pytest-cov** - Coverage reporting
- **black** - Code formatting
- **ruff** or **flake8** - Linting
- **mypy** - Type checking
- **pre-commit** - Git hooks for quality enforcement

### Free Resources Only
- Use **GitHub Actions** for CI/CD (free tier)
- Use **public model lists** from g4f providers (no paid API keys required for discovery)
- Mock authentication flows for testing
- No paid services unless explicitly approved

---

## 3. Requirements (Numbered)

### Core Functionality

1. **Provider Discovery Engine**
   - Automatically load all providers available in g4f
   - Query each provider's `/v1/models` endpoint (or equivalent) to fetch available models
   - Handle providers that don't expose a models endpoint by falling back to known model lists
   - Support custom providers by implementing a standard ProviderAdapter interface

2. **Model Metadata Extraction**
   - Parse response to extract: model ID, name, capabilities, context window size, pricing info (if available), token limits
   - Handle varying response formats across different providers
   - Normalize model names to a standard format

3. **Pricing/Multiplier Detection**
   - Attempt to extract pricing from provider responses or known pricing tables
   - Default to "unknown" pricing when not available
   - Support manual pricing overrides via configuration

4. **Dynamic Router**
   - Select provider based on: model requirements, availability, pricing preference (cheapest/fastest)
   - Implement round-robin or least-loaded selection for load balancing
   - Support fallback chains (try provider A, then B, then C)

5. **Caching Layer**
   - Cache model lists with TTL (default: 1 hour)
   - Invalidate cache on provider errors
   - Support manual cache refresh

6. **Unified API**
   - Expose a simple Python API: `router.get_model(provider=None, preferences={})`
   - Expose CLI tool: `python -m llm_router list-models --provider openai --filter gpt-4`
   - HTTP API wrapper (optional, low priority)

### Extensibility

7. **Custom Provider Adapter**
   - Define abstract base class `ProviderAdapter` with methods:
     - `async get_available_models() -> List[ModelInfo]`
     - `async list_models() -> List[str]` (model IDs only)
     - `get_provider_name() -> str`
   - Allow registering custom providers via configuration

8. **Configuration Management**
   - Use environment variables or YAML config file
   - Support: cache TTL, preferred providers, pricing weights, timeout settings

### Error Handling

9. **Graceful Degradation**
   - If a provider is unreachable, log warning and continue with other providers
   - Implement retry with exponential backoff for transient failures
   - Circuit breaker pattern for providers that repeatedly fail

10. **Logging**
    - Log all provider queries, successes, and failures
    - Include timing information for performance monitoring
    - Use structured logs for easy parsing

---

## 4. File Structure

```
llm-router/
├── .github/
│   └── workflows/
│       └── ci.yml                    # GitHub Actions CI workflow
├── .pre-commit-config.yaml            # Pre-commit hooks
├── ruff.toml                         # Linter configuration
├── pyproject.toml                    # Python project metadata
├── requirements.txt                  # Pip dependencies
├── requirements-dev.txt             # Dev dependencies
├── requirements-test.txt             # Test dependencies
├── README.md                         # Project overview
├── AGENTS.md                         # This file
├── QUESTIONS.md                      # Unresolved questions (create if needed)
├── llm_router/
│   ├── __init__.py                   # Package root
│   ├── main.py                       # CLI entry point
│   ├── config.py                     # Configuration management
│   ├── logging_config.py            # Logging setup
│   ├── exceptions.py                 # Custom exceptions
│   ├── models.py                    # Pydantic models/schemas
│   ├── router.py                     # Main router logic
│   ├── cache.py                     # Caching layer
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py                  # ProviderAdapter abstract base class
│   │   ├── g4f_provider.py          # g4f integration
│   │   ├── openai_compatible.py     # OpenAI-compatible endpoint adapter
│   │   └── custom/                  # Custom provider implementations
│   │       ├── __init__.py
│   │       └── example.py           # Example custom provider
│   └── utils/
│       ├── __init__.py
│       ├── http_client.py           # Async HTTP client wrapper
│       ├── pricing.py               # Pricing utility functions
│       └── retry.py                 # Retry logic utilities
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # Pytest fixtures
│   ├── test_router.py               # Router unit tests
│   ├── test_providers/
│   │   ├── __init__.py
│   │   ├── test_g4f_provider.py     # g4f provider tests
│   │   └── test_provider_adapter.py # Base adapter tests
│   ├── test_cache.py                # Cache layer tests
│   └── test_integration/
│       ├── __init__.py
│       └── test_provider_discovery.py # Integration tests
└── docs/
    ├── architecture.md              # Architecture decision notes
    ├── providerProtocol.md          # Custom provider protocol guide
    └── cli_usage.md                  # CLI usage documentation
```

---

## 5. Testing Requirements

### Unit Tests
- **Minimum Coverage**: 80% line coverage
- Test all provider adapters with mocked HTTP responses
- Test router selection logic with various preferences
- Test caching layer TTL and invalidation

### Integration Tests
- Test actual provider discovery (if providers arereachable)
- Use VCR/pyrecorder to record/replay HTTP responses for reproducibility
- Mock-free tests that run in CI only when explicit feature flags are set

### Test Fixtures
- Mock provider responses stored in `tests/fixtures/`
- Use `@pytest.fixture` for reusable test objects
- Ensure tests can run in parallel (no shared mutable state)

### CI Requirements
- All tests must pass on GitHub Actions
- Linting (ruff) must pass with no errors
- Type checking (mypy) must pass with no errors
- Coverage report must be generated and posted as PR comment

### Running Tests
```bash
# Install dev dependencies
pip install -r requirements-dev.txt
pip install -r requirements-test.txt

# Run all tests with coverage
pytest --cov=llm_router --cov-report=html --cov-report=term

# Run specific test categories
pytest tests/test_router.py -v
pytest tests/test_providers/ -v

# Run with verbose output
pytest -vv --tb=long
```

---

## 6. Git Protocol

### Branching Strategy
- Use **Trunk-Based Development**: Small, short-lived branches directly on main for fixes
- Feature branches: `feature/add-new-provider` or `fix/cache-invalidation`
- Delete branches after merge

### Commit Messages
Follow Conventional Commits:
```
