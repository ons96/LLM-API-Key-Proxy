# AGENTS.md — LLM Provider Gateway

## 1. Role/Mission

**Mission**: Build a centralized, provider-agnostic gateway system that enables seamless switching between multiple LLM providers for AI coding agents. The system must provide a unified API abstraction layer, robust provider configuration management, and intelligent fallback routing logic to ensure reliability and continuity in AI-assisted coding workflows.

**Autonomous Agent Directive**: You are an autonomous coding agent operating on GitHub Actions. Your mission is to design, implement, and deliver a fully functional LLM Provider Gateway that meets all requirements outlined below. You must make independent technical decisions, prioritize free-tier resources, and document any blockers or unresolved questions in `QUESTIONS.md`.

---

## 2. Technical Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Language** | Python 3.11+ | Extensive LLM library support, async capabilities, ease of testing |
| **HTTP Client** | `httpx` | Async-first, supports HTTP/2, built-in timeouts and retries |
| **Configuration** | `pydantic` + YAML | Type-safe config validation, environment variable expansion |
| **Async Runtime** | `asyncio` + `python-dotenv` | Native async support, seamless env var loading |
| **Testing** | `pytest` + `pytest-asyncio` + `pytest-mock` | Industry-standard async testing |
| **Mocking** | `requests-mock` / `unittest.mock` | Free, no external API calls during testing |
| **Documentation** | Markdown + `mdformat` (optional) | Self-documenting code, GitHub-native display |
| **CI/CD** | GitHub Actions | Free tier available, native to GitHub |

**Allowed Free LLM Providers for Development**:
- OpenAI (GPT-3.5 Turbo — free tier available via API credits)
- Anthropic (Claude — limited free tier / sandbox)
- Google AI (Gemini — free tier available)
- Groq ( free inference endpoint)

> **Note**: The gateway must support all major providers abstractly. Actual API calls should be mocked in tests to avoid external dependencies. For integration testing, use free-tier API keys or environment-variable-based credentials.

---

## 3. Requirements (Numbered)

### Core Functionality

1. **Unified API Abstraction Layer**
   - Implement a generic `LLMClient` abstract base class that defines the contract for all LLM providers.
   - Support a standard request format: `prompt`, `system_message`, `temperature`, `max_tokens`, `stream`.
   - Support a standard response format: `content`, `usage`, `model`, `finish_reason`, `provider`.
   - Handle provider-specific quirks (e.g., model naming, endpoint URLs, auth headers) transparently.

2. **Provider Configuration Management**
   - Create a YAML-based configuration file (`providers.yaml`) that stores API keys, base URLs, model mappings, and rate limits per provider.
   - Support environment variable substitution for sensitive credentials (e.g., `$OPENAI_API_KEY`).
   - Validate config on startup using Pydantic models; fail fast with clear error messages.

3. **Fallback Routing Logic**
   - Implement a `Router` class that accepts a list of providers in priority order.
   - On provider failure (timeout, 429 rate limit, 5xx error, invalid auth), automatically attempt the next provider.
   - Preserve conversation context/state across fallback attempts to avoid data loss.
   - Log all fallback events with timestamps, provider names, and error reasons.

4. **Provider-Agnostic Agent Integration**
   - Expose a simple `Gateway` class that accepts a prompt and returns a response, hiding all routing and provider details.
   - Allow users to specify preferred provider(s) via config or at runtime.
   - Support both synchronous (`await`) and asynchronous (generator-based streaming) interfaces.

5. **Streaming Support**
   - Implement streaming responses using Python generators or `asyncstream`.
   - Ensure streaming works consistently across all providers with a unified callback interface.

6. **Error Handling & Retries**
   - Implement exponential backoff for transient errors (429, 503).
   - Define clear error types: `ProviderError`, `AuthError`, `RateLimitError`, `TimeoutError`, `FallbackExhaustedError`.
   - Raise descriptive exceptions that include provider context and original error messages.

### Operational Requirements

7. **Logging & Observability**
   - Use Python's `logging` module with configurable levels (DEBUG, INFO, WARNING, ERROR).
   - Log request/response metadata (token counts, latency, provider, model) without exposing full API keys.
   - Include correlation IDs to track requests across fallbacks.

8. **Request/Response Validation**
   - Validate all inputs (prompt length, token limits) before sending to providers.
   - Reject requests that exceed provider-specific token limits with clear validation errors.

9. **Extensibility**
   - Design the system to be pluggable: adding a new provider should require only a new subclass of `LLMClient` and a config entry.
   - Provide a registration mechanism (decorator or registry pattern) for provider plugins.

### Documentation & Quality

10. **Self-Documenting Code**
    - docstrings for all public classes, methods, and functions following Google or NumPy style.
    - Include usage examples in docstrings for the `Gateway` and `Router` classes.

11. **Example Usage Script**
    - Provide a `demo.py` script that demonstrates the gateway with mocked providers (for safety/cost savings) and real providers (with env var selection).

---

## 4. File Structure

```
llm-provider-gateway/
├── .github/
│   └── workflows/
│       └── test.yml                # GitHub Actions CI workflow
├── src/
│   └── llm_gateway/
│       ├── __init__.py
│       ├── client.py               # Abstract LLMClient base class
│       ├── clients/
│       │   ├── __init__.py
│       │   ├── openai_client.py    # OpenAI implementation
│       │   ├── anthropic_client.py # Anthropic implementation
│       │   ├── google_client.py    # Google AI implementation
│       │   ├── groq_client.py       # Groq implementation
│       │   └── registry.py         # Provider registry
│       ├── config.py               # Configuration loading & validation
│       ├── router.py                # Fallback routing logic
│       ├── gateway.py               # Unified Gateway facade
│       ├── exceptions.py            # Custom exceptions
│       ├── models.py                # Pydantic request/response models
│       └── logging_config.py         # Logging setup
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # Pytest fixtures
│   ├── test_clients/
│   │   ├── __init__.py
│   │   ├── test_openai_client.py
│   │   ├── test_anthropic_client.py
│   │   └── test_google_client.py
│   ├── test_router.py
│   ├── test_gateway.py
│   └── test_config.py
├── config/
│   └── providers.yaml.example      # Example provider config template
├── demo/
│   └── demo.py                      # Example usage script
├── docs/
│   └── README.md                   # Project overview
├── QUESTIONS.md                    # Autonomous agent questions
├── requirements.txt
├── pyproject.toml
└── AGENTS.md                       # This file
```

---

## 5. Testing Requirements

### Test Coverage Goals

| Category | Minimum Coverage | Description |
|----------|-----------------|-------------|
| **Unit Tests** | >90% | Test individual classes/methods in isolation |
| **Integration Tests** | Core flows only | Test router fallback, config loading |
| **Mock Strategy** | 100% external calls mocked | No real API calls in CI |

### Testing Guidelines

1. **Mock All External LLM APIs**
   - Use `unittest.mock.patch` to mock `httpx.AsyncClient` responses.
   - Create fixture JSON responses that match real provider response schemas.

2. **Router Fallback Tests**
   - Test successful fallback when primary provider returns 429.
   - Test successful fallback when primary provider times out.
   - Test failure when all providers fail (raise `FallbackExhaustedError`).

3. **Config Validation Tests**
   - Test missing required fields raises `ValidationError`.
   - Test environment variable substitution works.
   - Test invalid provider name raises error.

4. **Async Tests**
   - Use `pytest-asyncio` for all async test functions.
   - Ensure proper event loop handling in fixtures.

5. **Running Tests**
   - Execute locally: `pytest`
   - Execute in CI: `.github/workflows/test.yml`

---

## 6. Git Protocol

### Commit Convention

Format: