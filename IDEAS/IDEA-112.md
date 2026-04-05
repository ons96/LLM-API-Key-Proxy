# AGENTS.md

## LLM Gateway Virtual Model Names with Fallbacks

---

## 1. Role/Mission

**Mission**: Implement an LLM API key proxy that provides virtual model name aliases with automatic fallback logic. The proxy should abstract away the specific LLM provider and model details, allowing clients to request logical model categories (coding-best, coding-fast, chat-smart, chat-fast) and have the system automatically route to the best available model with fallback behavior when primary choices are unavailable or rate-limited.

**Goals**:
- Create a unified API gateway that accepts virtual model names
- Implement intelligent fallback routing based on model capability categories
- Handle rate limiting, model availability, and cost optimization
- Provide a seamless abstraction layer for downstream applications

---

## 2. Technical Stack

**Language**: Python 3.10+

**Core Dependencies**:
- `fastapi` - High-performance async API framework
- `httpx` - Async HTTP client for upstream LLM provider calls
- `pydantic` - Data validation and settings management
- `python-dotenv` - Environment variable management
- `aiofiles` - Async file operations for logging

**Testing**:
- `pytest` - Testing framework
- `pytest-asyncio` - Async test support
- `pytest-cov` - Coverage reporting

**Dev Tools** (Free tier):
- `uv` - Fast Python package manager (or pip)
- GitHub Actions - CI/CD (free tier available)
- Local development with mock services

---

## 3. Requirements

1. **API Gateway Service**: Create a FastAPI-based HTTP proxy service that accepts Chat Completions API requests

2. **Virtual Model Registry**: Implement a configuration-driven registry mapping virtual names to actual provider models:
   - `coding-best` → Primary: GPT-4, Fallback: Claude-3-Opus, Fallback: GPT-3.5-Turbo
   - `coding-fast` → Primary: GPT-3.5-Turbo, Fallback: Claude-3-Haiku
   - `chat-smart` → Primary: GPT-4, Fallback: Claude-3-Sonnet
   - `chat-fast` → Primary: GPT-3.5-Turbo, Fallback: Claude-3-Haiku

3. **Fallback Logic**: Implement sequential fallback that attempts next-available model when:
   - Rate limit is hit (429 response)
   - Model is unavailable
   - API error returned
   - Timeout occurs

4. **Provider Abstraction**: Build an abstraction layer supporting multiple providers:
   - OpenAI API format
   - Anthropic API format
   - Design pattern for adding more providers

5. **Configuration Management**: Use YAML/JSON config file for model mappings (no hardcoding)

6. **Request Transformation**: Transform requests between client format and provider format

7. **Response Proxying**: Pass through responses from provider to client unchanged

8. **Logging**: Log all requests with virtual model used, actual model called, and fallback status

9. **Health Check**: Implement `/health` and `/ready` endpoints

10. **Metrics**: Track request counts by virtual model and fallback occurrence

---

## 4. File Structure

```
llm-gateway/
├── .github/
│   └── workflows/
│       └── test.yml              # GitHub Actions CI
├── src/
│   └── llm_gateway/
│       ├── __init__.py
│       ├── main.py               # FastAPI application entry
│       ├── config.py             # Configuration loading
│       ├── models.py             # Pydantic request/response models
│       ├── registry.py           # Virtual model registry
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── base.py            # Base provider interface
│       │   ├── openai.py          # OpenAI provider implementation
│       │   ├── anthropic.py       # Anthropic provider implementation
│       │   └── registry.py        # Provider factory
│       ├── router.py             # Request routing logic
│       ├── fallback.py           # Fallback execution logic
│       └── logging_config.py     # Logging setup
├── config/
│   └── models.yaml               # Model mappings configuration
├── tests/
│   ├── __init__.py
│   ├── test_providers.py
│   ├── test_registry.py
│   ├── test_fallback.py
│   ├── test_integration.py
│   └── fixtures/
│       ├── __init__.py
│       └── mock_responses.py
├── env.example                    # Environment template
├── pyproject.toml               # Project metadata and dependencies
├── uv.lock                      # Locked dependencies (or pip.lock)
├── README.md
└── AGENTS.md                   # This file
```

---

## 5. Testing Requirements

**Unit Tests**:
- Test virtual model registry loading and lookup
- Test provider abstraction interface contract
- Test fallback logic decision making
- Test request/response transformation

**Integration Tests**:
- Test end-to-end request flow with mock providers
- Test fallback behavior when primary fails
- Test multiple sequential fallbacks
- Test configuration reload

**Mock Strategy**:
- Use `responses` library or custom mock classes for HTTP
- Mock provider responses (success and error cases)
- No live API calls during testing (use mocks)

**Coverage Target**:
- Minimum 80% code coverage
- All fallback paths must have test coverage
- All provider transformations must have test coverage

**Test Execution**:
```bash
# Run all tests with coverage
pytest --cov=src.llm_gateway --cov-report=html

# Run with verbose output
pytest -v
```

---

## 6. Git Protocol

**Branch Strategy**: Trunk-based development
- Work directly on `main` branch for single-agent scenario
- Create feature branches if multiple agents working: `feature/*`

**Commit Message Format**:
```
