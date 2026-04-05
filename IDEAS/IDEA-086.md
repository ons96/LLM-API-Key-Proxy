# AGENTS.md - Virtual Model Provider Abstraction Layer

## 1. Role/Mission

**Mission:** Design and implement a virtual model provider abstraction layer that decouples client code from specific LLM API providers, enabling seamless fallback, dynamic model selection, and provider portability.

**Primary Objectives:**
- Create a unified abstraction layer that exposes virtual model names (e.g., `coding-elite`, `coding-fast`, `coding-best`, `chat-elite`, `chat-fast`, `chat-best`) to consumers
- Implement intelligent fallback ordering so that if a primary provider fails or is unavailable, the system automatically attempts the next model in the priority chain
- Support dynamic configuration loading so that provider mappings and fallback orders can be modified without code changes
- Enable context transfer between conversations/sessions and implement context pruning for memory efficiency
- Use only free-tier or freely available API resources for development and testing

**Scope:**
- Build a Python library/package that can be imported into other projects
- Provide configuration via JSON/YAML files (no hardcoded provider credentials in source)
- Support at least 2-3 free-tier providers (OpenAI free tier, Anthropic free tier if available, Ollama local, etc.)
- Include comprehensive test coverage

---

## 2. Technical Stack

**Language:** Python 3.11+ (for modern typing and async support)

**Key Dependencies:**
- `pydantic` (v2.x) - Configuration validation and settings management
- `pyyaml` - Configuration file parsing
- `aiohttp` - Async HTTP client for API calls
- `python-dotenv` - Environment variable management (for API keys in dev)
- `tenacity` - Retry/fallback logic
- `pytest` - Testing framework
- `pytest-asyncio` - Async test support
- `pytest-mock` - Mocking for tests
- `httpx` - Sync/async HTTP client alternative (optional)

**Optional/Maybe:**
- `httpx` as alternative HTTP client
- `tiktoken` - Token counting for context management
- `litellm` - For comparison/reference (research only, not direct dependency)

**Free Resources for Development:**
- OpenAI API (free tier with credits)
- Anthropic API (free tier with credits)
- Ollama (local, free, open-source)
- Groq (free tier available)
- Any other providers with free tiers

**Environment:**
- GitHub Actions for CI/CD
- Python virtual environments (venv)

---

## 3. Requirements (Numbered)

### 3.1 Core Architecture

1. **Virtual Model Registry** - Create a registry that maps virtual model names (e.g., `coding-elite`) to one or more actual provider model identifiers, with priority ordering
2. **Provider Interface** - Define an abstract `Provider` base class/interface that all specific providers (OpenAI, Anthropic, Ollama, etc.) must implement
3. **Client Factory** - Build a factory pattern implementation that returns the appropriate provider client based on the virtual model request

### 3.2 Configuration System

4. **Config File Support** - Implement loading configuration from YAML files (default: `vm-config.yaml` in project root or user home)
5. **Environment Override** - Allow environment variables to override config file settings (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`)
6. **Schema Validation** - Use Pydantic models to validate configuration structure at load time with clear error messages

### 3.3 Fallback Logic

7. **Fallback Chain** - Implement automatic fallback: if primary model/provider fails, attempt the next in the priority chain until success or exhaustion
8. **Failure Detection** - Define what constitutes a "failure" (API error, rate limit, timeout, invalid response)
9. **Exponential Backoff** - Implement configurable retry with exponential backoff between fallback attempts

### 3.4 Context Management

10. **Conversation Context** - Create a `Conversation` class that maintains message history and context window
11. **Context Pruning** - Implement smart context pruning (e.g., remove older messages, summarize, or truncate when approaching token limits)
12. **Session Persistence** - Allow conversation objects to be serialized (JSON) and restored for session continuity

### 3.5 API Implementation

13. **Async Interface** - All provider calls must support async/await pattern
14. **Streaming Support** - Support streaming responses where provider supports it
15. **Token Counting** - Include token counting utility to estimate context window usage

### 3.6 Provider Implementations (Minimum Viable)

16. **OpenAI Provider** - Implement provider for OpenAI-compatible APIs (OpenAI, Groq, local昆仑)
17. **Anthropic Provider** - Implement provider for Anthropic API
18. **Ollama Provider** - Implement provider for local Ollama installation

### 3.7 Developer Experience

19. **Logging** - Implement structured logging for debugging and monitoring
20. **Error Hierarchy** - Create custom exception classes for different failure modes
21. **Health Check** - Provide a utility to check provider availability/health

---

## 4. File Structure

```
virtual-model-provider/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI workflow
├── .env.example                # Example environment file (no real keys)
├── .gitignore                  # Ignore patterns
├── pyproject.toml              # Project metadata and dependencies
├── uv.lock                     # Lock file (if using uv)
├── README.md                   # Project overview
├── AGENTS.md                   # This file (for autonomous agent)
├── QUESTIONS.md                # Questions for human review (agent saves here)
│
├── vm_provider/
│   ├── __init__.py             # Public API exports
│   ├── config.py               # Configuration loading and models
│   ├── exceptions.py           # Custom exception classes
│   ├── logging.py              # Logging setup
│   ├── token_counter.py       # Token counting utilities
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── message.py          # Message model
│   │   ├── conversation.py    # Conversation/context management
│   │   └── virtual_model.py   # Virtual model definitions
│   │
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract Provider base class
│   │   ├── openai.py           # OpenAI-compatible provider
│   │   ├── anthropic.py       # Anthropic provider
│   │   ├── ollama.py           # Ollama local provider
│   │   └── registry.py         # Provider registry/factory
│   │
│   └── fallback/
│       ├── __init__.py
│       ├── chain.py            # Fallback chain logic
│       └── retry.py           # Retry/backoff logic
│
├── config/
│   └── vm-config.yaml.example  # Example configuration file
│
└── tests/
    ├── __init__.py
    ├── conftest.py             # Pytest fixtures
    ├── test_config.py          # Configuration tests
    ├── test_providers/         # Provider-specific tests
    │   ├── __init__.py
    │   ├── test_openai.py
    │   ├── test_anthropic.py
    │   └── test_ollama.py
    ├── test_fallback.py        # Fallback logic tests
    ├── test_conversation.py    # Conversation/context tests
    ├── test_integration.py     # Integration tests
    └── test_cli.py              # CLI utility tests (if implemented)
```

---

## 5. Testing Requirements

### 5.1 Test Coverage Goals

- **Minimum:** 80% code coverage
- **Core modules:** 90%+ coverage (config, providers, fallback, conversation)

### 5.2 Test Types

1. **Unit Tests** - Test individual functions and classes in isolation
   - Configuration loading and validation
   - Provider client instantiation
   - Token counting logic
   - Fallback chain ordering

2. **Mock Tests** - Test with mocked external API calls
   - All provider client methods
   - Error handling paths
   - Fallback triggers

3. **Integration Tests** - Test with actual free-tier API calls (marked with `@pytest.mark.integration`)
   - Run these selectively in CI with environment setup
   - Use environment variables for API keys in CI secrets

4. **Contract Tests** - Verify provider interface adherence
   - Each provider implements the base interface correctly
   - Request/response formats match expected structure

### 5.3 Test Fixtures (conftest.py)

- `mock_provider_configs` - Sample provider configurations
- `sample_conversation` - Pre-populated conversation for testing
- `mock_api_responses` - Sample API responses for mocking

### 5.4 CI Test Execution

```yaml
# In .github/workflows/ci.yml
- name: Run unit tests
  run: pytest tests/ -v --cov=vm_provider --cov-report=xml -m "not integration"

- name: Run integration tests
  if: github.event_name == 'schedule'  # Or manual trigger
  run: pytest tests/ -v -m integration
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

---

## 6. Git Protocol

### 6.1 Branch Strategy

- **Main branch:** `main` - Always deployable, passing all tests
- **Development:** `develop` - Integration branch for features
- **Feature branches:** `feature