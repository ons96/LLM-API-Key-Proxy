# AGENTS.md - Multi-LLM Dynamic Router for Coding Tasks

## 1. Role/Mission

### Purpose
Build an intelligent routing system that automatically selects the optimal LLM (Large Language Model) for different coding tasks. The router analyzes incoming requests, classifies the task type, and dispatches the request to the most suitable model provider.

### Core Mission
- **Task Classification**: Automatically classify coding requests into categories (planning, code generation, search/grep, debugging, code review, refactoring)
- **Dynamic Model Selection**: Route requests to the best-suited LLM based on task type, cost, speed, and capability
- **Unified API Interface**: Provide a single API endpoint that abstracts away the complexity of multiple LLM providers
- **Cost Optimization**: Prioritize free/local models when appropriate to minimize costs
- **Resilience**: Graceful fallback when primary models are unavailable

### Target Users
- Developer tools and IDEs
- Automated coding agents
- CI/CD pipelines requiring AI assistance
- Code review automation systems

---

## 2. Technical Stack

### Core Framework
- **FastAPI** (v0.109+) - Modern high-performance Python web framework
- **httpx** (v0.26+) - Async HTTP client for LLM API calls
- **Python 3.11+** - Required for modern async features

### Task Classification
- **LangChain** (v0.1+) - For building task classification chains
- **scikit-learn** - Optional lightweight classifier for keyword-based routing

### LLM Integration
- **Ollama** - Local free LLM inference (primary recommendation for free usage)
- **OpenAI Compatible API** - Interface for various open-source models
- Free tier APIs where available

### Configuration & Logging
- **python-dotenv** - Environment variable management
- **structlog** - Structured logging
- **pydantic** - Data validation and settings management

### Testing & Quality
- **pytest** - Testing framework
- **pytest-asyncio** - Async testing support
- **pytest-cov** - Coverage reporting

### Development Tools
- **ruff** - Fast Python linter
- **mypy** - Type checking
- **pre-commit** - Git hooks

---

## 3. Requirements

### 3.1 Core Functionality

1. **Task Classifier**
   - Analyze incoming request text and/or code
   - Classify into categories: `planning`, `code_generation`, `search_grep`, `debugging`, `code_review`, `refactoring`, `general`
   - Return confidence score alongside classification
   - Support custom task types via configuration

2. **Model Router**
   - Map task types to optimal model configurations
   - Support multiple model providers per task type
   - Implement priority-based selection (preferred → fallback)
   - Consider parameters: cost, speed, capability, context window

3. **Unified API Endpoint**
   - POST `/v1/chat/completions` - Main chat completion endpoint
   - POST `/v1/completions` - Legacy completion endpoint
   - GET `/health` - Health check endpoint
   - GET `/models` - List available virtual models
   - POST `/classify` - Task classification only (no execution)

4. **Streaming Support**
   - Support server-sent events (SSE) for streaming responses
   - Configurable stream chunk size

### 3.2 Model Providers

5. **Ollama Integration**
   - Connect to local Ollama instance
   - Support for multiple models: codellama, llama2, mistral
   - Configurable base URL
   - Connection health checks

6. **Provider Abstraction**
   - Unified provider interface
   - Easy addition of new providers
   - Request/response normalization

### 3.3 Configuration

7. **Configurable Routing Rules**
   - YAML-based routing configuration
   - Task type to model mappings
   - Provider priority ordering
   - Default fallback chain

8. **Environment-based Settings**
   - Model provider API keys
   - Endpoint URLs
   - Timeout configurations
   - Rate limiting settings

### 3.4 Observability

9. **Logging**
   - Structured request/response logging
   - Task classification logging
   - Model selection reasoning logged
   - Configurable log levels (DEBUG, INFO, WARNING, ERROR)

10. **Metrics**
    - Request count by task type
    - Model usage statistics
    - Average response time
    - Error rates by provider

### 3.5 Reliability

11. **Error Handling**
    - Graceful provider fallback
    - Timeout handling (configurable per provider)
    - Rate limit detection and backoff
    - Clear error messages with troubleshooting hints

12. **Health Checks**
    - Overall system health
    - Per-provider availability
    - Connection verification

---

## 4. File Structure

```
multi-llm-router/
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI workflow
├── .pre-commit-config.yaml         # Pre-commit hooks
├── pyproject.toml                # Project configuration
├── ruff.toml                     # Linter configuration
├── mypy.ini                      # Type checker configuration
├── uv.lock                       # Dependency lock file
├── README.md                     # Project documentation
├── QUESTIONS.md                  # Questions for human review
├── AGENTS.md                      # This file
├── config/
│   └── routing.yaml               # Routing configuration
├── src/
│   └── multi_llm_router/
│       ├── __init__.py
│       ├── main.py                # FastAPI application entry
│       ├── config.py              # Configuration management
│       ├── models/
│       │   ├── __init__.py
│       │   ├── requests.py        # Pydantic request models
│       │   ├── responses.py       # Pydantic response models
│       │   └── types.py           # Enums and type definitions
│       ├── classifier/
│       │   ├── __init__.py
│       │   ├── base.py            # Base classifier interface
│       │   ├── keyword_classifier.py  # Keyword-based classifier
│       │   └── registry.py        # Classifier registry
│       ├── router/
│       │   ├── __init__.py
│       │   ├── base.py            # Base router interface
│       │   ├── dynamic_router.py  # Dynamic routing logic
│       │   └── config.py          # Routing rules config
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── base.py            # Base provider interface
│       │   ├── ollama.py          # Ollama provider
│       │   ├── openai_compat.py   # OpenAI-compatible provider
│       │   └── registry.py        # Provider registry
│       ├── services/
│       │   ├── __init__.py
│       │   ├── chat_service.py    # Chat completion service
│       │   ├── metrics_service.py # Metrics collection
│       │   └── health_service.py  # Health check service
│       └── utils/
│           ├── __init__.py
│           ├── logger.py         # Logging setup
│           └── async_utils.py    # Async helper utilities
├── tests/
│   ├── __init__.py
│   ├── conftest.py               # Pytest fixtures
│   ├── test_classifier/
│   │   ├── __init__.py
│   │   ├── test_keyword_classifier.py
│   │   └── test_classifier_registry.py
│   ├── test_router/
│   │   ├── __init__.py
│   │   ├── test_dynamic_router.py
│   │   └── test_routing_config.py
│   ├── test_providers/
│   │   ├── __init__.py
│   │   ├── test_ollama_provider.py
│   │   └── test_provider_registry.py
│   ├── test_services/
│   │   ├── __init__.py
│   │   ├── test_chat_service.py
│   │   └── test_health_service.py
│   └── test_integration/
│       ├── __init__.py
│       └── test_api_endpoints.py
├── scripts/
│   ├── install_ollama.sh        # Ollama installation helper
│   └── pull_models.sh           # Pull default models
└── docs/
    ├── architecture.md          # Architecture documentation
    ├── api_reference.md          # API reference
    └── routing_config.md        # Routing configuration guide
```

---

## 5. Testing Requirements

### 5.1 Unit Tests

**Test Coverage Goals**
- Classifier: ≥90% coverage
- Router: ≥90% coverage
- Provider interface: ≥85% coverage
- Services: ≥85% coverage

**Critical Test Cases**
1. **Task Classifier Tests**
   - Classification of planning requests
   - Classification of code generation requests
   - Classification of debugging requests
   - Default fallback for unknown task types
   - Confidence threshold behavior

2. **Router Tests**
   - Correct model selection per task type
   - Fallback chain execution
   - Provider priority ordering
   - Invalid configuration handling

3. **Provider Tests**
   - Successful request/response handling
   - Timeout handling
   - Error propagation
   - Streaming response handling

### 5.2 Integration Tests

**API Endpoint Tests**
- `/health` returns healthy status
- `/models` returns configured models
- `/classify` returns valid classification
- `/v1/chat/completions` returns valid response
- Streaming responses are valid SSE
- Error responses have proper format

**Test Environment**
- Use Ollama with `llama2:7b` or `codellama:7b` for integration tests
- Mock external API calls for CI speed (use pytest-mock)
- Provide test configuration for offline testing

### 5.3 CI Pipeline Requirements

**Automated Tests on GitHub Actions**
- Run unit tests on every push
- Run integration tests on pull requests
- Type checking with mypy
- Linting with ruff
- Coverage reporting to codecov (optional)

**Test Execution Time**
- Unit tests: