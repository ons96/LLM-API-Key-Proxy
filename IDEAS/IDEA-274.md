# AGENTS.md: Multi-Provider LLM Gateway Manager

## 1. Role/Mission

### Mission Statement

Build an autonomous, self-healing LLM gateway system that intelligently routes requests across multiple AI providers and API keys, with automatic failover, dynamic load balancing, and optimal fallback chain management.

### Core Objectives

1. **Multi-Provider Abstraction**: Create a unified interface for interacting with OpenAI, Anthropic, Google, Azure, Ollama, and third-party providers (e.g., OpenRouter, Together AI, LiteLLM proxies).

2. **Key Rotation & Load Balancing**: Implement intelligent rotation of multiple API keys per provider to distribute load, avoid rate limits, and handle quota exhaustion gracefully.

3. **Dynamic Fallback Chains**: Design configurable "virtual models" that define fallback sequences (e.g., GPT-4 → Claude-3 → Gemini → local Ollama), with automatic reordering based on real-time performance metrics.

4. **Resilient Error Handling**: Implement comprehensive error recovery with exponential backoff, circuit breakers, and automatic provider switching on failures.

5. **Tool Call Routing**: Handle function calling across different provider schemas with translation layers between provider-specific formats.

### Success Criteria

- System can route any LLM request through 3+ provider types seamlessly
- Automatic failover completes requests within 30 seconds when primary provider fails
- API key rotation maintains 99.5% successful request rate under load
- Dynamic reordering improves average latency by 20% over static configurations

---

## 2. Technical Stack

### Language & Runtime

- **Python 3.11+** with type hints
- **uv** for package management (faster than pip, better lockfiles)

### Core Dependencies

```
# Core framework
pydantic>=2.0          # Data validation & settings
pydantic-settings     # Configuration management
httpx>=0.27           # Async HTTP client
aiohttp>=3.9          # WebSocket support fallback

# Resilience & reliability
tenacity>=8.2         # Auto-retry logic
cachetools>=5.4       # In-memory caching
structlog>=24.0      # Structured logging

# Configuration & secrets
pyyaml>=6.0           # YAML configuration
python-dotenv>=1.0    # Environment variable loading

# Monitoring & metrics
prometheus-client>=0.19  # Metrics export
opentelemetry-api>=1.20 # Observability

# Testing
pytest>=8.0           # Testing framework
pytest-asyncio>=0.23  # Async test support
pytest-mock>=3.12      # Mocking utilities
```

### Optional Provider SDKs (Lazy-loaded)

```
openai>=1.10           # OpenAI API
anthropic>=0.18        # Anthropic API
google-generativeai>=0.5  # Google AI
azure-openai>=1.1      # Azure OpenAI
ollama>=0.1            # Ollama local
```

### Dev Dependencies

```
ruff>=0.3             # Linting
mypy>=1.8             # Type checking
pre-commit>=3.6       # Git hooks
```

---

## 3. Requirements (Numbered)

### Core Gateway Requirements

#### R1: Unified Provider Interface
- [ ] Define abstract base class `LLMProvider` with standard methods: `chat_completion()`, `embeddings()`, `list_models()`
- [ ] Implement concrete providers for: OpenAI, Anthropic, Google Gemini, Azure OpenAI, Ollama, OpenRouter (third-party)
- [ ] Support provider-specific authentication (API keys, OAuth, Azure AD tokens)
- [ ] Handle provider-specific request/response transformations

#### R2: Multi-Key Management
- [ ] Store multiple API keys per provider in encrypted configuration
- [ ] Implement round-robin, least-used, and weighted key selection strategies
- [ ] Track per-key usage counts and last-used timestamps
- [ ] Detect key-level rate limits (429 errors) and automatically skip exhausted keys
- [ ] Support key refresh/reload without service interruption

#### R3: Fallback Chain Configuration
- [ ] Define `FallbackChain` data structure with ordered model list and failure tolerance
- [ ] Support "virtual model" aliases that expand to fallback sequences
- [ ] Configure per-chain settings: max retries, timeout, cost weighting
- [ ] Support provider grouping (e.g., "any gpt-4 compatible")

#### R4: Dynamic Reordering
- [ ] Implement performance tracking per (provider, model) tuple
- [ ] Calculate running metrics: success rate, latency p50/p95/p99, cost per token
- [ ] Automatically demote poor-performing providers/models after configurable threshold
- [ ] Implement "healing" logic to slowly promote previously-failed providers
- [ ] Persist performance data to disk for restart resilience

#### R5: Error Handling & Recovery
- [ ] Classify errors into recoverable (rate limit, timeout,502) vs non-recoverable (auth, quota)
- [ ] Implement exponential backoff with jitter for rate limits
- [ ] Add circuit breaker pattern: track failure rate, stop routing to provider when threshold exceeded
- [ ] Support dead letter queue for failed requests after all fallbacks exhausted
- [ ] Log detailed error context for debugging

#### R6: Tool/Function Calling
- [ ] Abstract tool call schema across providers (OpenAI function calling, Anthropic tool use)
- [ ] Implement automatic format conversion between provider schemas
- [ ] Support tool call streaming and partial execution
- [ ] Handle tool call timeouts and max iterations

#### R7: Streaming Support
- [ ] Implement Server-Sent Events (SSE) streaming for chat completions
- [ ] Handle stream interruption and reconnection
- [ ] Support partial response buffering for fallback switching mid-stream

#### R8: Configuration Management
- [ ] Support YAML-based configuration files
- [ ] Support environment variable overrides
- [ ] Support runtime configuration reloading via signal or API
- [ ] Validate configuration on load (provider availability, key format)

### Observability Requirements

#### R9: Logging & Tracing
- [ ] Structured logging with contextual fields (request_id, provider, model, duration)
- [ ] Request/response logging with PII handling (redact API keys)
- [ ] Integration with OpenTelemetry for distributed tracing

#### R10: Metrics & Monitoring
- [ ] Export Prometheus metrics: request count, success/failure rate, latency histogram
- [ ] Track per-provider and per-model metrics
- [ ] Alert on sustained failure rate > 10%

### Testing Requirements

#### R11: Unit Tests
- [ ] Test provider interface implementations with mock responses
- [ ] Test key rotation logic
- [ ] Test fallback chain traversal
- [ ] Test error classification and recovery

#### R12: Integration Tests
- [ ] Test against mock server responses
- [ ] Test configuration loading
- [ ] Test metrics export

#### R13: Load/End-to-End Tests
- [ ] Simulate realistic traffic patterns
- [ ] Test failover under injected failures
- [ ] Verify performance under concurrent load

---

## 4. File Structure

```
llm_gateway/
├── __init__.py
├── pyproject.toml
├── .env.example
├── .gitignore
├── ruff.toml
├── mypy.ini
├── .pre-commit-config.yaml
│
├── src/
│   └── llm_gateway/
│       ├── __init__.py
│       ├── main.py                    # Application entry point
│       ├── config/
│       │   ├── __init__.py
│       │   ├── settings.py             # Pydantic settings
│       │   ├── models.py               # Config data models
│       │   ├── loader.py               # Config loading
│       │   └── validators.py           # Config validation
│       │
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── base.py                 # Abstract provider base
│       │   ├── registry.py             # Provider registry
│       │   ├── openai.py               # OpenAI provider
│       │   ├── anthropic.py           # Anthropic provider
│       │   ├── google.py               # Google Gemini provider
│       │   ├── azure.py                # Azure OpenAI provider
│       │   ├── ollama.py                # Ollama local provider
│       │   └── openrouter.py           # Third-party (OpenRouter)
│       │
│       ├── keys/
│       │   ├── __init__.py
│       │   ├── manager.py              # Key manager interface
│       │   ├── rotation.py             # Rotation strategies
│       │   └── stats.py                # Key usage tracking
│       │
│       ├── routing/
│       │   ├── __init__.py
│       │   ├── chain.py                # Fallback chain logic
│       │   ├── reorder.py              # Dynamic reordering
│       │   ├── circuit.py              # Circuit breaker
│       │   └── selector.py             # Model/provider selection
│       │
│       ├── errors/
│       │   ├── __init__.py
│       │   ├── classification.py       # Error classification
│       │   ├── retry.py                # Retry policies
│       │   └── dead_letter.py          # Failed request handling
│       │
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── schema.py               # Tool schema abstraction
│       │   ├── router.py              # Tool call routing
│       │   └── executor.py             # Tool execution
│       │
│       ├── streaming/
│       │   ├── __init__.py
│       │   ├── handler.py               # Stream handling
│       │   └── buffer.py              # Partial response buffer
│       │
│       ├── metrics/
│       │   ├── __init__.py
│       │   ├── collector.py            # Metrics collection
│       │   ├── prometheus.py           # Prometheus export
│       │   └── tracking.py            # Performance tracking
│       │
│       ├── observability/
│       │   ├── __init__.py
│       │   ├── logging.py             # Structured logging
│       │   └── tracing.py             # OpenTelemetry setup
│       │
│       └── utils/
│           ├── __init__.py
│           ├── pii.py                  # PII redaction
│           └── sync.py                 # Sync utilities
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_providers/
│   │   │   ├── __init__.py
│   │   │   ├── test_base.py
│   │   │   ├── test_openai.py
│   │   │   └── test_anthropic.py
│   │   ├── test_keys/
│   │   │   ├── __init__.py
│   │   │   ├── test_rotation.py
│   │   │   └── test_stats.py
│   │   ├── test_routing/
│   │   │   ├── __init__.py
│   │   │   ├── test_chain.py
│   │   │   └── test_reorder.py
│   │   └── test_errors/
│   │       ├── __init__.py
│   │       └── test_classification.py
│   │
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_config.py
│   │   └── test_providers.py
│   │
│   ├── fixtures/
│   │   ├── __init__.py
│   │   └── responses/
│   │       ├── openai_chat.json
│   │       ├── anthropic_message.json
│   │       └── error_429.json
│   │
│   └── e2e/
│       ├── __init__.py
│       ├── test_fallback_chain.py
│       └── test_load.py
│
├── scripts/
│   ├── validate_config.py
│   └── benchmark.py
│
├── docs/
│   ├── architecture.md
│   ├── configuration.md
│   └── providers.md
│
├── QUESTIONS.md                          # Save questions here
└── COMPLETION.md                       # Track progress
```

---

## 5. Testing Requirements

### Test Coverage Goals

- **Unit Tests**: 90%+ coverage on core routing and key management logic
- **Integration Tests**: All provider implementations tested against mocked APIs
- **E2E Tests**: Fallback chain tests verify full request lifecycle

### Testing Standards

1. **Async Testing**: All I/O-bound operations tested with `pytest-asyncio`
2. **Fixtures**: Use shared fixtures for common configurations and mock responses
3. **Parameterization**: Use `@pytest.mark.parametrize` for provider/variation coverage
4. **Isolation**: Each test cleans up its own state; no shared global fixtures
5. **Clarity**: Test names follow `test