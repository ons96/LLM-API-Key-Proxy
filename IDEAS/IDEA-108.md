# AGENTS.md - Provider Redundancy & Failover System

---

## 1. Role/Mission

### Purpose
The Provider Redundancy & Failover System ensures continuous AI service availability by maintaining redundant provider mappings for every model name supported by the gateway. When a primary provider experiences an outage (whether due to API failures, rate limits, or service degradation), the system automatically redirects requests to pre-configured fallback providers without user intervention.

### Mission
- **Eliminate single points of failure**: No model request should fail due to a single provider going offline
- **Maintain uptime**: Achieve 99.9%+ availability through automated failover
- **Preserve functionality**: Ensure fallback providers support equivalent model capabilities
- **Provide visibility**: Notify administrators of outages and failover events
- **Enable control**: Allow manual provider override when automatic behavior is undesirable

### Core Philosophy
The system operates on a "configure once, run forever" basis - all redundancy mappings are defined in configuration, and the failover logic executes automatically without human intervention during outages.

---

## 2. Technical Stack

### Implementation Technology
- **Language**: Python 3.10+
- **Framework**: FastAPI (for health check endpoints and admin API)
- **Configuration**: YAML-based configuration files
- **Async Operations**: `asyncio` for concurrent health checks
- **HTTP Client**: `httpx` for provider health probing
- **State Management**: In-memory with Redis persistence for distributed deployments

### Dependencies
```
# requirements.txt
fastapi>=0.104.0
uvicorn>=0.24.0
httpx>=0.25.0
pyyaml>=6.0
pydantic>=2.0
redis>=5.0
python-dotenv>=1.0
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-mock>=3.12.0
```

### External Services
- **Redis** (optional): For distributed health state across multiple gateway instances
- **Provider APIs**: Anthropic, OpenAI, Cohere, Mistral, Anyscale, etc.

---

## 3. Requirements (Numbered)

### 3.1 Provider Health Checking
- [ ] Implement periodic health checks for all configured providers
- [ ] Health check should probe provider endpoints with a minimal request (e.g., lightweight model completion)
- [ ] Default check interval: 60 seconds (configurable)
- [ ] Track health state: `healthy`, `degraded`, `unhealthy`
- [ ] Implement exponential backoff for repeated failures
- [ ] Health check timeout: 10 seconds per provider

### 3.2 Redundant Provider Mapping
- [ ] Create `provider_mappings.yaml` configuration file
- [ ] Define primary and fallback providers for each model name
- [ ] Support multiple fallback levels (e.g., primary -> fallback1 -> fallback2 -> ...)
- [ ] Support virtual models (e.g., `coding-smart` -> actual model mapping)
- [ ] Implement mapping lookup with fallback chain traversal

### 3.3 Automatic Failover Triggers
- [ ] Trigger failover when health check returns provider as `unhealthy`
- [ ] Trigger failover on HTTP 5xx errors from provider API
- [ ] Trigger failover on rate limit (429) errors persisting > 60 seconds
- [ ] Trigger failover on timeout errors (> 30 seconds)
- [ ] Implement circuit breaker pattern: after 5 failures in 60 seconds, open circuit
- [ ] Auto-recovery: attempt to reintegrate provider after 5 successful health checks

### 3.4 Outage Notification System
- [ ] Implement notification webhook system
- [ ] Fire notifications on: failover event, provider recovery, extended outage
- [ ] Include in notification: provider name, model name, action taken, timestamp
- [ ] Support multiple webhook endpoints (configurable)
- [ ] Implement notification batching (max 1 notification per provider per 5 minutes)

### 3.5 Manual Override Capabilities
- [ ] Admin API endpoint to force use of specific provider for a model
- [ ] Admin API endpoint to disable a provider entirely
- [ ] Admin API endpoint to view current provider health status
- [ ] Override persistence: survive restarts (stored in Redis or file)
- [ ] Admin API endpoint to list all active failovers

### 3.6 Provider Request Routing
- [ ] Integrate with gateway request flow
- [ ] Intercept request before sending to provider
- [ ] Check provider health before routing
- [ ] Handle provider response errors with retry on fallback
- [ ] Preserve request context across failover attempts

### 3.7 Configuration Management
- [ ] Support environment variable substitution in config
- [ ] Validate configuration on startup
- [ ] Hot-reload configuration without restart (optional)
- [ ] Provide default configuration with common model mappings

---

## 4. File Structure

```
ai-gateway/
├── README.md
├── requirements.txt
├── .env.example
├── config/
│   ├── provider_mappings.yaml       # Core redundancy mappings
│   ├── providers.yaml              # Provider endpoints and configs
│   └── health_check.yaml            # Health check settings
├── src/
│   ├── __init__.py
│   ├── main.py                      # FastAPI application entry
│   ├── config/
│   │   ├── __init__.py
│   │   ├── loader.py               # Configuration loader
│   │   └── settings.py             # Settings management
│   ├── models/
│   │   ├── __init__.py
│   │   ├── provider.py              # Provider data models
│   │   ├── mapping.py               # Mapping data models
│   │   └── health.py                # Health state models
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py                  # Base provider abstraction
│   │   ├── anthropic.py             # Anthropic provider
│   │   ├── openai.py                # OpenAI provider
│   │   ├── cohere.py                 # Cohere provider
│   │   └── registry.py              # Provider registry
│   ├── failover/
│   │   ├── __init__.py
│   │   ├── manager.py                # Failover orchestration
│   │   ├── circuit_breaker.py        # Circuit breaker logic
│   │   └── health_checker.py         # Health check executor
│   ├── routing/
│   │   ├── __init__.py
│   │   ├── router.py                # Request router
│   │   └── selector.py              # Provider selector
│   ├── notifications/
│   │   ├── __init__.py
│   │   ├── webhook.py                # Webhook notifier
│   │   └── manager.py               # Notification manager
│   └── api/
│       ├── __init__.py
│       ├── admin.py                 # Admin API routes
│       └── health.py                # Health status API
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # Test fixtures
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_config.py           # Config tests
│   │   ├── test_failover.py         # Failover logic tests
│   │   ├── test_circuit_breaker.py  # Circuit breaker tests
│   │   └── test_selector.py         # Provider selector tests
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_health_check.py     # Health check integration
│   │   ├── test_failover_flow.py   # End-to-end failover
│   │   └── test_notifications.py   # Notification flow
│   └── fixtures/
│       ├── mock_providers.yaml      # Mock provider configs
│       └── sample_mappings.yaml    # Sample mappings
├── scripts/
│   ├── check_health.py             # Manual health check script
│   └── generate_mappings.py         # Mapping generator helper
├── QUESTIONS.md                     # Questions for human review
└── AGENTS.md                       # This file
```

---

## 5. Testing Requirements

### 5.1 Unit Tests
- [ ] Test configuration loading and validation
- [ ] Test provider mapping lookup with fallback chain
- [ ] Test circuit breaker state transitions (closed -> open -> half-open)
- [ ] Test health check scheduling and execution
- [ ] Test provider selector logic (prefer healthy, respect overrides)
- [ ] Test notification batching logic

### 5.2 Integration Tests
- [ ] Test complete failover flow: healthy -> unhealthy -> failover -> recovery
- [ ] Test multi-level fallback (primary -> secondary -> tertiary)
- [ ] Test manual override persists across restarts
- [ ] Test health check concurrent execution
- [ ] Test notification webhook firing

### 5.3 Mock-Based Tests
- [ ] Mock provider APIs to simulate success/failure/timed-out responses
- [ ] Mock health check responses
- [ ] Mock Redis for state persistence tests

### 5.4 Test Coverage
- [ ] Minimum 80% code coverage required
- [ ] All critical failover paths must have tests
- [ ] All public API endpoints must have tests

### 5.5 Test Fixtures
```python
# tests/conftest.py key fixtures
@pytest.fixture
def sample_mappings():
    """Sample provider mappings for testing."""
    return {
        "coding-smart": {
            "primary": "anthropic-claude-3-5-sonnet",
            "fallbacks": ["openai-gpt-4", "cohere-command-r"]
        },
        "coding-fast": {
            "primary": "anthropic-claude-3-haiku",
            "fallbacks": ["openai-gpt-4o-mini", "mistral-small"]
        }
    }

@pytest.fixture
def mock_provider_response():
    """Mock successful provider response."""
    return {"choices": [{"message": {"content": "test response"}}]}

@pytest.fixture
def unhealthy_provider():
    """Mock unhealthy provider (raises exception)."""
    raise ProviderError("Provider unavailable")
```

---

## 6. Git Protocol

### 6.1 Branch Strategy
- **Main branch**: `main` - stable, production-ready code
- **Development branch**: `develop` - integration branch
- **Feature branches**: `feature/provider-redundancy-*` - individual features
- **Hotfix branches**: `hotfix/*` - urgent fixes

### 6.2 Commit Messages
Follow conventional commits format:
```
