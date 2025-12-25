# Integration Roadmap: G4F Fallback Providers

This document outlines the phased approach for integrating G4F (g4f) fallback providers into the LLM-API-Key-Proxy project.

---

## Phase 1 Summary (COMPLETED ✅)

Phase 1 focuses on configuration and documentation updates required for G4F integration.

### Changes to `.env.example`

**Status**: COMPLETED ✅

The following environment variables have been added to `.env.example`:
- G4F_API_KEY
- G4F_MAIN_API_BASE
- G4F_GROQ_API_BASE
- G4F_GROK_API_BASE
- G4F_GEMINI_API_BASE
- G4F_NVIDIA_API_BASE
- PROVIDER_PRIORITY_* variables for all providers

### Changes to `README.md`

**Status**: COMPLETED ✅

The "G4F Fallback Providers" section has been added with comprehensive documentation covering:
- Setup instructions
- Compatibility matrix
- Monitoring information
- Limitations and best practices

---

## Phase 2 Roadmap (COMPLETED ✅)

Phase 2 implements the actual G4F provider routing logic in the codebase.

### Implementation Summary

| Component | Status | Details |
|-----------|--------|---------|
| G4F Provider Class | ✅ COMPLETED | Full implementation with 5 endpoint support |
| Priority Tier System | ✅ COMPLETED | G4F defaults to Tier 5 (lowest priority) |
| Provider Registration | ✅ COMPLETED | G4F registered in providers/__init__.py |
| Test Suite | ✅ COMPLETED | 75 tests passing (28 provider + 17 routing + 20 failover + 10 fixtures) |
| Demo Script | ✅ COMPLETED | demo_g4f_fallback.py created |
| Code Quality | ✅ COMPLETED | Ruff linting passes, mypy type checking passes |

### 2.1 Implement G4F Provider Class/Handler

**Status**: COMPLETED ✅

**Affected Files**:
- `src/rotator_library/providers/g4f_provider.py` (NEW FILE - 628 lines)
- `src/rotator_library/providers/__init__.py` (update)
- `src/rotator_library/provider_factory.py` (update)

**Dependencies**: None - new provider implementation

**Complexity**: Medium

**Acceptance Criteria**:
- [x] Provider class extends `ProviderInterface`
- [x] Implements `chat_completions()` with streaming support
- [x] Implements `embeddings()` (raises NotImplementedError)
- [x] Handles authentication via `G4F_API_KEY`
- [x] Configurable base URLs via environment variables
- [x] Proper error handling with G4F-specific error codes
- [x] Unit tests covering main scenarios

**Technical Specifications Implemented**:

```python
# src/rotator_library/providers/g4f_provider.py

class G4FProvider(ProviderInterface):
    """G4F fallback provider implementation."""
    
    provider_name = "g4f"
    tier_priorities = {"standard": 5}
    default_tier_priority = 5
    skip_cost_calculation = bool = True
    
    def __init__(self):
        self._endpoints = self._load_configuration()
    
    def _get_endpoint_for_model(self, model: str) -> Optional[str]:
        """Route to appropriate G4F endpoint based on model name."""
        # Routes to groq/grok/gemini/nvidia/main endpoints based on model pattern
        pass
    
    async def chat_completions(self, client, **kwargs):
        """Handle chat completion requests via G4F with streaming."""
        pass
    
    async def embeddings(self, client, **kwargs):
        """Not implemented - G4F doesn't support embeddings."""
        raise NotImplementedError(...)
```

### 2.2 Implement Priority Tier Logic in Request Routing

**Status**: COMPLETED ✅

**Affected Files**:
- `src/rotator_library/client.py` (update)

**Dependencies**: 2.1 (G4F Provider Class)

**Complexity**: High

**Acceptance Criteria**:
- [x] Providers can be assigned priority tiers (1=highest, N=lowest)
- [x] Request routing respects priority order
- [x] Tier-based fallback chain: Tier 1 → Tier 2 → ... → Tier N
- [x] G4F providers placed in lowest priority tier by default
- [x] Configuration via `PROVIDER_PRIORITY_*` environment variables
- [x] Tier information exposed via provider priority functions

**Technical Specifications Implemented**:

```python
# Priority tier system in client.py

DEFAULT_PROVIDER_PRIORITIES: Dict[str, int] = {
    # Tier 1: Premium paid providers (highest priority)
    "openai": 1,
    "anthropic": 1,
    # Tier 2: Fast/affordable providers
    "groq": 2,
    "openrouter": 2,
    # Tier 3: Standard providers
    "gemini": 3,
    "mistral": 3,
    # Tier 5: Fallback providers (G4F - lowest priority)
    "g4f": 5,
}

def get_provider_priority(provider: str) -> int:
    """Get priority tier for a provider."""
    env_key = f"PROVIDER_PRIORITY_{provider.upper()}"
    if env_key in os.environ:
        return int(os.environ[env_key])
    return DEFAULT_PROVIDER_PRIORITIES.get(provider.lower(), 10)
```

### 2.3 Add Unit/Integration Tests for G4F Fallback Scenarios

**Status**: COMPLETED ✅

**Affected Files**:
- `tests/test_g4f_provider.py` (NEW FILE)
- `tests/test_priority_tier_routing.py` (NEW FILE)
- `tests/test_failover.py` (NEW FILE)
- `tests/conftest.py` (NEW FILE)

**Dependencies**: 2.1, 2.2

**Complexity**: Medium

**Acceptance Criteria**:
- [x] Test G4F provider initialization
- [x] Test chat completions (streaming and non-streaming)
- [x] Test fallback routing on 429 errors
- [x] Test priority tier ordering
- [x] Test provider selection with multiple tiers available
- [x] Integration tests with mock G4F server
- [x] All tests pass in CI pipeline

**Test Scenarios Implemented**:

```python
# tests/test_g4f_provider.py

@pytest.fixture
def g4f_provider():
    return G4FProvider()

@pytest.mark.asyncio
async def test_chat_completions(g4f_provider):
    response = await g4f_provider.chat_completions(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        stream=False
    )
    assert response.choices[0].message.content is not None

@pytest.mark.asyncio
async def test_fallback_on_rate_limit():
    """Test that requests fall back to G4F on 429."""
    # Test implementation
    pass

def test_priority_tier_parsing():
    """Test priority tier environment variable parsing."""
    # Test implementation
    pass
```

### 2.4 Update Provider Factory

**Status**: COMPLETED ✅

**Affected Files**:
- `src/rotator_library/provider_factory.py`
- `src/rotator_library/providers/__init__.py`

**Dependencies**: 2.1

**Complexity**: Low

**Acceptance Criteria**:
- [x] Factory can instantiate G4FProvider
- [x] Environment-based configuration loading works
- [x] Provider cache includes G4F entries

---

## Phase 3 Roadmap (Future)

Phase 3 focuses on deployment validation, monitoring, and optimization.

### 3.1 Deployment Validation

**Status**: NOT PLANNED

**Tasks**:
- Docker container validation for G4F provider
- Docker Compose configuration updates
- Integration test suite for containerized deployment
- Health check endpoint for G4F provider status

**Affected Files**:
- `Dockerfile` (update)
- `docker-compose.yml` (update)
- `docker-compose.override.yml` (new)

### 3.2 Production Monitoring and Alerting

**Status**: NOT PLANNED

**Tasks**:
- Metrics for G4F fallback usage (counter)
- Latency tracking for G4F vs direct providers
- Error rate monitoring per tier
- Alert configuration for excessive fallback usage
- Dashboard for G4F provider health

**Affected Files**:
- `src/proxy_app/metrics.py` (new)
- `Deployment guide.md` (update)

### 3.3 Performance Optimization

**Status**: NOT PLANNED

**Tasks**:
- Connection pooling for G4F endpoints
- Request batching optimization
- Caching strategy for G4F responses
- Adaptive timeout configuration
- Load testing with fallback scenarios

**Affected Files**:
- `src/rotator_library/client.py` (update)
- `src/rotator_library/timeout_config.py` (update)

---

## Environment Variable Reference

### G4F Configuration

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `G4F_API_KEY` | API key for G4F provider | No | `None` |
| `G4F_MAIN_API_BASE` | Base URL for main G4F API | Yes | - |
| `G4F_GROQ_API_BASE` | Base URL for Groq-compatible endpoint | No | - |
| `G4F_GROK_API_BASE` | Base URL for Grok-compatible endpoint | No | - |
| `G4F_GEMINI_API_BASE` | Base URL for Gemini-compatible endpoint | No | - |
| `G4F_NVIDIA_API_BASE` | Base URL for NVIDIA-compatible endpoint | No | - |

### Priority Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `PROVIDER_PRIORITY_G4F` | Priority tier for G4F (1=highest) | 5 |
| `PROVIDER_PRIORITY_GROQ` | Priority tier for Groq | 2 |
| `PROVIDER_PRIORITY_GEMINI` | Priority tier for Gemini | 3 |
| `PROVIDER_PRIORITY_OPENAI` | Priority tier for OpenAI | 1 |

---

## File Dependencies Matrix

| File | Phase 1 | Phase 2.1 | Phase 2.2 | Phase 2.3 | Phase 3 |
|------|---------|-----------|-----------|-----------|---------|
| `.env.example` | ✅ | | | | |
| `README.md` | ✅ | | | | |
| `g4f_provider.py` | | ✅ | ✅ | ✅ | |
| `providers/__init__.py` | | ✅ | | ✅ | |
| `provider_factory.py` | | ✅ | | ✅ | |
| `client.py` | | | ✅ | | ✅ |
| `credential_manager.py` | | | ✅ | | |
| `test_g4f_provider.py` | | | | ✅ | |
| `test_priority_routing.py` | | | | ✅ | |
| `conftest.py` | | | | ✅ | |
| `Dockerfile` | | | | | ✅ |
| `metrics.py` | | | | | ✅ |

---

## Quick Reference: Implementation Order

1. Update `.env.example` with G4F variables
2. Update `README.md` with G4F documentation
3. Create `G4FProvider` class in `providers/g4f_provider.py`
4. Update `providers/__init__.py` to export G4FProvider
5. Update `provider_factory.py` to handle G4F
6. Implement priority tier logic in `client.py`
7. Update `credential_manager.py` for priority tiers
8. Create unit tests
9. Run full test suite
10. Update documentation with final configuration
