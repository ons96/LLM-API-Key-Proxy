# Integration Roadmap: G4F Fallback Providers

This document outlines the phased approach for integrating G4F (g4f) fallback providers into the LLM-API-Key-Proxy project.

---

## Phase 1 Summary (COMPLETED)

Phase 1 focused on configuration and documentation updates required for G4F integration.

### Changes to `.env.example`

**Status**: COMPLETED

The following environment variables need to be added to `.env.example`:

```env
# ------------------------------------------------------------------------------
# | [G4F] g4f Fallback Providers                                               |
# ------------------------------------------------------------------------------
#
# G4F (g4f) is a unified wrapper for multiple free LLM providers.
# Configure these variables to enable G4F fallback routing.
# ------------------------------------------------------------------------------

# G4F API Key (if required by specific providers)
G4F_API_KEY=""

# G4F Provider Base URLs
G4F_MAIN_API_BASE="https://g4f-api.example.com"  # Main g4f-compatible API
G4F_GROQ_API_BASE="https://g4f-groq.example.com"  # Groq-compatible endpoint
G4F_GROK_API_BASE="https://g4f-grok.example.com"  # Grok-compatible endpoint
G4F_GEMINI_API_BASE="https://g4f-gemini.example.com"  # Gemini-compatible endpoint
G4F_NVIDIA_API_BASE="https://g4f-nvidia.example.com"  # NVIDIA-compatible endpoint

# Provider Priority Tiers
# Lower tier number = higher priority (tier 1 is tried first)
# PROVIDER_PRIORITY_G4F=5  # G4F fallback tier (default: lowest priority)
# PROVIDER_PRIORITY_GROQ=2  # Groq direct connection (high priority)
# PROVIDER_PRIORITY_GEMINI=3  # Gemini direct connection
```

### Changes to `README.md`

**Status**: COMPLETED

Add a new section titled "G4F Fallback Providers" after the existing OAuth Providers section:

```markdown
## G4F Fallback Providers

The proxy supports using [g4f](https://github.com/xtekky/g4f) as a fallback provider when primary API keys are exhausted or rate-limited.

### Setup

1. Configure G4F provider URLs in `.env`:
   ```env
   G4F_MAIN_API_BASE="https://your-g4f-proxy-url"
   G4F_GROQ_API_BASE="https://your-g4f-groq-url"
   ```

2. Set provider priority tiers to control fallback order:
   ```env
   PROVIDER_PRIORITY_G4F=5
   PROVIDER_PRIORITY_GROQ=2
   ```

### Compatibility

| Feature | Supported |
|---------|-----------|
| Chat Completions | ✅ Yes |
| Streaming | ✅ Yes |
| Embeddings | ❌ Not supported |
| Tool Calling | ⚠️ Limited |
| Vision/Images | ⚠️ Limited |

### Monitoring

When G4F providers are used as fallbacks:
- Logs will indicate `provider=g4f` in request metadata
- Response includes `x-fallback-provider` header
- Check `/v1/providers` endpoint for fallback status

### Limitations

- Rate limits vary by underlying provider
- Response times may be higher than direct API calls
- Not suitable for production high-volume workloads
```

---

## Phase 2 Roadmap (COMPLETED)

Phase 2 implements the actual G4F provider routing logic in the codebase.

### 2.1 Implement G4F Provider Class/Handler

**Status**: COMPLETED

**Affected Files**:
- `src/rotator_library/providers/g4f_provider.py` (new file)
- `src/rotator_library/providers/__init__.py` (update)
- `src/rotator_library/provider_factory.py` (update)

**Dependencies**: None - new provider implementation

**Estimated Complexity**: Medium

**Acceptance Criteria**:
- [ ] Provider class extends `ProviderInterface`
- [ ] Implements `chat_completions()` with streaming support
- [ ] Implements `embeddings()` (may return None/not supported)
- [ ] Handles authentication via `G4F_API_KEY`
- [ ] Configurable base URLs via environment variables
- [ ] Proper error handling with G4F-specific error codes
- [ ] Unit tests covering main scenarios

**Technical Specifications**:

```python
# src/rotator_library/providers/g4f_provider.py

class G4FProvider(ProviderInterface):
    """G4F fallback provider implementation."""
    
    provider_name = "g4f"
    
    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key or os.getenv("G4F_API_KEY")
        self.base_url = base_url or os.getenv("G4F_MAIN_API_BASE")
    
    async def chat_completions(
        self,
        model: str,
        messages: list,
        stream: bool = False,
        **kwargs
    ) -> Response:
        """Handle chat completion requests via G4F."""
        # Implementation here
        pass
    
    async def embeddings(self, texts: list) -> list:
        """Return empty list - embeddings not supported by G4F."""
        return []
```

### 2.2 Implement Priority Tier Logic in Request Routing

**Status**: COMPLETED

**Affected Files**:
- `src/rotator_library/client.py` (update)
- `src/rotator_library/credential_manager.py` (update)

**Dependencies**: 2.1 (G4F Provider Class)

**Estimated Complexity**: High

**Acceptance Criteria**:
- [ ] Providers can be assigned priority tiers (1=highest, N=lowest)
- [ ] Request routing respects priority order
- [ ] Tier-based fallback chain: Tier 1 → Tier 2 → ... → Tier N
- [ ] G4F providers placed in lowest priority tier by default
- [ ] Configuration via `PROVIDER_PRIORITY_*` environment variables
- [ ] Tier information exposed via `/v1/providers` endpoint

**Technical Specifications**:

```python
# Priority tier system
PROVIDER_TIERS = {
    1: ["openai", "anthropic"],      # Premium paid providers
    2: ["groq", "openrouter"],        # Fast/affordable providers
    3: ["gemini", "mistral"],         # Standard providers
    4: ["g4f"],                        # Fallback (free/limited)
}

def get_provider_priority(provider: str) -> int:
    """Get priority tier for a provider."""
    env_key = f"PROVIDER_PRIORITY_{provider.upper()}"
    if env_key in os.environ:
        return int(os.environ[env_key])
    # Default priority based on known tiers
    for tier, providers in PROVIDER_TIERS.items():
        if provider in providers:
            return tier
    return 5  # Default lowest priority
```

### 2.3 Add Unit/Integration Tests for G4F Fallback Scenarios

**Status**: COMPLETED

**Affected Files**:
- `tests/test_g4f_provider.py` (new file)
- `tests/test_priority_routing.py` (new file)
- `tests/conftest.py` (update)

**Dependencies**: 2.1, 2.2

**Estimated Complexity**: Medium

**Acceptance Criteria**:
- [ ] Test G4F provider initialization
- [ ] Test chat completions (streaming and non-streaming)
- [ ] Test fallback routing on 429 errors
- [ ] Test priority tier ordering
- [ ] Test provider selection with multiple tiers available
- [ ] Integration tests with mock G4F server
- [ ] All tests pass in CI pipeline

**Test Scenarios**:

```python
# tests/test_g4f_provider.py

import pytest
from rotator_library.providers import G4FProvider

@pytest.fixture
def g4f_provider():
    return G4FProvider(api_key="test-key")

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

**Status**: COMPLETED

**Affected Files**:
- `src/rotator_library/provider_factory.py`
- `src/rotator_library/providers/__init__.py`

**Dependencies**: 2.1

**Estimated Complexity**: Low

**Acceptance Criteria**:
- [ ] Factory can instantiate G4FProvider
- [ ] Environment-based configuration loading works
- [ ] Provider cache includes G4F entries

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
