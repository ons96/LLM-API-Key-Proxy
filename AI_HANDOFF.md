# AI Agent Handoff Guide

This document provides a quick reference for any AI agent taking over development of the LLM-API-Key-Proxy project.

---

## Quick Start for AI Agents

### Project Purpose and Current State

**Purpose**: Universal OpenAI-compatible LLM proxy server that provides:
- Single API endpoint for multiple LLM providers
- Automatic key rotation and failover
- Intelligent request routing with priority tiers
- OAuth support for various providers

**Current State**: Phase 1 of G4F fallback integration is **NOT STARTED**.
- Configuration variables not added to `.env.example`
- Documentation not added to `README.md`
- Provider implementation not started

**Branch**: `docs-g4f-phase1-verification-ai-handoff`

### How to Run the Proxy Locally

```bash
# Clone and setup
git clone https://github.com/Mirrowel/LLM-API-Key-Proxy.git
cd LLM-API-Key-Proxy
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your API keys

# Run with TUI (interactive)
python src/proxy_app/main.py

# Run directly with arguments
python src/proxy_app/main.py --host 0.0.0.0 --port 8000
```

### How to Test Changes

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_g4f_provider.py -v

# Run with coverage
pytest --cov=src/rotator_library tests/

# Test proxy manually
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_PROXY_API_KEY" \
  -d '{
    "model": "gemini/gemini-2.5-flash",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## Key Codebase Locations

### Provider Logic

| Component | Location | Purpose |
|-----------|----------|---------|
| **Main Client** | `src/rotator_library/client.py` | RotatingClient class - handles request routing, key rotation, failover |
| **Credential Manager** | `src/rotator_library/credential_manager.py` | API key discovery and management |
| **Provider Interface** | `src/rotator_library/providers/provider_interface.py` | Base class for all providers |
| **Provider Factory** | `src/rotator_library/provider_factory.py` | Factory for creating provider instances |
| **Error Handler** | `src/rotator_library/error_handler.py` | Error handling and cooldown logic |

### Configuration Loading

| Component | Location | Purpose |
|-----------|----------|---------|
| Environment Config | `src/proxy_app/config.py` (if exists) or direct `os.getenv` calls | Loads `.env` variables |
| Provider Config | `src/rotator_library/providers/__init__.py` | Provider registry |

### Request Routing

| Component | Location | Purpose |
|-----------|----------|---------|
| Router | `src/rotator_library/client.py` | Routes requests to appropriate provider |
| Request Sanitizer | `src/rotator_library/request_sanitizer.py` | Validates and normalizes requests |
| Timeout Config | `src/rotator_library/timeout_config.py` | HTTP timeout settings |

### Tests

| Location | Purpose |
|----------|---------|
| `tests/` | All unit and integration tests |
| `tests/test_priority_routing.py` | Tests for priority tier routing (needs creation) |
| `tests/test_g4f_provider.py` | Tests for G4F provider (needs creation) |

---

## For Next Phase Implementation

### Phase 2 Items (Priority Order)

1. **Add G4F configuration to `.env.example`**
   - Add `G4F_API_KEY`, `G4F_MAIN_API_BASE`, etc.
   - Add `PROVIDER_PRIORITY_*` variables
   - Reference: `INTEGRATION_ROADMAP.md` Phase 1 section

2. **Add G4F documentation to `README.md`**
   - Create "G4F Fallback Providers" section
   - Document setup, compatibility, monitoring
   - Reference: `INTEGRATION_ROADMAP.md` for template

3. **Create G4F Provider class**
   - File: `src/rotator_library/providers/g4f_provider.py`
   - Must extend `ProviderInterface`
   - Implement `chat_completions()` and `embeddings()`

4. **Implement priority tier routing**
   - Modify `src/rotator_library/client.py`
   - Add `get_provider_priority()` function
   - Update request routing to respect tiers

5. **Write unit tests**
   - Create `tests/test_g4f_provider.py`
   - Create `tests/test_priority_routing.py`
   - Ensure all tests pass before committing

### Critical Files to Modify

| File | Change Type | Description |
|------|-------------|-------------|
| `.env.example` | Add | G4F provider variables |
| `README.md` | Add | G4F documentation section |
| `src/rotator_library/providers/g4f_provider.py` | Create | G4F provider implementation |
| `src/rotator_library/providers/__init__.py` | Modify | Export G4FProvider |
| `src/rotator_library/provider_factory.py` | Modify | Handle G4F provider creation |
| `src/rotator_library/client.py` | Modify | Priority tier routing logic |
| `tests/test_g4f_provider.py` | Create | Unit tests for G4F |
| `tests/test_priority_routing.py` | Create | Tests for priority system |

### Testing Requirements

Before pushing changes, ensure:

```bash
# Run full test suite
pytest tests/ -v

# Verify no linting errors
flake8 src/rotator_library/ --max-line-length=120

# Verify type hints (if mypy is configured)
mypy src/rotator_library/

# Check formatting
black src/rotator_library/ --check
```

### Validation Steps

1. Create feature branch from `main`
2. Implement changes
3. Write/update tests
4. Run full test suite
5. Verify documentation is complete
6. Commit with conventional commit message
7. Push and create PR

---

## Important Context

### G4F Provider Priority Tier System

The priority tier system controls the order in which providers are tried:

- **Tier 1** (Highest): Premium providers (OpenAI, Anthropic)
- **Tier 2**: Fast/affordable providers (Groq, OpenRouter)
- **Tier 3**: Standard providers (Gemini, Mistral)
- **Tier 4**: Fallback providers (G4F - free/limited)
- **Tier 5+**: Additional fallback tiers

**How it works**:
1. Request comes in for `provider/model`
2. Get priority tier of primary provider
3. If primary fails (429, 5xx), try next provider in same tier
4. If all providers in tier fail, move to next tier
5. G4F providers are placed in lowest tier (default: 5)

**Configuration**:
```env
PROVIDER_PRIORITY_G4F=5      # G4F in tier 5 (fallback)
PROVIDER_PRIORITY_GROQ=2     # Groq in tier 2
PROVIDER_PRIORITY_OPENAI=1   # OpenAI in tier 1 (highest)
```

### How the RotatingClient Works

The `RotatingClient` is the core component managing API key rotation:

```python
# Key flow:
1. Request comes in: client.acompletion(model="provider/model", messages=[...])
2. Extract provider name from model identifier
3. Get available credentials for provider from CredentialManager
4. Select credential based on:
   - Priority tier (if configured)
   - Rotation mode (balanced/sequential)
   - Current load (concurrent request limits)
5. Execute request with timeout
6. On error (429, 5xx):
   - Apply cooldown to credential
   - Rotate to next credential in tier
   - Retry up to max_retries
7. Return response or raise exception
```

**Key Classes**:
- `RotatingClient`: Main async client
- `CredentialManager`: Manages API key discovery and rotation
- `CooldownManager`: Tracks rate limits and cooldowns
- `UsageManager`: Tracks usage per credential

### Multi-Provider Failover Mechanism

```
Request → Provider Selection → Credential Selection → Request Execution
                                    ↓
                              [Error: 429/5xx]
                                    ↓
                            Apply Cooldown
                                    ↓
                         Rotate to Next Credential
                                    ↓
                         Retry (up to max_retries)
                                    ↓
                       [Still Failing: All Credentials]
                                    ↓
                    Move to Next Priority Tier (if available)
                                    ↓
                       Try All Providers in Tier
                                    ↓
                    [All Tiers Exhausted]
                                    ↓
                    Raise ProviderExhaustedError
```

### OpenAI SDK Compatibility Requirements

The proxy must produce responses compatible with OpenAI SDK:

```python
# Response format must match:
{
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1677858242,
    "model": "gpt-4",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "Hello!"
        },
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 13,
        "completion_tokens": 7,
        "total_tokens": 20
    }
}
```

**Streaming format**:
```python
# Each chunk:
{
    "id": "chatcmpl-abc123",
    "object": "chat.completion.chunk",
    "created": 1677858242,
    "model": "gpt-4",
    "choices": [{
        "index": 0,
        "delta": {
            "content": "Hello"
        },
        "finish_reason": null
    }]
}
```

---

## Common Commands

### Install Dependencies

```bash
# Basic installation
pip install -r requirements.txt

# With dev dependencies
pip install -r requirements.txt -r requirements-dev.txt

# Individual packages
pip install fastapi uvicorn httpx litellm python-dotenv rich
```

### Run Tests

```bash
# All tests
pytest tests/ -v

# Specific test
pytest tests/test_g4f_provider.py -v

# With coverage report
pytest --cov=src/rotator_library --cov-report=html tests/

# Fast run (no coverage)
pytest tests/ -q
```

### Start the Proxy

```bash
# Interactive TUI
python src/proxy_app/main.py

# Direct server start
python src/proxy_app/main.py --host 127.0.0.1 --port 8000

# With verbose logging
python src/proxy_app/main.py --enable-request-logging

# Custom environment file
PROXY_ENV=.env.production python src/proxy_app/main.py
```

### Validate Changes

```bash
# Python syntax check
python -m py_compile src/rotator_library/client.py

# Linting
flake8 src/rotator_library/ --max-line-length=120 --extend-ignore=E203

# Type checking
mypy src/rotator_library/ --ignore-missing-imports

# Formatting check
black src/rotator_library/ --check

# Import sorting
isort src/rotator_library/ --check-only
```

### Git Operations

```bash
# Create feature branch
git checkout -b feat/g4f-provider

# Stage changes
git add src/rotator_library/providers/g4f_provider.py

# Commit with conventional message
git commit -m "feat(provider): Add G4F fallback provider implementation"

# Push to origin
git push -u origin feat/g4f-provider

# Create PR (via GitHub CLI)
gh pr create --title "feat: Add G4F fallback provider" --body "..."
```

---

## Additional Resources

| Resource | Link |
|----------|------|
| Project README | `README.md` |
| Integration Roadmap | `INTEGRATION_ROADMAP.md` |
| Project Status | `PROJECT_STATUS.md` |
| Full Documentation | `DOCUMENTATION.md` |
| Deployment Guide | `Deployment guide.md` |
| Library README | `src/rotator_library/README.md` |
| LiteLLM Providers | https://docs.litellm.ai/docs/providers |
| g4f GitHub | https://github.com/xtekky/g4f |

---

## Summary of Verification Results

| Check | Status | Notes |
|-------|--------|-------|
| G4F variables in `.env.example` | ❌ Not found | Need to add G4F_API_KEY, G4F_MAIN_API_BASE, etc. |
| PROVIDER_PRIORITY_* variables | ❌ Not found | Need to add priority tier configuration |
| G4F section in README.md | ❌ Not found | Need to add "G4F Fallback Providers" section |

**Overall Phase 1 Status**: NOT STARTED

The Phase 1 configuration and documentation changes must be completed before beginning Phase 2 implementation work.
