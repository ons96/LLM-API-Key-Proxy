# Provider Adapter Architecture

This document describes the architecture of the provider adapter system in the LLM API Proxy.

## Overview

The provider adapter system allows the gateway to communicate with multiple LLM providers through a unified interface. Each provider (Groq, Gemini, G4F, etc.) has its own adapter that handles provider-specific authentication, API calls, and response formatting.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI Gateway                          │
│                        (main.py)                                 │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RotatingClient                              │
│              (rotator_library/client.py)                         │
│                                                                  │
│  - Manages credential rotation                                   │
│  - Tracks usage per API key                                      │
│  - Handles rate limit backoff                                    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PROVIDER_PLUGINS                              │
│           (providers/__init__.py)                                │
│                                                                  │
│  Dict[str, Type[ProviderInterface]]                              │
│  {                                                               │
│    "groq": GroqProvider,                                         │
│    "gemini_cli": GeminiCliProvider,                              │
│    "g4f": G4FProvider,                                           │
│    ...                                                           │
│  }                                                               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │  Groq    │   │ Gemini   │   │   G4F    │
    │ Provider │   │ Provider │   │ Provider │
    └──────────┘   └──────────┘   └──────────┘
          │               │               │
          ▼               ▼               ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ Groq API │   │Gemini API│   │ G4F Lib  │
    └──────────┘   └──────────┘   └──────────┘
```

## Core Components

### 1. ProviderInterface (Abstract Base Class)

**Location**: `src/rotator_library/providers/provider_interface.py`

All provider adapters inherit from this abstract base class. It defines the contract that every provider must follow.

```python
from abc import ABC, abstractmethod
from typing import Dict, List

class ProviderInterface(ABC):
    """Base class for all provider adapters."""

    # Configuration attributes
    skip_cost_calculation: bool = False
    default_rotation_mode: str = "balanced"  # or "sequential"
    provider_env_name: str = ""

    # Tier management
    tier_priorities: Dict[str, int] = {}
    default_tier_priority: int = 10

    # Usage tracking
    usage_reset_configs: Dict[UsageConfigKey, UsageResetConfigDef] = {}
    model_quota_groups: Dict[str, List[str]] = {}
    model_usage_weights: Dict[str, int] = {}

    @abstractmethod
    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """Fetch list of available models from the provider."""
        pass

    def has_custom_logic(self) -> bool:
        """Return True if provider implements custom completion logic."""
        return False

    async def acompletion(self, client, **kwargs):
        """Custom completion logic (override if has_custom_logic() is True)."""
        raise NotImplementedError()
```

### 2. Provider Plugin Registry

**Location**: `src/rotator_library/providers/__init__.py`

The `PROVIDER_PLUGINS` dictionary maps provider names to their adapter classes:

```python
PROVIDER_PLUGINS: Dict[str, Type[ProviderInterface]] = {
    "groq": GroqProvider,
    "gemini": GeminiProvider,
    "gemini_cli": GeminiCliProvider,
    "g4f": G4FProvider,
    "cerebras": CerebrasProvider,
    "openai": OpenAIProvider,
    "openrouter": OpenRouterProvider,
    "qwen_code": QwenCodeProvider,
    # ... more providers
}
```

#### Plugin Discovery

Plugins are auto-discovered on import:

1. Scan all `*_provider.py` files in the package
2. Import and check for `ProviderInterface` subclasses
3. Derive provider name from filename (e.g., `groq_provider.py` → `groq`)
4. Register in `PROVIDER_PLUGINS`

### 3. Dynamic Provider Creation

For custom endpoints, providers can be created dynamically:

```python
# If CUSTOM_API_BASE environment variable is set:
# A DynamicOpenAICompatibleProvider is created automatically

# Example: Set env vars
# MY_CUSTOM_API_BASE=https://api.custom.com/v1
# MY_CUSTOM_API_KEY=xxx
#
# Provider is accessible as: my_custom
```

## Provider Types

### Simple Provider (Standard API)

For providers using standard OpenAI-compatible APIs:

```python
# groq_provider.py
class GroqProvider(ProviderInterface):
    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        response = await client.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        return [f"groq/{model['id']}" for model in response.json().get("data", [])]

    # No need to implement acompletion() - uses litellm by default
```

### Complex Provider (Custom Logic)

For providers with non-standard APIs:

```python
class GeminiCliProvider(ProviderInterface):
    def has_custom_logic(self) -> bool:
        return True

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        # Custom model discovery logic
        pass

    async def acompletion(self, client, **kwargs):
        # Custom completion logic bypassing litellm
        # Handles streaming, special auth, etc.
        pass
```

### OAuth-Based Provider

For providers requiring OAuth authentication:

```python
class GeminiCliProvider(ProviderInterface, GeminiAuthBase):
    # Inherits OAuth handling from GeminiAuthBase
    # OAuth tokens stored in oauth_creds/ directory
    pass
```

## Provider Configuration

### Tier Priorities

Providers can have multiple credential tiers with different priorities:

```python
tier_priorities = {
    "tier1": 1,    # Highest priority
    "tier2": 5,    # Medium priority
    "free": 10,    # Lowest priority
}
```

### Usage Reset Configuration

Define how usage windows reset:

```python
usage_reset_configs = {
    UsageConfigKey(priority=1): UsageResetConfigDef(
        window_seconds=60,
        mode="credential",
        description="Rate limit window"
    ),
    UsageConfigKey(priority=10): UsageResetConfigDef(
        window_seconds=86400,  # 24 hours
        mode="per_model",
        description="Daily quota"
    ),
}
```

### Model Quota Groups

Models can share quotas:

```python
model_quota_groups = {
    "premium": ["gpt-4", "gpt-4-turbo"],
    "standard": ["gpt-3.5-turbo", "gpt-3.5-turbo-16k"],
}

model_usage_weights = {
    "gpt-4": 6,        # Uses 6x quota
    "gpt-3.5-turbo": 1,  # Uses 1x quota
}
```

## Creating a New Provider

### Step 1: Create the Provider File

Create `src/rotator_library/providers/my_provider.py`:

```python
from typing import List
import httpx
from .provider_interface import ProviderInterface

class MyProvider(ProviderInterface):
    """Adapter for MyProvider API."""

    provider_env_name = "my_provider"
    skip_cost_calculation = True  # If it's a free provider

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """Fetch available models from MyProvider."""
        response = await client.get(
            "https://api.myprovider.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        models = response.json().get("data", [])
        return [f"my_provider/{model['id']}" for model in models]
```

### Step 2: Add Configuration

Add to `config/router_config.yaml`:

```yaml
providers:
  my_provider:
    enabled: true
    free: true
    free_tier_models:
      - my_provider/model-1
      - my_provider/model-2
```

### Step 3: Add Environment Variables

Update `.env.example`:

```env
MY_PROVIDER_API_KEY=your-api-key
```

### Step 4: Test

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "my_provider/model-1", "messages": [{"role": "user", "content": "test"}]}'
```

## Authentication Patterns

### API Key (Simple)

```python
class GroqProvider(ProviderInterface):
    # API key from environment: GROQ_API_KEY
    provider_env_name = "groq"
```

### OAuth (Complex)

```python
from .gemini_auth_base import GeminiAuthBase

class GeminiCliProvider(ProviderInterface, GeminiAuthBase):
    # OAuth tokens from oauth_creds/gemini_cli_oauth_*.json
    # Auto-refreshes expired tokens
```

### Multi-Credential Rotation

```python
# Environment variables:
# GROQ_API_KEY_1=gsk_xxx
# GROQ_API_KEY_2=gsk_yyy
# GROQ_API_KEY_3=gsk_zzz

# RotatingClient automatically rotates through these
```

## Error Handling

Providers should raise specific exceptions:

```python
from fastapi import HTTPException

# Rate limit
raise HTTPException(status_code=429, detail="Rate limit exceeded")

# Auth error
raise HTTPException(status_code=401, detail="Invalid API key")

# Model not found
raise HTTPException(status_code=404, detail="Model not found")

# Provider error
raise HTTPException(status_code=502, detail="Upstream provider error")
```

## Best Practices

1. **Use `skip_cost_calculation=True`** for free providers
2. **Define `tier_priorities`** if you have multiple credential tiers
3. **Implement `get_models()`** to enable dynamic model discovery
4. **Override `has_custom_logic()`** only when litellm doesn't support the provider
5. **Use `model_quota_groups`** when models share quotas
6. **Add usage reset configs** for rate limit windows

## Provider Registry Reference

| Provider | File | Custom Logic | Auth Type |
|----------|------|--------------|-----------|
| Groq | `groq_provider.py` | No | API Key |
| Gemini | `gemini_provider.py` | No | API Key |
| Gemini CLI | `gemini_cli_provider.py` | Yes | OAuth |
| G4F | `g4f_provider.py` | Yes | None |
| OpenAI | `openai_provider.py` | No | API Key |
| OpenRouter | `openrouter_provider.py` | No | API Key |
| Cerebras | `cerebras_provider.py` | No | API Key |
| Qwen Code | `qwen_code_provider.py` | Yes | OAuth |
| HuggingFace | `huggingface_provider.py` | No | API Key |
| Cohere | `cohere_provider.py` | No | API Key |
| Mistral | `mistral_provider.py` | No | API Key |
| NVIDIA NIM | `nvidia_provider.py` | No | API Key |
| Antigravity | `antigravity_provider.py` | Yes | OAuth |
| iFlow | `iflow_provider.py` | Yes | OAuth |
