# Design: Deduplicated Model Selector with Multi-Provider Auto-Retry

## Problem Statement

Currently, the gateway lists models per-provider, causing duplication:
- `groq/llama-3.3-70b`
- `cerebras/llama-3.3-70b`
- `g4f/llama-3.3-70b`

Users see 3 entries for the same model. When one provider fails, they must manually try another.

## Proposed Solution

### 1. Unified Model Registry

Create a central registry that maps canonical model names to available providers:

```yaml
models:
  llama-3.3-70b:
    providers:
      - provider: cerebras
        model_id: llama-3.3-70b
        priority: 1
        free: true
        rate_limit: 30
      - provider: groq
        model_id: llama-3.3-70b-versatile
        priority: 2
        free: true
        rate_limit: 30
      - provider: g4f
        model_id: llama-3.3-70b
        priority: 5
        free: true
        unreliable: true
```

### 2. Model Selector Behavior

When user requests `llama-3.3-70b`:
1. Look up in registry
2. Get ordered provider list by priority
3. Try providers in order until success
4. Track success rates for dynamic reordering

### 3. Implementation Plan

#### Phase 1: Model Registry
- Create `ModelRegistry` class
- Load from config + auto-discover from providers
- Normalize model names across providers

#### Phase 2: Selector Logic
- Update router to use registry lookup
- Implement `resolve_model(model_name)` → `[(provider, model_id), ...]`
- Add success rate tracking

#### Phase 3: API Changes
- `/v1/models` returns deduplicated list
- Each model includes `providers: [...]` metadata
- Add `X-Provider-Used` header in responses

### 4. Config Format

```yaml
model_registry:
  normalize_names: true

  aliases:
    llama-3.3-70b:
      - llama-3.3-70b-versatile
      - llama3.3-70b

  providers:
    cerebras:
      free_models: [llama-3.3-70b, llama-3.1-8b, qwen-3-235b]
      priority_boost: 2

    groq:
      free_models: [llama-3.3-70b-versatile, llama-3.1-8b-instant]
      priority_boost: 1

    g4f:
      unreliable: true
      priority_penalty: 5
```

### 5. Fallback Strategy

```python
async def execute_with_fallback(model_name: str, messages: list):
    providers = registry.get_providers(model_name)

    for provider, model_id in providers:
        try:
            result = await provider.execute(model_id, messages)
            registry.record_success(provider, model_name)
            return result
        except (RateLimitError, ProviderError) as e:
            registry.record_failure(provider, model_name)
            continue

    raise AllProvidersExhaustedError(model_name)
```

### 6. Success Rate Tracking

```python
class ProviderStats:
    success_count: int
    failure_count: int
    avg_latency_ms: float
    last_success: datetime

    def get_score(self) -> float:
        success_rate = self.success_count / (self.success_count + self.failure_count)
        return success_rate * (1 / self.avg_latency_ms)
```

### 7. Benefits

1. **Simpler UX**: One model entry, automatic fallback
2. **Better Reliability**: Auto-retry on provider failure
3. **Cost Optimization**: Prioritize free providers
4. **Analytics**: Track which providers work best

### 8. Migration Path

1. Keep existing `/v1/models` behavior with `?dedup=false`
2. Add new `?dedup=true` (default) for deduplicated view
3. Virtual models continue to work as-is
4. Gradually phase out provider-prefixed model names

## Open Questions

- How to handle provider-specific features (e.g., Groq's `json_mode`)?
- Should we expose provider preference to users?
- How to handle regional differences in model availability?
