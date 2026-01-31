# Provider Exclusion & Availability Filtering Policy

## What Providers Are Excluded (score = 0, not considered in fallback)?

### PAID Providers (Excluded by default):
**Excluded:**
- tavily (marked as paid in providers_database.yaml)
- exa (marked as paid in providers_database.yaml)

**Not Excluded:**
- Groq (free_tier: true, enabled: true)
- Gemini (free_tier: true, enabled: true)
- Cerebras (free_tier: true, enabled: true)
- OpenAI (free_tier: true, enabled: true)
- Nvidia (free_tier: true, enabled: true)
- Mistral (free_tier: true, enabled: true)
- OpenRouter (free_tier: true, enabled: true)
- G4F (free, no API key required)
- GitHub Models (enabled: false by default, can be enabled)
- Cloudflare (enabled: false by default, can be enabled)

### TEMPORARILY Unavailable Providers (Auto-Excluded):
**Auto-excluded (will be included again when available):**
- Rate-limited providers (429 responses)
- Unhealthy providers (5+ consecutive failures)
- Down providers (connection failures)
- Usage limit exceeded

**Reset logic:**
- Rate limits: reset at provider's window (usually 1 min)
- Health status: reset on successful API call/health check
- Usage limits: reset at midnight UTC (daily) or month start (monthly)

## How Availability Filtering Works

### current Implementation (score_engine.py):

`is_provider_available()` returns False (score=0, excluded) if:

```python
if not health.get("is_healthy", True):
    return False

if health.get("consecutive_failures", 0) > 5:
    return False

is_limited, _ = self.telemetry.check_rate_limit(provider, model)
if is_limited:
    return False
```

**This is CORRECT** - only temporary issues cause exclusion.

### No Paid Provider Filter Implementation

There is NO current code that filters out paid providers in the scoring engine.
All free_tier providers with `enabled: true` are included in fallback by default.

## Recommended Additions (for filtering paid providers):

If you want to explicitly exclude paid providers, add this to score_engine.py:

```python
def is_provider_free(self, provider: str) -> bool:
    """Check if provider offers free tier."""
    # Would need to check providers_database.yaml for free_tier flag
    return True  # All your current providers are free tier
```

And modify `is_provider_available()` to check:

```python
if not self.is_provider_free(provider):
    return False  # Exclude paid providers
```

## Current Provider Status (from config/router_config.yaml):

| Provider | Free Tier | Enabled | Status |
|----------|-----------|---------|--------|
| Groq | ✅ Yes | ✅ true | **Included** |
| Cerebras | ✅ Yes | ✅ true | **Included** |
| OpenAI | ✅ Yes | ✅ true | **Included** |
| Gemini | ✅ Yes | ✅ true | **Included** |
| Nvidia | ✅ Yes | ✅ true | **Included** |
| Mistral | ✅ Yes | ✅ true | **Included** |
| OpenRouter | ✅ Yes | ✅ true | **Included** |
| G4F | Free (no key) | ✅ true | **Included** |
| GitHub Models | ✅ Yes | ❌ false | **Disabled** (can enable) |
| Cloudflare | ✅ Yes | ❌ false | **Disabled** (can enable) |
| tavily | ❌ Paid | ❌ false | **Paid** |
| exa | ❌ Paid | ❌ false | **Paid** |

## Summary

- **Currently NO PAID providers** - all your configured providers are free tier
- Only temporary unavailability (rate limits, health issues) causes exclusion
- GitHub Models and Cloudflare are disabled by default (opt-in features), NOT paid
- All your 8+ free providers (Groq, Cerebras, OpenAI, Gemini, Nvidia, Mistral, OpenRouter) are included in fallback

The availability filter is working correctly - it only excludes temporarily unavailable providers, which is exactly what you want.
