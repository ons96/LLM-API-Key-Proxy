#HN|# Rotator Library Knowledge Base
#KM|
#VQ|**Generated:** 2026-02-24
#ZB|**Commit:** a55806d
#SY|
#ZH|## OVERVIEW
#SK|
Async Python library for API key rotation, OAuth management, and provider abstraction. Used by proxy_app for resilient LLM API calls with automatic failover.

#HV|## STRUCTURE
#XB|
#VK|```
#WV|rotator_library/
#RW|├── client.py              # RotatingClient - main entry point (2717 lines)
#SY|├── credential_tool.py     # OAuth credential management (2255 lines)
#HR|├── usage_manager.py       # Usage tracking & cooldowns (1792 lines)
#XW|├── model_info_service.py  # Model metadata (1352 lines)
#RK|├── error_handler.py       # Error classification (976 lines)
#NQ|├── providers/             # Provider adapters (26 files)
#TJ|│   ├── provider_interface.py   # Base class for all providers
#TX|│   ├── antigravity_provider.py # Gemini 3, Claude via Google internal (3611 lines)
#RX|│   ├── gemini_cli_provider.py  # Gemini OAuth flow (2344 lines)
#KZ|│   └── ...                    # Other provider implementations
#QW|└── utils/                 # Shared utilities
#KW|```
#VK|
#KS|## WHERE TO LOOK
#QN|
#PS|| Task | Location | Notes |
#RJ||------|----------|-------|
#SW|| Add new provider | `providers/<name>_provider.py` | Inherit ProviderInterface, auto-discovered |
#SP|| Key rotation logic | `client.py:acquire_key()` | Weighted random selection |
#MX|| OAuth flow | `credential_tool.py` | Gemini, Qwen, iFlow OAuth |
#QB|| Error classification | `error_handler.py:classify_error()` | Maps exceptions to types |
#SZ|| Cooldown tracking | `usage_manager.py` | Escalating per-model cooldowns |
#XQ||
#TW|## KEY CONCEPTS
#YS|
#ZM|### Provider Plugin System
#SJ|Auto-discovers `*_provider.py` files in `providers/`. Each provider:
#KJ|- Inherits `ProviderInterface`
#QY|- Implements `get_models(api_key, client) -> List[str]`
#MK|- Optionally: `get_model_options()`, `has_custom_logic()`, `get_auth_header()`
#TX|
#RR|### Dynamic OpenAI-Compatible Providers
#SR|Any provider with `<NAME>_API_BASE` env var gets auto-registered. No code needed.
#TP|
#MS|### Credential Rotation
#WJ|- **rotation_tolerance=0.0**: Deterministic (least-used always)
#SH|- **rotation_tolerance=2.0** (default): Weighted random (harder to fingerprint)
#KW|- **rotation_tolerance=5.0+**: High randomness
#QB|
#HQ|### Error Escalation
#YT|1. **Server error (5xx)**: Retry same key with backoff
#NN|2. **Rate limit**: Cooldown key for that model (10s→30s→60s→120s)
#NH|3. **Multi-model failure**: Global 5-min lockout
#MW|4. **Auth error**: Immediate 5-min global lockout
#BN|
#QS|## CONVENTIONS
#PZ|
#XP|- **Async-first**: All I/O uses `asyncio` + `httpx`
#YM|- **Type hints**: Required on all public methods
#VB|- **Logging**: `logging.getLogger('rotator_library')`
#TW|- **Context managers**: Use `async with RotatingClient() as client:`
#TM|
#PJ|## ANTI-PATTERNS
#JK|
#QH|❌ **DO NOT** use `requests` library - use `httpx.AsyncClient`
#HQ|❌ **DO NOT** hardcode provider names - use `PROVIDER_PLUGINS` dict
#YV|❌ **DO NOT** skip cooldown checks - call `usage_manager.is_available()`
#TM|❌ **DO NOT** block on async operations - always `await`
#YN|
#ZN|## LARGE FILES (Complexity Hotspots)
#XJ|
#MB|| File | Lines | Complexity |
#XN||------|-------|------------|
#RT|| antigravity_provider.py | 3611 | Gemini 3 OAuth, thought signatures |
#VM|| client.py | 2717 | Core rotation logic |
#TQ|| gemini_cli_provider.py | 2344 | Google OAuth flow |
#PB|| credential_tool.py | 2255 | Multi-provider OAuth |
#MB|| router_core.py (proxy_app) | 1885 | Fallback chains |
#HV||
#XZ|## TESTING
#VP|
#JW|```python
#VY|import asyncio
#HK|from rotator_library import RotatingClient
#HK|
#HS|async def test():
#PR|    async with RotatingClient(api_keys={"groq": ["key1"]}) as client:
#WW|        models = await client.get_available_models("groq")
#WT|        print(models)
#ZR||
#YN|asyncio.run(test())
#WV|```
