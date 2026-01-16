# Known Bugs
**Date:** January 15, 2026

## 1. Syntax Errors in Provider Adapter
- **File:** `src/proxy_app/provider_adapter.py`
- **Issue:** Unterminated string literals and missing quotes in `_convert_response` and `_convert_chunk` methods.
- **Status:** Fixed locally during verification.

## 2. Dormant Router Logic
- **File:** `src/proxy_app/main.py`
- **Issue:** The main entry point bypasses the entire `RouterCore` / `RouterIntegration` system, rendering virtual models and advanced fallback logic inactive for the primary `/v1/chat/completions` endpoint.
- **Impact:** Users configuring `router_config.yaml` will see no effect. Virtual models like `router/best-coding` are unreachable via the standard API.

## 3. Missing Dependencies
- **Issue:** `httpx` and other requirements were not installed in the environment initially.
- **Status:** Resolved by creating a virtual environment and installing `requirements.txt`.
