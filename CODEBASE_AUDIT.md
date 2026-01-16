# Codebase Audit
**Date:** January 15, 2026

## 1. Architecture & Request Flow
The codebase is currently in a "dual-state" architecture:
- **Legacy Path (Active):** `src/proxy_app/main.py` directly uses `RotatingClient` from the `rotator_library` for the `/v1/chat/completions` endpoint. This path handles basic provider rotation and retry logic but lacks advanced routing.
- **Router Path (Dormant/Parallel):** A sophisticated routing system exists in `src/proxy_app/router_core.py`, `router_integration.py`, and `router_wrapper.py`.
  - It supports virtual models (`router/best-coding`), MoE (Mixture of Experts), search augmentation, and detailed capability matching.
  - **Crucially**, `main.py` does NOT import or initialize this router system, meaning these features are effectively unimplemented in the running application.

## 2. Existing Integrations
- **Core Providers:** Google (Gemini, Antigravity), OpenAI, Anthropic, Groq.
- **Specialized Providers:**
  - **G4F (GPT4Free):** Fully integrated as a fallback layer (Tier 5) in `rotator_library`.
  - **OAuth Providers:** Gemini CLI, Antigravity, Qwen Code, iFlow (supported via `rotator_library/credential_manager.py`).
- **Search Providers:** Brave, Tavily, Exa (configured in `router_config.yaml` but only accessible via the dormant Router Path).

## 3. Fallback Logic
- **Library Level (`RotatingClient`):**
  - Uses a "Provider Priority" system (Premium -> Fast/Affordable -> Standard -> Fallback).
  - Handles rate limit (429) errors by rotating keys within the same provider.
  - Can fallback to G4F if explicitly prioritized.
- **Router Level (Dormant):**
  - Implements `candidate` chains in `router_config.yaml`.
  - Explicitly defines fallback order (e.g., Groq -> Gemini -> G4F for coding tasks).
  - This logic is NOT currently active for standard API requests.

## 4. Configuration Formats
- **`.env`**: Secrets (API keys) and high-level feature toggles.
- **`config/router_config.yaml`**: Defines virtual models, routing behavior (timeouts, retries), and search provider settings.
- **`config/aliases.yaml`**: Defines short aliases (e.g., `coding` -> `router/best-coding`) and their fallback strategies.
- **`oauth_creds/*.json`**: Dynamic storage for OAuth tokens.

## 5. Virtual Models & Aliases
- **Status:** Defined but Inactive.
- **Implementation:** `router_config.yaml` defines complex virtual models like `router/best-coding` with MoE support.
- **Gap:** Since `main.py` bypasses `RouterCore`, requests to `router/best-coding` will likely fail or be handled generically unless the user manually points to a different endpoint (which doesn't exist yet).

## 6. Recommendations
1.  **Activate Router:** Modify `main.py` to use `RouterWrapper` / `RouterIntegration` instead of `RotatingClient` directly.
2.  **Unify Configuration:** Ensure `aliases.yaml` and `router_config.yaml` are loaded and respected by the main entry point.
3.  **Verify Fallbacks:** Test that falling back from a primary provider to G4F actually works end-to-end.
