# LLM-API-Key-Proxy Codebase Structure

This document provides a directory map and outlines the key entrypoints and important files of the `LLM-API-Key-Proxy` project.

## Directory Map

*   `config/`: Declarative YAML and JSON configuration files (routing rules, virtual models, provider settings).
*   `src/`: Main source code root.
    *   `proxy_app/`: The FastAPI gateway application and terminal UI.
    *   `rotator_library/`: Core business logic, key rotation, and provider API abstractions.
        *   `providers/`: Specific API provider implementations (adapters).
        *   `utils/`: Helper utilities (I/O, path management, headless detection).
*   `tests/`: Test fixtures and test cases.
*   `docs/`: Documentation for deployments, APIs, and tools.
*   `scripts/`: Automation and diagnostic scripts (deployments, benchmarking, scraping).
*   `deploy/`: Deployment configuration (systemd services, webhooks).

## Key Entrypoints

*   **FastAPI Application**: `src/proxy_app/main.py`
    *   Starts the web server, defines the `/v1/chat/completions`, `/v1/responses`, and `/v1/embeddings` routes.
*   **Terminal UI**: `src/proxy_app/launcher_tui.py`
    *   Interactive terminal application for managing configurations, keys, and monitoring the proxy.
*   **Test Runner**: Run via `pytest` from the root directory or `run_tests.bat`/`scripts/test_health_check.py`.

## Important Files

### Proxy Gateway (`src/proxy_app/`)
*   `main.py`: Application initialization, endpoint definitions, and FastAPI dependency injection.
*   `router_core.py`: Implementation of virtual models, fallback chains, and cross-provider routing logic.
*   `settings_tool.py`: Dynamic settings application and configuration management.
*   `provider_urls.py`: Management of base URLs for various providers.

### Rotator Library (`src/rotator_library/`)
*   `client.py`: The `RotatingClient` interface. Orchestrates API calls, key rotation, and invokes provider adapters.
*   `credential_tool.py` & `credential_manager.py`: OAuth flows, secret management, and API key handling.
*   `error_handler.py`: Error classification (rate limit vs. invalid request) to determine retry/fallback behavior.
*   `usage_manager.py` & `cooldown_manager.py`: Rate limit tracking and provider cooldowns to avoid bans.
*   `providers/provider_interface.py`: Base class contract that all provider plugins must implement.
*   `providers/openai_compatible_provider.py`: Standard implementation for providers utilizing OpenAI-compatible endpoints.

### Configurations (`config/`)
*   `virtual_models.yaml`: Defines "smart" model names (e.g., `coding-elite`, `chat-fast`) and their corresponding provider fallback sequences.
*   `router_config.yaml`: Global router settings, enabling/disabling specific providers and configuring timeouts.
*   `aliases.yaml`: Simple model name mapping (e.g., translating `gpt-4` to a specific provider's `gpt-4-turbo` string).
*   `providers_database.yaml`: Source of truth for provider metadata, endpoints, and capabilities.
