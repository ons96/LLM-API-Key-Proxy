# Integrations

## External APIs / Providers
- **Groq**, **Gemini**, **Together**, **Kilo**, **Supacoder**, **G4F** variants, and OpenAI-compatible providers (noobrouter, wiwi, aihubmix, opencode_zen, iflow) via `src/proxy_app/provider_adapter.py` and `rotator_library/providers/`.
- Virtual models resolve to provider/model chains defined in `config/virtual_models.yaml` (and backups/generated variants).
- Provider metadata and base URLs in `config/providers_database.yaml`; routing behavior in `config/router_config.yaml` and rankings in `config/model_rankings.yaml` / `config/chat_model_rankings.yaml`.
- API key verification for gateway requests: `verify_api_key` in `src/proxy_app` (see `main.py` usage on endpoints).
- Provider-specific keys pulled from environment (env vars) and credential tools under `rotator_library/credential_tool.py` / `credential_manager.py`.
- Free-only mode enforced by configuration (FREE_ONLY_MODE expected in env/config).
- `.env` (not committed) loaded via `python-dotenv`.
- Core configs: `config/router_config.yaml`, `config/virtual_models.yaml`, `config/providers_database.yaml`, `config/aliases.yaml`, `config/virtual_models_generated.yaml` (derived), `config/virtual_models_backup.yaml`.
- Settings management: `src/proxy_app/settings_tool.py`.
