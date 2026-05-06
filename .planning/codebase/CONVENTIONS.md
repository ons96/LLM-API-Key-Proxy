# Conventions

- **Style/format**: Black (88-char per KB); prefer type hints everywhere. Async-first, no blocking I/O in request paths.
- **Logging**: `logging.getLogger(__name__)`; colorlog/rich for console; strip ANSI when needed.
- **FastAPI patterns**: Dependency injection via `Depends`; parse bodies with `await request.json()`; raise `HTTPException` with explicit status/detail.
- **Error handling**: Keep structured errors; avoid silent excepts; map provider errors to HTTP codes.
- **Config**: YAML-driven (`config/router_config.yaml`, `virtual_models.yaml`, `providers_database.yaml`, rankings/aliases). Environment via `.env` (python-dotenv); no secrets in code.
- **Routing/Adapters**: OpenAI-compatible payloads; virtual model resolution + fallback chains; adapters translate to provider-native formats and convert back.
- **Testing mindset**: Use pytest markers (asyncio/slow/load/integration) and fixtures under `tests/fixtures/`; prefer deterministic mocks over live providers.
