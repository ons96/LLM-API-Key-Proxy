# Stack

## Languages & Runtime
- Python 3.x (async-first); FastAPI ASGI app (`src/proxy_app/main.py`).
- Async HTTP stack: `httpx`, `aiohttp`, `aiofiles`, `curl_cffi` (for some providers).

## Frameworks & Core Libraries
- **FastAPI** for HTTP API and dependency injection.
- **Uvicorn** as ASGI server (dev/runtime entry).
- **LiteLLM** as common OpenAI-compatible client wrapper across providers.
- **Rotator Library** (editable install: `-e src/rotator_library`) for routing, provider adapters, credential/usage/error handling.
- **G4F** fallback provider; **curl_cffi** for certain provider flows.
- **python-dotenv** for environment loading.

## Application Structure
- `src/proxy_app/`: FastAPI app, router wrapper, settings, provider adapter factory.
- `src/rotator_library/`: Rotating client, provider adapters, usage/rate/error management.
- `config/*.yaml`: Router, virtual models, provider DB, rankings, aliases.

## Tooling
- Formatting/style: Black (88-char per KB), type hints expected; logging via `logging.getLogger(__name__)`, `colorlog`, `rich` for console output.
- Tests: `pytest` with async markers; fixtures in `tests/fixtures/`.

## Key Dependencies (requirements.txt)
- fastapi, uvicorn, python-dotenv
- litellm, httpx, aiohttp, aiofiles
- g4f, curl_cffi, filelock, colorlog, rich
- Local editable: `src/rotator_library`

## Notable Services/Providers
- Groq, Gemini, G4F variants, Together, Kilo, Supacoder, plus custom OpenAI-compatible providers (noobrouter, wiwi, aihubmix, opencode_zen, iflow) defined in configs and provider_adapter factory.
