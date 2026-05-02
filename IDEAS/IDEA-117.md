# AGENTS.md  
**Autonomous Agent Specification: OpenAI Proxy Optimization & Render Deployment Refactor**  

---

## 1. Role/Mission

You are an autonomous software architect agent tasked with optimizing and refactoring a FastAPI-based OpenAI-compatible proxy for deployment on Render’s **free tier** with zero cost and maximum efficiency. Your mission includes:

- Modernizing the dependency management pipeline using `uv` for faster, safer, and lighter `pip install` operations.
- Rigorously auditing and minimizing dependencies to reduce attack surface and cold start time.
- Refactoring asynchronous code (`async`/`await`) for performance and correctness.
- Ensuring full OpenAI SDK compatibility across key endpoints:  
  - `POST /v1/chat/completions`  
  - `GET /v1/models`  
  - `POST /v1/embeddings`  
- Guaranteeing successful, zero-cost deployment via `render.yaml` on Render without exceeding free tier limits.

You must operate **autonomously** within cost and resource constraints. No paid or external services may be used. Any unresolved issues or ambiguities must be documented in `QUESTIONS.md`. Do not block workflow — use best-effort decision-making based on evidence and performance.

---

## 2. Technical Stack

| Component             | Technology / Tool                          | Purpose |
|-----------------------|--------------------------------------------|--------|
| Backend Framework      | FastAPI (ASGI)                             | Lightweight, async Python web framework |
| Server                 | Uvicorn (asynchronous worker)              | ASGI server for FastAPI |
| Dependency Manager     | `uv` (by Astral)                           | High-speed alternative to `pip` and `virtualenv` |
| Packaging              | `requirements.txt` (optimized)             | Minimal, deterministic dependency list |
| Deployment             | Render (Free Tier) via `render.yaml`       | Zero-cost cloud hosting with no persistent build cache |
| Containerization       | Docker (multi-stage, minimal image)        | Ensures consistency and small image size |
| Python Version         | 3.11+ (Render-compatible)                  | Stable and widely supported |
| Core Integration       | OpenAI Python SDK (for outbound calls)     | Proxy relay to official OpenAI API |
| Language               | Python 3 (async/await)                     | Enforced for non-blocking performance |

---

## 3. Requirements

1. **Use `uv` for all Python dependency operations**:
   - Replace `pip install -r requirements.txt` with `uv pip install -r requirements.txt` in Docker and CI.
   - Use `uv venv` to create virtual environments locally if needed (optional for CI).
   - Ensure `uv` is installed via `RUN curl -LsSf https://astral.sh/uv/install.sh \| sh` in Docker.

2. **Optimize `requirements.txt`**:
   - Remove all unused, redundant, or indirect dependencies.
   - Pin versions strictly (e.g., `fastapi==0.104.1`).
   - Prefer smaller, audited packages. Remove dev-only tools (e.g., `pytest`, `mypy`) from prod.

3. **Refactor async code for correctness and performance**:
   - Audit all async endpoints in `src/proxy_app/main.py`.
   - Ensure no blocking calls in async routes.
   - Use `async with`, `await` properly; avoid `asyncio.run()` inline.
   - Leverage `httpx` with async client for outbound OpenAI calls.

4. **Ensure OpenAI SDK compatibility**:
   - Validate that all proxy responses match OpenAI API schema exactly.
   - Support streaming responses for `/v1/chat/completions` via `text/event-stream`.
   - Mock or skip optional features not required for free usage.

5. **Deploy via `render.yaml` on Render’s free tier**:
   - Define Web Service in `render.yaml`.
   - Use Free instance type; no background workers.
   - Set health check to `/` or `/health`.
   - Environment variables: `OPENAI_API_KEY`, `LOG_LEVEL=INFO`.

6. **Docker image optimization**:
   - Use multi-stage builds: builder with `uv`, slim runtime (e.g., `python:3.11-slim`).
   - Minimize layers and remove cache/temp files.
   - Target image size **< 150MB**.

7. **Zero external costs**:
   - No Redis, Postgres, or persistent storage unless free and included in tier.
   - No third-party monitoring unless open-source and self-contained.
   - Rate limiting must be memory-based or disabled (no Redis).

8. **Code cleanup**:
   - Remove unused imports, dead code, debug logs.
   - Add type hints where missing.
   - Add structured logging (use `structlog` or `logging` with JSON format optional).

---

## 4. File Structure

```
.
├── src/
│   └── proxy_app/
│       ├── main.py              # FastAPI app, all endpoints
│       ├── config.py            # Settings (e.g., OPENAI_API_KEY)
│       ├── clients.py           # Async OpenAI client wrapper (httpx or openai>=1.0)
│       └── utils.py             # Shared helpers (rate limiting, logging, etc.)
├── requirements.txt             # Minimal, sorted, pinned
├── requirements-dev.txt         # Dev-only (optional, for local use)
├── Dockerfile                   # Multi-stage, uv-optimized
├── render.yaml                  # Web service config for free tier
├── .github/
│   └── workflows/
│       └── deploy.yml           # CI: build, test, push to Render via webhook (if needed)
├── QUESTIONS.md                 # UNTOUCHED by humans — agent logs unresolved issues here
└── AGENTS.md                    # This file
```

> **NOTE**: All changes **must** preserve `/v1/chat/completions`, `/v1/models`, and `/v1/embeddings` endpoint behavior to match OpenAI.

---

## 5. Testing Requirements

You **must** validate the following before marking completion:

1. **Local Smoke Test**:
   - Run `docker build -t openai-proxy . && docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... openai-proxy`
   - Hit each endpoint with `curl` or `httpx` and verify functional responses.

2. **Endpoint Validation**:
   - `/v1/models`: Returns `{"data": [...], "object": "list"}`
   - `/v1/chat/completions`: Accepts a valid chat request; streams if `stream=true`
   - `/v1/embeddings`: Accepts `input` and `model`, returns embeddings

3. **Performance**:
   - Cold start on Render must be < 10 seconds.
   - No timeouts reported on free instance.

4. **Security & Efficiency**:
   - `docker images` shows final image size ≤ 150MB.
   - No secrets hardcoded.
   - Dependencies scanned (via `pip-audit` or equivalent if possible in free tier).

5. **Render Deployment Test**:
   - Push to `main` triggers Render deploy via `render.yaml`.
   - Service becomes healthy within 2 minutes.
   - Public URL serves `/` and API endpoints.

---

## 6. Git Protocol

- Branch: work on `refactor/render-uv-optimize` (create if not exists).
- Commit Messages:
  - Use conventional commits: `feat:`, `fix:`, `perf:`, `refactor:`, `chore:`
  - Be specific: e.g., `refactor: async client using httpx in clients.py`
- Pull Requests:
  - Auto-create PR if not exists.
  - Assign to self (virtual assignment via label: `agent/todo`).
  - Merge only when **all tests pass** and `QUESTIONS.md` is empty or non-blocking.
- Conflict Resolution:
  - If conflicts arise, rebase interactively and resolve toward minimal, functional code.
  - Do **not** revert working features unless broken.

> ⚠️ **Never commit to `main` directly.**

---

## 7. Completion Criteria

This task is complete **only when all** of the following are true:

1. ✅ `uv` is used in `Dockerfile` for dependency installation.
2. ✅ `requirements.txt` has 30% fewer lines (minimum), all pinned, no dev deps.
3. ✅ `async` code in `main.py` and `clients.py` is refactored for safety and speed.
4. ✅ All OpenAI proxy endpoints (`/v1/chat/completions`, `/v1/models`, `/v1/embeddings`) return correct responses.
5. ✅ `Dockerfile` produces image < 150MB.
6. ✅ `render.yaml` deploys a free-tier Web Service that passes health check.
7. ✅ Service is publicly accessible on Render (https://your-project.onrender.com).
8. ✅ GitHub Actions CI builds and deploys without error.
9. ✅ `QUESTIONS.md` is up-to-date with any open issues (even if unresolved).
10. ✅ No cost incurred: instance is free, no add-ons enabled.

When complete, label PR as `agent/ready` and trigger final CI run.

---  
**END OF AGENTS.md**  
_Execute with precision. Operate autonomously. Save questions. Maximize efficiency. Zero cost.**