# AI Agent Handoff Guide

**Last Updated**: 2026-02-05 20:30
**Session Status**: Phase 1 Implementation Review & Fixes

---

## 🚨 IMMEDIATE CONTEXT FOR NEXT AI

### Current Goal
Deploy a **stable, free-hosted LLM API Gateway** on Render.com that works reliably for:
1. AI Agentic Coding Tools (Opencode, Cursor, etc.)
2. AI Chatbots (SillyTavern, KoboldAI Lite, Agnaistic)

### Why G4F is Problematic
- **G4F scrapes** web-based LLM providers (ChatGPT, Claude, etc.).
- **Render's IPs are datacenter IPs** → instantly blocked by Cloudflare ("Just a moment..." HTML).
- **Result**: G4F returns HTML instead of JSON, crashing clients.

### The Solution: Phased Approach
1. **Phase 1 (CURRENT)**: Deploy with **API-Key Providers ONLY** (Groq, Cerebras, HuggingFace).
   - These **never get blocked** on Render.
2. **Phase 2 (LATER)**: Add G4F as a **low-priority fallback** (best-effort).

---

## 📋 IMMEDIATE NEXT STEPS

### Phase 1: Core Providers (API Keys)
| Task | Status | File(s) |
|------|--------|---------|
| ✅ Groq Provider | DONE | `groq_provider.py` (litellm handles it) |
| ✅ Cerebras Provider | DONE | `cerebras_provider.py` |
| ✅ HuggingFace Provider | DONE | `huggingface_provider.py` |
| ✅ Verify `/v1/models` | DONE | 19 providers registered, models verified |
| ⬜ Deploy to Render | TODO | Push to GitHub → Auto-deploy |
| ✅ .env.example Updated | DONE | Added HuggingFace section |
| ✅ Test Suite Fixed | DONE | 29/30 tests passing (1 pre-existing)  |
| ✅ README Updated | DONE | Added HuggingFace provider documentation |

### Cerebras Provider Implementation
```python
# File: src/rotator_library/providers/cerebras_provider.py
# URL: https://api.cerebras.ai/v1
# Model: llama3.1-70b (extremely fast)
# Auth: Bearer Token (CEREBRAS_API_KEY)
```

### HuggingFace Provider Implementation
```python
# File: src/rotator_library/providers/huggingface_provider.py
# URL: https://api-inference.huggingface.co/models
# Models: Qwen/Qwen2.5-72B-Instruct, meta-llama/Llama-3.3-70B-Instruct
# Auth: Bearer Token (HUGGINGFACE_API_KEY)
```

---

## 🔧 HOW TO CONTINUE

### 1. Clone the Repo
```bash
git clone https://github.com/ons96/LLM-API-Key-Proxy.git
cd LLM-API-Key-Proxy
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Tests
```bash
pytest tests/ -v
```

### 4. Implement Cerebras Provider
Create `src/rotator_library/providers/cerebras_provider.py`:
- Extend `ProviderInterface`
- Implement `get_models()` and handle chat completions via litellm
- Register in `src/rotator_library/providers/__init__.py`

### 5. Implement HuggingFace Provider
Create `src/rotator_library/providers/huggingface_provider.py`:
- Same pattern as Cerebras
- Note: HF uses different endpoint structure

### 6. Test Locally
```bash
python src/proxy_app/main.py --port 8000
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer YOUR_PROXY_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "cerebras/llama3.1-70b", "messages": [{"role": "user", "content": "Hi"}]}'
```

### 7. Commit and Push
```bash
git add .
git commit -m "feat: Add Cerebras and HuggingFace providers for Phase 1"
git push
```

---

## 📂 KEY FILES

| File | Purpose |
|------|---------|
| `src/rotator_library/client.py` | Core RotatingClient (request routing) |
| `src/rotator_library/providers/` | All provider implementations |
| `src/rotator_library/providers/g4f_provider.py` | G4F provider (Phase 2 - has WAF detection) |
| `src/proxy_app/main.py` | FastAPI entry point |
| `.env.example` | Environment variable template |
| `tests/test_g4f_resilience.py` | G4F WAF detection tests |

---

## 🧪 RECENT CHANGES (This Session)

1. **G4F WAF Detection** (`g4f_provider.py`)
   - Added `_is_waf_html()` to detect Cloudflare blocks
   - Raises `litellm.APIConnectionError` instead of leaking HTML
   - Tests: `tests/test_g4f_resilience.py` (3 tests, all pass)

2. **G4F Retry Logic** (`g4f_provider.py`)
   - Internal retry loop with exponential backoff (0.1s → 0.25s → 0.5s)
   - "Quota exhausted" treated as soft rate-limit, not hard lockout

3. **Implementation Plan** (Artifacts)
   - Phase 1: API-Key providers (Cerebras, HF, Groq)
   - Phase 2: G4F as experimental fallback

---

## 🔗 RELEVANT DOCS

- **Full Documentation**: `DOCUMENTATION.md`
- **Deployment Guide**: `Deployment guide.md`
- **Integration Roadmap**: `INTEGRATION_ROADMAP.md`
- **Project Status**: `PROJECT_STATUS.md`
- **LiteLLM Providers**: https://docs.litellm.ai/docs/providers

---

## 📞 USER PREFERENCES

- **Package Manager**: Use `uv pip` over `pip`
- **Git Workflow**: Commit frequently, push to `main` branch
- **Testing**: Run `pytest -q` before committing
- **Linting**: Run `ruff check .` before committing
- **Goal**: 100% free hosting on Render, no paid APIs

---

## ⚙️ ENVIRONMENT VARIABLES (.env)

```env
# Required
PROXY_API_KEY="your-secret-proxy-key"

# Groq (working)
GROQ_API_KEY_1="your-groq-key"

# Cerebras (Phase 1 - TODO)
CEREBRAS_API_KEY_1="your-cerebras-key"

# HuggingFace (Phase 1 - TODO)
HUGGINGFACE_API_KEY_1="your-hf-key"

# G4F (Phase 2 - experimental)
G4F_API_KEY="optional-g4f-key"
PROVIDER_PRIORITY_G4F=5  # Lowest priority
```

---

## 🏁 SUCCESS CRITERIA

Phase 1 is complete when:
1. ✅ Groq models work via gateway
2. ⬜ Cerebras models work via gateway
3. ⬜ HuggingFace models work via gateway
4. ⬜ `/v1/models` returns all available models
5. ⬜ Deployed to Render and accessible via public URL
6. ⬜ Tested with Opencode or SillyTavern

---

**Good luck, successor AI! You've got this. 🤖**
