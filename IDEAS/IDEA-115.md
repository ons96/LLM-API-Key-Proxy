# LLM Gateway Instant Fallback System

## AGENTS.md

---

## 1. Role/Mission

You are an autonomous software agent responsible for implementing an LLM Gateway system with instant fallback capabilities. Your mission is to create a robust gateway that:

- Routes LLM requests to available providers without waiting on rate limits
- Instantly detects failures (connection errors, API errors, timeouts) and immediately switches to a fallback provider
- Never blocks on waiting for rate limit resets - when rate limits are detected, switch providers immediately
- Only performs minimal delays (under 2 seconds) for very short expected waits when specifically configured
- Maintains high availability and resilience for LLM API calls

**Your core principle**: Fail fast, fallback faster. Never wait on a struggling provider when another is available.

---

## 2. Technical Stack

- **Language**: Python 3.10+
- **Async Framework**: `asyncio` with `aiohttp` for async HTTP requests
- **LLM Providers**: OpenAI, Anthropic, Google Gemini, Mistral, Groq (all have free tiers)
- **Configuration**: YAML-based configuration file (`config.yaml`)
- **Environment**: Single Python file implementation for simplicity, no external package dependencies beyond standard library + aiohttp
- **Testing**: Built-in pytest with mock providers
- **CI/CD**: GitHub Actions (free tier)

---

## 3. Requirements

1. **Async Request Handling**: All LLM API calls must use async/await for non-blocking operations

2. **Instant Failure Detection**:
   - Connection timeouts