# AGENTS.md - LLM API Key Proxy with Multi-Provider Routing

## 1. Role/Mission

You are an autonomous software engineering agent responsible for implementing and maintaining a sophisticated LLM API proxy system. Your mission is to build a production-ready OpenAI-compatible API proxy that intelligently routes LLM requests across multiple providers with the following core capabilities:

1. **Provider Routing & Fallback**: Route requests to optimal providers (Groq, g4f, Google Gemini) based on capability requirements, with intelligent fallback when providers fail or hit rate limits.

2. **Virtual Router Models**: Expose synthetic "router models" that map to pools of real models, enabling intelligent selection without exposing backend architecture.

3. **Free/Paid Gating**: Enforce `FREE_ONLY_MODE` to prevent accidental billing by hard-gating paid providers when free-tier alternatives exist.

4. **Web Search Augmentation**: Implement search-powered augmentation with deterministic "when to search" rules and provider fallbacks (Brave Search, Tavily).

5. **MoE Committee Mode**: Implement mixture-of-experts voting for specific router models with bounded token usage and failure tolerance.

6. **Observability**: Track EWMA latency, cooldowns, consecutive failures, and provide structured logging without leaking sensitive data.

## 2. Technical Stack

- **Framework**: FastAPI (Python 3.10+)
- **Server**: Uvicorn with async/await patterns
- **Configuration**: JSON/YAML config files with environment variable interpolation
- **HTTP Client**: httpx for async provider requests
- **Serialization**: Pydantic for request/response models
- **Web Search APIs**: Brave Search API, Tavily API
- **LLM Providers**:
  - Groq (free tier available)
  - g4f (free, open-source proxy)
  - Google Gemini Developer API (free tier available)
- **Streaming**: Server-Sent Events (SSE) with proper `data: {...}\n\n` format
- **Testing**: pytest, httpx TestClient, shell scripts

## 3. Requirements (Numbered)

### 3.1 OpenAI Compatibility (Must Preserve)

1. Implement `/v1/models` endpoint returning OpenAI-style list JSON with both real and virtual models.
2. Implement `/v1/chat/completions` accepting standard OpenAI fields:
   - `messages` (required)
   - `temperature` (optional, default 0.7)
   - `max_tokens` (optional)
   - `tools` / `tool_choice` (where supported)
   - `response_format` (where supported)
3. Implement streaming via SSE with format `data: {...}\n\n` followed by final `data: [DONE]`.
4. Do not switch models mid-stream; maintain model consistency throughout response.

### 3.2 Virtual Router Models

5. Expose these synthetic model IDs in `/v1/models`:
   - `router/best-coding`
   - `router/best-reasoning`
   - `router/best-research`
   - `router/best-chat`
   - `router/best-coding-moe` (explicit MoE/committee mode)
6. Each virtual model maps to an ordered candidate pool of `(provider, model)` tuples defined in configuration.
7. Virtual models must appear in both `/v1/models` and routing logic with proper metadata.

### 3.3 Provider Adapter Architecture

8. Create clean adapter interface with methods:
   - `list_models()` - returns available models with metadata
   - `chat_completions(request, stream)` - handles completion request
9. Each adapter must expose capability metadata:
   - `supports_tools` (function calling)
   - `supports_vision`
   - `supports_structured_output`
   - `max_context_tokens`
   - `free_tier_available`
10. Implement adapters for:
    - **Groq** (free tier: 14-28 GB/day throughput)
    - **g4f** (free, open-source proxy)
    - **Google Gemini Developer API** (free tier: 15 RPM, 1M TPM)

### 3.4 Router + Fallback + Cooldowns (Core Logic)

11. Create central router that:
    - Determines `CapabilityRequirements` from request (needs_tools, needs_vision, needs_structured_output, min_context)
    - Resolves requested model into candidate list
    - Filters candidates by capability + FREE_ONLY_MODE + cooldown status
    - Attempts candidates in order until success or exhaustion
12. Track per `(provider, model)`:
    - `cooldown_until` (timestamp)
    - `consecutive_failures` (count)
    - `last_success_ts` (timestamp)
    - `last_error_ts` (timestamp)
    - `ewma_latency_ms` (Exponential Weighted Moving Average)
13. On HTTP 429 (rate limit):
    - Use `retry-after` header if present
    - Apply safe default cooldown (30 seconds minimum)
    - Immediately try next candidate
14. On HTTP 401/403: Switch candidates immediately (authentication failures are provider-side)
15. On transient network errors: Retry same candidate once, then switch
16. Ordering rules:
    - Prefer "best" models for task category (coding, reasoning, research, chat)
    - Break ties by reliability (lower consecutive failures) + EWMA latency
    - Keep selection deterministic/stable unless failure/cooldown forces switch

### 3.5 Web Search Augmentation

17. Implement `SearchProvider` plugin system with interface:
    - `search(query, num_results)` - returns search results
    - `get_capabilities()` - returns free/paid tier info
18. Implement deterministic "when to search" rules:
    - Default: 0 searches
    - Do 1 search when prompt indicates shopping, "latest", "best-value", or explicit sources/citations
    - Allow up to N additional searches only if results are insufficient/conflicting
    - If no live search available: respond clearly that live search is unavailable; do not fabricate citations
19. Implement providers (with FREE_ONLY_MODE gating):
    - **Brave Search API** - has official free tier (2,000 searches/month)
    - **Tavily** - has free tier but also paid usage → hard-gate under FREE_ONLY_MODE
    - Optional: Exa (limited free credits, keep disabled by default)
20. Use Gemini "grounding/search" as another search backend with quota-aware fallbacks if implemented.

### 3.6 MoE / Committee Mode

21. Only enable MoE for `router/best-coding-moe` virtual model.
22. Run 2-3 experts max in parallel from config pool.
23. Execute 1 aggregator pass after expert responses.
24. Hard cap tokens and expert count (configurable).
25. If some experts fail, proceed with remaining and note reduced committee size in response metadata.

### 3.7 Configuration + Safety

26. Create config file (JSON or YAML) or env-driven configuration defining:
    - Valid proxy API keys (support rotation/multiple keys per provider)
    - Enabled providers and their environment variable names
    - Virtual model pools with priority ordering
    - Timeouts, retries, cooldown durations
    - Search providers with priority ordering
    - FREE_ONLY_MODE default (default: `true`)
27. Implement safe defaults for all timeouts (provider: 120s, search: 30s).
28. Logging requirements:
    - Include: request ID, chosen candidate, fallback attempts, timings, error types
    - Never log: upstream API keys or full prompts
    - Hash prompts if diagnostic logging is needed
29. Implement proper error handling with OpenAI-compatible error responses.

### 3.8 Tests + Smoke Scripts + Documentation

30. Create `scripts/smoke_test.sh` for local testing:
    - Test `/v1/models` endpoint
    - Test non-stream completion
    - Test streaming completion
    - Test virtual model call
    - Test forced fallback (disable primary provider)
31. Create unit tests for:
    - Routing decisions with capability filtering
    - FREE_ONLY_MODE gating (assert no paid providers selected)
    - Fallback chain exhaustion
    - Cooldown application on 429 errors
32. Provide README updates covering:
    - Configuration file format
    - Proxy key rotation procedure
    - Virtual model usage guide
    - Fallback/cooldown mechanism explanation
    - Render redeploy workflow (if applicable)

## 4. File Structure

```
llm-api-proxy/
├── AGENTS.md                 # This specification
├── QUESTIONS.md              # Questions for human clarification
├── README.md                 # Documentation
├── config/
│   ├── default.json          # Default configuration
│   └── example.json         # Example configuration template
├── src/
│   ├── __init__.py
│   ├── main.py               # FastAPI application entry point
│   ├── config.py             # Configuration loading
│   ├── models.py             # Pydantic request/response models
│   ├── router/
│   │   ├── __init__.py
│   │   ├── core.py           # Central router logic
│   │   ├── candidate.py     # Candidate selection & filtering
│   │   └── metrics.py       # EWMA, cooldowns, health tracking
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py          # Abstract provider adapter
│   │   ├── groq.py          # Groq adapter
│   │   ├── g4f.py           # g4f adapter
│   │   ├── gemini.py        # Google Gemini adapter
│   │   └── registry.py      # Provider registry
│   ├── search/
│   │   ├── __init__.py
│   │   ├── base.py          # Abstract search provider
│   │   ├── brave.py         # Brave Search adapter
│   │   ├── tavily.py        # Tavily adapter
│   │   └── registry.py      # Search provider registry
│   ├── moe/
│   │   ├── __init__.py
│   │   └── committee.py     # Mixture-of-experts logic
│   └── utils/
│       ├── __init__.py
│       ├── logging.py      # Structured logging
│       └── security.py     # Key hashing, sanitization
├── tests/
│   ├── __init__.py
│   ├── test_router.py      # Router logic tests
│   ├── test_providers.py   # Provider adapter tests
│   ├── test_free_mode.py   # FREE_ONLY_MODE gating tests
│   └── test_search.py      # Search integration tests
├── scripts/
│   └── smoke_test.sh       # Smoke test script
└── requirements.txt        # Python dependencies
```

## 5. Testing Requirements

### 5.1 Smoke Tests (`scripts/smoke_test.sh`)

The smoke test script must verify:
1. `/v1/models` returns valid JSON with at least 5 virtual models + real models
2. Non-stream completion works with `router/best-coding`
3. Streaming completion works with proper SSE format
4. Virtual model routing produces valid response schema
5. Forced fallback works (temporarily disable primary to verify secondary is used)

### 5.2 Unit Tests

Required test coverage:

| Test Category | What to Verify |
|--------------|----------------|
| `test_router.py` | Capability filtering excludes incompatible models; fallback selects next best; cooldown triggers on 429 |
| `test_providers.py` | Each adapter's `list_models()` returns valid metadata; `chat_completions()` handles stream/non-stream |
| `test_free_mode.py` | With `FREE_ONLY_MODE=true`, only free providers are selected; paid providers are filtered out |
| `test_search.py` | Search triggers based on prompt keywords; Brave is used when Tavily is gated; fallback on search failure |

### 5.3 Test Execution

```bash
# Run smoke tests
./scripts/smoke_test.sh

# Run unit tests
pytest tests/ -v

# Run specific test
pytest tests/test_free_mode.py -v
```

## 6. Git Protocol

### 6.1 Branch Strategy

- **Main branch**: `main` - production-ready code
- **Development branch**: `develop` - integration branch
- **Feature branches**: `feature