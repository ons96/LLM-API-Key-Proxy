# Model Council: Multi-Model + Judge Routing (Design)

Status: **DESIGN ONLY — not implemented.** Future enhancement behind a flag.
Refs: task-board #204, #194 (epic). Builds on #197 (semantic `auto`) + #195 (telemetry reorder) + #196 (penalty store).

## When to invoke the council

Two trigger modes, both off by default. `COUNCIL_MODE=off` (default) → zero overhead, requests go through the normal `auto` router.

| Mode | When it fires | Use case |
|---|---|---|
| `flag` | Client sets `metadata.council=true` or `model="council"` | Explicit opt-in for high-stakes prompts |
| `heuristic` | Council fires when `auto` resolves to `coding-elite` or `chat-smart` AND prompt length > 2000 chars AND `COUNCIL_MODE=heuristic` | Long-form reasoning where a second opinion is cheap relative to value |

`flag` mode is the v1. `heuristic` is v2 at the earliest — premature without telemetry showing which prompts benefit.

## Fan-out

Send the prompt to N candidate models concurrently. N is tunable, default 3.

Candidate pool: top-N entries from the `auto`-resolved chain (post-penalty, post-capability-filter). Never fan out to more than 5 — latency + free-tier rate limits.

```python
async def council_route(request, chain, n=3):
    candidates = chain[:n]
    tasks = [router.acompletion(model=c["chain"], messages=request["messages"]) for c in candidates]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    return [(c, r) for c, r in zip(candidates, responses) if not isinstance(r, Exception)]
```

Failed candidates drop silently — council proceeds with whatever returned. If zero return, fall back to the original `auto` single-route (graceful degradation, never a hard failure).

## Judge selection

Judge model = a separate virtual model `council-judge`, NOT one of the candidates. It receives:

```
system: You are a code review judge. Pick the best response. Output only the letter (A, B, C) and a one-sentence rationale.
user:
<prompt>
---
Response A (from {model_a}):
{response_a}
---
Response B (from {model_b}):
{response_b}
---
Response C (from {model_c}):
{response_c}
```

Judge candidate: `chat-smart` chain (high-reasoning, free tier). Specific defaults:

- **coding prompts**: `chat-smart` (DeepSeek-R1-class reasoning model)
- **chat prompts**: `chat-elite` (GPT-5 / Claude-class, if free tier available) or `chat-smart` as fallback
- **fast prompts**: council not invoked (heuristic mode skips, flag mode warns caller)

Judge output is parsed via simple regex `^\s*([A-C])\b`. If parse fails, return the **first** candidate (deterministic fallback — never random).

## Cost guard

Total token budget per council call: `COUNCIL_MAX_TOKENS=8000` (4K candidates × ~1K each + 1K judge prompt + 1K judge response). If estimated prompt + expected responses exceed budget, downgrade to `auto` single-route with a log line.

Free-tier constraint: both candidates and judge MUST come from `free_unlimited` or `free_daily` providers (per `llm_providers.db` `providers` table). Reject `free_one_time` providers — they exhaust credits faster than a single-route call.

Rate-limit guard: if any candidate chain has a provider currently in `penalty_store` with `invalid_key` or `rate_limit` failure type in the last 60s, skip that candidate.

## Configuration

| Env var | Default | Description |
|---|---|---|
| `COUNCIL_MODE` | `off` | `off` \| `flag` \| `heuristic` |
| `COUNCIL_N` | `3` | Number of candidate models (2-5) |
| `COUNCIL_JUDGE_CHAIN` | `chat-smart` | Virtual model for judge |
| `COUNCIL_MAX_TOKENS` | `8000` | Total budget per council call |
| `COUNCIL_MIN_PROMPT_LEN` | `2000` | Min prompt chars for heuristic mode |
| `COUNCIL_ENABLED_CHAINS` | `coding-elite,chat-smart` | Chains that trigger heuristic mode |

## Wiring point

Same as `auto` — `router_wrapper.py:handle_chat_completions`. If `model_id == "council"` or (`COUNCIL_MODE=heuristic` and prompt matches heuristic), call `council_route()` instead of `resolve_auto()`. Mutate `request_data["model"]` to the judge's chosen chain, then proceed normally so telemetry + penalty store record the winning provider.

If council is off, the wiring point is never reached — zero overhead.

## Latency model

Council latency = max(candidate latencies) + judge latency. With 3 candidates from `coding-elite` (avg TTFT ~2s, TPS ~30 tok/s, 1K-token response) + judge (`chat-smart`, 2K-token input, 200-token response):
- Parallel candidates: ~35s (1K tokens / 30 TPS)
- Judge: ~10s (200 tokens / 20 TPS)
- **Total: ~45s** vs ~35s for single-route

Acceptable for high-stakes (coding design, refactoring decisions). Not acceptable for chat — heuristic mode restricts to long-form only.

## Future: self-improvement loop

Council outcomes feed back into telemetry: winning provider gets a `+1` success event, losers get a `+0.5` (didn't win but didn't fail). Over time, `reorder_chains.py` (#195) promotes consistent winners.

This is the "feedback loop" mentioned in #204. Deferred — requires council outcome logging in `telemetry.db` schema (`council_winner`, `council_candidates`, `council_judge_decision` columns).

## Implementation estimate

- `src/proxy_app/council.py` (~150 LOC): `council_route()`, `judge_response()`, cost + rate-limit guards.
- `tests/test_council.py` (~200 LOC): mock `router.acompletion`, verify fan-out, judge parsing, fallback on parse failure, fallback on all-candidates-fail, cost guard downgrade, off-by-default zero overhead.
- `config/virtual_models.yaml`: add `council-judge:` stub chain.
- `router_wrapper.py`: +10 LOC for council dispatch.
- Total: ~360 LOC, ~10 tests.

**Defer implementation until:** (1) #197 merged + `auto` proven in production for ≥2 weeks, (2) user signal that high-stakes prompts need second opinions, (3) `COUNCIL_MODE=flag` prototype validated on VPS-155 before enabling on gateway-40.

## Acceptance criteria (#204)

- [x] Design doc: when to invoke council (flag / heuristic), N-model fan-out, judge selection, cost guard — THIS DOC.
- [ ] Prototype behind a flag; off by default — DEFERRED (see implementation estimate above).
- [x] Free models only for both candidates and judge — documented (reject `free_one_time`, prefer `free_unlimited` / `free_daily`).
- [ ] Suggested Verification: behind flag, council fan-out + judge returns single best answer on test prompt; off by default has zero overhead — DEFERRED with prototype.

## Why this is P6

Council adds 3-5× token consumption per call. On a free-only gateway with rate limits, this is expensive. The single-route `auto` (PR #272) + telemetry-driven reorder (PR #274) + penalty store (PR #273) already cover 95% of cases. Council only earns its cost on rare high-stakes prompts where one wrong answer costs more than 5 free API calls.

Defer until the single-route system has telemetry showing which prompt classes actually fail. Then target council at exactly those classes.
