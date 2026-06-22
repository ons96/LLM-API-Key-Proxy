# OSS Routing & Key-Mgmt Tooling Evaluation

Sibling task under EPIC #194 ("Self-improving zero-cost LLM router").
Informs: #195 (dynamic reorder), #196 (persistent penalty + decay), #197 (semantic `auto`), #207 (free embeddings chain).

## Constraints (non-negotiable)

- **Free only** — no paid SaaS, no proprietary models.
- **Local embeddings** — all-MiniLM-L6-v2 already in stack (used by `journal-embedder`, leaderboard).
- **Lightweight** — gateway-40 ~260MB RAM, VPS-155 1GB. No new heavyweight services (Redis is borderline; SQLite preferred).
- **Never remove providers/keys on failure** — deprioritize only (memory of stable free-provider fleet matters more than purity).
- **Already LiteLLM-based** — gateway is built on LiteLLM Router; we cannot drop it, only extend.

## Tool comparison

| Tool | Free | Lightweight | Solves our need? | Key gaps |
|---|---|---|---|---|
| LiteLLM native Router | yes | yes (already in stack) | Partial — cooldown + latency-rank + per-error policy | No composite quality+tps+reliability scoring; no decay; per-deployment not per-provider; no auto-discovery |
| semantic-router (aurelio-labs) | yes | yes (`[local]` extras, ~50MB) | Yes for embedding classification | No cooldown / fallback / multi-key / quality |
| RouteLLM (lm-sys) | yes | yes (small server) | No — only 2-model strong-vs-weak | Requires OpenAI key for embeddings; trained on GPT-4 vs Mixtral; not for 100+ free providers |
| OptiLLM | yes | medium (Flask + techniques) | No — inference-time compute, not provider routing | Adds per-request overhead; not a multi-provider fallback router |
| OpenCode plugins (rtk, title-fallback, dcp, caveman, fix-nim) | yes | yes | No — none manage LLM routing/keys/refresh | Already evaluated 2026-06-18; rate-limit-fallback REMOVED (caused bad model drops) |

### LiteLLM Router — what we get for free

- **Strategies**: `simple-shuffle` (default, recommended), `least-busy`, `latency-based-routing` (ttl + `lowest_latency_buffer`), `cost-based-routing`, `usage-based-routing-v2` (Redis), custom subclass `CustomRoutingStrategyBase`.
- **Cooldowns**: per-deployment, `allowed_fails` (default 3) + `cooldown_time` (default 5s), auto on 429 / >50% failure / 401 / 404 / 408.
- **Policies**: `RetryPolicy` + `AllowedFailsPolicy` — per-error-type customization (`AuthErrorRetries=0`, `RateLimitErrorRetries=3`, `RateLimitErrorAllowedFails=100`, etc.).
- **Order-based priority** with auto-failover across `order` tiers; `enable_weighted_failover` for in-group retry.
- **`enable_health_check_routing`** — proactive background health probes (reactive cooldown is the default).
- **`routing_groups`** — per-model-group strategy (e.g. `gpt-4o` → latency, `cheap` → simple-shuffle).
- **Traffic mirroring** for A/B.
- **Redis** for cross-instance rate-limit tracking + response cache.

### LiteLLM Router — what it does NOT give us

- No composite **quality + TPS + reliability** scoring across heterogeneous free models.
- No **time-decay** half-life penalty (cooldown is binary: in or out).
- No **auto-discovery** of new free models from `models.dev` / provider manifests.
- No **multi-key rotation** across many providers (1 LiteLLM deployment = 1 key).
- No **persistent penalty history** across restarts (in-memory unless Redis; Redis is one more service to babysit on VPS-155).

### semantic-router (aurelio-labs) — perfect fit for #197

- `pip install -qU "semantic-router[local]"` — pulls `fastembed` for local ONNX embeddings, **no API key, no cost, no network**.
- `Route` + `RouteLayer` (now `SemanticRouter`) with cosine similarity over utterance lists.
- `rl("query").name` → matched route in <1ms after model load (~80MB).
- Encoders: FastEmbed (local), HuggingFace, OpenAI, Cohere, LlamaCpp.
- Threshold training via `06-threshold-optimization.ipynb`.
- **Drops cleanly behind our existing `intent_detector.py`** — keyword regex stays as fallback for low-confidence embeddings.

### RouteLLM — doesn't fit

- 2-model strong-vs-weak router. We have 100+ heterogeneous providers; binary split loses 95% of the signal.
- `mf` and `sw_ranking` routers require OpenAI key for embeddings. Against free-only constraint.
- Trained on GPT-4 vs Mixtral preference data — calibration for "free tier on VPS" is unsupported.
- Verdict: skip. Re-evaluate only if we ever ship a 2-tier "free premium" model.

### OptiLLM — orthogonal, but useful as a downstream proxy

- 20+ inference-time techniques (CoT-reflection, MARS, MoA, MCTS, CePO, Best-of-N, LEAP, R*, self_consistency).
- Slug-prefixed: `moa-gpt-4o-mini`, `mars-gemini-flash`. Auto-router plugin (`optillm-modernbert-large`) picks technique per prompt.
- **NOT a multi-provider fallback router** — it's an inference-time optimizer on a single chosen model.
- Possible future use: chain OptiLLM behind our gateway for `code-quality` intent → wrap free model output with CoT-reflection / Best-of-N. Defer to follow-up issue if needed.

### OpenCode plugins — none relevant

- `rtk`, `title-fallback`, `dcp`, `caveman`, `fix-nim`, `opencode-auth`, `nim-fix` — UX/devtools, not LLM routing.
- `@azumag/opencode-rate-limit-fallback` — REMOVED 2026-06-18 (cycled through 30 fallbacks on any error; dropped GLM 5.2 xhigh → llama-3.3-70b). Confirmed removal correct for our use case.
- `@openauthjs/opencode-auth` — auth helper, not routing. Skip.

## Build-vs-reuse decisions (per sibling task)

| Task | Decision | Rationale |
|---|---|---|
| **#195** Dynamic reorder | **BUILD** (telemetry feed loop + composite scorer) | LiteLLM `latency-based-routing` is too coarse; needs quality×TPS×reliability×penalty composite. LiteLLM cooldown = binary threshold, not scored. Our `DynamicScoringEngine` + telemetry SQLite already has the inputs; just need the reorder trigger. |
| **#196** Persistent penalty + decay | **BUILD thin layer + EXTEND LiteLLM via `AllowedFailsPolicy`** | LiteLLM cooldown = in-memory (Redis-backed if we add Redis = +1 service). We already have telemetry SQLite (`/dev/shm/telemetry.db`). Keep `cooldown_manager.py` (1.4KB in-memory dict), persist `{provider, ts, half_life, score}` rows, score = `tps_failure_rate × exp(-age / half_life)`. Wire `AllowedFailsPolicy` thresholds from our decay table so LiteLLM's runtime cooldowns stay in sync with our penalty score. |
| **#197** Semantic `auto` routing | **REUSE semantic-router** | `semantic-router[local]` + FastEmbed ONNX = free, local, ~80MB, <1ms route lookup. Perfect fit for our 5-intent taxonomy (CODING / COMPLEX / FAST / UNSENSORED / AGENTIC). Wire `RouteLayer` ahead of `intent_detector.py`; keyword detector stays as fallback for low-confidence embeddings. No new paid dep. |
| **#207** Free embeddings chain | **BUILD** (MTEB × free-tier × telemetry join) | No OSS tool covers: (1) MTEB retrieval benchmark scores, (2) free-tier embedding provider uptime, (3) join to our telemetry SQLite. semantic-router is the **consumer** of an embedding chain, not the source. Build a `embeddings_chain.yaml` analogous to `virtual_models.yaml` (providers → MTEB score → fallback chain). |
| Multi-key verify/rotation CLI | **BUILD** | No OSS manages keys across 19+ free providers. Our `credential_manager.py` + `provider_target_models` + 19 secret files already exist. Need ~100 LOC CLI wrapping `credential_manager.list_unhealthy()` → ping each → write back to `provider_status.db`. |
| Check-in tracking | **BUILD** | Not covered by any reviewed OSS. Free daily-token providers (hapuppy, choosechat, etc.) need a check-in daemon. Standalone `checkin_scheduler.py` reading `~/.secrets/` + provider manifest. |
| Cost-effectiveness ranking | **REUSE llm-leaderboard-aggregate** | Already in stack (not external OSS). Phase 1 YAML configs shipped PR #47 (commit `65b7b6d`). Don't re-build. |

## Net work plan (post this doc)

1. **#197** (REUSE semantic-router): add `semantic-router[local]` to gateway requirements.txt, write `src/proxy_app/semantic_router.py` wrapping `RouteLayer` for 5 intents, fall back to `intent_detector.py` on low confidence. ~200 LOC + tests.
2. **#196** (BUILD + EXTEND): add `decay_penalty` table to telemetry SQLite; replace in-memory `cooldown_manager` dict with SQLite-backed read-through cache; emit LiteLLM `AllowedFailsPolicy` thresholds at router init. ~150 LOC + tests.
3. **#195** (BUILD): write `src/optimization/reorder_trigger.py` — reads telemetry SQLite every N minutes, recomputes composite score for each provider, regenerates `config/virtual_models.yaml` chain order, hot-reloads via LiteLLM `Router.update_settings()`. ~250 LOC + tests.
4. **#207** (BUILD): write `src/embeddings/chain_builder.py` + `config/embeddings_chain.yaml` analogous to `virtual_models.yaml`. ~200 LOC + tests.

Sequence: #197 first (smallest, biggest UX win), then #196 (foundation for #195), then #195, then #207.

## Verification

- [x] Comparison doc committed: this file.
- [ ] Each sibling (#195, #196, #197, #207) annotated with reuse/build decision → handled by comments on the task-board issues.
- [ ] All picks free + lightweight for gateway-40 (260MB) / VPS-155 (1GB) → confirmed: semantic-router (~80MB), LiteLLM already loaded, others are our own code.