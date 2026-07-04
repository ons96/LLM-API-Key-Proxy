# SPEC.md -- LLM-API-Key-Proxy

Canonical specification for the LLM-API-Key-Proxy gateway. Status tags: `[built]` = live in production, `[partial]` = implemented with gaps, `[planned]` = not yet implemented.

Truth source: live code in `src/proxy_app/` + `src/rotator_library/` + `config/` on VPS 40.233.101.233. DESIGN.md covers the dynamic-fallback subsystem in depth; this SPEC consolidates all subsystems.

**Repo:** `ons96/LLM-API-Key-Proxy` (private). **VPS:** 40.233.101.233 (Tailscale 100.71.95.75:8000, SSH-only firewall). **Task-board:** `ons96/task-board` (issues = tracker).

---

## Quickstart

### Local bring-up

```
git clone https://github.com/ons96/LLM-API-Key-Proxy.git
cd LLM-API-Key-Proxy
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill provider API keys
python -m proxy_app.main
```

- **Entry point:** `src/proxy_app/main.py` (FastAPI app, `app = FastAPI()`).
- **Config dir:** `config/` (11 YAML files -- see Virtual Models / Providers sections).
- **Env vars:** loaded via python-dotenv from `.env` at repo root. Keys: `GROQ_API_KEY`, `GEMINI_API_KEY`, `NVIDIA_API_KEY`, `OPENAI_API_KEY`, `TOGETHER_API_KEY`, `MISTRAL_API_KEY`, `OPENROUTER_API_KEY`, `CEREBRAS_API_KEY`, `SAMBANOVA_API_KEY`, `VPS_GATEWAY_API_KEY` (gateway auth), `TELEMETRY_DB_PATH` (override default `/tmp/llm_proxy_telemetry.db`).
- **Default port:** 8000.

### VPS deployment

- **systemd unit:** `llm-gateway.service` (managed via systemctl, NOT PM2 -- #258 [built]).
- **Deploy:** `cd ~/LLM-API-Key-Proxy && git pull origin main && sudo systemctl restart llm-gateway.service`.
- **Logs:** `journalctl -u llm-gateway.service -f`.
- **Health check:** `curl http://localhost:8000/v1/models` (returns 401 without auth = healthy; 200 with `Authorization: Bearer $VPS_GATEWAY_API_KEY`).
- **Tunnel (laptop access):** `vps-gateway-tunnel.service` (systemd --user) forwards `localhost:8000` -> VPS Tailscale `100.71.95.75:8000`.

### OpenCode provider config

```
"vps-gateway": {
  "baseURL": "http://localhost:8000/v1",
  "apiKey": "{env:VPS_GATEWAY_API_KEY}",
  "models": { "coding-elite": {...}, "coding-fast": {...}, ... }
}
```

---

## Routing

The core request path. Entry: `POST /v1/chat/completions` at `src/proxy_app/main.py:1085`.

**Call chain:** `main.py:chat_completions()` -> `RouterWrapper.handle_chat_completions()` (`src/proxy_app/router_wrapper.py`) -> `RouterIntegration.chat_completions()` (`src/proxy_app/router_integration.py`) -> `RouterCore.route_request()` (`src/proxy_app/router_core.py:1942`).

| Feature | Status | File | Notes |
|---|---|---|---|
| `/v1/chat/completions` handler | `[built]` | `main.py:1085` | Delegates to RouterWrapper, no short-circuit (#340 verified) |
| RouterWrapper (auto/semantic dispatch) | `[built]` | `router_wrapper.py` | `auto`/`router/auto` -> `semantic_router.resolve_auto()`, all branches -> `router_integration` |
| Semantic routing (intent -> VM) | `[built]` | `semantic_router.py` | `resolve_auto()` mutates `request_data["model"]`, tool-aware fallback to `DEFAULT_TOOL_CHAIN` (#197) |
| RouterIntegration (adapter init) | `[built]` | `router_integration.py` | Builds `RouterCore`, `ProviderAdapterFactory`, skips `_SEARCH_ONLY={brave_search,tavily,duckduckgo,exa,jina}` + `router/` prefix + virtual groups |
| RouterCore.route_request | `[built]` | `router_core.py:1942` | -> `_route_with_retry` (if `GATEWAY_RETRY_UNTIL_DONE`) -> `_route_request_inner:1977` |
| Candidate resolution | `[built]` | `router_core.py:1313` `_get_candidates` | Virtual model -> multi-candidate chain from `virtual_models.yaml`; filters health/cooldown + `FREE_ONLY_MODE`; sorts by success_rate then latency |
| Non-stream fallback loop | `[built]` | `router_core.py:2107-2193` | Loops ALL candidates; error -> `_classify_error` -> `_apply_error_cooldown` -> `continue` |
| Stream fallback loop | `[built]` | `router_core.py:1885` `_stream_with_fallback` | Loops all candidates, yields chunks; exception -> classify -> cooldown -> continue (#340 fixed missing cooldown) |
| Error classification | `[built]` | `router_core.py` `_classify_error` | Returns `(error_category, retry_after)`; categories: AUTH/INVALID/TRANSIENT/PROVIDER |
| Error cooldown application | `[built]` | `router_core.py` `_apply_error_cooldown` | Marks provider+model unhealthy for retry window |
| Retry-until-done wrapper | `[built]` | `router_core.py` `_route_with_retry` | Env `GATEWAY_RETRY_UNTIL_DONE=1` enables; `GATEWAY_FORCE_TRY_COOLDOWN=1` forces cooldown bypass |
| cache_control strip (non-Anthropic) | `[built]` | `router_core.py` `_strip_cache_control` | Recursive strip in `_execute_single_candidate` + stream variant (#346) |
| X-Phase-Router headers | `[built]` | `router_core.py` + plugin | Gateway routes on `X-Phase` header for per-request model swap (#252/#347) |

**Other endpoints:** `/v1/responses` (main.py:947), `/v1/embeddings` (main.py:1166, #207), `/v1/token-count` (main.py:1442), `/v1/cost-estimate` (main.py:1469), `/v1/models` (main.py:1316, lists virtual models via `router_core`).

---

## Virtual Models

Virtual models are aliases that resolve to ordered fallback chains of concrete (provider, model) pairs. Config: `config/virtual_models.yaml`.

**Chain generation:** `scripts/generate_virtual_models.py` (the builder, #341 rebalanced weights). **Runtime reorder:** `scripts/reorder_chains.py --use-u-formula` (telemetry-driven, delegates to `rank_models.py`).

| VM | Status | Weights (agentic/tps/other) | SWE-bench floor |
|---|---|---|---|
| `coding-elite` | `[built]` | 0.80 agentic family / 0.15 tps / 0.05 halluc | >= 70.0 |
| `coding-smart` | `[built]` | 0.80 / 0.15 / 0.05 | >= 65.0 |
| `coding-fast` | `[built]` | 0.60 tps-weighted / 0.40 agentic | none |
| `chat-elite` | `[built]` | intelligence 0.50 / mmlu 0.20 / arena_elo 0.20 / halluc 0.10 | n/a |
| `chat-smart` | `[built]` | intelligence 0.38 / mmlu 0.15 / arena_elo 0.15 / tps 0.15 / halluc 0.17 | n/a |
| `chat-fast` | `[built]` | tps 0.45 / intelligence 0.30 / halluc 0.25 | n/a |
| `chat-rp` | `[built]` | tps 0.35 / ugi 0.40 / writing 0.25 | n/a |
| `auto` / `router/auto` | `[built]` | semantic-routed to one of above | inherited |
| `glm5-elite` | `[built]` | GLM-family coding chain | inherited |
| `agent-*` | `[partial]` | agent task chains | varies |

**Scoring config:** `config/scoring_config.yaml` -- `virtual_models.weights = {agentic_score: 0.80, tps_score: 0.15, availability_score: 0.05}` (#341). `specific_models.weights = {tps_score: 0.70, agentic_score: 0.30}`. `roleplay.weights = {tps: 0.80, agentic: 0.20}`.

**Runtime scoring engine:** `src/rotator_library/scoring_engine.py` `DynamicScoringEngine` -- `CATEGORY_WEIGHTS` (all 6 categories 80/15/5), `THRESHOLDS = {coding-elite: 70.0, coding-smart: 65.0}`, `_compute_free_baseline` (floor 30.0, excludes models worse than worst free model). Wired into `router_core.py:24` (import), `:654` (instantiation), `:1328` (`rank_models_for_virtual` in `_get_candidates`). 16 tests in `tests/test_scoring_engine_341.py`.

**Benchmark data:** `config/model_rankings.yaml` (5155L, 101K) -- per-model `scores.{swe_bench, swe_bench_verified, agentic_coding, livebench_coding, humaneval, speed_tps, hallucination_rate, verified}`. Unverified -> swe_bench/agentic * 0.5.

**Chain-gen weights:** `scripts/generate_virtual_models.py:30-78` `WEIGHTS` dict (hardcoded per VM type, rebalanced #341). `calculate_score(model, weights)` (line 140): log2 TPS scaling, arena_elo normalize (1200-1500), benchmarks /100. `generate_fallback_chain()` (line 278): filters by min thresholds, scores, sorts desc, dedups, caps at max_models.

**U-formula reorder:** `scripts/rank_models.py` + `scripts/unified_ranking.py` `rank_candidates()` -- elite mode 0.75/0.20/0.05 (closest to 80/15/5), smart 0.65/0.25/0.10, fast 0.60/0.30/0.10.

| Feature | Status | Issue |
|---|---|---|
| 80/15/5 scoring weights | `[built]` | #341 |
| SWE-bench thresholds (elite/smart) | `[built]` | #341 |
| Free-model baseline exclusion | `[built]` | #343 |
| Dynamic uptime-weighted chain | `[built]` | #251 |
| Semantic `auto` routing | `[built]` | #197 |
| Dynamic fallback reorder | `[built]` | #195 |
| Cost-effectiveness ranking | `[built]` | #200 |
| Model Council (MoA/best-of-N) | `[planned]` | #339 |
| Category leaderboard endpoints | `[planned]` | #331 |

---

## Providers

Provider registry + adapters. Config: `config/router_config.yaml` (providers block, line 4). Database: `config/providers_database.yaml` (richer metadata: free_quota, rate limits, active_days_windows).

**Adapter factory:** `src/proxy_app/provider_adapter.py` (46.8K) -- `ProviderAdapterFactory` builds per-provider adapters from `router_config.yaml`. Legacy no-key providers force-added: `{g4f, g4f_ollama, g4f_pollinations, g4f_nvidia, g4f_gemini, g4f_groq}`.

**Enabled LLM providers (sample from router_config.yaml):** groq, gemini, cerebras, openai, nvidia, mistral, openrouter, sambanova, bluesminds, together, cloudflare, wiwi, iflow, kilo, aihubmix, supacoder, freemodel, freemodel-anthropic, swiftrouter, g4f_*.

**Search-only providers (skipped in LLM routing):** brave_search, tavily, duckduckgo, exa, jina (in `_SEARCH_ONLY` set, `router_integration.py`).

**Free providers (chain-gen dedup whitelist):** `scripts/generate_virtual_models.py:80` `FREE_PROVIDERS = {groq, cerebras, gemini, together, g4f, nvidia, github-models, kilo, modal}`.

| Feature | Status | File | Issue |
|---|---|---|---|
| Provider adapter factory | `[built]` | `provider_adapter.py` | -- |
| Multi-key management | `[built]` | `rotator_library/credential_tool.py` | #198 |
| Daily check-in tracking | `[built]` | `rotator_library/credential_manager.py` | #199 |
| Provider-category database | `[built]` | `config/providers_database.yaml` | #201 |
| Auto portability (config-driven) | `[built]` | `config_watcher.py` | #203 |
| Re-enable disabled providers | `[built]` | -- | #345 |
| Free-type taxonomy + credits | `[partial]` | `rotator_library/telemetry.py` `llm_provider_credits` table | #290 |
| Cross-ref free providers (OmniRoute/9router) | `[planned]` | -- | #257 |
| Tool scout (9router/OmniRoute/octo) | `[planned]` | -- | #256 |
| add-provider.sh helper | `[built]` | `scripts/add-provider.sh` | #99 |

**Provider URLs:** `src/proxy_app/provider_urls.py` (canonical base URLs per provider).

---

## Telemetry

Two telemetry systems:

1. **Passive LiteLLM callback** `[built]` -- `src/proxy_app/telemetry/logger.py` (DB_PATH env override, default `/dev/shm/telemetry.db`). `llm_events` table. Registered via `litellm.callbacks = [TelemetryLogger()]` in `main.py`. LiteLLM CustomLogger API: `log_pre_api_call`, `log_post_api_call`, `async_log_success_event`, `async_log_failure_event`, `async_log_stream_event`. Captures real TTFT/TPS from live traffic (#140, #141, #142).

2. **Active rotator telemetry** `[built]` -- `src/rotator_library/telemetry.py` `TelemetryManager` (default db_path `/tmp/llm_proxy_telemetry.db`). Tables: `api_calls`, `rate_limits`, `provider_health`, `tps_metrics`, `search_api_credits`, `search_api_usage`, `llm_provider_credits`, `llm_provider_usage`. #342 added rate_limits + provider_health schema.

**`api_calls` schema** (`telemetry.py:60-77`): id, timestamp, provider, model, success, error_reason, response_time_ms, time_to_first_token_ms, tokens_per_second, input_tokens, output_tokens, cost_estimate_usd. Indexes: idx_timestamp, idx_provider_model, idx_success.

**Insert API:** `TelemetryManager.record_call(provider, model, success, response_time_ms, error_reason=None, time_to_first_token_ms=None, tokens_per_second=None, input_tokens=None, output_tokens=None, cost_estimate_usd=None)` (`telemetry.py:227`).

**Active probe** `[built]` (#332) -- `scripts/probe_providers.py`: sends canonical prompt to each enabled provider+model, measures TTFT/TPS/latency, writes via `record_call`. Scheduled every 30min on VPS-40 cron (`/home/ubuntu/logs/probe-providers.log`). 16 tests in `tests/test_probe_providers_332.py`.

**Telemetry export** `[built]` -- `src/rotator_library/tps_export.py` `aggregate_tps_stats(db_path, days=7, min_samples=5)`: queries `api_calls WHERE success=1 AND tokens_per_second>0`, groups by (provider,model), computes avg/median/p95 TPS + avg TTFT.

**Telemetry rotation** `[built]` -- `scripts/telemetry-rotate.sh` (VPS cron weekly). Laptop snapshot: `~/CodingProjects/vps-gh-agent-loop/scripts/telemetry-snapshot.sh` (cron every 6h).

| Feature | Status | File | Issue |
|---|---|---|---|
| LiteLLM TelemetryLogger | `[built]` | `telemetry/logger.py` | #140 |
| SQLite WAL mode | `[built]` | `telemetry/logger.py` | #141 |
| tmpfs DB + snapshot | `[built]` | `/dev/shm/telemetry.db` | #142 |
| rate_limits + provider_health tables | `[built]` | `telemetry.py` | #342 |
| Active provider probe | `[built]` | `scripts/probe_providers.py` | #332 |
| TPS stats export | `[built]` | `tps_export.py` | -- |
| HF keepalive cron | `[built]` | `vps-gh-agent-loop/scripts/hf-keepalive.sh` | #137 |
| VPS+gateway health collector | `[built]` | -- | #111 |
| Telemetry dashboard | `[built]` | `llm-speedrun/scripts/telemetry-dashboard.py` | -- |

---

## Free-Tier Rules

The gateway enforces free-tier-only operation. No paid providers in routing chains.

**Free provider whitelist:** `scripts/generate_virtual_models.py:80` `FREE_PROVIDERS`. Chain dedup loop (line 315-330) skips providers NOT in `FREE_PROVIDERS`.

**Auto-skip conditions** (DESIGN.md): `rate_limited`, `usage_limit_exceeded`, `unhealthy` (>50% failure rate), `down` (5+ consecutive failures). Config: `config/scoring_config.yaml` `skip_conditions` block.

**Free-model baseline exclusion** `[built]` (#343) -- `scoring_engine.py` `_compute_free_baseline` (floor 30.0): models with SWE-bench worse than worst free-tier model are excluded from coding chains. Recomputed on `model_rankings.yaml` reload (`test_baseline_recomputed_on_reload`).

**FREE_ONLY_MODE** `[built]` -- `router_core.py` `_get_candidates` filters candidates when `FREE_ONLY_MODE` env is set.

**Rate-limit awareness** `[built]` -- `src/proxy_app/rate_limiter.py` (22.6K) tracks per-provider RPM/TPM. `config/providers_database.yaml` has `rate_limit_rpm` + `rate_limit_daily_tokens` per provider. Active probe paces at `interval_s=3.0` (~1 req/3s, #207 free-tier safe).

**Free embeddings chain** `[built]` -- `/v1/embeddings` (main.py:1166, #207).

| Feature | Status | Issue |
|---|---|---|
| Free-only provider filtering | `[built]` | #16 |
| Free-model baseline exclusion | `[built]` | #343 |
| Rate-limit tracking | `[built]` | #207 |
| Free embeddings chain | `[built]` | #207 |
| Daily check-in (checkin_required providers) | `[built]` | #199 |
| Free-type taxonomy | `[partial]` | #290 |

---

## Search

Search provider integration for web-augmented routing.

**Search-only providers:** brave_search, tavily, duckduckgo, exa, jina -- in `_SEARCH_ONLY` set (`router_integration.py`), skipped from LLM candidate resolution.

**Search API credits tracking** `[built]` -- `telemetry.py` tables `search_api_credits` + `search_api_usage`.

| Feature | Status | Issue |
|---|---|---|
| Search provider adapters | `[built]` | `provider_adapter.py` |
| Search credit tracking | `[built]` | `telemetry.py` |
| AI web search/research tool | `[planned]` | #20 |
| Super-fast AI chatbot w/ search | `[planned]` | #19 |

---

## Health

Provider health monitoring + cooldown management.

**Health checker** `[built]` -- `src/proxy_app/health_checker.py` (5.7K): periodic health checks (5min interval per DESIGN.md). Marks providers unhealthy after `consecutive_failures_threshold=5` or `failure_rate_threshold=0.50` (config/scoring_config.yaml `availability` block).

**Penalty store** `[built]` -- `src/proxy_app/penalty_store.py` (17K): persistent health penalty tracking. `router_core.py:1355` sorts candidates by penalty score (ponytail: global-lock O(n) on ~5-10 candidates).

**Cooldown manager** `[built]` -- `src/rotator_library/cooldown_manager.py` (5.8K) + `config/cooldown_policy.yaml` (#218).

**Distributed cooldown + gating** `[built]` -- `src/rotator_library/distributed_gate.py` (15.3K) + `config/concurrency_policy.yaml` (#233): `SharedCooldownStore` (cross-process sqlite cooldown) + `ConcurrencyGate` (in-process non-blocking per-(provider,model) slots). Wired into `client.py` (commit 915f0b6).

**Persistent health penalty** `[built]` (#196) -- penalties persist across restarts via `penalty_store.py`.

| Feature | Status | File | Issue |
|---|---|---|---|
| Health checker | `[built]` | `health_checker.py` | -- |
| Penalty store | `[built]` | `penalty_store.py` | #196 |
| Cooldown policy | `[built]` | `cooldown_manager.py` + `cooldown_policy.yaml` | #218 |
| Distributed cooldown + gating | `[built]` | `distributed_gate.py` | #233 |
| Dynamic uptime-weighted chain | `[built]` | `dynamic_chain.py` | #251 |

---

## Status API

Endpoints exposing gateway state for monitoring + router consumption.

| Endpoint | Status | File | Purpose |
|---|---|---|---|
| `GET /v1/models` | `[built]` | `main.py:1316` | Lists virtual models (via `router_core`) |
| `GET /v1/tps-stats` | `[built]` | `main.py:1410` | TPS/TTFT stats from `api_calls` (via `tps_export.aggregate_tps_stats`) |
| `GET /v1/token-count` | `[built]` | `main.py:1442` | Token counting |
| `GET /v1/cost-estimate` | `[built]` | `main.py:1469` | Cost estimation |
| Status API module | `[built]` | `src/proxy_app/status_api.py` (8.4K) | Provider status + health summary |
| Model API | `[built]` | `src/proxy_app/model_api.py` (5.4K) | Model metadata endpoints |
| Category leaderboard endpoints | `[planned]` | -- | #331 |

---

## Deployment

**VPS-40** (40.233.101.233): gateway-only, ~954MB RAM, Tailscale firewall (port 8000 restricted to 100.64.0.0/10). SSH key `~/.ssh/oracle.key`, user `ubuntu`.

**systemd unit:** `llm-gateway.service` (NOT PM2 -- #258 explicitly resolved). Manage: `sudo systemctl [start|stop|restart|status] llm-gateway.service`.

**Deploy flow:**
1. Laptop: `cd ~/CodingProjects/LLM-API-Key-Proxy && git push origin main --no-verify` (gitleaks 8.21 FP on pre-existing history -- known pattern).
2. VPS: `cd ~/LLM-API-Key-Proxy && git pull origin main && sudo systemctl restart llm-gateway.service`.
3. Smoke: `curl http://localhost:8000/v1/chat/completions -H "Authorization: Bearer $VPS_GATEWAY_API_KEY" -d '{"model":"chat-rp","messages":[{"role":"user","content":"say ok"}]}'` -> 200 with content.

**VPS crons (alongside gateway):**
- `*/30 * * * *` -- provider perf probe (#332): `cd /home/ubuntu/LLM-API-Key-Proxy && set -a; . .env; set +a; venv/bin/python scripts/probe_providers.py --max-models 2 --interval 3 --db /tmp/llm_proxy_telemetry.db >> /home/ubuntu/logs/probe-providers.log 2>&1`
- `*/30 * * * *` -- HF keepalive (#137)
- `0 3 * * 0` -- telemetry rotate
- `0 3 * * *` -- llm-speedrun nightly benchmark (VPS-155)

**Laptop crons:**
- `0 */6 * * *` -- telemetry snapshot: `~/CodingProjects/vps-gh-agent-loop/scripts/telemetry-snapshot.sh`

**Tunnel (laptop -> VPS):** `vps-gateway-tunnel.service` (systemd --user, auto-start, auto-restart). Forwards `localhost:8000` -> `100.71.95.75:8000`.

| Feature | Status | Issue |
|---|---|---|
| systemd gateway (not PM2) | `[built]` | #258 |
| Render deploy | `[built]` | #42 |
| Auto portability | `[built]` | #203 |
| OCI A1 capture | `[built]` | #206 |
| HF Space worker + keepalive | `[built]` | #137 |
| Free gateway capacity monitoring | `[built]` | #205 |

---

## Deviations from DESIGN.md

DESIGN.md (331L) is scoped to the dynamic-fallback subsystem. This SPEC is broader. Deviations:

1. **Scoring weights:** DESIGN.md lines 16-37 specify 80/15/5 (AgenticScore/TPS_Score/AvailabilityScore). `config/scoring_config.yaml` previously had 0.70/0.30/0.0 -- corrected to 0.80/0.15/0.05 per #341. DESIGN.md was updated in #341 to reflect this.

2. **SWE-bench thresholds:** DESIGN.md specifies coding-elite >= 70.0, coding-smart >= 65.0. Implemented in `scoring_engine.py` `THRESHOLDS` dict + `rank_models_for_virtual` (sorts meets_threshold first). Chain-gen script (`generate_virtual_models.py`) does NOT enforce per-metric SWE-bench floor -- it uses composite `min_score`. Runtime `scoring_engine.py` is the authoritative threshold path. This is intentional: runtime filter catches models added after chain generation.

3. **Specific models scoring:** DESIGN.md says 70/30 (TPS/Availability, no AgenticScore). `scoring_config.yaml` `specific_models.weights = {tps_score: 0.70, agentic_score: 0.30}` -- includes AgenticScore, not AvailabilityScore. Deviation: agentic is weighted, availability is not. Rationale: specific models are direct provider/model pairs where agentic quality matters more than uptime (uptime handled by health checker separately).

4. **Telemetry DB path:** DESIGN.md references `/dev/shm/telemetry.db` (passive LiteLLM). Active rotator telemetry uses `/tmp/llm_proxy_telemetry.db`. Two separate DBs -- passive captures real traffic, active captures probe data. `/v1/tps-stats` reads the active DB.

5. **Scheduled task intervals:** DESIGN.md specifies health check 5min, rate limit reset 1min, daily reset midnight UTC, TPS recalc hourly, ranking update 5min. Active probe (#332) adds a 30min interval not in DESIGN.md.

6. **AGENTS.md in repo is STALE** (2026-02-05, commit 0342491): claims `main.py` 1357L / `router_core.py` 1683L vs actual 62.5K / 90.0K (~2352L). References G4F/legacy virtual models. Treat AGENTS.md as historical only; this SPEC + DESIGN.md are canonical.

7. **RouterCore bypass (BUGS.md item #2):** BUGS.md (Jan 15 2026) claimed `/v1/chat/completions` bypasses RouterCore. #340 investigation proved this STALE -- full chain verified: main.py -> RouterWrapper -> RouterIntegration -> RouterCore.route_request -> `_route_request_inner` with complete fallback traversal. Real bug found+fixed: streaming fallback exception handler was not applying cooldown (now fixed, router_core.py:1925-1943).

---

## Open Gaps (task-board issues)

| Issue | Priority | Status | Title |
|---|---|---|---|
| #339 | P2 | new | Model Council: Multi-LLM ensemble (MoA / best-of-N) |
| #331 | P2 | new | Category-specific rankings endpoints |
| #330 | P2 | new | Evaluate Bradley-Terry pairwise ranking vs Bayesian shrinkage |
| #329 | P2 | new | Data-driven weight optimization for benchmark dimensions |
| #334 | P2 | new | Shared alias map data source |
| #333 | P2 | new | Consolidate dup leaderboard repos |
| #338 | P2 | new | Leaderboard fetcher |
| #257 | P2 | new | Cross-ref free LLM providers (OmniRoute/9router/etc) |
| #256 | P2 | new | Tool scout: evaluate/forks for 9router, OmniRoute, octo |
| #290 | -- | done/new | Add free_type taxonomy + llm_provider_credits table |
| #258 | P3 | new | Keep llm-gateway on systemd (not PM2) -- resolved, issue still open |
| #8 | P5 | new | OpenRouter paid-via-puter |
| #39 | P5 | new | Puter.js custom provider |
| #19 | P2 | new | Super fast AI chatbot w/ search |
| #20 | P2 | new | AI web search/research tool |
| #21 | P2 | new | Uncensored AI UGI |

---

## Related Documentation

- `DESIGN.md` -- dynamic-fallback system (325L, canonical for scoring + telemetry schema)
- `README.md` -- project overview
- `DEPLOYMENT_GUIDE.md` -- deployment steps
- `ROUTER_DOCUMENTATION.md` -- router internals
- `PROJECT_STATUS.md` -- status snapshot
- `BUGS.md` -- known bugs snapshot (Jan 15 2026, item #2 resolved via #340)
- `HANDOFF_OPUS_2026-06-24.md` -- dynamic_chain/distributed_gate/cost_efficiency wiring
- `AGENTS.md` -- STALE (2026-02-05), historical only

**Task-board:** `ons96/task-board` (issues = tracker). Labels: `status:new|in_progress|done|blocked`, `priority:P0-P9`, `project:*`, `tag:*`, `category:*`.
