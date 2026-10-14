# New-API vs LiteLLM vs Keep-Current: Provider Dashboard + Free/Paid Tagging

Issue: #1083. Research-only — no code migration.
Siblings: `docs/oss-routing-key-mgmt-eval.md` (routing brain), `docs/vercel-sdk-eval.md` (app-layer SDK, REJECT). This eval covers a different layer: **control-plane ergonomics** (dashboard, tagging, quota metadata, sync), not request-path routing.

## Question

Should we adopt New-API (QuantumNous/new-api, the one-api fork) over LiteLLM — or keep the current custom gateway — for the provider dashboard, free/paid model tagging, and OMP/gateway sync?

## TL;DR

**Keep current + extend.** New-API is the best *dashboard product* of the three, but its tagging/quota model is billing-shaped (resale credits, USD-ish quotas), not free-fleet-shaped (quota classes, daily resets, key rotation, check-ins). LiteLLM Proxy needs Postgres and exceeds the VPS-155 budget. Our tagging gap is already solved on `feat/1069-quota-class`; the dashboard gap is a thin read-only HTML page over the existing `status_api.py` JSON endpoints. Sync stays git-native.

## Constraints (non-negotiable)

- **Free only** — no paid SaaS; all $0 licenses are fine, but license *terms* differ (below).
- **VPS-155 budget: ~273MiB avail** (1 GB box, weak CPU, nightly `llm-speedrun`). VPS-40 gateway already systemd-capped at `MemoryMax=400M`.
- **Git as source of truth** — provider/model metadata is YAML in this repo (PR review, VPS auto-pull, `reorder_chains.py` cron).
- **33 exotic adapters** — g4f, gemini_cli OAuth, antigravity, cloudflare, kilo OAuth, etc. Any replacement must still sit *in front of* these.

## Candidates (as of 2026-10)

| | New-API | LiteLLM Proxy | Current gateway |
|---|---|---|---|
| What it is | Go+Gin/React self-hosted AI hub (one-api fork), 49.9k★, AGPLv3 + attribution terms, very active (6.5k commits) | Python proxy server + Admin UI, MIT core, key UI features enterprise-gated | Custom FastAPI (`src/proxy_app`) + 33 adapters; we already use the LiteLLM **SDK** inside adapters — that is separate from the Proxy product |
| Data plane | OpenAI/Anthropic/Gemini-format relays, channels, model mapping, priorities/weights, retries | OpenAI-compatible proxy, Router (cooldowns, budgets) | Virtual model chains, cooldown/quota_class policy, telemetry SQLite, health probes |
| Storage | SQLite OK; MySQL/Postgres optional; Redis optional; multi-node needs shared DB+Redis | **Postgres required** for keys/budgets/UI | YAML (git) + SQLite |

## Feature comparison — the six #1083 axes

| Axis | New-API | LiteLLM Proxy | Keep current + extend |
|---|---|---|---|
| **Free-model tagging** | Per-channel model lists, groups, "model square" pricing page; tags are billing-shaped (price/quota), no free-tier semantics — free = "price 0", loses quota-class fidelity | `model_info` metadata + key tags; cost model is USD from `model_prices_and_context_window.json`; unknown providers → no cost data; no free-only enforcement | `free_tier` + `free_models` in `providers_database.yaml` (89 true / 2 false today) + `FREE_ONLY_MODE` enforcement; #1069 adds `quota_class: unlimited/credit/trial/paid` + `supports_cache` — already coded on `feat/1069-quota-class` |
| **Quota/type metadata** | Best-in-class quota UIs, but denominated in site credits (resale-oriented); expression-based pricing; per-key/user/team budgets | Per-key/user/team `max_budget`, tpm/rpm; USD spend tracking; state in Postgres | `provider_caps.yaml` (context/rpm/tpm), `rate_limits` in providers DB (daily resets), quota-class table driving 429 wait-vs-fallback (#218/#1068/#1069) |
| **Dashboarding** | The reason it's tempting: channel table, model square, usage dashboards, logs, playground, channel testing, plugin marketplace | Decent key/spend UI; SSO/audit/advanced admin enterprise-gated; UI requires the DB | JSON-only today: `/status`, `/status/{p}`, `/best`, `/history`, `/export/csv`, `/health` (`status_api.py`). Gap is real but thin: one read-only HTML page over these endpoints ≈ 150–300 LOC, zero new deps/services |
| **OMP/gateway sync** | Source of truth moves into DB rows; needs YAML→REST push job + drift/conflict handling; channel edits bypass git review/rollback | `config.yaml` stays declarative (good), but key/budget edits in UI/DB bypass git | Sync *is* git: PR-reviewed YAML → VPS auto-pull → cron reorder → `sync-opencode-config.sh` for the opencode side. Adding a DB plane makes sync bidirectional and worse |
| **Self-host cost (273MiB)** | Go binary + SQLite ≈ 50–200MiB (estimate, unverified on A1-class CPU); fits only without Postgres/Redis; tight next to nightly benchmarks | Python proxy ≈ 300–500MiB **plus Postgres ≈ 60–120MiB** → does not fit VPS-155; hosting it on VPS-40 would fight the 400M cap | ~0 additional: HTML served by the existing FastAPI process |
| **Maintenance** | Active but fast-moving: frequent releases, migrations, mixed Chinese/English docs; must pin versions on a 1 GB box | Active; frequent breaking changes; enterprise gating churn | Ours — but the dashboard need is read-only and small; no upgrade treadmill |

## Why not New-API in front (hybrid panel)?

The only configuration that makes sense without losing the routing brain: New-API on VPS-155 (SQLite-only), our gateway registered as a single OpenAI-compatible channel, dashboard consumed by humans only. Buys: a polished panel + playground + channel testing. Costs: a second always-on service on the tightest box, AGPL+attribution terms if ever exposed, a push job to mirror `providers_database.yaml` → channel rows (and drift risk the moment either side is hand-edited), and its tagging still can't express quota classes. It replaces a ~200-LOC static page. Not worth it now.

## Recommendation: keep current + extend

1. **Merge `feat/1069-quota-class`** — `quota_class` + `supports_cache` tagging and the policy table are exactly the #1083 "free/paid tagging + quota/type metadata" workstream, already implemented and tested in-repo.
2. **Thin read-only dashboard**: one HTML endpoint in `status_api.py` rendering provider, enabled, quota_class, free-model count, health/uptime, chain membership from existing JSON APIs + `virtual_models.yaml`. No new service, no new dependency, no auth surface beyond the existing key.
3. **Keep git as the sync backbone.** If the OMP side needs a machine-readable feed, auto-generate `providers_export.csv` from the existing `/status/export/csv` on the cron instead of introducing a database plane.

**Revisit triggers:** (a) we ever need multi-user key management / resale-style quotas → New-API wins that future outright; (b) VPS-155 gains ≥512MiB headroom and we want playground + automated channel testing; (c) LiteLLM Proxy ships free-tier-aware (non-USD) quota metadata.

## Migration sketch (if we ever switch anyway)

- **Phase 0 — canary:** VPS-155, `docker run` SQLite-only, bind `127.0.0.1:3000`, one channel = existing gateway `base_url`, zero-cost group. Gateway untouched.
- **Phase 1 — metadata push:** idempotent script: `providers_database.yaml` → channels via New-API REST (upsert by name), all model prices 0, quota classes recorded in channel notes/tags as best-effort.
- **Phase 2 — opt-in consumption:** one opencode profile points at a New-API-issued token; virtual chains still resolved by our gateway behind it.
- **Phase 3 — decide:** keep only if the panel earns its RAM; otherwise stop container. Every phase rolls back by stopping the container; the gateway never changes.

## Acceptance criteria — #1083

- [x] Comparison doc (six axes: tagging, quota/type metadata, dashboarding, OMP sync, self-host cost, maintenance)
- [x] Recommendation (keep current + extend) with migration sketch
- [x] No code migration — research-only, no deps added
- [x] QUESTIONS.md entry for the one blocking ambiguity (OMP referent) — see `workload-router/QUESTIONS.md`

## No code changes

No files modified outside this doc and the QUESTIONS.md entry. No new deps. No tests.
