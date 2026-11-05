# New-API sidecar handoff for OmniRush

**Saved local continuation (2026-11-04):** focused startup-proof changes are on
`feat/newapi-sidecar`, based on remote `546952e`. Read the local continuation
section below and `workload-router/STATE.md` first; preserve any newer work.

> **Historical GitHub checkpoint (2026-11-03).** The saved continuation point is
> `origin/feat/newapi-sidecar`; it was clean at that checkpoint. Read
> `workload-router/docs/newapi-v1-v2-roadmap.md` first. The old sandbox paths
> below are historical and may not exist on another machine.
>
> Safety boundary: the user has authorized commits, pushes, VPS file placement,
> temporary service stops and bounded test calls. Do not stop, replace or delete
> the LiteLLM gateway, cut over traffic, rotate credentials or delete data
> without explicit approval.

Snapshot: 2026-11-03. This file supersedes the stale conversation summaries about
an unapplied benchmark fix. The current verifier contains that fix, and a saved
full local integration report passes. Production readiness is NOT established.

## Local continuation - 2026-11-04

Read `workload-router/STATE.md`'s 2026-11-04 checkpoint first. The older
`temp-omnirush` handoff is historical: this session fetched the remote and
fast-forwarded the clean tracked worktree from `5863e02` to `546952e`, preserving
`.serena/`. Startup/helper/operator and compatibility work already exists on the
remote; do not recreate it or reset this worktree to `fde9784`.

This continuation saves focused startup-proof changes on the feature branch.
The real binary's 579-byte proof exceeded the helper's old 512-byte bound; it now
matches the sidecar's 4096-byte printable-ASCII limit. The isolated verifier
launches the actual entrypoint and checks fail-closed proofs, TOTP-authorized
boot, chat/coding/SSE, SIGTERM and durable restart. See
`workload-router/docs/channel-proof-helper.md`. No VPS, provider account,
credential store, production data or cutover was touched. The user subsequently
authorized committing/pushing this milestone as a backup, without merging or
deploying it.

Git fetch and PR refs were checked. PR state/reviews were not queried through
the API because no approved fine-grained PAT was available; stored OAuth/classic
authentication was not used for API access. Recheck remote freshness before
continuing. Real-provider/OpenCode and VPS approval gates still apply.

## Saved operator/OpenCode milestone - 2026-11-04

- Startup milestone was committed and pushed as `b8b882b` on the existing
  `feat/newapi-sidecar` branch. Normal secret-scan hooks passed. No merge or
  production deployment occurred.
- The next local milestone now runs the operator CLI against fresh New-API:
  route seeding, catalog preview/apply, empty-catalog preservation, audit,
  invalid-order rejection and priority-only updates. Fixed `boot-check` returning
  exit 0 despite `ok:false`; missing/empty providers now exit 1.
- Optional verifier `--opencode /home/osees/.opencode/bin/opencode` boots current
  OpenCode with isolated HOME/XDG/config and completes a scripted read tool loop
  through the actual sidecar/New-API. It uses no real provider or permanent
  client config. Latest full suite: 357 passed. Exact local evidence/commands
  are in `workload-router/STATE.md` and `docs/operator-commands.md`.
- Saving this verified milestone on the same feature branch is a backup, not a
  cutover. PR review/API access still lacks approved fine-grained authentication.
  Keep v1 and the production gateway unchanged; real-provider, compatibility,
  load and rollback gates remain.

## Current checkpoint

- Branch `feat/newapi-sidecar` tracks `origin/feat/newapi-sidecar`.
- The branch contains the sidecar, compatibility/tooling, pilot/failover and
  staged-roadmap docs. Run `git log --oneline` for the exact HEAD; no local-only
  work is required to resume.
- The staged decision is recorded in
  `workload-router/docs/newapi-v1-v2-roadmap.md`: v1 is a native New-API
  model-first prototype; v2 is the sidecar/full-parity path. No cutover has
  occurred.
- Full provider-free suite: 336 passed (231 at the earlier checkpoint), with
  the existing `Unknown config option: asyncio_mode` warning.
- Release, resource, and 50-request/8-worker concurrency smoke checks passed.
- Targeted provider-fault, runtime, server, quota, adapter, streaming, and
  benchmark tests passed: 77 + 25 tests.
- No deployment of *this sidecar*, credential provisioning, merge, or traffic
  cutover has occurred. Bounded live New-API relay calls and a temporary local
  OpenCode test were performed; VPS-40 also runs the separately authorized
  New-API build described below.

### Staged v1/v2 direction - 2026-11-03

The user accepted a staged path: v1 may prove basic model-first fallback with
native New-API (`RetryTimes=3`) without prompt classification; v2 must add the
old gateway behavior. The full scope, alias matrix and gates are in
`workload-router/docs/newapi-v1-v2-roadmap.md`.

Parsed v1 gates now pass for the direct model `agnes-3.0-flash`: streamed
content and usage, both-direction KiosAPI/CrowLLM failover, a tool call, and an
OpenCode 1.19.5 loop. `router/fast` and `router/balanced` are effectively
Groq-only (OpenRouter returned 429/paid-only 404), so they are not fallback
examples. The 19 legacy aliases remain v2 only, and legacy `auto` must not be
mapped to New-API's literal `auto`.

A v2 deterministic baseline now classifies the legacy `auto` alias from the
latest user message (code/verify/debug vs chat), while explicit `X-Route-Step`
and all other aliases keep their existing behavior. Targeted tests: 88; full
provider-free suite: 336. This is not semantic-router parity and is not deployed.

### 2026-11-03 compatibility increment

- Provider-free suite is now 300 passed (231 in the snapshot above).
- Legacy gateway aliases are accepted and advertised as `deprecated`;
  `/v1/responses` and `/v1/embeddings` serve the documented subsets. See
  `workload-router/docs/responses-and-embeddings.md`.
- Operator CLI and channel-proof helper exist but are not installed on VPS-40.
- Added read-only `GET /v1/models/{model_id}` and `GET /v1/providers`; that
  increment raised the surface to 9 paths.
- Added ledger-backed `GET /v1/tps-stats` and the read-only model-management
  routes `GET /v1/model/list`, `GET /v1/model/status`, and
  `POST /v1/model/parse-command`; the sidecar previously exposed 16 paths, with
  16 gateway paths absent. Full provider-free suite: 336 passed.
- Added read-only `POST /v1/cost-estimate` from the cheapest configured ledger
  price. Python remains the sidecar language after a measured 26,984 KiB RSS
  baseline; see `workload-router/docs/sidecar-language-decision.md` for the
  128 MiB canary guardrail and rewrite triggers.
- Added free-tier classification: `free_unlimited`, `free_limited`,
  `free_credits`, `paid`, `unknown`. Actual prices stay zero for free routes;
  one-time credits use a separate notional credit tariff so they deplete and
  rank below unlimited routes without appearing as real spend. See
  `workload-router/docs/free-tier-classes.md`.
- Added conservative `POST /v1/token-count`; it returns a UTF-8 byte upper bound
  with `estimated:true`, not an exact provider tokenizer count.
- Added unauthenticated `GET /` with the legacy `Status` field for probes.
- Added read-only `provider_classification.py` and
  `newapi_operator.py classify`, using `llm_providers.db` as the source for
  free-tier flags. The command reports route mismatches but never rewrites them.
  Also fixed catalog loading for the live `providers.key_name` schema.
- Added authenticated read-only `/api/providers/*` compatibility routes backed
  by the gateway's `provider_status.db`: status, single-provider status, best,
  health, history, single-provider history, and CSV. Every row is marked
  `stale` after the configured threshold; `health` is explicitly
  `running:false`. The sidecar does not probe or write that database. See
  `workload-router/docs/provider-status.md`.
- Added process-local bounded session switching, session lookup,
  deterministic intent detection, and ledger-only model-info stats. The five
  `/api/cliproxyapi/*` routes remain absent because they require an external
  service and credentials. See `workload-router/docs/model-management.md`.
- Review map and unverified list: `workload-router/docs/review-package.md`.
- Later on 2026-11-03: real New-API relay calls to both usable providers (Groq,
  OpenRouter) passed at `max_tokens=15`. Sidecar code is staged read-only at
  `/opt/newapi-sidecar` but not running: strict audit is blocked on New-API
  `RetryTimes=3`, forced-stream channel overrides, and disabled 2FA; the login
  helper initially read the wrong env file (corrected below). See
  `workload-router/STATE.md`.
- Provider catalog located: 204 providers / 6,827 models, with 44 enabled in the
  gateway config. Live probes found many stale entries; New-API now has
  `gw:hcnsec`, `gw:iamhc` (`auto`) and `gw:bluesminds`, all 200. The earlier
  "login limiter" was a wrong env-file read; login works. The goal is now
  model-first routing through New-API channels, so the sidecar's strict audit
  and 2FA remain parked. See `workload-router/docs/provider-import-plan.md`.
- Failover proven 2026-11-03: disabled `gw:hcnsec` with
  `POST /api/channel/6/status`; `auto` still returned 200 via `gw:iamhc`;
  re-enabled and confirmed `status: 1`. The generic channel update rejects a
  `status` field.
- Added `gw:crowllm` (id 9, now 73 chat models), `gw:kiosapi` (id 10, 4 models)
  and `gw:crowllm-canonical` (id 11, priority 40) mapping `agnes-3.0-flash` to
  `agnes-3.0-flash:free`. Channels 9/10 passed through New-API; both 10 and 11
  are in the abilities table for that model. New-API has 9 channels. The 7
  embedding ids were removed from channel 9; it still returned 200 and has 73
  abilities.
- Canonical failover test passed 2026-11-03: with `gw:kiosapi` (id 10)
  disabled, `agnes-3.0-flash` still returned 200 via `gw:crowllm-canonical`
  (id 11); channel 10 was re-enabled and confirmed `status: 1`.

### VPS-40 state as of 2026-11-03 (Stage 1 inventory, read-only)

- Host: Ubuntu 24.04.4, amd64, 2 vCPU, 954 MiB RAM, 128 MiB available at sample,
  5 GiB swap (2 GiB file + 3 GiB zram at ~494 MiB), 48 GiB disk at 31%.
- `llm-gateway.service` active on `*:8000`, socket-activated, effective
  `MemoryMax=512M` via drop-in `memory.conf`. `/health` returns 200.
  Live traffic and rollback stay with this service.
- **New-API already installed:** `new-api.service` on `*:8001` since 2026-10-20,
  `/opt/new-api`, pinned `v1.0.0-rc.36` with sha256 matching
  `5d9b2547...e957f3`, user `newapi`, SQLite `one-api.db` (41 MiB, 3 channels,
  29 abilities, 1 user). Simplified deploy notes: `docs/new-api-deploy.md`
  (task-board #862, EPIC #876). Support timers: healthcheck 5 min, watchdog
  5 min, backup daily (all active); discovery daily is enabled but inactive.
- Ports 8000/8001 are admitted only from loopback and Tailscale by iptables
  (no `ufw`). The `*:8001` bind is deliberate: the dashboard is reached over
  Tailscale, and the firewall, not the bind address, is the access control.
- Existing `ai-workload-router-canary.service` on `127.0.0.1:8123` runs the older
  router release, not this sidecar. No sidecar install, unit, or channel-key
  proof helper exists on the host.
- Effective pipeline: Stage 2 is gap-closing on this install, not a fresh
  install. Stage 3 is blocked on the operator proof helper.

## Resume prompt

Read this handoff and `workload-router/docs/newapi-v1-v2-roadmap.md`, then
continue the New-API work from the GitHub branch `feat/newapi-sidecar`.
Preserve existing work and leave VPS-40 and the current gateway untouched until
the canary stage is explicitly authorized. Start with the remaining external
validation plan below. Do not merge, deploy, change credentials, or cut over
traffic without explicit authorization.

## Goal and safety boundary

The final stack is an unmodified New-API gateway plus a quota-aware, latency-aware,
cost-aware Python sidecar on VPS-40. It will replace the current LLM gateway only
after testing and confirmation that the replacement works. Parallel operation is
the validation stage, not the final architecture.

The user's replacement goal does not authorize an immediate service restart,
deployment, traffic switch, or deletion of the old gateway. Preserve the legacy
gateway, configuration and state for a tested rollback.

Use Python stdlib, SQLite WAL and bounded concurrency for the 1 GB VPS. Do not
introduce a Go rewrite, Docker, Redis, a LiteLLM/New-API fork, or a resale layer.
All tools and service usage must remain free. New-API is AGPL-3.0; preserve its
notices. The sidecar's distribution/license scope remains unresolved; do not
relabel it MIT.

## Authoritative locations

| Purpose | Location and verified state |
| --- | --- |
| Implementation worktree | `/home/osees/CodingProjects/LLM-API-Key-Proxy-worktrees/wt-newapi-sidecar` |
| Implementation branch/base HEAD | `feat/newapi-sidecar`, `fde9784` |
| Python component | Implementation worktree's `workload-router/` directory |
| Migration documentation worktree | `/home/osees/CodingProjects/llm-gateway-worktrees/wt-newapi-sidecar` |
| Documentation branch/base HEAD | `work/newapi-sidecar`, `078ce55` |
| Provider metadata, READ ONLY | `/home/osees/CodingProjects/llm-provider-manager/llm_providers.db` |
| Unmodified test binary | `/home/osees/new-api/bin/new-api` |
| Passing full local proof report | `/tmp/opencode/newapi-proof-7uclf86q/report.json` |
| OmniRush launch directory | this worktree root: launch `omnirush` here |

Implementation work is committed on the feature branch. No production database
writes or VPS deployment were performed by this workstream. Original checkouts
can contain other intentional edits; do not switch to them to work.

Read `workload-router/AGENTS.md`, `workload-router/STATE.md`,
`workload-router/README.md`, `workload-router/docs/specs/p0-router.md`, and
`workload-router/docs/newapi-design.md`. The older `HANDOFF.md`
describes a different VPS task; do not treat its "next task" as this assignment.
Read the migration worktree's `docs/sidecar-decisions.md` for the accepted
reuse/build decisions and licensing gates.

## Implementation completed

| File in workload-router | What exists |
| --- | --- |
| `quota_ledger.py` | Atomic SQLite reservations; per-key and shared-provider RPM/RPD/TPM/TPD, credit and daily-money limits; conservative settlement; persistent cooldowns; only-tighten limit learning; route catalog and per-attempt telemetry. |
| `route_policy.py` | Hard capability/context/quota/price gates, requested cost scorer, top-N Thompson exploration, cache-aware switching, coding affinity, no chat affinity, conservative text bounds and opt-in lexical JSON tool-output compaction. |
| `newapi_runtime.py` | Request validation, private-alias dispatch, independent reservation for each attempt, no same-route 429 retry, accounted completions and streams, separate text/reasoning TTFT, pre-header priming and cleanup. |
| `newapi_server.py` | Separate authenticated loopback entrypoint; bounded worker creation, 1 MiB request cap, SSE, model discovery, diagnostics, route preview and chat completions. Legacy entrypoint stays unchanged. |
| `provider_adapter.py` | Existing transport reused; bounded 429 hints, optional response cap and redirect denial. Legacy defaults preserved. New runtime uses an 8 MiB response/stream cap and no redirects. |
| `newapi_catalog.py` | Management audit, private-channel creation, authorized channel-key proofs, relay-token audit, priority-only updates, additive ordinary-channel refresh, read-only canonical metadata, last-good catalog fetch and symlink-safe additive OpenCode refresh. |
| `newapi_benchmark.py` | Five deterministic plan/edit/debug/tool/review cases, non-executing graders, fragmented-tool reconstruction, seeded request budget, persisted grades, timing metrics and before/after rankings. |
| `tools/verify_newapi.py` | Fresh network namespace and private database, real unmodified New-API binary, three scripted local providers, authorized 2FA enrollment, strict audit, profile/SSE/failover/config-refresh/benchmark checks and RSS samples. |

Tests added: `tests/test_quota_ledger.py`, `test_route_policy.py`,
`test_newapi_runtime.py`, `test_newapi_server.py`, `test_newapi_catalog.py`,
and `test_newapi_benchmark.py`.

Tracked edits also include `.gitignore` and `STATE.md`. The new modules, tests,
`tools/verify_newapi.py` and `docs/newapi-design.md` are untracked. The migration
worktree has untracked `docs/sidecar-decisions.md`. Inspect current status before
editing or staging; none of this may be discarded.

## Accounting and API contracts to preserve

1. One private alias identifies one provider/model/key channel. Audit requires
   `RetryTimes=0`, unique aliases, enabled single-key channels, exact model
   mapping, credential fingerprint and endpoint identity. Never bypass New-API.
2. The dedicated relay token uses group `default` and `cross_group_retry=false`.
   Unknown identities, prices, required capabilities or limits fail closed.
   Free-only defaults true; paid routes are excluded. Free credits are budgeted,
   not called unlimited.
3. `QuotaLedger.reserve()` precedes every dispatch; `start()` marks network risk;
   `finish()` settles it. Only unstarted leases can be cancelled. Failed or
   truncated output without trustworthy final usage retains the full reservation.
   Crashed started reservations stay charged; do not silently refund them.
4. Public profiles are `auto-coding`, compatibility alias `auto-code`, and
   `auto-chat`. The 19 legacy gateway aliases are also accepted and mapped to a
   profile plus a step hint (see
   `workload-router/docs/alias-mapping-proposal.md`). Step hints use
   `X-Route-Step`; sessions use `X-Session-Id`.
   `RouterState.set_preferred(session_id, "auto-coding", route_id)` is the correct
   argument order. Chat does not reuse coding preference/cache state.
5. The default output cap is 2048 and is forwarded upstream. Multi-generation
   requests are rejected. Streams prime before headers, replay buffered data once,
   and never fail over after output is exposed. Grading happens after settlement.

Endpoints currently exposed: `/healthz`, `/v1/models`, `/v1/models/{model_id}`,
`/v1/providers`, `/v1/tps-stats`, `/v1/model/list`, `/v1/model/status`,
`/v1/router/diagnostics`, `/v1/router/route`, `/v1/model/parse-command`,
`/v1/chat/completions`, `/v1/cost-estimate`, `/v1/responses`,
`/v1/embeddings`. Model detail, provider list, TPS, cost estimation, and model
management are read-only; TPS covers measured streaming ledger samples only,
and command parsing never switches a session.
Route preview performs no reservation or generation. Text-only conservative
token accounting intentionally rejects multimodal payloads.

## Verified evidence and its limits

The report at `newapi-proof-7uclf86q/report.json` was read directly when preparing
this handoff. It contains `passed: true`, not merely a truncated success message:

- Chat, coding and SSE returned 200 through the real New-API binary.
- Injected 429 dispatched exactly `one-model`, then `two-model`; no hidden retry.
- Three local catalogs refreshed the disposable OpenCode configuration.
- Grading changed all five case rankings from `one, two` to `two, one`.
  Route one graded 0/5 correct; route two graded 5/5; paid route three had zero
  attempts. These are deliberately scripted fixture results, NOT LLM scores.
- RSS sample: New-API 128900 KiB; Python fixture plus sidecar 42996 KiB. This is
  a local sample, not a VPS load or pure-sidecar memory guarantee.

The current full Python suite was rerun from `workload-router/` with
`PYTHONPATH=.` and `uv run --with pytest`: **234 passed**. Pytest still emits the
existing `Unknown config option: asyncio_mode` warning because the environment
does not install `pytest-asyncio`; this is non-blocking for the current suite.

The real-binary verifier also now passes using the documented pinned
New-API `v1.0.0-rc.36` amd64 release (SHA-256 verified before execution).
It exercised the disposable local New-API instance, scripted 429 failover,
three-provider catalog refresh, additive OpenCode refresh, benchmark ranking
changes, paid-route exclusion, and resource sampling. This remains local
fixture evidence; no live provider, VPS, or traffic cutover was performed.

Earlier isolated runs intermittently returned 503 before upstream dispatch.
The cause was not established; a later passing run does not prove that flakiness
was eliminated. Reproduce with fresh instances and inspect the correct report.

## Resolved proof failure - do not undo the real ledger

The old fixture returned `RPM Limit: 1` in its 429 body. Limit learning correctly
installed an RPM ceiling of one, so every later benchmark request on that route
was infeasible. Clearing cooldowns alone could not remove quota usage. Earlier
transport evidence also already favored route two before grading.

The current fixture now returns a generic `rate limit exceeded` message and
retains `Retry-After: 120`. Before grading, the verifier clears quota windows,
decisions, cooldowns and benchmark history ONLY in its newly created disposable
database. This creates a clean controlled ranking experiment. Earlier phase
results stay in the report. NEVER transplant those DELETE statements into a
production maintenance path or weaken only-tighten learning to make a test pass.

## Outstanding gates and next work

1. Run the real-binary verifier using the exact command below. The binary and
   artifact parent must exist on the execution host. Inspect only redacted
   report fields; do not print generated credentials or temporary logs.
2. Provision the operator-controlled channel-key proof helper. It must receive
   only a channel ID on stdin and print exactly one proof token. Configure
   `NEWAPI_CHANNEL_KEY_PROOF_COMMAND` and verify valid startup plus fail-closed
   behavior for missing or invalid proofs. Never store credentials in this repo.
3. Finish catalog/benchmark operator orchestration and boot validation. Primitives
   exist, but real three-provider refreshes, current OpenCode boot checking,
   repeatable operator commands and end-to-end priority updates remain unverified.
   The additive OpenCode writer assumes providers already exist and never guesses
   model context, tool capability or free pricing.
4. Test real free providers and complete OpenCode tool loops with intact tools,
   usage, SSE and fault behavior. Inventory the old gateway's actual clients,
   virtual aliases and API routes; the new HTTP surface is not yet a full
   compatibility replacement. Synthetic microbenchmarks are not this gate.
5. After separate authorization, validate side-by-side on VPS-40: resource/load,
   restart/persistence, strict quotas, compatibility and rollback. Confirm every
   gate before traffic cutover. Keep the old gateway intact afterward until its
   retirement is separately approved. Note the inventory above: New-API already
   runs on `:8001`, so this stage is gap-closing plus sidecar validation, not a
   new install.

The host-integrated execution plan and secret-free prompt for this stage are in
`docs/VPS-CANARY-HANDOFF-PROMPT.md`. Use a host-integrated harness for SSH
operations, then return here for local review before advancing stages.

## Exact local checks

Set the command runner's working directory to the absolute `workload-router`
path above. Do not rely on OmniRush's launch directory or a stale MCP index.

```sh
python3 -m pytest tests -o addopts= --quiet
git diff --check
python3 tools/verify_newapi.py --binary /home/osees/new-api/bin/new-api --artifacts /tmp/opencode
```

The verifier enters its own network namespace because the binary otherwise
listens on all interfaces. It requires Linux `unshare` and `ip`; no Go compiler
or Docker is required. It writes a new owner-only `newapi-proof-*` directory
each run. Select reports by modification time, NOT lexicographic path order.
Temporary logs/databases contain generated credentials; do not dump them or
commit them. Use bounded redacted inspection and the report's public fields.

Current entrypoint environment names: `SIDECAR_DB`, `SIDECAR_PORT`,
`SIDECAR_API_KEY`, `NEWAPI_BASE_URL`, `NEWAPI_ADMIN_TOKEN`,
`NEWAPI_ADMIN_USER_ID`, `NEWAPI_API_KEY`, `NEWAPI_RELAY_TOKEN_ID`.
Optional: `SIDECAR_HOST` (loopback), `SIDECAR_FREE_ONLY` (true),
`SIDECAR_COMPRESS_JSON` (false). Names only; no credential values belong here.

## Resume gotchas

- `rtk pytest` repeatedly reported `No tests collected`; that is not verification.
  Use the exact `python3 -m pytest` command above, without piping to head/tail.
  Preserve exit status and stop to inspect a real failure instead of repeating
  the same command.
- leanctx's overview/checkpoint index targets an unrelated old project. Use
  absolute file paths and task-local searches. Do not follow its stale task label.
- Do not stash/reset/clean these worktrees. New files are untracked and depend on
  tracked adapter edits; an earlier stash removed `_NoRedirect` and broke imports.
- Earlier failed temporary debug scripts may have left a test New-API process on
  port 8099 outside the verifier namespace. Identify PID, argv, owner and artifact
  directory before cleanup; never use broad process-kill patterns or touch a live
  service. Process cleanup has not been verified by this handoff.
- Coverage, lint/typecheck/format and current full release checks are not claimed
  complete. Historical counts and model-fallback messages are not new evidence.

First action: run the real-binary verifier if the binary and artifact paths exist;
otherwise document that environment gap and proceed with proof-helper and
operator-validation planning. Do not repeat passing local tests unless code
changes.
