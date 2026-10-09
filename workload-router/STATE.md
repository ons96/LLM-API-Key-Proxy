# Router Session State

## Current status

This project folder contains prior router artifacts and a handoff from the VPS
implementation. A standalone P0 implementation has now started locally with a
dependency-free deterministic routing core.

## Source documents

- Architecture plan: `/home/osees/CodingProjects/temp-omnirush/docs/ai-workload-router-architecture.md`
- Existing implementation handoff: `HANDOFF.md`
- Existing validation entry point: `python3 tools/validate_session.py`

## Known existing next task

`HANDOFF.md` says the next VPS-repository task is fixing
`scripts/reorder_chains.py --dry-run` and adding a regression test proving that
dry-run preserves file bytes. That work belongs to the VPS repository and must
not be assumed complete here.

## Completed checkpoint: deterministic core

Added `router_core.py` and `tests/test_router_core.py`. The core classifies
obvious requests, applies tool/structured-output/context/health hard filters,
and selects an eligible deployment using expected latency divided by smoothed
success rate. Added `router_server.py`, a dependency-free local route-inspection
HTTP boundary; it makes no provider calls.

Verification:

- `python3 -m pytest -q` -> `4 passed in 0.04s`
- `python3 tools/validate_session.py` -> `router session validation: OK`

Added `HealthRegistry` with bounded in-memory operational cooldowns. Selection
now excludes cooled-down deployments while keeping operational failover
separate from semantic capability escalation.

Added session stickiness. SQLite stores the last deployment per session and
phase; eligible tool-loop continuations reuse it, while phase changes or
cooldowns permit reselection. The preference is only a soft state hint and
never overrides hard capability filters.

Added an `outcomes` SQLite table and bounded outcome recording for success,
operational failure, semantic failure, and escalation. The HTTP route records
successful route decisions; unsupported outcome categories are rejected so raw
prompt/error content cannot become an accidental telemetry category.

Added bounded phase normalization for `explore`, `plan`, `implement`, `verify`,
`debug`, and `deep_debug`, including a few safe aliases. Unknown header values
become the neutral default and cannot inject arbitrary session-state keys.

Added `POST /v1/router/outcome`. It accepts bounded outcome categories and
records them separately. Only `operational_failure` marks a deployment for
cooldown; semantic failures are telemetry and do not cause an implicit
operational health change.

Added end-to-end HTTP tests covering route selection, invalid empty-message
requests, operational outcome reporting, and cooldown exclusion.

Added a 1 MiB request-body limit enforced before JSON parsing; oversized route
and outcome requests return HTTP 413.

Added per-deployment cache observations in SQLite. The route boundary accepts
provider-neutral cached/write token counts, hashes only a bounded prefix for
identity, and keeps cache state separate for each deployment.

Added prompt-free `GET /healthz` and `GET /v1/router/diagnostics` endpoints for
basic liveness and aggregate decision counts.

Diagnostics now include aggregate success/failure categories and cache-hit
counts, with no prompt or hash fields.

Added a version constant (`0.1.0`) and included it in the prompt-free health
response for deployment diagnostics.

Deployment profiles and decision telemetry now distinguish provider labels and
configuration fingerprints in addition to deployment IDs. These fields are
non-secret metadata only.

SQLite startup now migrates legacy `decisions` tables by adding the new metadata
columns with safe defaults.

Deployment profiles now support optional quota class and remaining-quota fields.
Exhausted quota receives a soft score penalty rather than becoming a hard
filter, so incomplete quota telemetry cannot take the router fully offline.

Catalog loading now validates positive context/latency, success rates in the
closed interval 0..1, and non-negative quota counts before accepting profiles.

Added `stall_detector.py` with bounded repeated-action and failed-verification
signals, escalation ceilings, and progress reset. The HTTP route now accepts
bounded action/verification headers and returns an escalation level without
retaining raw tool history.

Successful outcome reports reset the session's stall signals. Repeated failed
actions can raise escalation only up to the configured ceiling, preventing
unbounded escalation or oscillation.

Added `trace_replay.py` and `examples/sample-traces.jsonl` for offline shadow
evaluation. Replay uses request metadata only, makes no provider calls, and
reports selected deployments against a named fixed baseline.

Added a dedicated `verify` deployment to the safe default catalog so verify
phase traces have an explicit capability target rather than falling into debug.

Replay results now report baseline-match counts and can write JSON aggregate
reports with `write_report`; reports contain no prompt contents.

The current verified local baseline is 25 passing tests. Runnable commands are
documented in `README.md`; no provider credentials or upstream service are
needed for the test suite or route-inspection server.

Deployment target is VPS 40 (`40.233.101.233`). This workspace remains the
development/test location; no VPS deployment has been performed by this
session. Resource decisions must remain suitable for 1 GB RAM, zram, weak CPU,
and no GPU. See `docs/DEPLOYMENT.md`.

Hardened JSON responses with `no-store` and `nosniff`, and added route metadata
headers (`X-Router-Deployment` and `X-Router-Capability`) for diagnostics.

Added `router_config.py` and `config/deployments.json` for validated,
credential-free deployment profiles. Invalid or missing optional catalogs fall
back to the built-in safe defaults.

Added `provider_adapter.py` with a provider-neutral execution boundary. It
tries eligible same-capability deployments once each, marks operational
failures unhealthy, and never silently falls back to a different capability.

The HTTP boundary accepts `POST /v1/router/route` with a messages list and
returns the selected deployment/capability. It is intentionally not yet a chat
proxy, provider adapter, telemetry store, or streaming endpoint.

Added `router_state.py` and wired it into the route boundary. Requests now
require a non-empty list of message objects, and each successful route records
session/prompt hashes plus route metadata in SQLite. SQLite access is protected
for the threaded HTTP server; raw prompt text is not persisted.

README now documents local startup and a credential-free example request.

Additional verification:

- `python3 -m pytest -q` -> `5 passed in 0.02s`
- `python3 -m py_compile router_core.py router_server.py tests/test_router_core.py` -> passed
- Local server smoke test returned `{"deployment": "tool", "capability": "fast_tool"}`
- `python3 -m pytest -q` -> `6 passed in 0.06s`
- `python3 -m py_compile router_core.py router_server.py router_state.py tests/test_router_core.py tests/test_router_state.py` -> passed
- `python3 tools/validate_session.py` -> `router session validation: OK`
- `python3 -m pytest -q` after cooldown support -> `7 passed in 0.05s`
- `python3 -m py_compile router_core.py router_server.py router_state.py tests/test_router_core.py tests/test_router_state.py` -> passed
- `git diff --check` -> passed
- `python3 -m pytest -q` after offline replay batch -> `32 passed in 4.22s`
- `python3 tools/validate_session.py` -> `router session validation: OK`
- `git diff --check` -> passed
- `python3 -m pytest -q` after stall/escalation batch -> `30 passed in 4.21s`
- `python3 tools/validate_session.py` -> `router session validation: OK`
- `git diff --check` -> passed
- `python3 -m pytest -q` after catalog range validation -> `28 passed in 4.24s`
- `python3 tools/validate_session.py` -> `router session validation: OK`
- `git diff --check` -> passed
- `python3 -m pytest -q` after quota scoring -> `27 passed in 4.28s`
- `python3 tools/validate_session.py` -> `router session validation: OK`
- `git diff --check` -> passed
- `python3 -m pytest -q` after SQLite metadata migration -> `26 passed in 4.19s`
- `python3 tools/validate_session.py` -> `router session validation: OK`
- `git diff --check` -> passed
- `python3 -m pytest -q` after provider metadata -> `25 passed in 4.17s`
- `python3 tools/validate_session.py` -> `router session validation: OK`
- `git diff --check` -> passed
- `python3 -m pytest -q` after telemetry diagnostics -> `25 passed in 4.18s`
- `python3 tools/validate_session.py` -> `router session validation: OK`
- `git diff --check` -> passed

Added dependency-free context estimation from message content. Explicit
`context_tokens` remains authoritative; otherwise the estimate feeds hard
context-window filtering.

Added `POST /v1/chat/completions` with an OpenAI-compatible response shape when
the embedding process installs provider adapters. The standalone process keeps
the adapter map empty and returns bounded 503 errors instead of making
unconfigured upstream calls.

Chat requests now validate `response_format` objects and explicitly return HTTP
501 for unsupported streaming instead of silently downgrading the request.

Chat requests now validate the public model aliases `auto`, `auto-code`, and
`auto-chat`, and return the requested alias in the OpenAI-compatible response.
`auto-code` also supplies an implementation-phase hint when the client has not
provided an explicit phase, so it selects the code implementation capability.
- `python3 -m pytest -q` after provider failover boundary -> `20 passed in 2.12s`
- `python3 tools/validate_session.py` -> `router session validation: OK`
- `git diff --check` -> passed
- `python3 -m pytest -q` after alias routing -> `25 passed in 3.80s`
- `python3 tools/validate_session.py` -> `router session validation: OK`
- `git diff --check` -> passed
- `python3 -m pytest -q` after alias handling -> `25 passed in 4.18s`
- `python3 tools/validate_session.py` -> `router session validation: OK`
- `git diff --check` -> passed
- `python3 -m pytest -q` after chat validation -> `24 passed in 3.69s`
- `python3 tools/validate_session.py` -> `router session validation: OK`
- `git diff --check` -> passed
- `python3 -m pytest -q` after chat-completions boundary -> `22 passed in 2.66s`
- `python3 tools/validate_session.py` -> `router session validation: OK`
- `git diff --check` -> passed
- `python3 -m pytest -q` after deployment catalog -> `18 passed in 2.13s`
- `python3 tools/validate_session.py` -> `router session validation: OK`
- `git diff --check` -> passed
- `python3 -m pytest -q` after HTTP response hardening -> `16 passed in 2.12s`
- `python3 tools/validate_session.py` -> `router session validation: OK`
- `git diff --check` -> passed
- `python3 -m pytest -q` after diagnostics endpoints -> `16 passed in 2.14s`
- `python3 tools/validate_session.py` -> `router session validation: OK`
- `git diff --check` -> passed
- `python3 -m pytest -q` after cache metadata -> `15 passed in 1.75s`
- `python3 tools/validate_session.py` -> `router session validation: OK`
- `git diff --check` -> passed
- `python3 -m pytest -q` after request-size hardening -> `14 passed in 1.60s`
- `python3 tools/validate_session.py` -> `router session validation: OK`
- `git diff --check` -> passed
- HTTP smoke test recorded `operational_failure` for `general`; the following
  route returned `no eligible deployment for fast_general`, proving cooldown
  exclusion rather than retrying the dead deployment.
- `python3 -m pytest -q` after HTTP integration tests -> `13 passed in 1.11s`
- `python3 tools/validate_session.py` -> `router session validation: OK`
- `git diff --check` -> passed
- `python3 -m pytest -q` after failure handling -> `11 passed in 0.06s`
- `python3 tools/validate_session.py` -> `router session validation: OK`
- `git diff --check` -> passed
- `python3 -m pytest -q` after phase normalization -> `10 passed in 0.08s`
- `python3 tools/validate_session.py` -> `router session validation: OK`
- `git diff --check` -> passed
- `python3 -m pytest -q` after session stickiness -> `8 passed in 0.08s`
- `python3 tools/validate_session.py` -> `router session validation: OK`
- `python3 -m py_compile router_core.py router_server.py router_state.py` -> passed
- `python3 -m pytest -q` after outcome telemetry -> `9 passed in 0.08s`
- `python3 tools/validate_session.py` -> `router session validation: OK`
- `python3 -m py_compile router_core.py router_server.py router_state.py` -> passed
- `git diff --check` -> passed

## Latest checkpoint

- Added replay report distribution metrics in `trace_replay.py`: selected
  deployment counts and baseline match rate, while retaining prompt-free JSON
  output.
- Extended `tests/test_trace_replay.py` to cover both metrics.
- `python3 -m pytest -q` -> `32 passed in 4.23s`
- `python3 tools/validate_session.py` -> `router session validation: OK`
- `git diff --check` -> passed

## Current batch checkpoint

- Fixed compatibility SSE chunking to preserve exact whitespace, newlines, and
  code indentation. Added a multi-chunk reconstruction regression assertion.
- `python3 -m pytest -q tests/test_router_server.py` -> 8 passed.
- `git diff --check` -> passed. These changes are not yet committed.
- Added `stream_execute()` and server wiring for native upstream SSE streams;
  failover occurs before the first event, while completion-only adapters retain
  the compatibility stream fallback.
- Full suite after native streaming wiring -> 38 passed.
- Added native provider stream server regression coverage and environment
  adapter wiring tests; suite now reports 41 passed.
- Added lightweight GitHub Actions test workflow at
  `.github/workflows/test.yml` and documented native-stream operations.
- Added `tools/verify_release.py` as a repeatable provider-free release gate
  for local operators.
- Release gate initially caught an indentation error in the new provider test;
  fixed it, then `python3 tools/verify_release.py` passed with 42 tests,
  session validation OK, resource smoke health/route 200, and compilation OK.
- Hardened native streaming failover by priming each provider iterator before
  sending headers; a provider that fails before its first event now permits the
  next eligible deployment to run. Added regression coverage; suite now has 43
  tests.
- Corrected telemetry semantics so route inspection records decisions without
  falsely recording successful completions; chat completion success is recorded
  only after an adapter response is emitted.

- Added `.gitignore` for generated Python/test/runtime artifacts.
- Added `tools/resource_smoke.py`, a provider-free startup, health, route, and
  RSS smoke test; latest result was health 200, route 200, RSS 24748 KB.
- Added `run_router.sh` as a deployment-safe local entrypoint.
- Added `ROUTER_HOST` and `ROUTER_PORT` support to the server entrypoint.
- Expanded README replay metrics and resource-smoke documentation.
- No VPS files were changed and no credentials or provider calls were added.
- Removed a duplicate server import and made `ROUTER_STATE_DB` configure
  durable SQLite state; default remains in-memory for tests.
- Documented durable-state setup in README and `docs/DEPLOYMENT.md`.
- `python3 -m pytest -q` -> `32 passed in 4.22s`
- `python3 tools/resource_smoke.py` -> health 200, route 200, RSS 24748 KB
- `python3 tools/validate_session.py` -> `router session validation: OK`
- Added compatibility SSE streaming for `/v1/chat/completions` and tests;
  upstream token streaming remains provider-adapter work.
- Added `tools/calibrate_outcomes.py` and a redacted sample outcome file.
- Added generic OpenAI-compatible adapter configuration, systemd packaging,
  OpenCode signal documentation, and concurrency smoke tooling.
- Found and fixed a real SQLite concurrency bug by locking read operations;
  `tools/concurrency_smoke.py --requests 20 --workers 4` -> 20 successful.
- Latest suite after streaming and concurrency fixes -> 34 passed.
- Wired environment-configured OpenAI-compatible adapters into server startup;
  unconfigured profiles remain provider-free.
- Added local install/rollback scripts and `docs/OPERATIONS.md`; neither script
  was run against VPS 40.
- Final local checks for this batch: 34 passed, resource smoke health/route 200,
  session validation OK.
- Added mocked HTTP tests for the OpenAI-compatible adapter, upstream SSE line
  handling, and missing-endpoint failures.
- Added `integrations/opencode_headers.py` with bounded signal-only header
  construction and tests.
- Latest suite after original-plan integration additions -> 38 passed.

## VPS 40 canary checkpoint

- SSH access to `ubuntu@40.233.101.233` was confirmed with the existing Oracle
  key. Existing gateway services and ports were inspected without stopping or
  modifying them.
- The clean `cc65db6` release was copied to
  `/home/ubuntu/ai-workload-router/releases/cc65db6`.
- A user-level `ai-workload-router-canary.service` was installed and enabled.
  It runs the standard-library server on `127.0.0.1:8123`, uses durable state
  at `/home/ubuntu/ai-workload-router/state/router-state.sqlite3`, and has a
  160 MiB memory limit.
- VPS checks: health 200; route 200; 100/100 concurrent routes with 8 workers;
  resource smoke health/route 200 and RSS 25360 KB; persisted decision/cache
  data survived restart; existing `llm-gateway.service` stayed active on port
  8000.
- The first route probe used an invalid payload and returned 400; the supported
  payload was then verified successfully. No source or provider credentials
  were changed on the VPS.
- The canary was then configured with a VPS-only gateway credential and a
  runtime catalog mapping `general` to `chat-smart`; the credential was never
  printed or copied into the repository. Ordinary chat and upstream SSE both
  returned 200. `auto-code` returned the intended 503 because the current
  runtime has no configured code deployment.

## VPS structured-forwarding checkpoint

- Copied the forwarding release to
  `/home/ubuntu/ai-workload-router/releases/forwarding-20261008` and updated
  only the user-level canary unit. The existing gateway service and port 8000
  remained active throughout.
- VPS checks after restart: health 200, route 200 selecting `general`, ordinary
  chat 200 with an OpenAI-shaped response, upstream SSE 200 with `[DONE]`, and
  `auto-code` 503. A tool/structured request correctly returned 503 because no
  eligible `fast_tool` deployment is configured.
- The canary remains loopback-only on port 8123 with durable state and the same
  memory cap. No credential was printed, copied into the repository, or added
  to the release archive.

## Structured provider forwarding checkpoint

- Added `ProviderRequest` and `ProviderCompletion` to preserve full message
  history, tool definitions/calls, structured-output fields, standard options,
  finish reasons, and provider usage metadata across the adapter boundary.
- The router still owns aliases, selected deployment model, stream mode, and
  router-only telemetry fields; those are not forwarded as upstream controls.
- Added adapter and HTTP boundary regression tests for tool calls, response
  format, options, usage, and finish reasons.
- `python3 -m pytest -q` -> `46 passed in 5.22s`
- A temporary local server passed `python3 tools/concurrency_smoke.py --requests
  20 --workers 4` with 20/20 successful requests; session validation,
  compileall, and `git diff --check` also passed.

## VPS code-profile checkpoint

- The canary release `forwarding-20261008` now maps `tool`, `implement`, and
  `debug` to the existing gateway's `coding-smart` model and `verify` to
  `coding-fast`. The old gateway and port 8000 were not changed.
- Provider-backed probes passed for tool-call output, structured JSON output,
  implementation, verification, debug, and native SSE. The router preserved
  deployment metadata and returned the provider's tool-call/usage fields.
- An explicit `operational_failure` outcome caused the affected deployment to
  become cooldown-excluded with HTTP 503; the canary was restarted afterward
  and returned to active state.
- The VPS has no `opencode` executable. Direct canary requests using the
  documented OpenCode headers kept the implementation deployment stable while
  repeated actions raised bounded escalation from 0 to 2.
- Added stable `GET /v1/models` discovery for `auto`, `auto-chat`, and
  `auto-code`, plus explicit rejection of non-object JSON bodies across route,
  chat, and outcome endpoints.
- Credentials remain only in the VPS mode-0600 environment file. No secret was
  printed, copied locally, or included in the release archive.

## Next action for a new implementation session

Run provider-specific timeout/429/5xx/quota checks and verify the deployed
OpenCode signal/header path against a real client session. Keep the existing
gateway as fallback and do not perform reverse-proxy cutover until those checks
pass.

## VPS compatibility-hardening checkpoint

- Deployed `hardening-20261008` to the loopback canary and changed only the
  canary user unit; the existing gateway remained active on port 8000.
- Added prompt-free `GET /v1/models` discovery for `auto`, `auto-chat`, and
  `auto-code`. The canary returned all three aliases with HTTP 200.
- Route, chat, tool-call, and verification probes passed after restart. A JSON
  array sent to the route endpoint returned bounded HTTP 400 instead of raising
  an unhandled server exception. Legacy gateway health remained HTTP 200.
- Local verification: 48 tests, session validation, compilation, and diff
  checks passed.
- `python3 tools/verify_release.py` passed with 48 tests, resource smoke health
  and route HTTP 200, and approximately 25 MiB RSS. A temporary local server
  then passed `python3 tools/concurrency_smoke.py --requests 20 --workers 4`
  with 20/20 successful requests.

Next: exercise provider-specific error injection and quota behavior without
cutting over traffic; keep the old gateway as fallback.

## Provider-failure hardening checkpoint

- `ProviderError` now classifies HTTP 429, timeout, 4xx, 5xx, network, invalid
  response, and empty-stream failures using bounded categories and optional
  status codes; upstream error text is never copied into telemetry.
- Same-capability failover reports failed deployment/category pairs through a
  callback and records operational failures while preserving a successful later
  attempt. Native malformed SSE closes after any already-sent events without a
  fake success record.
- Added regression coverage for HTTP/transport classification, failure
  telemetry, failover reporting, and malformed native SSE. Targeted suite:
  `25 passed`.

Next: run provider-specific error injection and quota behavior without cutting
over traffic; keep the old gateway as fallback.

## VPS provider-failure release checkpoint

- Deployed `failures-20261008` to the loopback canary and changed only the
  canary user unit; the existing gateway remained active on port 8000.
- Health and model discovery returned HTTP 200. Chat returned HTTP 200, a
  forced tool-choice request returned a `tool_calls` response through the
  `tool` deployment, native SSE returned HTTP 200 with `[DONE]`, and a
  non-object route body returned bounded HTTP 400.
- The explicit operational-failure probe returned HTTP 200 and diagnostics
  remained available with operational-failure telemetry. Legacy gateway health
  remained HTTP 200 and the canary service remained active.
- Full verification passed: 53 tests, release gate, compilation, diff checks,
  and resource smoke at approximately 25 MiB RSS.

Next: exercise provider-specific error injection and quota behavior without
cutting over traffic; keep the old gateway as fallback.

## Model-group chain and cache-routing checkpoint

- Added explicit deployment groups and chain priorities. Fresh requests begin
  at the lowest configured priority, while explicit continuation requests may
  stay on an eligible preferred deployment. Explicit model groups remain hard
  boundaries; automatic classification can use metadata groups such as
  `chat-fast` when no capability-named group exists.
- Added per-deployment cache hints with TTLs, cache-savings break-even logic,
  and a bounded warm-cache wait for short rate-limit responses. Timeout,
  upstream, network, and long-quota failures fail over without waiting.
- Added error-aware cooldowns, Retry-After and rate-reset parsing, cooldown
  recovery after a successful retry, bounded retry response headers, and
  provider-failure telemetry without upstream error text.
- Added read-only `provider_metadata.py` loading from the
  `llm-provider-manager` SQLite database. It imports ordered free/no-key
  provider/model metadata, endpoint labels, capability hints, and environment
  prefixes without reading credentials. `ROUTER_DEPLOYMENTS` remains the
  explicit-config precedence; `GET /v1/router/groups` exposes safe chain
  diagnostics.
- Added regression coverage for chain priority, strict groups, automatic
  metadata groups, cache TTL/hints, warm-cache retries, Retry-After parsing,
  cooldown recovery, and continuation-aware native streaming.
- Local verification: `python3 tools/verify_release.py` passed 68 tests,
  session validation, resource smoke health/route 200 at 26040 KB RSS, and
  compilation. A 50-request/8-worker concurrency smoke passed 50/50, the
  provider metadata smoke loaded 11 `chat-fast` deployments, and the Python
  secret-pattern scan found no matches.
- The standalone batch is committed locally as `60c3822`; it was synchronized
  into the vendored PR branch as `747cca4` and pushed to `feat/workload-router-p0`.
  No VPS deployment, gateway change, task-board duplicate, or credential access
  was performed.

## Next action for a new implementation session

Review PR #315 and its merge policy. Before enabling provider-manager metadata
on VPS 40, run actual provider-specific quota/429/timeout/5xx tests and a real
OpenCode client session; keep the current gateway and loopback canary as
fallback.

## Provider-fault and OpenCode compatibility checkpoint

- Added `tests/test_provider_faults.py`, a local scripted OpenAI-compatible HTTP
  provider harness covering 429/Retry-After, quota reset, timeout, 4xx/5xx,
  malformed or truncated SSE, empty streams, same-group failover, cooldown
  recovery, cache-aware retry, option forwarding, and usage/cache telemetry.
  It uses no live provider, quota, or credential.
- Hardened provider transport normalization for incomplete reads, malformed
  events, finite retry/quota hints, closeable primed streams, and post-header
  stream failure handling. Updated the stream-execution regression to the
  structured provider-stream contract.
- Full verification passed: `python3 -m pytest -q` -> 89 passed (one existing
  warning); `python3 tools/verify_release.py` -> 89 passed, session validation
  OK, resource smoke health/route 200 at 26116 KB RSS, and compilation OK.
- OpenCode 1.18.35 was run through an SSH tunnel to the loopback canary using a
  temporary provider config. The real request reached the router, but the
  existing gateway's upstream chain exhausted/cooldowned during the large
  real tool-context request; a sanitized equivalent completed. No tools were
  stripped in the router, and no gateway or provider configuration was changed.
- This is not a successful real OpenCode session yet. Re-run after provider
  availability is restored, then test a real tool loop before any cutover.

## Next action for a new implementation session

Push this fault-hardening checkpoint into PR #315, then re-run a real OpenCode
tool loop when the existing gateway has a usable upstream. Keep the existing
gateway as fallback and do not perform reverse-proxy cutover until that client
test and provider-specific checks pass.

## Large-request and adapter-lifecycle checkpoint

- Normalized unexpected adapter exceptions during completion, stream creation,
  first-event priming, warm-cache retry, and later stream iteration into bounded
  provider error classes so same-group failover and cooldown behavior remains
  deterministic.
- Corrected generic `RateLimit-Reset` epoch parsing and added regression tests
  for OSError/ValueError completion and stream failures, including pre-header
  stream failover and health categorization.
- Added an OpenCode-shaped provider-free regression with 69 tools, a 78,297
  character system message, `tool_choice=auto`, `stream_options.include_usage`,
  and `max_tokens=32000`; it verifies an intact serialized request over 150 KB
  and a complete SSE response.
- Verification passed: `python3 -m pytest -q` -> 95 passed (one existing
  warning); `python3 tools/verify_release.py` -> 95 passed, session validation
  OK, resource smoke health/route 200 at 26224 KB RSS, and compilation OK.
  A temporary local server passed `python3 tools/concurrency_smoke.py
  --requests 100 --workers 8` with 100/100 successful requests.
- The real OpenCode client test remains externally blocked by upstream provider
  availability in the legacy gateway; no router-side tool stripping, gateway
  edit, VPS deployment, or credential change was made.

## Next action for a new implementation session

Push this checkpoint to PR #315 and re-run a real OpenCode tool loop after the
legacy gateway has a usable upstream. Keep the existing gateway as fallback;
then review/merge PR #315 before gated integration work.

## Stream iterator lifecycle checkpoint

- Audited the provider execution boundary after the adapter-failure release and
  found that an iterator acquired before the first stream event could remain
  open when priming raised `StopIteration`, `ProviderError`, or an unexpected
  exception.
- `_prime_stream()` now closes an acquired iterator on every pre-header failure;
  successful priming still leaves it open for `_prepend_stream()`, which owns
  later iteration and final cleanup.
- Added a regression proving a pre-header iterator is closed while same-group
  failover proceeds. Targeted tests passed: `10 passed`.
- Full verification passed: `python3 -m pytest -q` -> 96 passed (one existing
  warning); `python3 tools/verify_release.py` -> 96 passed, session validation
  OK, resource smoke health/route 200 at 26132 KB RSS, and compilation OK.
  A temporary local server passed `python3 tools/concurrency_smoke.py
  --requests 100 --workers 8` with 100/100 successful requests.
- No gateway/VPS files, provider credentials, live quotas, or deployment state
  were touched.

## Next action for a new implementation session

Run the full release/concurrency/secret checks, push this lifecycle fix to PR
#315, and wait for a usable upstream before repeating the real OpenCode tool
loop. Keep the existing gateway as fallback and do not cut over traffic.

## Checkpoint protocol

After every milestone, record changed files, exact commands and results,
unresolved issues, and the next concrete action in this file. Keep checkpoints
small enough that another agent can resume from this file alone.
