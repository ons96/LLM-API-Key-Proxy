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

Deployment target is VPS 40 (`<VPS40_IP>`). This workspace remains the
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

- SSH access to `ubuntu@<VPS40_IP>` was confirmed with the existing Oracle
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

## Local New-API sidecar release checkpoint

- The pushed `feat/newapi-sidecar` branch remains clean and tracks its remote.
- `python3 tools/verify_release.py` passed: 231 tests, session validation,
  resource smoke (health/route 200, approximately 25 MiB RSS), and compilation.
- `python3 tools/resource_smoke.py` passed: health 200, route 200, RSS 25568 KB.
- A temporary local router passed
  `python3 tools/concurrency_smoke.py --requests 50 --workers 8` with 50/50
  successful requests.
- `git diff --check` passed. No New-API binary, live provider, OpenCode client,
  credentials, VPS, or cutover was exercised in this sandbox.

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

## Compatibility stream and cleanup checkpoint

- Hardened cleanup across provider HTTP responses, primed iterators, native
  streams, and compatibility streams so cleanup failures cannot mask the
  original provider result or failover error.
- Compatibility SSE now preserves tool calls and function calls when a
  completion-only adapter returns `content: null`; it emits the provider finish
  reason and usage event while still completing with `[DONE]`.
- Added regressions for cleanup exceptions and textless completion-only tool
  calls. Targeted tests passed: 39 passed.
- Full verification passed: `python3 -m pytest -q` -> 99 passed (one existing
  warning); `python3 tools/verify_release.py` -> 99 passed, session validation
  OK, resource smoke health/route 200 at 26236 KB RSS, and compilation OK.
  A direct resource smoke also returned health/route 200 at 26052 KB RSS.
- No gateway/VPS files, provider credentials, live quotas, or deployment state
  were touched.

## Next action for a new implementation session

Run the full release/concurrency/secret checks, push this compatibility and
cleanup checkpoint to PR #315, and keep the existing gateway as fallback while
real OpenCode tool-loop validation waits for upstream availability.

## Finite cooldown metadata checkpoint

- Hardened `HealthRegistry` and `/v1/router/outcome` parsing against NaN,
  positive infinity, negative infinity, malformed, and negative retry/quota
  hints. Invalid values now fall back safely instead of poisoning cooldown
  timestamps or response headers.
- Added direct core and HTTP regression coverage proving non-finite failure
  metadata leaves the normal bounded cooldown intact.
- Full verification passed: `python3 -m pytest -q` -> 103 passed (one existing
  warning); `python3 tools/verify_release.py` -> 103 passed, session validation
  OK, resource smoke health/route 200 at 26044 KB RSS, and compilation OK.
  Direct resource smoke returned health/route 200 at 26132 KB RSS, and a
  temporary local server passed `python3 tools/concurrency_smoke.py
  --requests 100 --workers 8` with 100/100 successful requests.
- No gateway/VPS files, provider credentials, live quotas, or deployment state
  were touched.

## Next action for a new implementation session

Run the staged secret scan, commit and push this finite-metadata hardening to
PR #315, then continue with review/merge preparation while the real OpenCode
tool-loop test remains blocked on upstream availability.

## Finite deployment catalog checkpoint

- Hardened `router_config.load_deployments()` against JSON NaN and infinity in
  latency, success rate, cache TTL, and quota reset fields. Invalid non-finite
  values now trigger the existing safe-default catalog fallback instead of
  entering scoring, freshness, or quota-reset calculations.
- Added parametrized configuration regressions for NaN and positive infinity;
  parsing remains credential-free and provider-free.
- Full verification passed: `python3 -m pytest -q` -> 111 passed (one existing
  warning); `python3 tools/verify_release.py` -> 111 passed, session validation
  OK, resource smoke health/route 200 at 26248 KB RSS, and compilation OK.
  Direct resource smoke returned health/route 200 at 26188 KB RSS, and a
  temporary local server passed `python3 tools/concurrency_smoke.py
  --requests 100 --workers 8` with 100/100 successful requests.
- No gateway/VPS files, provider credentials, live quotas, or deployment state
  were touched.

## Next action for a new implementation session

Run the staged secret scan, commit and push this catalog-validation fix to PR
#315, then continue review/merge preparation while the real OpenCode tool-loop
test remains blocked on legacy upstream availability.

## Finite cache observation checkpoint

- Hardened `CacheHint.is_fresh()` against non-finite timestamps, TTLs, and
  cached-token values; invalid or non-positive cache evidence is never treated
  as fresh routing state.
- Hardened `RouterState.record_cache()` to reject non-finite, fractional,
  negative, or otherwise invalid cache counts/TTLs. `cache_hints()` skips
  corrupt legacy SQLite rows instead of poisoning routing decisions.
- Added core and SQLite regressions for NaN, positive/negative infinity,
  invalid cache counts, and corrupt persisted observations.
- Full verification passed: `python3 -m pytest -q` -> 127 passed (one existing
  warning); `python3 tools/verify_release.py` -> 127 passed, session validation
  OK, resource smoke health/route 200 at 26328 KB RSS, and compilation OK.
  Direct resource smoke returned health/route 200 at 26232 KB RSS, and a
  temporary local server passed `python3 tools/concurrency_smoke.py
  --requests 100 --workers 8` with 100/100 successful requests.
- No gateway/VPS files, provider credentials, live quotas, or deployment state
  were touched.

## Next action for a new implementation session

Run the staged secret scan, commit and push this cache-state hardening to PR
#315, then continue review/merge preparation while the real OpenCode tool-loop
test remains blocked on legacy upstream availability.

## Provider metadata boundary checkpoint

- Hardened the read-only provider-manager metadata loader against non-finite or
  malformed context-window, tokens-per-second, and chain-priority values. Safe
  finite defaults now keep one corrupted metadata row from aborting the whole
  eligible chain catalog.
- Added a provider-free SQLite regression with one valid row and one corrupted
  row; the valid deployment remains available and the corrupted deployment gets
  bounded context, latency, and priority defaults.
- Verification passed: `python3 -m pytest -q` -> 128 passed (one existing
  warning); `python3 tools/verify_release.py` -> 128 passed, session validation
  OK, resource smoke health/route 200 at 26128 KB RSS, and compilation OK.
  Direct resource smoke returned health/route 200 at 26292 KB RSS, and a
  temporary local server passed `python3 tools/concurrency_smoke.py
  --requests 100 --workers 8` with 100/100 successful requests.
- No gateway/VPS files, provider credentials, live quotas, or deployment state
  were touched.

## Next action for a new implementation session

Run the staged secret scan, commit and push this provider-metadata hardening to
PR #315, then finish review/merge preparation while real OpenCode tool-loop
validation remains blocked on legacy upstream availability.

## Checkpoint protocol

After every milestone, record changed files, exact commands and results,
unresolved issues, and the next concrete action in this file. Keep checkpoints
small enough that another agent can resume from this file alone.

## New-API replacement target clarification

The final goal is deployment on VPS-40 and replacement of the current LLM
gateway, only after the new stack is tested and confirmed working. Parallel
operation is the validation stage. Do not infer immediate deployment/cutover
permission from this goal, and preserve the old gateway as rollback.

The active implementation worktree is
`LLM-API-Key-Proxy-worktrees/wt-newapi-sidecar`, branch `feat/newapi-sidecar`.
The quota ledger, scorer, accounted runtime and opt-in HTTP boundary are written.
The admin/catalog module is in progress and not yet integration-verified.
No production gateway or VPS files have been changed by this implementation.

Verified commands:

- `python3 -m pytest -q tests/test_newapi_runtime.py tests/test_quota_ledger.py tests/test_route_policy.py`
  -> 46 passed.
- `python3 -m pytest -q tests/test_newapi_server.py tests/test_newapi_runtime.py tests/test_provider_http_adapter.py`
  -> 53 passed.
- The broader suite found one upload/early-close race in the oversized-body test
  (198 passed, 1 failed). The test was corrected to send headers only; the HTTP
  suite then passed 15 tests. Re-run the full suite before claiming it passes.

Next: finish and test the management/catalog preflight, then verify against a
fresh isolated New-API instance. Real-provider/OpenCode and VPS-40 validation,
client compatibility and tested rollback remain required before replacement.

## New-API OmniRush handoff checkpoint - 2026-11-03

The continuation handoff is now at:
`/home/osees/CodingProjects/temp-omnirush/HANDOFF-newapi-sidecar.md`.
Use it from OmniRush's launch directory, but run implementation commands in this
isolated worktree's `workload-router/` directory. Preserve all uncommitted files.

Implementation branch/base HEAD: `feat/newapi-sidecar`, `fde9784`.
Migration docs branch/base HEAD: `work/newapi-sidecar`, `078ce55`.
No commit, push, production deployment or VPS modification occurred.

The catalog, benchmark, authorized management-proof flow and isolated verifier
are written. A directly inspected report at
`/tmp/opencode/newapi-proof-7uclf86q/report.json` records a passing real-binary
local proof: chat/coding/SSE 200, exactly one 429 attempt then failover, three
fixture catalogs, additive disposable OpenCode refresh, paid-route exclusion and
all five rankings changed by grades. Scripted route one graded 0/5; route two
graded 5/5. These are LOCAL FIXTURES, not live-provider validation.

The benchmark-isolation fix is applied in the verifier. Its history resets are
restricted to a fresh owned fixture database; do not weaken production accounting
or copy these resets into production. Earlier conversation summaries saying this
fix was unapplied are stale.

Last recorded full suite before catalog/benchmark additions: 199 passed.
Later focused benchmark/catalog suite: 31 passed. A new full-suite result is
still needed; repeated `rtk pytest` runs collected no tests and prove nothing.
`git diff --check` passed immediately before writing this handoff.

Outstanding: intermittent early 503 proof runs are not causally resolved;
production `serve()` supplies no channel-key proof callback and therefore fails
closed when keys are hidden; real catalog/operator orchestration, OpenCode tool
loops, client compatibility, lint/type checks and VPS resource/restart/rollback
gates remain. The old gateway must stay live until replacement is confirmed.

Next: run `python3 -m pytest tests -o addopts= --quiet` once in this directory;
record the actual result, then fix the supported production startup proof path.

## New-API startup authorization checkpoint - 2026-11-03

- Ran the handoff's exact full suite: `230 passed, 1 warning in 11.54s`.
- Ran the handoff's exact `git diff --check`: passed.
- The exact isolated verifier was attempted twice. It first lacked `ip`; after
  installing the free `iproute2` package, it stopped because this sandbox does
  not mount `/home/osees/new-api/bin/new-api` or `/tmp/opencode`.
- Added `NEWAPI_CHANNEL_KEY_PROOF_COMMAND` support to `newapi_server.py`.
  The configured helper receives only a channel ID on stdin, returns one
  authorized proof token on stdout, and is bounded by a 30-second subprocess
  timeout. No password, backup code, enrollment secret, or proof is stored or
  passed in argv. If unset, startup remains fail-closed.
- Added `test_channel_key_proof_helper_uses_stdin_and_validates_output`.
  Targeted server tests: `16 passed`; Python compilation passed.
- No deployment, credential, gateway, VPS, commit, or push was performed.

Next: document the operator proof-helper contract, run the full suite again,
and rerun the exact isolated verifier in an environment containing its binary
and artifact parent.


## 2026-11-03 resume verification

- GitHub state and task-board discovery completed after authenticating `gh`; no lost sidecar work was found.
- `PYTHONPATH=. uv run --with pytest pytest tests -o addopts= --quiet` passed: **231 tests**.
- The existing `Unknown config option: asyncio_mode` warning remains because pytest-asyncio is not installed.
- The real New-API verifier is blocked on this machine because `/home/osees/new-api/bin/new-api` is absent. No deployment, credential, or live-provider checks were performed.
- Project instructions now prefer `uv` for Python tooling, with pip only as a documented fallback.
- Exact pinned New-API `v1.0.0-rc.36` verifier passed.
  The SHA-256 matched `5d9b254721c205f4f442848a4af5b58617a8c0ee090366732cfff90455e957f3`; local fixture checks covered chat, coding, SSE, 429 failover, catalog/OpenCode refresh, benchmark ranking changes, paid exclusion, and RSS.
- The verifier used only disposable local state and scripted providers; VPS-40, LiteLLM, live providers, and credentials remain untouched.
- Added the operator-facing channel proof-helper contract at `docs/channel-proof-helper.md` and malformed-output regression coverage.
- Full provider-free suite after this milestone: `PYTHONPATH=. uv run --with pytest pytest tests -o addopts= --quiet` -> **234 passed**, one existing `asyncio_mode` warning.

## VPS-40 Stage 1 read-only inventory - 2026-11-03

- Host `ubuntu@40.233.101.233` (`llm-gateway`): Ubuntu 24.04.4 LTS, amd64, 2 vCPU,
kernel 6.17.0-1018-oracle, uptime 104 days, load 0.22.
- Resources: 954 MiB RAM (826 used, 128 available at sample), 5 GiB swap
(2 GiB swapfile + 3 GiB zram at ~494 MiB used), 48 GiB disk at 31% (33 GiB
free). Passwordless sudo available; python3.12; no `uv`.
- `llm-gateway.service` active on `*:8000`, socket-activated via
`llm-gateway.socket`. Effective `MemoryMax=512M` from drop-in `memory.conf`
(base unit declares 400M). `MemoryCurrent` sampled 77-259 MiB, `NRestarts=0`,
started 22:30:11 UTC roughly two minutes before this inventory.
- Gateway drop-ins: `smart-router.conf` (`ROUTER_ENABLED=1`, `ROUTER_DRY_RUN=0`)
and `workload-router-shadow.conf` (`WORKLOAD_ROUTER_DRY_RUN=1`). The VPS checkout
contains `src/proxy_app/workload_router_shadow.py` and `router_wrapper.py`.
- **Pre-existing New-API install, absent from the handoff:** `new-api.service`
active since 2026-10-20 on `*:8001`, root-owned binary in `/opt/new-api`
(536M total), pinned `v1.0.0-rc.36`, sha256 matches the documented
`5d9b2547...e957f3`, user `newapi`, RSS ~58-64 MiB.
`MemoryMax` was `infinity` at inventory time and was capped to 256M in the
gap-closing step below.
Documented in `docs/new-api-deploy.md` (task-board #862, EPIC #876).
- New-API support units: healthcheck every 5 min (active; exit status 2 means
unhealthy/deprioritized), watchdog every 5 min (active), backup daily (active,
last run 22h ago, 347M of weekly databases), discovery daily (enabled but
inactive).
- `one-api.db` is 41 MiB. Read-only counts: 3 channels (all type 1, enabled,
keyed), 29 abilities, 14,616 tokens, 1 user. Credential values were not read.
- Existing loopback canary: user unit `ai-workload-router-canary.service` on
`127.0.0.1:8123` from `releases/failures-20261008`, RSS 1.8 MiB,
`MemoryMax=160M`. This is the older router, not this sidecar.
- Other listeners: caddy `10.0.0.19:80/443` (reverse-proxies 8000 and 8080),
ngrok `127.0.0.1:4040` tunnelling 8000, cliproxyapi 8317, zeroclaw 8080,
docrender 8091, caddy admin 2019. Ports 8090, 8095 and 8081 bind `0.0.0.0`
with no iptables rule.
- Firewall: no `ufw` installed; iptables defaults to ACCEPT with explicit
loopback/Tailscale-only admission for 8000 and 8001, an SSH allowlist
(`BB-SSH`), and fail2ban running. No failed systemd units.
- No sidecar install, no sidecar unit, and no `newapi-channel-proof` helper
exist anywhere on the host.
- VPS gateway checkout `/home/ubuntu/LLM-API-Key-Proxy` is on
`work/gateway-catalog` (HEAD `22f96ad`) with four modified config files.
- Rollback assets: `~/gateway-backup-2026-10-20-vps.tgz`, `~/gateway-backups/`,
`~/backups/` (historical commit pins), `~/ai-workload-router/releases/` (four
releases), `~/LLM-API-Key-Proxy-archive/`, `/opt/new-api/backups/`.
- Nothing was installed, stopped, restarted, reconfigured, or cut over. Only
reads, unauthenticated `GET`s, and read-only SQLite counts were used.
- Stage 2 is therefore substantially implemented already, with two deviations
from the staging spec: New-API binds all interfaces rather than loopback, and
it has no memory bound. Stage 3 remains blocked on the proof helper.

## VPS-40 Stage 2 gap-closing - 2026-11-03

- Authorized next action executed against the pre-existing New-API install. No
reinstall, no replacement, no credential change, and no traffic cutover.
- Added `/etc/systemd/system/new-api.service.d/memory.conf` with
`MemoryMax=256M`. Before: `MemoryMax=infinity`, `MemoryCurrent` sampled at
101 MiB. `daemon-reload` and `systemctl restart new-api` both exited 0.
- Post-restart: `is-active=active`, `Result=success`, `MemoryMax=268435456`,
`MemoryCurrent` 45 MiB, new PID, `ActiveEnterTimestamp` 22:36:19 UTC,
`is-enabled=enabled` (boot persistence unchanged).
- `:8001/api/status` and `:8001/` returned 200 on the first probe; healthcheck,
watchdog and backup timers all remained active; `one-api.db` plus WAL are
intact and were checkpointed by the restart.
- `llm-gateway.service` stayed `active` with `/health` 200 throughout. Its unit
and drop-ins were not touched.
- The bind deviation was closed by documentation, not rebinding: `*:8001` is
intentional so the dashboard stays reachable over Tailscale, and iptables
(not the bind address) is the access control. No `ufw` is installed.
- Rollback for this change:
`sudo rm /etc/systemd/system/new-api.service.d/memory.conf && sudo systemctl
daemon-reload && sudo systemctl restart new-api`.
- Admin-password state remains unverified. Confirming it would require
handling credentials, so it stays an operator step; nothing was tested or
printed.
- Docs reconciled: `HANDOFF-newapi-sidecar.md` records the VPS-40 inventory and
qualifies the "no VPS deployment" claim to this sidecar;
`docs/VPS-CANARY-HANDOFF-PROMPT.md` now lists `docs/new-api-deploy.md`, states
that Stage 2 is gap-closing, records the unverified admin password, and keeps
Stage 3 blocked on the proof helper.

## Gateway compatibility inventory - 2026-11-03

- Added `docs/gateway-compatibility.md`. Read-only: the live gateway's
`/openapi.json` answered unauthenticated with 30 paths, and virtual-model names
came from `config/virtual_models.yaml` keys only.
- HTTP surface: 2 exact matches (`/v1/chat/completions`, `/v1/models`), 1 renamed
(`/health` -> `/healthz`), and 27 absent of 30. `/v1/models` is only partial:
the sidecar returns its 3 profile aliases, not a model list.
- Model aliases: zero overlap. The sidecar accepts only `auto-coding`,
`auto-code` and `auto-chat` (`newapi_runtime.py:88`); the gateway serves 19
virtual models (`auto`, `title-fast`, `safe-coding`, `coding-*`, `chat-*`,
`glm5-elite`, `p0-minimax-m3`, six `agent-*`). Every client pinned to a gateway
alias would receive HTTP 400 after a naive cutover.
- Recorded three options without choosing: alias translation in the sidecar,
permanent dual-path, or a Caddy front split. Alias translation is smallest but
changes the documented public profile contract, so it needs a decision.
- Client route *usage* is still unknown; no request log was reviewed. Only
availability is established.
- This closes the "inventory the old gateway's API routes and virtual aliases"
half of gate 4. Real-provider validation, OpenCode tool loops and cutover remain
after the sidecar can start.

## Channel-key proof helper - 2026-11-03

- Added `tools/newapi_channel_proof.py`, the operator-controlled helper the
startup audit needs whenever New-API hides channel keys. Two credential modes:
`interactive` (2FA or backup code at the controlling terminal) and `totp`
(base32 seed in a 0600 operator-owned file). Choosing between them is
configuration, not code.
- Configuration is environment-only (`NEWAPI_PROOF_ADMIN_TOKEN`,
`NEWAPI_PROOF_ADMIN_USER_ID`, `NEWAPI_PROOF_BASE_URL`,
`NEWAPI_PROOF_TOTP_SECRET_FILE`); no credential ever appears in argv. Hardening:
loopback-only base URL, redirects refused, 10-second HTTP timeout, exactly one
stdin line, 0600 seed enforced, and diagnostics that never echo a code or proof.
- The helper returns the proof without redeeming it. The sidecar redeems it
against `POST /api/channel/<id>/key`, so consuming it in the helper would break
that call.
- Added `tests/test_channel_proof_helper.py` (16 tests) against a loopback fake
New-API covering token shape, `scope=channel.key.read`, `context.channel_id`,
header forwarding, invalid stdin, management rejection, malformed proof, shared
seed file, missing admin token, remote base URL, interactive without a terminal,
and an end-to-end call through `channel_key_proof_from_command`.
- `PYTHONPATH=. uv run --with pytest pytest tests/test_channel_proof_helper.py
-o addopts= --quiet` -> **16 passed**.
- Full provider-free suite -> **250 passed**, one existing `asyncio_mode` warning.
- `docs/channel-proof-helper.md` documents the reference implementation and an
untested operational caveat: the audit runs once per boot and once per channel,
while 2FA backup codes are single use. Whether one TOTP window authorizes
several channels in one audit is not verified; the local real-binary verifier
cannot answer it on this host because the pinned binary is absent.
- Nothing was installed on the VPS, no credential was read or created, and no
commit or push was performed.

## Operator commands and alias proposal - 2026-11-03

- Added `newapi_operator.py` at the package root (matching `newapi_catalog.py`
and the other `newapi_*` modules) with four repeatable commands: `audit`,
`catalog`, `priorities` and `boot-check`. Read-only by default; only
`catalog --apply-opencode` and `priorities --apply` mutate anything.
- `audit` runs the same strict checks as `serve()` and prints a 64-character
digest; `catalog` refreshes provider `/models` into the ledger, resolves keys
from the metadata `env_var` (never argv) and preserves the last-good catalog on
an empty fetch; `priorities` validates an explicit best-first order and updates
priorities only after re-auditing; `boot-check` validates an OpenCode config
without writing.
- Priority order stays explicit rather than derived. Automatic ordering from
persisted grades or telemetry would encode a policy decision, so it remains open
work rather than a CLI default.
- Added `tests/test_newapi_operator.py` (17 tests): audit call ordering,
preview/apply separation, full order validation, catalog refresh against a
loopback `/models`, keyless versus keyed Authorization headers, empty-catalog
rejection, boot-check acceptance and rejection paths, and CLI exit codes.
- Targeted run -> **17 passed**. Full suite -> **267 passed**, one existing
`asyncio_mode` warning. `git diff --check` clean, CLI `--help` exits 0, and no
stale `tools/newapi_operator` references remain.
- Added `docs/operator-commands.md` and `docs/alias-mapping-proposal.md`. The
proposal maps all 19 legacy aliases onto the sidecar's profile/step vocabulary
and states plainly that it preserves acceptance and step hints, not the
gateway's fallback chains. It is NOT implemented and needs approval because it
changes the documented public profile contract.
- Initial placement in `tools/` failed pytest collection with
`ModuleNotFoundError`; the module was moved to the package root and the test and
documentation references were updated.
- No VPS change, no credential access, no commit and no push.

## Legacy alias compatibility - 2026-11-03

- Operator approved the compatibility shim (option 1 in
`docs/alias-mapping-proposal.md`).
- Added `LEGACY_ALIASES` (19 gateway names mapped to a profile and a default step)
and `MODEL_ALIASES` (the 22-name union) to `newapi_runtime.py`. `prepare()`
resolves through the union, applies the legacy step hint, still lets an explicit
`X-Route-Step` override it, and keeps the requested name in `Prepared.model` so
responses echo what the client asked for. Unknown names still fail closed.
- `/v1/models` now advertises all 22 names with the legacy ones marked
`"deprecated": true`; the discovery test asserts 19 deprecated out of 22.
- Added `tests/test_legacy_aliases.py`: every legacy alias accepted with its
mapped profile/step, inventory sizes 19 and 22, explicit step override, public
profile defaults and unknown-name rejection.
- Targeted run (`test_legacy_aliases`, `test_newapi_runtime`,
`test_newapi_server`) -> **71 passed**.
- Recorded the approved 2FA source: an operator-owned `0600` TOTP seed used by
`tools/newapi_channel_proof.py --mode=totp`. The single-use backup-code caveat in
`docs/channel-proof-helper.md` still applies to restarts.
- No VPS installation is planned until the replacement is demonstrably better
than the live LiteLLM gateway. The operator's reason: cutover would otherwise
interrupt the few clients that depend on the current gateway.
- Still no VPS change, credential access, commit or push.

## Compatibility routes and hardening - 2026-11-03

- Operator approved building the remaining compatibility routes.
- `POST /v1/responses` translates the text/function-tool subset of the Responses
API into a chat payload, so routing, quota reservation and telemetry are
unchanged. Supports string or message-list `input`, `instructions`, flat
function `tools`, `tool_choice` strings, `text.format`, `max_output_tokens`,
`temperature`, `top_p`, and text-only SSE streaming. `previous_response_id`,
non-function tools, image/audio parts and streaming-with-tools return 400
instead of degrading silently. The shim is stateless.
- `POST /v1/embeddings` forwards one text batch to the audited New-API relay using
an alias named in `SIDECAR_EMBEDDING_ALIASES`. Startup fails if the alias is not
an audited route or collides with a chat alias. Batches reserve estimated prompt
tokens and settle on upstream usage; responses are size- and type-validated.
- Server hardening: JSON 405 with `Allow` for PUT/PATCH/DELETE, no Python version
in the `Server` header, a single committed-response flag so a mid-stream failure
cannot emit a second status line, missing env vars exit with the variable name,
embedding aliases are validated before bind, and SIGTERM/SIGINT stop the server
through its normal close path.
- Docs: `docs/responses-and-embeddings.md` (contract and limits) and
`docs/review-package.md` (change map, reproduce commands, verified versus
unverified). `docs/gateway-compatibility.md` now records 25 of 30 gateway paths
absent instead of 27.
- Targeted run (`test_newapi_server`, `test_legacy_aliases`) -> 52 passed;
`py_compile` clean. Full suite -> 300 passed, 1 pre-existing warning; `git
diff --check` clean.
- No VPS service install or start, credential access, commit or push at that
time.

## Real-provider validation and VPS staging - 2026-11-03

- Read-only inventory of the existing New-API install: 3 channels
(`groq-free` id 3, `openrouter-free` id 4, `badkey-test-865` id 5). Two are
usable free providers (Groq, OpenRouter); the third is a deliberate broken-key
test. The 22 names in `/v1/models` are client aliases, not providers.
- Real calls through New-API with relay token id 21, `max_tokens=15`:
`openai/gpt-oss-20b` (Groq) and `cohere/north-mini-code:free` (OpenRouter) both
returned 200 in streaming and non-streaming mode. Groq reported 78 prompt / 15
completion tokens; OpenRouter 7 / 15. Four calls total.
- Non-streaming requests still returned `text/event-stream` because both live
channels carry `param_override: {"stream": true}`. The sidecar's non-stream
path expects JSON, so this must be cleared before a sidecar canary.
- Strict startup audit cannot pass on the current New-API configuration:
`RetryTimes=3` (must be 0), both live channels carry the stream override,
channel keys are hidden from the admin API, and 2FA is disabled. The key
endpoint returns 403 "security verification required".
- 2FA enrollment is itself gated by `/api/verify` with method `password` and
scope `2fa.setup`. That proof worked once, but repeated login attempts tripped
New-API's login rate limiter (429), so enrollment was stopped rather than
hammered. 2FA remains disabled; no seed or backup code was written.
- Sidecar code was staged read-only at `/opt/newapi-sidecar` (root-owned,
`lib/` modules and `bin/channel-proof`) and `/var/lib/newapi-sidecar`
(newapi-owned, 0700). There is no unit, no process, and no listener on 8099.
- Added the `routes` operator command (preview by default; `--apply` writes the
ledger only) with four tests. Full provider-free suite -> **307 passed**, one
pre-existing warning; `git diff --check` clean.
- LiteLLM remained active and healthy (`/health` 200); New-API remained active
and healthy (`/` 200). No traffic, credentials, channel settings, or service
state were changed. No commit or push.
- Recommendation: do not replace the LiteLLM gateway with this install. It
fronts 2 usable providers versus the gateway's 33, and the sidecar's
compatibility surface is still partial.

## Provider catalog inventory and pilot plan - 2026-11-03

- Located the gateway's provider store: `llm_providers.db` has 204 providers and
6,827 models; `config/router_config.yaml` has 60 entries with 44 enabled; `.env`
has 197 non-empty keys. The active gateway therefore uses far more than the 22
sidecar aliases and the 2 usable New-API channels.
- Rejected bulk import: many entries are internal, paid, dead, or share a base
URL, and copying 197 keys would spread credentials well beyond the providers in
use. New-API also stores channel keys in its SQLite file.
- Wrote `docs/provider-import-plan.md` and a no-credential plan at
`/home/ubuntu/newapi-import-plan.json`: pilot Cerebras, Mistral, NVIDIA and
Together as `gw:*` type-1 channels, then test one 15-token chat per provider and
expand in batches.
- Hardened the New-API data directory to `0700` and `one-api.db` plus WAL/SHM to
`0600`; healthcheck, watchdog and backup all run as `newapi`, and New-API stayed
healthy. This is now the only live VPS permission change.
- Restarted New-API to clear what looked like a login limiter (health 200), but
login still failed. Later correction: the helper read `ROOT_PASSWORD` from the
gateway `.env` (empty) instead of `/opt/new-api/new-api.env`; there was no
limiter. See the model-first section below.
- Recommendation on the security question: defer 2FA. It only unblocks the
sidecar's channel-key audit, which is still blocked by `RetryTimes=3` and the
forced-stream channel overrides.

## Model-first pivot and live pilot - 2026-11-03

- Corrected the earlier "login limiter" finding: the helper read
`ROOT_PASSWORD` from the gateway `.env` (empty) instead of
`/opt/new-api/new-api.env`. Login succeeds with the correct file; no limiter
blocked the import, and the 2FA attempts changed nothing.
- Live probes (15 output tokens) show the catalog is stale: Cerebras and Together
return 403, Supacoder 403, FreeTheAI 530, GCLI 401, Gemini 2.5 Flash is retired
(404; 3.8 Flash was 503 high-demand), Mistral chat was 429, and NVIDIA tested
models returned 404/410. Bluesminds, HCNSEC, IAMHC, and the existing
Groq/OpenRouter channels answered 200.
- The operator's goal is model-first routing: select a model, try any provider
that serves it, and fall back on failure. That is New-API's native channel
behavior: several channels expose the same model, `RetryTimes` stays above zero,
and priority/weight pick the order. The sidecar's strict one-alias-per-channel
audit is the opposite design; do not set `RetryTimes=0` for this goal.
- Seeded three live channels through the admin API: `gw:hcnsec` (id 6, `auto`),
`gw:iamhc` (id 7, `auto`), and `gw:bluesminds` (id 8,
`nvidia/ising-calibration-1.5-31b`). New-API served `auto` and the Bluesminds
model with 200 through relay token 21. Both `auto` channels remain enabled.
- Failover is proven: `POST /api/channel/6/status` `{"status":2}` disabled
`gw:hcnsec`; `auto` still returned 200 with "ok" through `gw:iamhc`; channel 6
was re-enabled and confirmed `status: 1`. The generic `PUT /api/channel/`
rejects a `status` field; the dedicated route is `POST /api/channel/:id/status`.
- Recommendation: park the sidecar strict deployment and 2FA. Use New-API as the
router, expand only live-verified providers/models, keep dead entries disabled,
and add canonical model mappings where provider model ids differ.
- LiteLLM remains the production gateway; no traffic cutover.

## Provider expansion - 2026-11-03

- Probed `/models` for 36 enabled providers with a base URL and key. 15 passed
`/models` and got a 16-token chat. Only CrowLLM (`qwen-3.6-27b:free`) and
KiosAPI (`agnes-3.0-flash`) returned 200. Everything else was quota-limited,
out of credit, paused, or returned malformed upstream responses.
- Created `gw:crowllm` (id 9, now 73 chat models), `gw:kiosapi` (id 10, 4
models), and `gw:crowllm-canonical` (id 11, priority 40) mapping
`agnes-3.0-flash` to `agnes-3.0-flash:free`. All are enabled; channels 9/10 were
verified through New-API relay token 21. The abilities table lists both channels
10 and 11 for `agnes-3.0-flash`. New-API now has 9 channels.
- Removed the 7 embedding ids from channel 9; its chat model still returned 200,
and the abilities table no longer contains those embeddings (73 abilities).
- The canonical fallback test passed: with `gw:kiosapi` (id 10) disabled,
`agnes-3.0-flash` still returned 200 with "ok" through
`gw:crowllm-canonical` (id 11); channel 10 was re-enabled and confirmed
`status: 1`.
- The status route is `POST /api/channel/:id/status` with `{"status": 1|2}`;
`PUT /api/channel/` rejects any `status` field. Failover is proven above.
- Re-probe the 429 providers after their daily reset; do not create channels for
  them until a chat succeeds. Keep dead catalog rows as disabled channels.

## Staged v1/v2 roadmap - 2026-11-03

- User accepted the staging: v1 is a native New-API model-first prototype
  (`RetryTimes=3`) for basic fallback; v2 is the sidecar/full-parity path with
  prompt classification, legacy aliases and quota/telemetry routing. No cutover
  is authorized by this decision.
- Added `docs/newapi-v1-v2-roadmap.md` with the v1 scope, the full 19-alias gap
  matrix, v2 blockers, migration checklist and rollback boundaries. Linked from
  `HANDOFF-newapi-sidecar.md`, `provider-import-plan.md` and
  `alias-mapping-proposal.md`.
- Read-only live inventory confirmed 9 channels, 110 abilities,
  `RetryTimes=3`, channels 3/4 `param_override={"stream":true}`, and 0 2FA rows.
  The five pre-existing `router/*` aliases were called at `max_tokens=15` with
  streaming: `router/fast`, `router/balanced`, `router/tools` and `router/code`
  returned 200; `router/reasoning` returned 404 because its mapped model is
  paid-only. These are status-only results; SSE content and tool calls were not
  asserted.
- Initial v1 candidates were `router/fast` and `router/balanced` because each
  exists on both channels 3 and 4. The live gate below found both are Groq-only;
  `agnes-3.0-flash` is the actual fallback route. `router/tools` and
  `router/code` are single-channel. The literal New-API `auto` route stays
  reserved: it is not the old semantic `auto` alias.
- No VPS configuration, channel state, credential, service or traffic was
  changed for the initial checkpoint. v2 still requires `RetryTimes=0`, cleared
  stream overrides and the channel-proof path.

## v1 live gate execution - 2026-11-03

- Parsed `router/fast` and `router/balanced` SSE: both returned `content:"ok"`,
  `finish_reason:"stop"` and usage. The upstream models were Groq
  `openai/gpt-oss-20b` and `openai/gpt-oss-120b` respectively.
- `router/tools` returned a valid `lookup_status` tool call with
  `{"service":"gateway"}`, `finish_reason:"tool_calls"` and usage.
- Channel 3/4 failover exposed a real limitation: both `router/fast` and
  `router/balanced` are Groq-only in practice. With Groq disabled, `router/fast`
  returned 429 and `router/balanced` returned a paid-only 404. With OpenRouter
  disabled, both still returned 200 via Groq. Do not present these as fallback
  routes.
- `agnes-3.0-flash` passed both directions: with channel 10 disabled it served
  via CrowLLM canonical; with channel 11 disabled it served via KiosAPI. A
  tool-shaped request also returned a valid `lookup_status` call. Both channels
  were restored to `status:1`.
- OpenCode 1.19.5 completed a local tool loop through a temporary SSH tunnel and
  temporary `XDG_CONFIG_HOME`: exit 0, one `tool_use` event, and output contained
  `hello-newapi`. No permanent OpenCode config or credential was written; the
  VPS has no OpenCode binary.
- Final live state: all 9 New-API channels `status:1`, LiteLLM `/health` 200,
  New-API `/` 200, and the temporary tunnel closed. No cutover or permanent
  service change occurred; the temporary channel status changes were restored.
  The narrow v1 route is `agnes-3.0-flash`, not the `router/*` aliases.

## V2 deterministic auto classifier - 2026-11-03

- Added `classify_auto()` to `newapi_runtime.py` for the legacy `auto` alias.
  It inspects only the latest user message and maps code/verify/debug intent to
  `auto-coding`, otherwise to `auto-chat`; an explicit `X-Route-Step` still
  wins. Other aliases and public profiles are unchanged.
- Code intent is checked before the tools bit, so an agent request such as
  "Implement the parser" stays on the coding profile even when tool definitions
  are present. `Work.tools` still carries the capability filter.
- Tests added for chat, implementation, failing tests, debug, verification,
  explicit steps, tool-bearing chat, and tools-enabled code intent. Targeted
  run: **88 passed**. Full provider-free suite: **317 passed**, one existing
  `asyncio_mode` warning; `git diff --check` clean.
- No VPS change, deployment, or cutover. This is a deterministic baseline, not
  the gateway's semantic-router parity.

## V2 read-only compatibility routes - 2026-11-03

- Added authenticated `GET /v1/models/{model_id}` and `GET /v1/providers` to
  `newapi_server.py`. Both are read-only: no quota reservation, dispatch, or
  ledger write.
- Model detail returns the old basic `{id, object, created, owned_by}` envelope
  for any non-empty id. Known aliases use `owned_by: sidecar` and carry
  `deprecated`; unknown ids use the first path segment as owner, or `unknown`.
  It deliberately does not claim upstream model-info statistics.
- Provider list returns sorted provider IDs referenced by configured ledger
  routes, not the gateway's provider-plugin catalog.
- GET routing now ignores query strings. Added an HTTP boundary test covering
  alias detail, unknown-id detail, provider list, and authentication. Targeted
  suite: **27 passed**. Full provider-free suite: **318 passed**, one existing
  `asyncio_mode` warning; `git diff --check` clean.
- Updated compatibility accounting to 9 exposed paths, 23 of 30 gateway paths
  absent, and 5 partially covered same-path routes. No VPS change, deployment,
  credential access, or cutover.

## V2 TPS and model-management compatibility - 2026-11-03

- Added `QuotaLedger.tps_stats()` and authenticated `GET /v1/tps-stats`. It
  aggregates only successful, measured streaming attempts by provider/model,
  matching the gateway's median/avg/p95/avg-TTFT/sample-count shape. Query
  bounds are 1-365 days and 1-10,000 samples; `export=true` is rejected so the
  endpoint cannot write files. No prompt or credential data is returned.
- Added `GET /v1/model/list`, `GET /v1/model/status`, and
  `POST /v1/model/parse-command`. They expose the 22 sidecar aliases, a default
  `auto-chat` status, and the legacy command forms (`/model`, `/switch`, `/use`,
  and shortcuts). Parsing never mutates session state; `/uncensored` remains
  invalid because the sidecar has no `uncensored-chat` profile.
- Targeted suites (`test_newapi_server`, `test_quota_ledger`): **42 passed**.
  Full provider-free suite: **320 passed**, one existing `asyncio_mode`
  warning; `git diff --check` clean.
- Compatibility surface is now 13 paths: 10 served at the same path (1 covered,
  9 partial), 1 renamed, and 19 of 30 gateway paths absent. No VPS change,
  deployment, credential access, or cutover.
