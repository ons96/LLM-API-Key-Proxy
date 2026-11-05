# Router Session State

Current concise cross-harness handoff: `../HANDOFF-CURRENT-newapi-sidecar.md`.
Standalone prompt: `../RESUME-newapi-sidecar.txt`. Implementation checkpoint
`27e1fd6` is pushed; 398 tests passed. The entries below are chronological, so
read the final checkpoints rather than an old repeated "next action" section.

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

## V2 cost estimate and language decision - 2026-11-03

- Added `QuotaLedger.estimate_cost()` and authenticated
  `POST /v1/cost-estimate`. It returns the cheapest known configured route
  price as per-token USD pricing, returns `cost:null`/`source:"unknown"` when no
  price exists, and never selects or dispatches a route.
- Kept Python for the sidecar. A local sidecar process using in-memory SQLite
  and serving 50 authenticated `/v1/models` requests measured **26,984 KiB**
  RSS. VPS-40 has approximately 954 MiB total and New-API used 58-64 MiB; a
  128 MiB sidecar cap is the proposed canary guardrail. The local measurement is
  not a concurrency or streaming benchmark.
- Recorded the decision in `docs/sidecar-language-decision.md`: no Go or Node
  rewrite without a measured resource or concurrency trigger.
- Focused suites: **43 passed**. Full provider-free suite: **321 passed**, one
  existing `asyncio_mode` warning; `git diff --check` clean.
- Compatibility surface is now 14 paths: 11 same-path (1 covered, 10 partial),
  1 renamed, and 18 of 30 gateway paths absent. No VPS change, deployment,
  credential access, or cutover.

## V2 free-tier classification - 2026-11-03

- Kept `quota_type` as the capacity mechanism and added derived `free_class`:
  `unlimited -> free_unlimited`, `rate_limited -> free_limited`,
  `credit_based -> free_credits`, `paid -> paid`, otherwise `unknown`.
- Added optional `credit_input_price`/`credit_output_price`. Actual
  `input_price`/`output_price` remain zero for free routes; the credit tariff is
  notional and is used only to deplete `credit_balance` and rank the route.
  A one-time-credit route with a balance but no credit tariff now fails capacity
  as `unknown_credit_price` instead of looking unlimited.
- Cost estimates now include `free_class` and `free`; zero-price free routes
  report `cost:0`, `source:"free_tier"`. Diagnostics expose `free_classes`, and
  the operator route preview includes class/quota metadata.
- Focused suites: **77 passed**. Full provider-free suite: **324 passed**, one
  existing `asyncio_mode` warning; `git diff --check` clean.
- Added `POST /v1/token-count` as a conservative UTF-8 byte upper bound with
  `estimated:true` and `method:"utf8_byte_upper_bound"`. It never dispatches and
  is not an exact cross-provider tokenizer.
- Compatibility surface is now 16 paths: 13 same-path (1 covered, 12 partial),
  1 renamed, and 16 of 30 gateway paths absent.
- Added unauthenticated `GET /` returning `{"Status": "New-API sidecar is
  running"}`; it is a probe-only compatibility shim.
- No VPS change, deployment, credential access, or cutover.

## V2 provider-classification adapter - 2026-11-03

- Added read-only `provider_classification.py`, which reads the existing
  `llm_providers.db` provider flags and maps them to sidecar quota/free classes:
  one-time credit, daily/limited, unlimited, paid, or unknown. It never copies
  credentials or writes the provider database.
- Added `newapi_operator.py classify --provider-db PATH`. It compares ledger
  routes with the provider metadata, preserves explicit route values, and
  reports `match`, `mismatch`, `unclassified`, or `missing_provider_metadata`.
- Fixed `newapi_catalog.load_provider_metadata` to support the live
  `providers.key_name` schema as well as the older `name` test schema.
- VPS-40 read-only inventory: 204 providers, 6,827 models, 18 virtual models,
  23,074 provider-health rows. Provider classifications are `legacy-audit`
  with `human_confirmed`; many quota amount/unit/reset fields are null.
- Focused affected suites: **70 passed**. Full provider-free suite: **329
  passed**, one existing `asyncio_mode` warning.
- No VPS change, deployment, credential access, or cutover.

## V2 provider status snapshot routes - 2026-11-03

- Added read-only `provider_status.py` over the gateway's existing
  `provider_status.db`. It reads the latest row per provider and supports
  bounded history, best-provider selection, health, and CSV export.
- Added authenticated `/api/providers/status`, `/status/{provider_name}`,
  `/best`, `/health`, `/history`, `/history/{provider_name}`, and
  `/export/csv`. Every current row has `stale` and `age_seconds`; health reports
  `running:false` because the sidecar does not probe providers.
- Configured by `SIDECAR_PROVIDER_STATUS_DB` and
  `SIDECAR_PROVIDER_STATUS_STALE_SECONDS` (default 900). Missing configuration
  returns 503; a configured missing file fails startup rather than serving an
  empty health picture.
- Compatibility surface is now 23 paths: 20 same-path (1 covered, 19 partial),
  1 renamed, and 9 of 30 gateway paths absent.
- Focused status suites: **4 passed**. Full provider-free suite: **333 passed**,
  one existing `asyncio_mode` warning.
- No VPS change, deployment, credential access, database write, or cutover.

## V2 model-management compatibility - 2026-11-03

- Added `model_management.py` with a bounded, expiring process-local session
  store (1,024 sessions, 24-hour TTL) and the legacy regex intent heuristic,
  constrained to the sidecar's 22 accepted aliases.
- Added `GET /v1/model-info/stats` (ledger summary, `ready:false`, no upstream
  metadata), `GET /v1/model/session/{session_id}`, `POST /v1/model/switch`, and
  `POST /v1/model/detect-intent`. No route dispatches a provider call.
- Compatibility surface is now 27 paths: 24 same-path (1 covered, 23 partial),
  1 renamed, and 5 of 30 gateway paths absent. The remaining five are all
  CLIProxyAPI routes requiring an external service and credentials.
- Focused model-management suites: **3 passed**. Full provider-free suite:
  **336 passed**, one existing `asyncio_mode` warning.
- No VPS change, deployment, credential access, database write, or cutover.

## Local remote reconciliation and startup proof - 2026-11-04

### Remote work preserved

- Started at `5863e02`, not stale base `fde9784`. Tracked files were clean;
  existing untracked `.serena/` was preserved.
- Fetched/pruned both project remotes. Implementation was 16 commits behind;
  fast-forwarded to `origin/feat/newapi-sidecar` at `546952e`. Reused its helper,
  operator CLI and compatibility work. `origin/main` is `76cd928` (one unrelated
  provider-policy commit outside this branch). Migration `origin/main` is
  `e983169`; its worktree/untracked decisions doc were not changed.
- Inspected GitHub PR refs via Git: none points at the current sidecar tip;
  latest exposed PR number 358. PR state/reviews/checks could not be queried
  without an approved fine-grained PAT. Stored gh auth was not used for API
  calls; deferred questions are in `QUESTIONS.md`.
- Before edits: exact full suite **336 passed**; three fresh isolated proofs
  passed with one 429 attempt then failover. This does not causally resolve
  historical intermittent early-503 reports.

### Focused change

- `tools/newapi_channel_proof.py`: the real pinned binary's 579-byte signed
  proofs exceeded the helper's old 512-byte limit. It now matches the sidecar's
  4096-byte printable-ASCII header bound; invalid shapes still fail closed.
- `tools/verify_newapi.py`: launches the actual `newapi_server.py` subprocess
  with a TOTP helper and `0600` seed from its freshly enrolled disposable
  account. Missing/invalid proofs fail before listener/upstream dispatch.
  Valid startup/restart audit three channels, serve chat/coding/SSE, require
  auth, exit cleanly on SIGTERM and preserve decisions/quota windows/cooldowns.
  Generated credentials/seed are checked absent from sidecar logs. Failure
  reports retain `passed:false`, bounded error and artifact location.
- Repeated audits hit verification HTTP 429 after enrollment. No same-provider
  retry was performed. Only the fresh fixture receives `CRITICAL_RATE_LIMIT=32`;
  limiting stays enabled. Never copy this test budget into production.
- Added malformed/579/4096-byte proof regressions to
  `tests/test_channel_proof_helper.py` and startup-probe false-pass/deadline
  checks in `tests/test_verify_newapi.py`. Updated README, proof docs, staged
  roadmap, root handoff and QUESTIONS. All changes remain uncommitted.

### Actual validation

- Binary SHA-256 matches pinned `v1.0.0-rc.36`:
  `5d9b254721c205f4f442848a4af5b58617a8c0ee090366732cfff90455e957f3`.
- `python3 -m pytest tests -o addopts= --quiet` -> **350 passed, 1 warning in
  26.58s** (existing `Unknown config option: asyncio_mode`).
- `python3 tools/verify_release.py` -> **350 passed**, session validation OK,
  health/route 200, RSS 26260 KB, compilation passed. Changed modules/tests
  also passed `python3 -m py_compile`.
- Exact `python3 tools/verify_newapi.py --binary /home/osees/new-api/bin/new-api
  --artifacts /tmp/opencode` passed on two final fresh instances:
  `/tmp/opencode/newapi-proof-mgfxu60y/report.json` and
  `/tmp/opencode/newapi-proof-hsj9s35w/report.json`. Read both directly:
  fail-closed helpers, zero unauthorized dispatches, same-window TOTP audits,
  clean SIGTERM, persisted accounting/quota/cooldowns, credentials absent from
  sidecar logs, original `one-model,two-model` 429 order, paid exclusion and
  five grade-driven ranking changes all passed. Dirs `0700`; reports/seeds `0600`.
- Fixture RSS: New-API 154536/154332 KiB; Python fixture plus in-process sidecar
  41260/41240 KiB. Not a VPS-load or pure-entrypoint memory guarantee.
- Cached offline uv Black changed-line check, flake8 and mypy passed on four
  changed/new Python files. Flake8 ignores the existing verifier E402 bootstrap;
  mypy skips imports. Whole-repo format/type compliance is not claimed.
- Real-namespace plus focused-test coverage: verifier **82%** combined
  line/branch; `wait_entrypoint` 18/18 lines, 10/10 branches; `request_proof`
  12/12 lines, 8/8 branches. Evidence:
  `/tmp/opencode/newapi-combined-coverage.json`. Initial coverage module selector
  collected nothing; corrected to file inclusion before reporting results.
- `git diff --check` and added-text ASCII/credential scan passed. No listeners
  remain on 8002, 8003 or 8099. No logs, seed or real key was dumped.

### Reuse and next move

- User confirmed existing free accounts/catalogs. Read-only local inventory:
  canonical DB 204 providers / 6827 models (137 enabled); OpenCode 117 provider
  entries / 1214 models. `~/.secrets` exists and was not opened. Reuse canonical
  metadata/approved credentials, not new signups or a bulk key import.
- Recheck remote/PR state before further edits. Next bounded milestone: verify
  catalog/operator/priority and actual OpenCode boot orchestration in a disposable
  instance, then plan a small already-owned free-provider validation subset.
- VPS v2 still needs explicit approval for zero retries, cleared overrides and
  authorized proof provisioning; review repeated-audit rate-limit budgets first.
  Keep native v1 and the legacy gateway unchanged. Live OpenCode tool-loop,
  compatibility exceptions, resource/load and rollback gates remain.
- No commit, push, PR, VPS SSH, deployment, live-provider call, production DB
  write, credential-store change or traffic cutover occurred during that
  validation pass.

## Authorized feature-branch backup - 2026-11-04

- User authorized commit/push to save the verified startup milestone.
- Re-fetched origin: `feat/newapi-sidecar` remains `546952e`, no newer changes.
  Keep this existing feature branch; do not merge main or trigger deployment.
  Oracle deployment workflow is main-only. A PR is for later review, not needed
  for backup, and approved GitHub API auth is still unavailable.
- Commit only the ten reviewed startup-proof/test/doc files. Preserve untracked
  `.serena/` and private `/tmp/opencode` reports. Use normal secret-scan hooks,
  no force push or hook bypass. Previous full/release checks passed 350 tests.
- Next: compose operator catalog/audit/priority commands and actual OpenCode
  1.19.5 boot/tool-loop checks against disposable New-API and scripted providers.
  Recommend no VPS mutation, real-account quota use or traffic switch for this
  milestone.

## Operator and installed OpenCode proof - 2026-11-04

- Committed/pushed the previous startup milestone as `b8b882b` on
  `feat/newapi-sidecar`; both normal secret-scan hooks passed. Remote tip was
  rechecked before more edits and contained no competing work. Main was not
  merged or pushed; no PR/deployment was created.
- `newapi_operator.py boot-check` previously printed `ok:false` but exited 0
  for a missing/empty provider. It now exits 1, with success still 0. Added two
  real CLI exit-status regressions in `tests/test_newapi_operator.py`.
- `tools/verify_newapi.py` now invokes operator subprocesses in the existing
  fresh loopback namespace: route preview/apply identity checks, three catalog
  preview/apply checks, unchanged canonical test metadata, curated config and
  symlink preservation, empty-catalog last-good preservation, audit and actual
  priority-only updates. Invalid orders fail before writes. These commands
  send no provider completions.
- Added optional `--opencode PATH`: installed OpenCode 1.19.5 loads all three
  refreshed fixture catalogs, then runs a scripted `read` call and returns
  `hello-sidecar-proof` through the actual sidecar/New-API. Exactly two upstream
  attempts settle successfully on one deployment; tool messages/definitions
  arrive intact. Only the disposable read tool is allowed. HOME/XDG/config,
  model cache, generated file-backed client key and work directory are isolated;
  global plugins, external skills, project config, updates and model fetch are
  disabled. No permanent OpenCode config, account key or live provider was used.
- Fixture command outputs stay private, are bounded, and reject generated key
  echoes; added timeout/exit/output/credential-leak regressions. The fixture's
  previously 32-request verification budget hit a protected key read during
  the larger audit workload. Pinned upstream source confirms critical limiting
  guards that route. Fresh-test budgets alone are now `CRITICAL_RATE_LIMIT=64`
  and `GLOBAL_WEB_RATE_LIMIT=512`, with limiting still enabled. No 429 retry or
  production rate-limit change was made.
- Verifier route comparison accounts for existing JSON tuple-to-list profile
  storage; no route/ledger behavior was changed to satisfy the fixture.

Validation:

- `python3 -m pytest tests -o addopts= --quiet` -> **357 passed, 1 warning in
  26.23s**. Existing `asyncio_mode` warning remains.
- `python3 tools/verify_release.py` -> **357 passed**, session validation OK,
  health/route 200, RSS 26244 KB and compilation passed.
- `python3 tools/verify_newapi.py --binary /home/osees/new-api/bin/new-api
  --artifacts /tmp/opencode --opencode /home/osees/.opencode/bin/opencode`
  passed twice on fresh instances. Reports read directly:
  `/tmp/opencode/newapi-proof-malzzq2d/report.json` and
  `/tmp/opencode/newapi-proof-df6gpmct/report.json`. Operator flags true;
  client version 1.19.5, 1 completed read loop, 2 accounted attempts, stable
  deployment; startup/restart, exact 429 fallback, paid exclusion and five
  grade-driven ranking changes still pass. Scripted responses are not LLM scores.
- Final client boot now uses the exact symlinked file written by the operator,
  not a separately refreshed equivalent. Repeated proofs after that final
  change passed: `/tmp/opencode/newapi-proof-pkfms7du/report.json` and
  `/tmp/opencode/newapi-proof-exot1afe/report.json`; focused suite 37 passed,
  lint/type/diff and full new-test formatting remained green.
- Exact verifier without `--opencode` also passed:
  `/tmp/opencode/newapi-proof-wpsvw62x/report.json`; client is optional, not
  silently claimed. Fixture RSS with client check: New-API 155588/156192 KiB,
  Python fixture/in-process sidecar 43924/43972 KiB; not a VPS-load guarantee.
- Focused flake8 (existing E402 bootstrap ignored), mypy with imports skipped,
  Black changed-line checks (full check for `test_verify_newapi.py` due to a
  range-boundary formatter artifact), py_compile and diff checks passed.
- Combined real-namespace/focused coverage: verifier 82%; `fixture_command`
  16/16 lines and 6/6 branches. Report:
  `/tmp/opencode/newapi-operator-combined.json`. No claim of whole-repo type or
  coverage compliance.
- Updated README, operator/proof docs, staged roadmap and root handoff. Preserve
  `.serena/`; only reviewed files are included in the feature-branch save.

Next recommendation:

- Save this verified milestone on the feature branch; do not merge into main,
  deploy or cut over. Then prepare a tiny existing-free-provider validation
  subset from canonical metadata with explicit live `/models` and price/capability
  checks. Stored free flags include stale/non-free model IDs, so flags alone
  cannot authorize generation. Do not bulk-import keys or spend unknown credits.
- A real-provider sidecar loop remains a distinct gate; the installed-client
  fixture does not substitute for it. VPS strict v2 config/proof approval,
  compatibility exceptions, load/restart/rollback and cutover approval remain.

## Budgeted live free-provider milestone - 2026-11-05

### Scope and safety

- Re-fetched origin before work and again before saving: remote
  `feat/newapi-sidecar` is `46aa2af`, no newer competing commits; no PR ref points
  to that tip. Preserve `.serena/`. GitHub API review state still lacks approved
  authentication. Continue saving only on the feature branch, not main.
- Added explicit `tools/verify_newapi_live.py` and
  `tests/test_verify_newapi_live.py`. Reuses the stdlib transport/catalog,
  ledger/runtime and disposable-bootstrap code. Default is catalog-only;
  `--execute` permits up to two 128-output-token chat/SSE requests per target
  (one or two explicit Groq/OpenRouter targets). Optional `--opencode` replaces
  that workload with one target and at most three 512-output-token requests.
- Canonical provider DB is read-only. Selected HTTPS endpoints/funding/rate
  limits come from it; OpenRouter must freshly advertise an explicit `:free`
  model with all zero prices, context and tools. No unknown trial credits,
  guessed endpoints, broad sweeps, new signups or canonical writes.
- Private file-referenced Groq/OpenRouter keys were read in memory from the
  existing `0600` secret store; never printed, copied into the gateway DB,
  passed as argv, rotated or committed. Host-only broker enforces model, request
  bytes, single generation, fixed request/output budgets and no retry after
  provider errors. It ignores proxies/refuses redirects. A socketpair bridges
  allowlisted responses into the new loopback-only namespace; New-API stores
  only generated bridge keys. HTTP bridge also authenticates those local keys.
- Started quota attempts settle independently. Real HTTP failures block a
  target for the remainder of the run. Cleanup targets only owned child
  process groups and closes the bridge. All reports/logs/DBs are owner-only
  private artifacts; no temporary raw response, prompt or credential was dumped.

### Actual live evidence

- Both catalogs passed: Groq 11 models; OpenRouter 418. Selected canonical
  Groq `openai/gpt-oss-20b` and live zero-priced OpenRouter
  `cohere/north-mini-code:free` (256000 context, tool support).
- Two-target smoke used **3 real attempts**: Groq generation 403 (no retry),
  OpenRouter chat 200 and SSE 200. Exact `ok` content and real usage passed:
  chat 5 input/90 output; SSE 5/62, text TTFT ~2757.5 ms and reasoning TTFT
  ~348.4 ms. Overall two-provider report correctly failed because Groq rejected:
  `/tmp/opencode/newapi-live-us6lrwkv/report.json`.
- First real client loop used **3 attempts** then hit its budget: model returned
  correct read calls but OpenCode rejected the test permission. Absolute-file
  permission was wrong for a non-Git worktree, which OpenCode treats as `/`.
  Report `/tmp/opencode/newapi-live-pp7vl34a/report.json` is failed, not provider
  unavailability. Do not reuse those attempts as successful loop evidence.
- Corrected both live and scripted verifier clients: initialize a disposable
  Git worktree, then allow only relative `proof.txt` and deny every other tool
  and file. Pinned OpenCode source confirmed relative permission semantics.
  Scripted proof passed first at
  `/tmp/opencode/newapi-proof-l8tj5qo1/report.json`; no extra live quota used
  while debugging the permission rule.
- Final actual OpenCode 1.19.5 loop passed with **3 real attempts**, all 200:
  exactly one completed read of generated `proof.txt`, exact response
  `hello-newapi-live-proof`, actual usage/settlement 2264/89, 2321/56,
  2326/69 input/output tokens (6911 input / 214 output total), no cooldown.
  Includes a small-model/title request in the same budget. Report read directly:
  `/tmp/opencode/newapi-live-gf3j9wix/report.json`, `passed:true`.
- Total this live session: **9 attempted generations** (1 rejected Groq, 2
  OpenRouter smokes, 3 permission-failed client calls, 3 final client calls).
  No further live provider traffic was used for safety/release checks.
- This is one working free route, not two-provider fallback, broader SWE
  quality, full compatibility, production entrypoint or VPS proof. Live runs
  used audited Runtime/Server; separate scripted gate tests the real entrypoint.

### Checks and saved artifacts

- `python3 -m pytest tests -o addopts= --quiet` -> **376 passed, 1 warning in
  29.15s**; existing `asyncio_mode` warning only.
- `python3 tools/verify_release.py` -> **376 passed**, session validation OK,
  health/route 200, RSS 26312 KB; compile gate passed. New/changed modules passed
  py_compile, flake8 (existing path-bootstrap E402 ignored), mypy with imports
  skipped, Black checks and diff whitespace checks.
- New live-policy suite: **19 tests**; quota/request ceilings, 401/403/429
  non-retry, unknown/free/price/capability boundaries, key permission/symlink
  safety, canonical read-only preflight, framed response caps and key echoes.
- Coverage of new critical policy: `Broker.dispatch` 48/51 lines, 13/16
  branches; `free_model` 25/27, 16/18; `private_key` 9/9, 4/4; `load_targets`
  42/47, 15/20. Overall live-harness coverage only ~52%, due to cross-process
  child/client paths; do not claim 80% overall. More subprocess error-path
  coverage remains. Evidence `/tmp/opencode/newapi-live-unit.json`.
- After the actual live pass, additional broker/authentication/process-cleanup
  hardening was verified with scripted host responses and real disposable
  binary/client only (no more quota). Final offline self-check:
  `/tmp/opencode/newapi-live-bot7lf38/report.json` (2 scripted attempts), not
  live provider evidence. These monkeypatched reports are explicitly labelled
  fixtures; the actual live evidence paths above are separate.
- Added `docs/live-provider-verification.md`; updated README, provider import
  plan, staged roadmap, QUESTIONS and root handoff. Private target JSON/key
  references stay outside git. Check staged diff for secrets/ASCII before save.

Next:

- Save the bounded live milestone on `feat/newapi-sidecar`. Keep main/VPS and
  existing v1/native retries unchanged; no cutover is recommended.
- Need one more working renewable free provider before live cross-provider
  failover. Groq inference 403 needs operator/account review, not retries or
  auto-rotation. Pick a tiny approved second target using canonical metadata
  and fresh catalog; avoid unbounded provider scans/credit depletion.
- Broader client/code workload, live quota/error failover, compatibility
  exceptions, VPS v2 config/proof provisioning, resource/load and tested rollback
  remain. No VPS SSH, production configuration/data or credential changes were
  performed. Current live successes do not authorize migration.

## Second free provider and controlled fallback - 2026-11-05

- Re-fetched origin before work and before saving: remote sidecar remains
  `5c4f323`, no competing work; existing `.serena/` untouched. Current PR refs
  do not point at sidecar tip; approved GitHub API auth still unavailable.
- Chose Mistral using existing human-confirmed canonical renewable-free metadata
  and current catalog, not a new signup or unknown one-time credits. Existing
  `MISTRAL_API_KEY` was resolved in memory by the approved dotenv loader; its
  value was not printed, copied into a file, modified or passed as argv.
- Extended `tools/verify_newapi_live.py` for Mistral active live chat/function
  support, canonical free/tools metadata and conservative min(live,canonical)
  context. Added mutually exclusive `key_file` or `key_env` source; environment
  name must exactly match canonical `env_var`, and the child gets no real keys.
- Added `--failover` (mutually exclusive with the client workload): exactly two
  providers, controlled primary-first order via fixture latency, local pre-output
  429 injection, maximum one real 128-output-token request per provider. Fresh
  owned ledger per direction; no clearing production/test quota state. Failed
  started attempts retain full reservations and cooldowns; successful fallback
  settles actual usage. Ledger reopening must retain reservation and cooldown.

Live results, four real requests total:

- Mistral catalog 55 models; `codestral-latest` active, chat/tool supported,
  live context 256000, conservative canonical context 200000. Two smoke calls
  passed, exact `ok`, 8 input/2 output tokens each. SSE text TTFT ~431.6 ms;
  no estimated usage/cooldown. Report read directly:
  `/tmp/opencode/newapi-live-vk3e2kqt/report.json`, `passed:true`.
- Both-direction fallback passed: Mistral local injected 429 -> OpenRouter chat
  (5 input/52 output); OpenRouter local injected 429 -> Mistral SSE (8/2).
  Exact primary/fallback order, no same-route hidden retry, conservatively
  charged failure and persisted cooldown. Only two real upstream requests:
  `/tmp/opencode/newapi-live-ad33w88d/report.json`, `passed:true`. Read each
  retained ledger directly: one charged failure, one success and one cooldown.
- Faults were generated in the local bridge; do not claim upstream spontaneous
  429s. These are different models within the same profiles, not same-model
  mapping parity. Groq was not retried. No VPS, production retry/override,
  client permanent configuration or traffic change occurred.
- After live success, one-call broker cap and exact persisted-reservation
  equality checks were tightened and verified only with scripted transport:
  `/tmp/opencode/newapi-live-123lfdk9/report.json` (2 scripted requests). No more
  live quota was spent for safety, coverage or formatting.

Validation and save:

- `python3 -m pytest tests -o addopts= --quiet` -> **384 passed, 1 warning in
  30.32s**. Existing `asyncio_mode` warning remains.
- `python3 tools/verify_release.py` -> **384 passed**, session validation OK,
  health/route 200, RSS 26152 KB; compilation passed. Final focused live suite
  **27 passed**, Black/flake8/mypy (imports skipped) and diff checks passed.
- Added tests for canonical env-source matching/missing vars, Mistral active
  tool/free/context gates, both failover directions, hidden-retry/missing-usage
  rejection, durable failures and one-request provider budget. New failover
  function: 45/50 covered lines, 15/20 branches. Evidence:
  `/tmp/opencode/newapi-failover-unit.json`. Whole-harness coverage still is
  not an 80% claim. One coverage run's wall-clock duration was negative due
  to a clock step; test exit/count and monotonic assertions passed, and a fresh
  exact focused run reported 27 passed in 1.07s.
- Updated live verification docs, README, provider plan, roadmap, QUESTIONS and
  root handoff; save only those and the two reviewed Python files on the feature
  branch. Keep target plans, secret paths/values and private reports outside git.

Next recommendation:

- Save this milestone, then validate a broader bounded coding/tool workload and
  compatibility exceptions before proposing any canary. Avoid more repetitive
  smoke probes; both currently selected free routes are now verified locally.
- VPS v2 still needs deliberate retry/override/proof configuration and separate
  resource/load, persistence/rollback and cutover decisions. Main deploy trigger
  remains untouched. New-API native v1 and the current gateway remain live.

## Cross-harness handoff package - 2026-11-05

- User requested a portable handoff. Added root
  `HANDOFF-CURRENT-newapi-sidecar.md` and `RESUME-newapi-sidecar.txt`.
  They capture current branch/remote, safety/accounting contracts, completed
  evidence and limitations, paths, prerequisites, exact checks and the next
  bounded workload/compatibility milestone. No provider call or code change
  was made for this handoff.
- Re-fetched origin: feature tip remains `ef6b803`, tracked work clean before
  docs edits; existing untracked `.serena/` preserved. Do not reset to the old
  `fde9784` handoff baseline or duplicate remote milestones.
- Older root and `temp-omnirush` handoffs now point to the current package.
  `temp-omnirush` is not the implementation worktree. A sandbox must mount the
  real worktree and binary; launch the new harness from the implementation root.
- Latest implementation verification remains 384 tests plus release/lint/type
  checks; no rerun needed for documentation-only changes. Validate current
  package ASCII, referenced paths, prompt consistency and git diff before save.
- Save this docs-only package on the same feature branch for GitHub portability;
  no merge, deployment, service change, credential access or traffic cutover.

## Deterministic coding/tool workload mode - 2026-11-05

- Re-fetched origin before work: local `feat/newapi-sidecar` and
  `origin/feat/newapi-sidecar` were both `675a199` with no divergence. Existing
  untracked `.serena/` was preserved; no stash/reset/clean was run.
- Added explicit `--workload` to `tools/verify_newapi_live.py`, reusing
  `newapi_benchmark.CASES`, the stdlib transport, ledger/runtime and live broker.
  Five cases (plan/edit/debug/tool/review) run once per verified target under a
  declared ceiling: at most two targets, ten requests, 2560 output tokens, 256
  output tokens per request and a 480-second child guard. The mode is mutually
  exclusive with `--opencode`/`--failover`; smoke, tool-loop and failover
  budgets are unchanged.
- `Case` now declares `requires` from a known capability set; the tool case
  keeps tools and forced `tool_choice`. JSON cases ask for JSON in the prompt
  but omit `response_format`, because live catalogs do not verify structured
  output. Payload output defaults to 512 for existing scripted use; only the
  workload passes 256.
- Live targets advertise only catalog-verified `tools`. Unverified case
  requirements and unknown capability names are skipped/rejected before
  dispatch. The broker enforces per-target and aggregate call ceilings plus the
  256-token cap; credential echo, redirect/proxy denial and owner-only artifacts
  are unchanged. Rate-limit/provider-rejection failures stop that target's
  remaining cases (`target_blocked_after:*`); cooldown-infeasible cases report
  `route_unavailable`.
- `passed` is transport/settlement/coverage only; deterministic correctness is
  reported separately in `quality` (`graded`, `correct`, `all_correct`,
  `by_route`). No prompt, key or raw response is stored in the report.

Validation:

- `python3 -m pytest tests -o addopts= --quiet` -> **392 passed, 1 existing
  `asyncio_mode` warning in 30.12s**.
- `python3 tools/verify_release.py` -> **392 passed**, session validation OK,
  health/route 200, RSS 26168 KB; compilation passed.
- Provider-free real-binary verifier passed on a fresh instance:
  `/tmp/opencode/newapi-proof-zg3ejfrg/report.json` (`passed:true`,
  scripted-local-only). Chat/coding/SSE, entrypoint startup/restart, operator
  catalog/priority, paid exclusion and grade-driven ranking changes passed; the
  benchmark still reports 0/5 for route one and 5/5 for route two.
- New workload tests cover complete five-case dispatch, transport/grade
  separation, unverified-capability skip, over-budget/unknown-capability
  rejection before dispatch, the 256-token/five-call broker budget and
  mutually exclusive modes. `git diff --check` clean.

Limits and next:

- No live provider generation, key access, VPS change, retry/override change or
  cutover occurred. This host has no private `/tmp/opencode/live-targets.json`
  plan and no canonical `MISTRAL_API_KEY`, so a live workload batch could not
  run; that is an environment gap, not a passing live result. Do not treat the
  scripted 0/5 vs 5/5 grades as LLM quality evidence.
- Updated `docs/live-provider-verification.md`, `docs/gateway-compatibility.md`,
  README and the root handoff. Next: run one budgeted live workload batch where
  the owner-provided plan and approved environment key exist, then resolve
  client-usage/compatibility exceptions before proposing any canary. VPS
  resource/load/rollback and authorization gates remain separate.

## Live workload attempt and child wiring fix - 2026-11-04

- Located the existing owner-created plans in `/tmp/opencode` and used
  `newapi-live-tool-target.json` (one OpenRouter `cohere/north-mini-code:free`
  target, owner-only key file). The canonical read-only provider DB was present.
- Found and fixed a real child-entrypoint bug: the parent forwarded `--workload`
  to the isolated child command, but the `child_verify()` call omitted
  `args.workload`, so the first invocation silently ran two smoke requests.
  Added keyword forwarding and a regression test that calls the child CLI path
  and asserts `workload=True`.
- Validation after the fix: focused live-policy suite **36 passed**; full suite
  **393 passed**, one existing `asyncio_mode` warning in 28.65s; `git diff
  --check` clean and no added non-ASCII lines.
- Live workload rerun: 5 planned, 3 attempted, 2 transport-usable and graded
  correct (plan 183 output tokens, edit 123). The debug case returned empty
  content after exactly 256 completion tokens with estimated usage and
  `invalid_response`; the route then cooled down, so tool and review reported
  `route_unavailable`. Workload `passed:false`; `quality` was 2/2 among graded
  cases. Private report: `/tmp/opencode/newapi-live-ht15k438/report.json`.
- Including the two accidental smoke calls from the first invocation, this
  continuation spent five live provider requests. No key was printed, copied
  into New-API, or committed; no VPS change, retry/override change, deployment
  or cutover occurred.
- Next: decide whether the 256-token cap is adequate for reasoning-heavy debug
  cases or declare a larger workload ceiling and rerun; then audit client usage
  and resolve compatibility exceptions before any canary proposal.

## Live workload cap and pacing calibration - 2026-11-04

- Re-fetched origin before work: local `feat/newapi-sidecar` and
  `origin/feat/newapi-sidecar` were both `74f1fb1` with no divergence. Existing
  untracked `.serena/` was preserved; no stash/reset/clean was run.
- Raised `WORKLOAD_OUTPUT_CAP` from 256 to 512 and the declared aggregate
  ceiling from 2560 to 5120 output tokens; the ten-request ceiling and
  five-case coverage are unchanged. Added `wait_for_workload_rpm`, which waits
  once for an exhausted per-minute window only when the retry hint is at most
  65 seconds, then rechecks capacity. Daily exhaustion and provider cooldowns
  still report `route_unavailable`.
- Provider-free tests: focused live-policy **38 passed**; full suite **395
  passed**, one existing `asyncio_mode` warning; `tools/verify_release.py`
  **395 passed**, session OK, health/route 200, RSS 26260 KB and compilation
  OK; `git diff --check` clean.
- Live one-target calibration on OpenRouter
  `cohere/north-mini-code:free`: the first 512-cap run completed 4/5 with 4/4
  correct; `debug` used 323 output tokens, confirming the earlier 256-token
  failure was cap exhaustion. `review` was skipped because the live harness
  caps the target at four requests per minute and the fifth case fell inside
  that window. After the bounded wait, the rerun completed all five cases:
  5/5 correct, actual usage, 0 skips, 233 input/773 output tokens, and
  `passed:true`.
- Private reports:
  `/tmp/opencode/newapi-live-e3rrmblj/report.json` (four-case pacing run) and
  `/tmp/opencode/newapi-live-s2tlwq_a/report.json` (passing five-case run).
  The two calibration batches used nine live provider requests. No key was
  printed, copied into New-API, or committed; no VPS change, deployment,
  retry/override change or cutover occurred.
- Workload coverage now passes on one live free route and one batch. Remaining
  gates: compatibility/client decisions, same-model mapping semantics,
  post-output stream failures with real providers, quotas under load, canary
  entrypoint provisioning, and VPS-40 resource/restart/rollback.

## Installed-client compatibility audit and /health alias - 2026-11-05

- Re-fetched origin before work: local and remote `feat/newapi-sidecar` were both
  `3d8aef9` at the start with no divergence; untracked `.serena/` preserved.
- Read-only OpenCode audit: `~/.config/opencode/opencode.json` is mode 0600 and
  has 117 provider entries. `custom` and `vps-gateway` use
  `http://localhost:8000/v1` and list 15 of 19 legacy aliases;
  `agentic-gateway` has an empty model list. The `summary` agent is pinned to
  `vps-gateway/coding-fast`.
- Read-only `opencode.db` query (1,578 sessions, 81,094 messages):
  `vps-gateway` assistant traffic for coding-elite 189, coding-fast 97 and
  coding-smart 17; latest coding-fast 2026-10-27. Session rows also use
  agent-oracle and auto. `agentic-gateway` has no messages.
- Log strings reference `http://127.0.0.1:8000/health`,
  `http://127.0.0.1:8000/v1/models` and `/v1/chat/completions`; none reference
  embeddings, CLIProxyAPI, token-count, cost-estimate or provider-status paths.
  Two `/v1/responses` strings had no gateway URL and are not counted as calls.
- Decision: Option A (alias translation); dual-path and front-proxy are not
  selected because they leave a second service in the request path or split
  accounting. Added unauthenticated `/health` as a same-path liveness alias with
  the same body as `/healthz`; it exposes no provider/config data. Updated the
  compatibility inventory to 28 sidecar paths, 25 same-path, 0 renamed, 5 absent.
- Verification: focused sidecar server/alias 67 passed; full suite 395 passed
  with one existing asyncio_mode warning; `tools/verify_release.py` 395 passed,
  health/route 200, RSS 26240 KB, compilation OK; `git diff --check` clean;
  real-binary scripted verifier `/tmp/opencode/newapi-proof-3qd538n_` passed
  with OpenCode 1.19.5, one tool loop and paid route excluded.
- Follow-up real-binary regression: `tools/verify_newapi.py` now requires
  `/health` to return the same unauthenticated body as `/healthz`, and sends
  `coding-elite` through the entrypoint with a response-model assertion. The
  `/tmp/opencode/newapi-proof-ubats_oc` run passed (`passed:true`), with both
  liveness paths 200, legacy alias 200, OpenCode 1.19.5 one tool loop, paid
  route excluded and grades 0/5 vs 5/5. Focused 81 passed; full and release
  suites 398 passed; release RSS 26280 KB and compilation OK.
- No key was printed or committed; no VPS change, deployment, retry/override
  change or cutover occurred. Remaining canary gates: external-consumer route
  exceptions, same-model mapping semantics, post-output stream failures, quotas
  under load, canary entrypoint provisioning and VPS-40 resource/restart/rollback.

Next: review the remaining partial/absent routes with each external consumer and
prepare the canary gate evidence; do not deploy or cut over traffic.
