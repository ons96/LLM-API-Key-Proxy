# New-API sidecar handoff for OmniRush

> **OmniRush sandbox note (2026-11-03).** This session's folder is
> `/home/osees/CodingProjects/LLM-API-Key-Proxy-worktrees/wt-newapi-sidecar`,
> and the sandbox mounts exactly that folder. The `workload-router` paths in
> this handoff are therefore valid. `/home/osees/CodingProjects/temp-omnirush`
> and the main `LLM-API-Key-Proxy` checkout are NOT visible from here.
>
> The base image is Debian 12 with python3.11 but no pytest, and its Python is
> marked externally managed. If `python3 -m pytest` reports `No module named
> pytest`, install it once in this folder's box (installed packages persist):
>
> ```sh
> python3 -m pip install --user --quiet --break-system-packages pytest
> ```
>
> Safety boundary unchanged: no commit, push, deploy, credential change or
> traffic cutover.

Snapshot: 2026-11-03. This file supersedes the stale conversation summaries about
an unapplied benchmark fix. The current verifier contains that fix, and a saved
full local integration report passes. Production readiness is NOT established.

## Current checkpoint

- Branch `feat/newapi-sidecar` tracks `origin/feat/newapi-sidecar`.
- Commits: `f00de12 feat: add New-API sidecar routing` and
  `1777e36 docs: record sidecar verification`.
- Worktree is clean.
- Full provider-free suite: 231 passed, with the existing
  `Unknown config option: asyncio_mode` warning.
- Release, resource, and 50-request/8-worker concurrency smoke checks passed.
- Targeted provider-fault, runtime, server, quota, adapter, streaming, and
  benchmark tests passed: 77 + 25 tests.
- No VPS deployment, live-provider test, OpenCode client test, credential
  provisioning, merge, or traffic cutover has occurred.

## Resume prompt

Read this handoff and continue the New-API sidecar in
`/home/osees/CodingProjects/LLM-API-Key-Proxy-worktrees/wt-newapi-sidecar/workload-router`.
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

Read `workload-router/AGENTS.md`, `STATE.md`, `README.md`,
`docs/specs/p0-router.md`, and `docs/newapi-design.md`. The older `HANDOFF.md`
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
   `auto-chat`. Step hints use `X-Route-Step`; sessions use `X-Session-Id`.
   `RouterState.set_preferred(session_id, "auto-coding", route_id)` is the correct
   argument order. Chat does not reuse coding preference/cache state.
5. The default output cap is 2048 and is forwarded upstream. Multi-generation
   requests are rejected. Streams prime before headers, replay buffered data once,
   and never fail over after output is exposed. Grading happens after settlement.

Endpoints currently exposed: `/healthz`, `/v1/models`,
`/v1/router/diagnostics`, `/v1/router/route`, `/v1/chat/completions`.
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

The last recorded full Python suite before the catalog/benchmark suites was
199 passed. The later benchmark/catalog run was 31 passed (12 plus 19). A fresh
full-suite result for the current tree still needs to be recorded. The existing
warning is `Unknown config option: asyncio_mode`.

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
   retirement is separately approved.

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
