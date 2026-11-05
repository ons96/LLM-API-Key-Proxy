# Current New-API sidecar handoff

Snapshot: 2026-11-05 UTC. Implementation checkpoint: `1586913`.
Authoritative continuation branch: `origin/feat/newapi-sidecar` in
`ons96/LLM-API-Key-Proxy`. A later handoff-only commit is expected.

This is the current cross-harness handoff. Older `HANDOFF-newapi-sidecar.md`,
`temp-omnirush/HANDOFF-newapi-sidecar.md`, and chronological STATE entries are
historical when they conflict with this snapshot. Always inspect newer remote
work before deciding what still needs implementation.

## Copy-paste resume prompt

```text
Continue the New-API sidecar from:
/home/osees/CodingProjects/LLM-API-Key-Proxy-worktrees/wt-newapi-sidecar/HANDOFF-CURRENT-newapi-sidecar.md

Work in that worktree's workload-router/ directory on feat/newapi-sidecar.
Implementation was last verified at 1586913; use the latest branch tip, not a
reset to that hash. First inspect git status, fetch origin, compare commits and
branches, and check relevant PRs if approved GitHub authentication is available.
Remote work may have completed more; do not duplicate it. Preserve .serena/ and
all other pre-existing work. Never stash/reset/clean this worktree blindly.

Read project AGENTS.md/SPEC.md, workload-router/AGENTS.md, this current handoff,
and the latest workload-router/STATE.md entries before editing. The older
workload-router/HANDOFF.md describes another VPS task and is not this assignment.

Next: the installed-client audit is complete and Option A (alias translation)
is selected; `/health` now serves the legacy liveness path. Review the remaining
partial/absent routes for each external consumer, then continue the canary gates
for VPS strict config/proof, resource/load, persistence/rollback and separate
authorization. The live workload passes 5/5 on one free route at a 512-token cap
with a bounded per-minute wait. Do not repeat completed smoke tests merely to
recreate evidence.

Already done: authorized entrypoint startup/restart, operator catalog/priority
and installed OpenCode checks, real OpenRouter chat/SSE/read loop, real Mistral
chat/SSE, controlled pre-output failover to real responses in both directions,
and a provider-free verified deterministic coding/tool workload mode; its child
wiring was fixed after the first live invocation silently ran smoke. The
2026-11-04 one-target live batch passed all five cases after the cap was raised
to 512 and a bounded 4-RPM wait was added (5/5 correct, actual usage, no skips).
The installed-client audit selected alias translation and added a `/health`
liveness alias; latest full and release suites passed 402 tests, and the
provider-free real-binary/OpenCode verifier now checks all 22 discovered
aliases and sends every legacy alias through the entrypoint. Read the handoff
for exact limitations; injected local 429s are not real upstream rate-limit
evidence.

Use existing provider-manager metadata read-only and existing approved secret
sources. Free-only; no paid calls or unknown one-time-credit spending. Never
print, copy into the gateway, rotate, or commit real credentials. Keep current
smoke/tool/failover budgets unchanged unless a separate explicit workload mode
has provider-free tests and a declared generation ceiling.

Commit and push small verified milestones to feat/newapi-sidecar to save work,
using normal secret-scan hooks. Do not merge into main, create duplicate PRs,
force-push, deploy, restart/reconfigure VPS services, change native New-API
retry/stream settings, or cut over traffic. Main has a deployment trigger.
Preserve the legacy gateway as rollback. Document blockers in QUESTIONS.md and
continue safe local work without asking for interactive input.

Update STATE.md and this handoff after the milestone. Report actual results,
remaining gates, and the next concrete action; never claim production readiness
from local or synthetic results.
```

Standalone prompt: `RESUME-newapi-sidecar.txt` at this worktree root.

## Locations and working state

| Asset | Location / state |
| --- | --- |
| Implementation worktree | `/home/osees/CodingProjects/LLM-API-Key-Proxy-worktrees/wt-newapi-sidecar` |
| Component and command cwd | Worktree's `workload-router/` |
| Branch | `feat/newapi-sidecar`, tracks `origin/feat/newapi-sidecar` |
| Last implementation commit | `1586913` (all-alias discovery + entrypoint regression) |
| Old baseline | `fde9784`; do not reset to it or redo its completed work |
| Remote main at snapshot | `76cd928`; not merged into this branch |
| Local changes before handoff | Tracked files clean; existing untracked `.serena/` preserved |
| Canonical metadata, read-only | `/home/osees/CodingProjects/llm-provider-manager/llm_providers.db` |
| Provider-manager schema guidance | Same project's `CLAUDE.md`; some documented counts are stale |
| Existing OpenCode config | `/home/osees/.config/opencode/opencode.json`; never overwritten by these checks |
| Secret-store overview | `/home/osees/CodingProjects/SECRETS.md` |
| Private runtime secrets | `~/.secrets/`, owner-only; Mistral uses existing `MISTRAL_API_KEY` env source |
| Unmodified New-API binary | `/home/osees/new-api/bin/new-api` |
| Installed OpenCode | `/home/osees/.opencode/bin/opencode`, version 1.19.5 |
| Migration decisions, reference only | `/home/osees/CodingProjects/llm-gateway-worktrees/wt-newapi-sidecar/docs/sidecar-decisions.md` |

Migration worktree remains on `work/newapi-sidecar` at `078ce55`, with its
untracked decisions document preserved. Its fetched main was `e983169`; do not
switch to that worktree to edit implementation. The primary legacy gateway
checkout also has unrelated work and must not be used as a clean substitute.

All implementation milestones below are committed and pushed:

| Commit | Milestone |
| --- | --- |
| `b8b882b` | Fixed valid signed channel proofs; verified actual startup/restart |
| `46aa2af` | Operator/catalog/priority checks and installed scripted OpenCode loop |
| `5c4f323` | Budgeted live verifier; real OpenRouter chat/SSE/read loop |
| `ef6b803` | Mistral support and controlled live fallback in both directions |
| `41d2b44` | Deterministic coding/tool workload mode; provider-free verified |
| `994395a` | Forwarded workload to child; live one-target batch recorded |
| `c6bc275` | Raised workload cap to 512; bounded 4-RPM wait; live 5/5 pass |
| `c6da14f` | Installed-client audit; `/health` liveness alias; Option A selected |
| `27e1fd6` | Real-binary `/health` alias + legacy `coding-elite` regression |
| `1586913` | Real-binary all-alias discovery + 19 legacy alias calls |

Git fetch and PR refs were checked before each milestone. No PR head pointed at
these tips. PR review/check/state metadata was not queried because no approved
machine-user fine-grained PAT was available. Do not use stored OAuth/classic
`gh` authentication for API calls. Normal Git pushes and secret hooks succeeded.
A feature-branch push is a backup, not permission to merge or deploy.

## Architecture and invariants

- Final goal: unmodified New-API plus a Python quota/latency/cost-aware sidecar
  on VPS-40, replacing the old gateway only after all gates pass. Current work
  is local validation, not a production replacement.
- Python stdlib, SQLite WAL, bounded workers; fit a 1 GB RAM/weak CPU/no GPU VPS.
  No Go rewrite, Docker, Redis, paid service, upstream fork, or resale layer.
- New-API is AGPL-3.0. Preserve notices. Sidecar distribution license remains
  unresolved; do not relabel it MIT.
- Strict sidecar routes have one private alias per provider/model/key, enabled
  single-key channels, exact model mapping/key fingerprint/endpoint audit and
  `RetryTimes=0`. Relay group is `default`, `cross_group_retry=false`.
- Never bypass New-API on a production dispatch. Real-provider test broker is
  test-only: keys remain in host memory; only allowlisted requests cross an
  inherited socket into the disposable gateway's isolated namespace. Its DB
  stores generated bridge keys, not actual provider keys.
- Reserve before every attempt; start marks network risk; finish settles.
  Cancel only unstarted leases. Unknown/truncated usage retains full charge.
  Started crashes never silently refund. Learned limits only tighten.
- Public profiles: `auto-coding`, `auto-code`, `auto-chat`; 19 legacy aliases
  also accepted. Mapping preserves profile/step intent, not exact old chains.
  `X-Route-Step` and `X-Session-Id` supply hints. Chat has no coding affinity.
- Default output cap 2048; multi-generation requests rejected. Prime streams
  before HTTP headers; never fail over after output has been exposed.
- Unknown price/capability/context/limits fail closed. No trial credit guessed
  into unlimited free usage. Text-only conservative accounting rejects multimodal.
- Keep test DB reset statements confined to the fresh scripted benchmark fixture.
  Live failover uses fresh separate ledgers, never deletes quota/cooldown history.

## Relevant code and docs

All paths in this section are relative to `workload-router/`.

| File | Responsibility |
| --- | --- |
| `quota_ledger.py` | Atomic per-key/shared quotas, conservative settlement, cooldowns/evidence |
| `route_policy.py` | Hard gates, cost/time scoring, bounded exploration/cache hints |
| `newapi_runtime.py` | Validation, aliases, accounted completion/SSE, response/embedding subsets |
| `newapi_server.py` | Authenticated loopback HTTP entrypoint, strict boot audit |
| `provider_adapter.py` | Existing transport, bounded responses/hints, no-redirect mode |
| `newapi_catalog.py` | Authorized channel audit, catalog fetch and additive symlink-safe config refresh |
| `newapi_operator.py` | Preview/apply, audit, priorities, classification, boot-check |
| `newapi_benchmark.py` | Five deterministic plan/edit/debug/tool/review cases, declared capabilities and persisted grades |
| `tools/newapi_channel_proof.py` | Operator TOTP/interactive proof helper |
| `tools/verify_newapi.py` | Scripted providers, real binary, actual entrypoint/operator/client proof |
| `tools/verify_newapi_live.py` | Explicit budgeted live checks with host-memory credential broker: chat/SSE, OpenCode loop, deterministic workload or failover |
| `tests/test_verify_newapi_live.py` | Free/credential/budget/failover/workload regression gates |

Read as needed: `docs/newapi-design.md`, `docs/newapi-v1-v2-roadmap.md`,
`docs/live-provider-verification.md`, `docs/operator-commands.md`,
`docs/channel-proof-helper.md`, `docs/gateway-compatibility.md`,
`docs/responses-and-embeddings.md`, `docs/alias-mapping-proposal.md`,
`docs/free-tier-classes.md`. `STATE.md` is a chronological log; its last entries
are authoritative, not old repeated "next action" sections.

## Verified results and limits

- Exact full suite: **402 passed, 1 existing asyncio_mode warning**.
  `tools/verify_release.py`: 402 tests, session check, health/route 200,
  26272 KB legacy resource smoke, compilation passed. Focused live-policy suite:
  38 passed; focused verifier/server/alias tests: 85 passed.
- Deterministic coding/tool workload mode is implemented, provider-free tested
  and live-passed: five plan/edit/debug/tool/review cases per target, a declared
  ten-request/5120-output-token ceiling, 512-token per-request cap, tools intact,
  catalog-verified capability gating, a bounded per-minute-window wait and
  separate transport/quality reporting. The 2026-11-04 one-target OpenRouter
  batch first confirmed cap exhaustion at 256 tokens, then completed 5/5 with
  actual usage and no skips after the cap/pacing calibration (`passed:true`,
  233 input / 773 output tokens). One live free route and one batch only; not
  sustained-load or canary evidence.
- Installed OpenCode audit (read-only): 117 provider entries; `custom` and
  `vps-gateway` use `localhost:8000`, with `vps-gateway/coding-fast` used by
  97 assistant messages and 263 sessions; coding-elite and coding-smart were
  also used. Logs reference `/health`, `/v1/models` and chat, with no embedding,
  CLIProxyAPI, token-count, cost-estimate or provider-status strings. Option A
  selected; `/health` now returns the liveness body unauthenticated.
- Real-binary scripted verifier passed after the alias-completeness follow-up:
  `/tmp/opencode/newapi-proof-imi83fh6/report.json` (`passed:true`,
  scripted-local-only). Startup and restart returned matching unauthenticated
  `/health` and `/healthz` bodies; authenticated `/v1/models` returned 22
  aliases with all 19 legacy names marked, and every legacy alias returned 200
  through the entrypoint. `coding-elite` still returned 200 after restart and
  preserved its response model. OpenCode 1.19.5 completed one tool loop, the
  paid route stayed excluded, and benchmark grades were 0/5 route one and 5/5
  route two.
- Focused Black/flake8/mypy passed; mypy skipped imported modules and flake8
  ignored existing E402 path bootstraps. No whole-repo type/format claim.
- Secret hooks and added-text ASCII scans passed. No remaining owned test
  processes/listeners at last inspection. `.serena/` remains untracked.
- Actual entrypoint rejects missing/invalid proofs before binding or dispatch,
  accepts pinned binary's 579-byte signed proofs under the 4096-byte header
  bound, and preserves quotas/cooldowns on SIGTERM/restart. Multi-channel TOTP
  same-window authorization was proven locally. Production provisioning is open.
- Scripted proof covers catalog/operator/priority writes, last-good preservation,
  symlink/curated config preservation, paid exclusion and grade-driven ordering.
  Synthetic grades are not real LLM quality scores. Fixture-only critical/web
  verification budgets are raised; do not copy them into production.
- Real OpenRouter `cohere/north-mini-code:free`: fresh all-zero catalog prices,
  tool support/context, chat/SSE `ok`, actual usage, and one real OpenCode
  read-only loop. Final client loop used 3 requests: 6911 input/214 output tokens.
- Groq catalog works but `openai/gpt-oss-20b` generation returned 403. It was
  not retried or automatically rotated. Do not hammer it to get a passing test.
- Real Mistral `codestral-latest`: canonical already-owned free account,
  live chat/function support/no deprecation; conservative context 200000.
  Chat and SSE returned `ok`, 8 input/2 output tokens each.
- Controlled failover: injected local 429 on Mistral -> live OpenRouter chat
  (5/52 tokens); injected local 429 on OpenRouter -> live Mistral SSE (8/2).
  Exact two-attempt order, zero hidden retries, whole failed reservation retained,
  persistent cooldown. Four real requests in the last continuation including
  the two Mistral smokes. No spontaneous upstream-429 claim.
- These providers serve different models within a profile. This is not
  same-model canonical parity, broader SWE quality, sustained load, or VPS proof.
- Historical early scripted 503 flakiness was not causally resolved. A later
  passing run is not proof that all startup races were eliminated.

## Private evidence and portability

Reports below were read directly, not inferred from a truncated success line:

| Report under `/tmp/opencode/` | Result |
| --- | --- |
| `newapi-proof-hsj9s35w/report.json` | Actual entrypoint startup/restart/accounting passed |
| `newapi-proof-exot1afe/report.json` | Operator-written catalog + scripted OpenCode loop passed |
| `newapi-proof-zg3ejfrg/report.json` | Post-workload real-binary scripted proof passed |
| `newapi-proof-3qd538n_/report.json` | Post-health real-binary/OpenCode verifier passed |
| `newapi-proof-ubats_oc/report.json` | Post-liveness health alias + legacy alias verifier passed |
| `newapi-proof-imi83fh6/report.json` | All-alias discovery + entrypoint verifier passed |
| `newapi-live-us6lrwkv/report.json` | Mixed smoke failed overall: Groq 403, OpenRouter chat/SSE passed |
| `newapi-live-pp7vl34a/report.json` | Failed client permission experiment; not successful evidence |
| `newapi-live-gf3j9wix/report.json` | Real OpenRouter OpenCode loop passed |
| `newapi-live-vk3e2kqt/report.json` | Real Mistral chat/SSE passed |
| `newapi-live-ad33w88d/report.json` | Both controlled live fallback directions passed |
| `newapi-live-123lfdk9/report.json` | Post-live hardening check with scripted transport only |
| `newapi-live-ht15k438/report.json` | One-target live workload failed coverage at 256-token cap |
| `newapi-live-2v7xsx0u/report.json` | Unintended smoke from child wiring bug; not workload evidence |
| `newapi-live-e3rrmblj/report.json` | 512-cap calibration: 4/4 correct, fifth blocked by 4-RPM window |
| `newapi-live-s2tlwq_a/report.json` | Live workload passed 5/5 after bounded RPM wait |

Artifacts are owner-only and contain generated credentials, logs and test DBs.
Do not copy/commit/dump those directories to move harnesses. Public summaries
are already in the committed docs. `/tmp` evidence may vanish on another host;
absence means evidence is unavailable there, not that the code is missing.
Select reports by modification time, never lexicographic directory order.

## Commands and environment gaps

Run checks from the absolute component path, not the old sandbox directory.

```sh
cd /home/osees/CodingProjects/LLM-API-Key-Proxy-worktrees/wt-newapi-sidecar

git status --short --branch
git fetch --prune origin
git log --oneline HEAD..origin/feat/newapi-sidecar
git rev-list --left-right --count HEAD...origin/feat/newapi-sidecar
```

Fast-forward only if tracked work is clean and the histories permit it; preserve
all other work. Never reset to an old handoff hash. After relevant code changes:

```sh
cd workload-router
python3 -m pytest tests/test_verify_newapi_live.py -o addopts= --quiet
python3 -m pytest tests -o addopts= --quiet
python3 tools/verify_release.py
git diff --check
python3 tools/verify_newapi.py --binary /home/osees/new-api/bin/new-api \
    --artifacts /tmp/opencode --opencode /home/osees/.opencode/bin/opencode
```

The last command is provider-free but needs Linux `unshare`/`ip`, installed
OpenCode and an existing artifact parent. Binary pinned to `v1.0.0-rc.36`, SHA-256:
`5d9b254721c205f4f442848a4af5b58617a8c0ee090366732cfff90455e957f3`.
Do not substitute a different binary silently. A sandbox must mount the worktree
and required binary paths; launch the harness in this worktree, not
`temp-omnirush`. If unavailable, report the limitation and continue unit work.
Do not run `rtk pytest` (historically collected no tests), pipe checks into
head/tail, or follow an unrelated stale MCP index. Preserve actual exit codes.

Live verifier usage and private target-plan format are in
`docs/live-provider-verification.md`. Default is catalog-only. Existing limits:
smokes <=2 requests/target with 128 output tokens; tool loop <=3 requests on one
target with 512 output tokens; failover <=1 real 128-token request per provider;
workload = five 512-token cases/target, max ten requests/5120 output tokens,
with a bounded wait for a short exhausted per-minute window.
The provider-free workload gate is `tests/test_verify_newapi_live.py`.
No key value in argv/JSON. `key_env` must equal canonical `env_var`; normal file
sources stay in `~/.secrets`. Use the existing approved secret loader, not manual
key copying. Do not auto-run live generation on resume.

## Next bounded milestone

1. ~~Broaden the bounded coding/tool workload~~ **Implemented and live-passed**
   at `c6bc275`. `--workload` runs the five deterministic cases per verified
   target with a ten-request/5120-output-token ceiling, 512-token cap, tools
   intact, verified-capability gating, a bounded per-minute-window wait and
   separate transport/quality reporting. The 2026-11-04 one-target OpenRouter
   batch passed 5/5 with actual usage and no skips after the cap/pacing
   calibration. This is one live free route and one batch; it is not sustained
   load or canary evidence. Stop on 401/403/429 and never substitute a
   paid/credit model.
2. ~~Update compatibility exceptions~~ **Audit complete; Option A selected** in
   `docs/gateway-compatibility.md` (28 vs 30 paths; 25 same-path, 0 renamed,
   5 absent). The installed OpenCode client uses legacy aliases over chat, and
   `/health` now serves the liveness path. Remaining partial/absent routes are
   explicit exceptions to review per external consumer; Responses/embeddings
   callers, process-local sessions, stale provider status and CLIProxyAPI remain
   separate gates. Do not add empty shims or claim parity from a 200 response.
3. Run a canary only after those compatibility decisions and the VPS strict
   config/proof provisioning, resource/load, persistence/rollback and separate
   authorization gates pass.
4. Save tested milestones to the feature branch, update STATE/current handoff.
   Do not promote to production from local or synthetic results.

## Production boundary (historical inventory only)

Prior remote work recorded legacy gateway on VPS-40 `:8000`, native New-API v1
on `:8001`, and older P0 canary on loopback `:8123`. This continuation never
SSH'ed or changed them. Last recorded native v1 had `RetryTimes=3`, forced
stream overrides on channels 3/4, and no 2FA enrollment; sidecar v2 was not
serving. v1 model-first fallback is separate from strict v2 accounting.
Changing these settings would alter v1 behavior, not merely unblock a test.

Require a separately scoped authorization for VPS service/configuration changes,
proof provisioning, load/rollback drills and traffic cutover. Keep legacy state
and configuration intact. Production inventory is historical, not freshly
verified. Main has a deployment workflow, so no main merge/push as a backup.
