# Deferred decisions and external actions

This file records work intentionally left for a later session because it needs
credentials, host access, or an explicit repository decision.

## Provider and VPS

- Code-capable VPS profiles are now mapped to the existing gateway: `coding-smart`
  handles tool/implementation/debug and `coding-fast` handles verification.
  Credentials remain outside git in the mode-0600 VPS environment file.
- The canary currently routes ordinary chat through the existing LLM gateway;
  decide later whether production traffic should remain there or use direct
  upstream adapters.
- Run provider-backed timeout, 429, 5xx, quota, and streaming tests on VPS 40.
  A local scripted fault harness now covers these paths without touching live
  quotas; live checks remain access-dependent.
- The provider-free side-by-side canary is installed and tested on port 8123;
  the existing gateway remains active on port 8000. Defer reverse-proxy
  cutover until provider configuration and compatibility testing are complete.
- Provider-backed tool-call, structured-output, code-phase, debug-phase, native
  stream, and cooldown checks passed on the side-by-side canary. A real
  OpenCode 1.18.35 request reached the canary through a local SSH tunnel, but
  the existing gateway upstream chain exhausted/cooldowned on the large live
  tool context; no router-side tool stripping was added. Repeat the real tool
  loop after upstream availability recovers before any cutover.
- The VPS has no `opencode` executable. The router's OpenCode signal-header path
  was exercised directly, but a real client session still requires the client
  to be installed or tested from another host.

## Repository integration

- Decide whether this project remains a separate private repository or is
  integrated into `LLM-API-Key-Proxy` through a reviewed branch/PR.
- The existing `LLM-API-Key-Proxy` checkout has unrelated uncommitted changes;
  do not merge or overwrite them without an explicit clean integration point.
- Configure a remote only after choosing the destination repository and access
  policy.

## OpenCode

- Confirm the target OpenCode plugin API/version and preferred installation
  location before adding a version-specific plugin package.
- Verify which phase, action, and verification events are available in the
  deployed OpenCode client.

## Calibration

- Supply redacted outcome JSONL or an approved telemetry export for personal
  success-rate calibration.
- Decide whether semantic outcomes are supplied by tests/tools, the client, or
  an operator review workflow.

## Local chain-routing batch

- The router now supports ordered provider-manager groups, free-only metadata
  import, per-deployment cache hints, bounded warm-cache waits, and same-group
  operational failover. The standalone work is committed as `60c3822` and the
  vendored PR branch contains the synchronized commit `747cca4`; neither has
  been deployed to VPS 40.
- Before enabling metadata-driven provider traffic on VPS 40, run controlled
  provider-specific 429, Retry-After, quota-reset, timeout, 5xx, malformed
  response, and native-stream fault tests using the existing gateway fallback.
- Confirm PR #315 review/merge before enabling the gated gateway integration;
  its current review check is skipped, while local validation remains green.

## New-API continuation - 2026-11-04

- Git fetch and PR refs were inspected, but PR state/review metadata could not
  be queried because no approved machine-user fine-grained PAT was available.
  Do not use the stored non-fine-grained gh token for API calls or create/rotate
  credentials autonomously.
- The user's existing provider accounts should be reused. Read-only local
  inventory: canonical `llm-provider-manager/llm_providers.db` has 204 providers
  and 6,827 models (137 providers enabled); OpenCode has 117 provider entries and
  1,214 model entries. `~/.secrets` exists and was not opened. Endpoint and model
  metadata must come from the canonical DB, not a credential-file search.
- Prior VPS inventory is historical; this session did not SSH to refresh it.
  Real-provider validation should use a small, explicitly free, already-owned
  subset after reviewing stale metadata. No bulk key/catalog import or new
  signup is needed.
- A bounded 2026-11-05 live pass found Groq `/models` available but generation
  403. Do not retry or rotate credentials automatically; catalog success is not
  inference authorization. OpenRouter zero-priced `cohere/north-mini-code:free`
  passed chat/SSE and a real OpenCode read loop. A later Mistral check passed
  chat/SSE, then Mistral/OpenRouter profile-level failover passed in both
  directions using locally injected 429s. Do not equate these different models
  with same-model parity or treat those injected faults as real provider limits.
- The second-provider local gate now passes; next validate broader coding/tool
  workloads and compatibility exceptions before a VPS canary. Continue choosing
  renewable free targets with canonical/live metadata, not bulk scans or unknown
  trial credits. No more real quota is needed for helper safety checks.
- Local TOTP startup works against the pinned disposable binary. Real v2 startup
  still needs approved zero retries, cleared channel overrides and the
  operator-owned proof flow; do not change the live v1 prototype to unblock it.
  Repeated audits can hit the verification limiter. Budget startup/refresh
  operations before enabling an unattended service; do not copy the fixture's
  raised verification budget into production.

## Issue #1083 evaluation (New-API vs LiteLLM, dashboard + free/paid tagging)

- The evaluation doc (`docs/newapi-vs-litellm-eval.md`) is complete, but it
  assumes "OMP" means the opencode-side model/provider catalog (omo/opencode
  config templates plus the shared provider metadata). The term is not defined
  anywhere in this repository — confirm the exact referent and its sync
  contract (what pushes, what pulls, and which side wins on conflict) before
  building any dashboard-to-OMP feed.
- Decide which host would serve a provider dashboard if the recommended thin
  read-only HTML page over `status_api.py` is built: gateway-40 (existing
  gateway process, no extra memory) versus VPS-155 (~273MiB avail, would need
  its own service or an SSH-tunnel reachability check from the dashboard
  consumer).
- If the quota-class tagging from branch `feat/1069-quota-class` is the
  accepted free/paid metadata model, confirm that PR's merge priority so the
  dashboard reads `quota_class` instead of duplicating a parallel tagging
  scheme.

## New-API canary follow-up - 2026-11-05

Resolved in `ec88ffd`:

- VPS strict-config provisioning is applied: `RetryTimes=0`, cleared channel
  3/4 stream overrides, private Agnes aliases on channels 10/11, and four
  audited sidecar routes. The backup is
  `/etc/newapi-sidecar/backups/v2-strict-20261105-052743/`.
- A tool-capable `auto-coding` route is live: authenticated `agnes-kios`
  tool-call request returned 200 with actual `tool_calls`.
- Same-model failover is live-verified: CrowLLM 403 -> `agnes-kios` 200 in two
  attempts. Quota burst/reset is live-verified: four successes, fifth
  `rate_limit` with `Retry-After=60`, success after 61 seconds.
- The 503/429 contract is fixed in `cb6824d`: exhausted local capacity returns
  429, while upstream failure/cooldown keeps its failure class and returns 503
  after `ec88ffd`; focused and full tests pass (405).
- The proof command points at the installed `ec88ffd` helper, and a clean
  startup audit passed. Local OpenCode `vps-gateway` is cut over to `:8124`;
  `custom` remains on legacy `:8000`.

Still open:

- The admin/proof token is short-lived; the current token expires at 05:40:39
  UTC and no unattended refresh flow exists. The canary is active but disabled.
- CrowLLM channel 11 is out of quota; KiosAPI has an observed 2 RPM upstream
  limit versus the local 4 RPM route setting. Recheck provider limits and the
  active route pair before production cutover.
- The real OpenCode process may need a restart to reload the changed provider
  URL; only direct HTTP through the new tunnel was verified.
- Do not claim production readiness or merge/deploy to main without a separate
  approval. Keep the legacy gateway and the timestamped config backup.
