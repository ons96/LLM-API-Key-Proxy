# Current New-API sidecar handoff

Snapshot: 2026-11-05 UTC. Implementation checkpoint: `ec88ffd`; feature branch
`feat/newapi-sidecar` is pushed. A later docs-only commit may advance the tip.

Authoritative continuation branch: `origin/feat/newapi-sidecar` in
`ons96/LLM-API-Key-Proxy`. Work in the component cwd
`workload-router/`; preserve the existing untracked `.serena/` directory and
all unrelated work.

## Copy-paste resume prompt

```text
Continue the New-API sidecar from:
/home/osees/CodingProjects/LLM-API-Key-Proxy-worktrees/wt-newapi-sidecar/HANDOFF-CURRENT-newapi-sidecar.md

Work in workload-router/ on feat/newapi-sidecar. Fetch origin, inspect git
status, and use the latest branch tip; do not reset or clean this worktree.
Preserve .serena/ and all pre-existing work.

The sidecar is active only as the loopback canary on VPS port 8124. The
legacy gateway on port 8000 is still the rollback path. Implementation release
is ec88ffd. Strict New-API configuration, a tool route, same-model failover,
local quota burst/reset, and a reversible local OpenCode vps-gateway cutover
have live evidence. Do not claim production readiness.

Remaining work: durable admin/proof token provisioning for unattended restart,
re-testing the real OpenCode process after its config reload, provider quota
limits on KiosAPI/CrowLLM, and an approved production cutover/PR to main.

Never print, copy into the repository, rotate, or commit credentials. Free-only.
Keep the legacy gateway and backups intact. Do not merge to main without
explicit approval; main has a deployment trigger.
```

Standalone prompt: `RESUME-newapi-sidecar.txt` at the worktree root.

## Current live state

| Asset | State |
| --- | --- |
| Implementation worktree | `/home/osees/CodingProjects/LLM-API-Key-Proxy-worktrees/wt-newapi-sidecar` |
| Component cwd | `workload-router/` |
| Branch | `feat/newapi-sidecar`, pushed through `ec88ffd` |
| Sidecar release | `/opt/newapi-sidecar/releases/ec88ffd` |
| Canary unit | `newapi-sidecar-canary.service`; **active, disabled** |
| Canary endpoint | `127.0.0.1:8124` on the VPS; loopback only |
| Legacy gateway | `:8000` active; `:8001` New-API active |
| Sidecar DB | `/var/lib/newapi-sidecar/sidecar.db` |
| Routes file | `/etc/newapi-sidecar/routes.json` |
| Strict-config backup | `/etc/newapi-sidecar/backups/v2-strict-20261105-052743/` |
| Unit backups | `/etc/systemd/system/newapi-sidecar-canary.service.bak-before-{bfe91ff,cb6824d,ec88ffd}` |
| Client config backup | `/home/osees/.config/opencode/opencode.json.bak-before-sidecar-cutover-20261105-054407` |
| Local sidecar tunnel | `127.0.0.1:8124 -> VPS 127.0.0.1:8124`; process `212370` at this snapshot |
| Local legacy tunnel | `127.0.0.1:8000 -> VPS 127.0.0.1:8000`; process `418` at this snapshot |

The canary is intentionally **not enabled**. If the VPS reboots, it will not
return until an operator refreshes the short-lived admin/proof token and starts
the unit. The current admin token was refreshed for this run and expires at
`05:40:39 UTC`, so treat it as expired by the time this is read. The proof
command now points at
`/opt/newapi-sidecar/releases/ec88ffd/tools/newapi_channel_proof.py --mode=totp`.

## Strict New-API configuration applied

- `RetryTimes=0` was applied through the admin option API and read back.
- Channels 3 (`groq-free`) and 4 (`openrouter-free`) have empty
  `param_override` fields; the previous stream overrides were cleared.
- Channel 3 retains the exact `router/tools -> openai/gpt-oss-120b` mapping.
- Channel 10 (`gw:kiosapi`) has private alias
  `router/agnes-kios -> agnes-3.0-flash`.
- Channel 11 (`gw:crowllm-canonical`) has private alias
  `router/agnes-crow -> agnes-3.0-flash:free`.
- Channel-key hashes were compared before and after the metadata updates and
  were unchanged. No key value was printed or committed.
- Four sidecar routes were applied:
  - `bluesminds-canary` — channel 8, `auto-chat`, chat-only
  - `groq-tools` — channel 3, `auto-coding`, tool-capable
  - `agnes-kios` — channel 10, `auto-coding`, tool-capable
  - `agnes-crow` — channel 11, `auto-coding`, tool-capable, currently upstream
    quota-limited
- Startup audit verified the configured channels with the TOTP proof helper.
  The sidecar has no provider key material in its route database.

## Verified live gates

- **Tool-capable route:** HTTP POST through `:8124/v1/chat/completions` with
  `tools` and `tool_choice=required` returned **200**, route `agnes-kios`, and
  a real `tool_calls` array.
- **Same-model failover:** a live sidecar-runtime harness forced `agnes-crow`
  first. It failed with `provider_rejection` (CrowLLM’s 403 quota response), then
  the second attempt succeeded on `agnes-kios`; result route `agnes-kios`,
  attempts `2`. This is same-underlying-model (`agnes-3.0-flash`) failover, but
  the pair is asymmetric because the CrowLLM side is currently out of quota.
- **Local quota burst and reset:** a live `north-mini-code:free` route with
  `limits={"rpm":4}` completed four calls, rejected the fifth locally with
  `rate_limit` and `retry_after=60.0` in 1 ms, and completed the next call after
  a 61-second wait. This exercises the new quota-429 path through the runtime,
  not the HTTP server; the server mapping has a focused passing test.
- **Client cutover:** local OpenCode provider `vps-gateway` now points at
  `http://localhost:8124/v1` with the sidecar client key. `/v1/models` returned
  22 model IDs, and all 15 `vps-gateway` models are in that inventory.
  `coding-fast` returned **200** in 0.88 s on `agnes-kios`.
- **Rollback:** `custom` still points at `http://localhost:8000/v1`; the legacy
  tunnel is untouched; legacy `/health` returned **200**. The old `vps-gateway`
  base URL and key are preserved in the timestamped config backup.

## Tests and commits

- Full provider-free suite: **405 passed**, with the existing
  `asyncio_mode` pytest warning.
- `cb6824d` — quota-only routing dead ends return HTTP 429 with
  `Retry-After`; generic failures remain 503.
- `ec88ffd` — cooldown after upstream failure keeps the real failure class
  instead of being masked as a synthetic quota 429.
- Focused runtime/server suite after the final code change: **61 passed**.
- No secret was printed, copied into the gateway, rotated, or committed.

## Known limitations and next action

- CrowLLM channel 11 returns a quota/balance 403. The `agnes-crow` route remains
  configured as a failover candidate, but the working same-model path is KiosAPI.
- KiosAPI reports an upstream limit of about **2 requests/minute**; the local
  route limit is 4 RPM. A tool request after recent KiosAPI traffic returned
  429. Do not present the local 4-RPM setting as the upstream limit.
- A 16-token `coding-fast` call initially returned 503 `invalid_response`; a
  normal 256-token request succeeded. Treat very small generation caps and
  transient provider errors as separate client checks.
- The cutover is local to this OpenCode config. The real OpenCode process may
  need a restart/reload to pick it up; only direct HTTP through the tunnel was
  verified. `custom` and the legacy gateway remain untouched for rollback.
- The admin/proof token flow is short-lived. Unattended restart remains the
  main operational blocker; refresh the token and proof command, then start the
  canary and verify the audit before relying on it after a reboot.
- No production cutover, main merge, or deployment approval is implied by this
  local canary evidence.

Immediate next action: implement or document a durable operator flow for the
short-lived admin/proof token, then retest a real OpenCode session through the
cutover and review provider limits before proposing production cutover.
