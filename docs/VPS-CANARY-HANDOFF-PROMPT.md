# VPS-40 canary handoff prompt

Use this prompt in a host-integrated coding harness that runs in WSL and can
use the user's existing SSH agent. Do not paste or expose private keys,
provider keys, API tokens, passwords, backup codes, or private payloads.

## Prompt

You are continuing the New-API sidecar project in:

`/home/owens/CodingProjects/LLM-API-Key-Proxy`

Target host: `ubuntu@40.233.101.233`.

The existing LiteLLM gateway is production/rollback infrastructure. Do not
stop it, replace it, delete it, change its credentials, or cut over traffic
without explicit approval from the user.

Read first:

- `AGENTS.md`
- `HANDOFF-newapi-sidecar.md`
- `workload-router/AGENTS.md`
- `workload-router/STATE.md`
- `workload-router/docs/newapi-design.md`
- `docs/new-api-deploy.md`
- this file

Current local evidence:

- Branch: `feat/newapi-sidecar`
- Local provider-free suite: 234 passed, one existing pytest warning.
- Pinned New-API fixture verifier passed with `v1.0.0-rc.36`.
- The pinned binary hash is documented in `docs/new-api-deploy.md`.
- No live-provider or traffic-cutover proof exists yet.
- Stage 1 inventory against VPS-40 completed 2026-11-03 (read-only). New-API is
  already installed and running there: `new-api.service` on `*:8001`, pinned
  `v1.0.0-rc.36` with a matching sha256, SQLite at `/opt/new-api/data`, separate
  `newapi` account, firewall-enforced Tailscale-only access. Stage 2 is therefore
  gap-closing on the existing install; do not reinstall or replace it. No sidecar
  install, sidecar unit, or channel-key proof helper exists on the host, so
  Stage 3 remains blocked on the operator proof helper.
- Admin password change on the existing New-API install is NOT verified (the
  deploy note records a default `root` credential). Confirm it interactively; do
  not test or print credentials from this harness.

## Rules

1. Inspect before changing anything: host status, services, ports, memory,
   disk, current gateway configuration locations, and rollback state.
2. Keep all new work side-by-side and loopback-only where possible.
3. Use the existing VPS service and credentials only as documented; never print
   secrets or copy them into the repository.
4. Prefer `nala` over `apt-get` for system packages when available.
5. Never use `latest`; pin New-API versions and verify checksums.
6. Do not modify CI files, delete data, rotate credentials, or cut over traffic
   without explicit user approval.
7. Make small commits on a dedicated branch if code changes are required.
8. After every change, run a narrow check first and record exact results.

## Staged work

### Stage 1 — Read-only inventory

Run read-only SSH checks for:

- OS/architecture and available memory/disk
- active services and listening ports
- existing LiteLLM gateway status
- whether New-API is already installed or has a prior data directory
- current firewall/bind addresses
- systemd unit names and rollback locations

Return a redacted report. Do not change anything in this stage.

### Stage 2 — New-API side-by-side install

Only after the user approves the inventory:

- install the pinned New-API binary in a separate root-owned location;
- create a separate unprivileged service account and SQLite data directory;
- bind New-API to a non-conflicting loopback port;
- configure a separate systemd unit with bounded memory/restart behavior;
- preserve the existing gateway unchanged;
- verify health, startup, restart persistence, and memory use.

Do not add real provider credentials yet unless the user explicitly approves
the credential step.

### Stage 3 — Sidecar canary

After Stage 2 passes:

- install the sidecar from the reviewed branch in a separate directory;
- configure its SQLite database and loopback New-API URL;
- configure the operator-controlled channel proof helper without putting its
  secret material in the repo or command-line arguments;
- run `/healthz`, `/v1/models`, route preview, and authenticated chat checks;
- verify the old gateway remains healthy and unchanged.

### Stage 4 — Real provider validation

Only after explicit approval:

- add one approved free provider/channel at a time;
- verify channel audit, free-only exclusion, quotas, cooldowns, 429 failover,
  streaming, tool calls, and usage accounting;
- run an actual OpenCode tool loop through the canary;
- record provider/model/channel identifiers but never keys or raw prompts.

### Stage 5 — Cutover decision

Stop and report evidence. Do not cut over automatically. Cutover requires a
separate explicit approval after rollback, persistence, resource, compatibility,
and failure tests pass.

## Return format

Report:

- commands run and redacted results;
- files/services changed;
- tests and smoke checks with exact outcomes;
- resource measurements;
- rollback command and whether it was tested;
- blockers and the single recommended next action.

After the host-integrated harness completes a stage, return to this project in
the current harness and ask for a review before advancing to the next stage.
