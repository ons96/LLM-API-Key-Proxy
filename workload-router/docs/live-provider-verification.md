# Budgeted live New-API verification

`tools/verify_newapi_live.py` is a separate, explicit live check. The default
only fetches selected provider catalogs; it never runs generation unless
`--execute` is supplied. It reuses the existing stdlib transport, quota ledger,
management audit, pinned binary and client helpers. No VPS or permanent gateway,
OpenCode configuration or canonical database is modified.

## Inputs and safeguards

Create an owner-controlled JSON target plan outside the repository:

```json
{
  "targets": [
    {
      "provider": "openrouter",
      "model": "cohere/north-mini-code:free",
      "key_file": "~/.secrets/openrouter-key"
    }
  ]
}
```

This example is a 2026-11-05 tested selection, not a permanent availability
promise. The live catalog must still list it with tools and zero prices.

- One or two explicit providers only: Groq or OpenRouter. Other providers need
  reviewed free-only selection policy before this tool can use them.
- Endpoints and funding/rate limits come from read-only `llm_providers.db`.
  Groq additionally needs canonical free/tool/context metadata. OpenRouter
  needs an explicit `:free` model, zero current pricing on all catalog axes,
  known context and advertised `tools` plus `tool_choice`.
- Key files must be bounded owner-only regular files in `~/.secrets` with no
  final-component symlink. No key value is accepted in arguments or target JSON.
- Catalog/auth/429 failures fail closed; no automatic same-provider retries or
  paid fallback. A failed generation blocks that target for the rest of the run.
- A host-process broker keeps real keys in memory. A local inherited socket
  bridges only allowlisted requests into a fresh loopback network namespace.
  New-API stores generated bridge credentials, not the real account keys.
  Provider HTTPS requests refuse redirects and ignore ambient HTTP proxies.
- The broker independently enforces selected model, one generation, request
  size, output cap and total attempts. It checks credential echoes and bounds
  response bytes/time. Started attempts still settle in the sidecar ledger.
- All databases/logs/configs/reports stay in newly created `0700` artifacts;
  files are `0600`. Reports contain timing/usage/status, never prompts or keys.
  Processes are cleaned up by exact owned PID/process-group, not broad kills.

## Commands

Run from `workload-router/`:

```sh
# Catalog preflight only: no generation tokens.
python3 tools/verify_newapi_live.py \
    --provider-db /home/osees/CodingProjects/llm-provider-manager/llm_providers.db \
    --targets /tmp/opencode/live-targets.json

# Chat and native SSE: at most two 128-output-token requests per target.
python3 tools/verify_newapi_live.py \
    --provider-db /home/osees/CodingProjects/llm-provider-manager/llm_providers.db \
    --targets /tmp/opencode/live-targets.json \
    --binary /home/osees/new-api/bin/new-api --artifacts /tmp/opencode --execute

# One target, actual OpenCode read loop: at most three 512-output-token requests.
python3 tools/verify_newapi_live.py \
    --provider-db /home/osees/CodingProjects/llm-provider-manager/llm_providers.db \
    --targets /tmp/opencode/live-targets.json \
    --binary /home/osees/new-api/bin/new-api --artifacts /tmp/opencode --execute \
    --opencode "$(command -v opencode)"
```

The client check replaces, rather than adds to, the chat/SSE smoke workload.
It allows only `read` of a generated `proof.txt` in a disposable Git worktree.
Initializing that worktree matters: OpenCode 1.19.5 uses `/` as the non-Git
worktree root and matches read permissions against paths relative to that root.
HOME/XDG/config/model-cache are isolated; plugins, model fetching and updates are
disabled. The real model's read call is actually executed; output is not scripted.
OpenCode may also request a small-model/title completion; it shares the same
three-request budget. Reported output tokens may include reasoning tokens.

Local tests remain provider-free:

```sh
python3 -m pytest tests/test_verify_newapi_live.py -o addopts= --quiet
```

## Evidence from 2026-11-05

- Groq catalog: 11 models. `openai/gpt-oss-20b` generation returned 403; no
  retry or second Groq request was made. Catalog availability is not inference
  authorization. This target remains blocked pending operator/account review.
- OpenRouter catalog: 418 models. `cohere/north-mini-code:free` had zero prices,
  context 256000 and tools. Chat and SSE returned exactly `ok` through audited
  New-API private aliases and the sidecar. Reported usage: 5 input / 90 output
  tokens (chat), 5 / 62 (SSE); reasoning/text TTFT remained separate.
- First live client attempt used an absolute permission rule and lacked a Git
  root. OpenCode denied all three read calls; the budget stopped generation.
  That was a test permission failure, not provider downtime. The stricter
  relative-file permission plus disposable Git root was verified using scripted
  providers before another live call.
- Final real OpenCode 1.19.5 loop completed one read of the disposable file and
  returned exactly `hello-newapi-live-proof`. Three requests all returned 200
  and settled with actual usage: 2264/89, 2321/56, 2326/69 input/output tokens.
  Total final-loop usage: 6911 input / 214 output. No estimated usage or cooldown.
- Total generation across this session: nine provider attempts, including the
  rejected Groq call, two OpenRouter smokes, three failed-permission client
  calls and three final client calls. No additional live requests were used for
  coverage/format/release checks.

Private evidence:

- `/tmp/opencode/newapi-live-us6lrwkv/report.json` (mixed-provider smoke, failed
  overall because Groq 403; OpenRouter chat/SSE passed).
- `/tmp/opencode/newapi-live-pp7vl34a/report.json` (permission failure, failed).
- `/tmp/opencode/newapi-live-gf3j9wix/report.json` (real client loop, passed).

These reports are not committed. The successful client run used an audited
Runtime/Server in the disposable instance; separate provider-free evidence
covers the production entrypoint. A post-run broker/cleanup hardening pass was
verified only with scripted host responses, not additional live quota.

## Remaining gates

Only one live free-provider route passed. Two-provider live fallback, broader
coding workloads, quotas under load, real canary entrypoint provisioning and
VPS-40 resource/restart/rollback gates remain. Do not switch production traffic,
clear live New-API overrides, or set its retries to zero based on this result.
