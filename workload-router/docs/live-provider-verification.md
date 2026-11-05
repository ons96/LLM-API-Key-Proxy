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

- One or two explicit providers only: Groq, OpenRouter or Mistral. Other
  providers need reviewed free-only selection policy before this tool can use them.
- Endpoints and funding/rate limits come from read-only `llm_providers.db`.
  Groq additionally needs canonical free/tool/context metadata. OpenRouter
  needs an explicit `:free` model, zero current pricing on all catalog axes,
  known context and advertised `tools` plus `tool_choice`. Mistral needs
  canonical free/tool/context metadata plus current `completion_chat` and
  `function_calling` capabilities, known live context and no deprecation; the
  smaller verified context wins. Mistral eligibility still relies on the
  already-owned free-tier account, not an inference-price endpoint.
- Key files must be bounded owner-only regular files in `~/.secrets` with no
  final-component symlink. Alternatively `key_env` may name exactly the
  provider's canonical `env_var`, already populated by an approved secret
  loader. Exactly one credential source is allowed. No key value is accepted
  in arguments or target JSON; no key is copied into a temporary file.
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

## Cross-provider fallback mode

A plan may combine a canonical environment key and an existing private file:

```json
{
  "targets": [
    {"provider": "mistral", "model": "codestral-latest", "key_env": "MISTRAL_API_KEY"},
    {"provider": "openrouter", "model": "cohere/north-mini-code:free", "key_file": "~/.secrets/openrouter-key"}
  ]
}
```

Populate `MISTRAL_API_KEY` using the existing approved environment/secret loader,
not a literal key in a shell command. This tool does not edit secret stores.

```sh
python3 tools/verify_newapi_live.py \
    --provider-db /home/osees/CodingProjects/llm-provider-manager/llm_providers.db \
    --targets /tmp/opencode/live-targets.json \
    --binary /home/osees/new-api/bin/new-api --artifacts /tmp/opencode \
    --execute --failover
```

This mode replaces the smokes and cannot combine with `--opencode`. It requires
two distinct verified providers, injects a local pre-output 429 in each
direction, and permits only one real 128-output-token completion per provider.
The injected primary never reaches a real provider. Primary order uses fixture
latency values so the test is deterministic, not a performance ranking.

Each direction uses a freshly created owner-only test ledger. Started failures
retain their whole reservation; successful fallbacks settle final usage; only
one attempt per route occurs. Reopening each ledger must retain the failed
reservation and cooldown. There is no quota/cooldown deletion. Do not transplant
fresh test ledgers or fixture latency values into production.

## Deterministic coding/tool workload

`--workload` runs the existing five `newapi_benchmark.CASES` (plan, edit, debug,
tool, review) once per verified target. It replaces, rather than adds to, the
chat/SSE, client and failover workloads.

```sh
python3 tools/verify_newapi_live.py \
    --provider-db /home/osees/CodingProjects/llm-provider-manager/llm_providers.db \
    --targets /tmp/opencode/live-targets.json \
    --binary /home/osees/new-api/bin/new-api --artifacts /tmp/opencode \
    --execute --workload
```

Budget and contracts:

- At most two verified targets, five non-streaming cases each: ten requests and
  2560 output tokens in total, each request capped at 256 output tokens. The
  host broker enforces per-target and aggregate request ceilings before any
  upstream call; the isolated child has a 480-second wall-clock guard.
- Temperature is 0. The tool case keeps its `tools` definition and forced
  `tool_choice`. The four JSON cases require JSON in the prompt but omit
  `response_format`, because live catalogs do not verify structured-output
  support; the scripted verifier still exercises `response_format`.
- Each case declares its required capabilities. Live targets advertise only
  `tools`, and only after `free_model()` verified it from current catalog
  metadata. A case whose requirement is unverified is skipped and reported, not
  guessed. Unknown capability names fail closed before dispatch.
- A target is not retried after a `rate_limit` or `provider_rejection` class
  failure; remaining cases are reported as `target_blocked_after:<class>`.
  Cases infeasible from cooldown/context are reported as `route_unavailable`.
  401/403/429 are never silently substituted or rotated.
- `passed` is a transport, settlement and coverage gate: every planned case
  must be dispatched, return usable output and settle with real (non-estimated)
  usage. Deterministic correctness is reported separately as `quality.graded`,
  `quality.correct`, `quality.all_correct` and per-route counts. A transport
  pass with low `quality.correct` is a model-quality result, not a pipeline
  failure. Reports contain no prompts.
- Broker response bounds, credential-echo rejection, no-redirect/no-proxy
  transport and owner-only artifacts are unchanged from the other live modes.

Provider-free gate:

```sh
python3 -m pytest tests/test_verify_newapi_live.py -o addopts= --quiet
```

The suite covers the complete five-case dispatch, transport/grade separation,
unverified-capability skips, aggregate budget rejection, the 256-token broker
cap and mutually exclusive workload flags. It uses scripted adapters and spends
no provider quota.

### Live workload attempt - 2026-11-04

An owner-provided one-target plan (`/tmp/opencode/newapi-live-tool-target.json`,
OpenRouter `cohere/north-mini-code:free`, existing owner-only key file) used the
declared workload budget. The first invocation exposed a child-entrypoint wiring
bug: the parent passed `--workload` to the isolated child, but `child_verify()`
did not receive it, so two smoke requests ran instead. The child call now
forwards `workload`, with a provider-free regression test covering the child CLI
path; the rerun used the workload path.

The rerun planned five cases but attempted three. `plan` and `edit` returned
correct JSON with actual usage (183 and 123 output tokens). `debug` returned no
content after exactly 256 completion tokens, reported estimated usage and
`invalid_response`; that failure put the route into cooldown, so `tool` and
`review` were reported `route_unavailable`. `quality` was 2/2 correct among
graded cases, but `passed:false` because coverage was incomplete. The cap
exhaustion is consistent with a reasoning-heavy or truncated completion, not
proven as the sole cause; do not raise the cap without a new declared ceiling
and a fresh batch. Private report:
`/tmp/opencode/newapi-live-ht15k438/report.json`. Including the two accidental
smoke calls, this attempt spent five live provider requests; no key was
printed, copied into New-API, or committed.

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

## Second provider and fallback evidence - 2026-11-05

- Mistral catalog: 55 models; active `codestral-latest` advertises chat/tools
  and 256000 context. Canonical free/tool metadata caps the test context at
  200000. Chat and SSE both returned exactly `ok` with actual 8 input/2 output
  tokens each. Two real attempts, no cooldown or estimated usage. Private
  report: `/tmp/opencode/newapi-live-vk3e2kqt/report.json` (`passed:true`).
- Both directions passed through audited New-API private aliases: Mistral
  injected 429 -> real OpenRouter chat (5 input/52 output); OpenRouter injected
  429 -> real Mistral SSE (8/2). Attempt order was exactly the primary then
  fallback; no hidden retry; failed reservation retained and cooldown persisted.
  Only two real provider attempts, because local injected faults consumed no
  upstream quota. Report: `/tmp/opencode/newapi-live-ad33w88d/report.json`.
- The local bridge introduced the faults. This does not claim that an upstream
  spontaneously returned 429 or prove same-model identity across providers;
  the tested routes serve the same sidecar profile but different models.
- Four additional real requests were used in this continuation (two Mistral
  smokes plus two live fallbacks). Groq was not retried and no new signup,
  real credit-funded provider, VPS change or permanent config change occurred.
- Final single-call broker ceiling and explicit persisted-reservation assertion
  were validated with scripted transport only after the live pass, to avoid
  spending more quota. Fixture report:
  `/tmp/opencode/newapi-live-123lfdk9/report.json`; not live-provider evidence.

## Remaining gates

Two live free routes and pre-output failover now pass locally. A live
single-target workload batch was attempted on 2026-11-04; it did not complete
all five cases (see above), so the workload coverage gate still fails. The
remaining work is to decide whether the 256-token cap is adequate for
reasoning-heavy cases or to declare a larger budget, then rerun. Same-model
mapping semantics, post-output stream failures with real providers, quotas under
load, real canary entrypoint provisioning and VPS-40 resource/restart/rollback
gates remain. Do not switch production traffic, clear live New-API overrides or
set its retries to zero based on these results.
