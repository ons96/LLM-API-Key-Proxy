# New-API staged roadmap: v1 model-first, v2 full behavior

Date: 2026-11-03
Branch: `feat/newapi-sidecar`
Status: the narrow v1 functional gates passed on 2026-11-03 (direct
`agnes-3.0-flash`), but v1 is **not a production replacement** and no cutover is
authorized. LiteLLM remains the live gateway.

This is the continuation point for a session started from another machine. Read
this with `../../HANDOFF-newapi-sidecar.md`, `provider-import-plan.md`,
`gateway-compatibility.md`, and `alias-mapping-proposal.md`.

## Decision

The replacement is staged deliberately:

| Stage | Runtime | What it proves | What it deliberately does not do |
| --- | --- | --- | --- |
| **v1 prototype** | New-API native routing, `RetryTimes=3` | Direct `agnes-3.0-flash` can chat, stream, call tools, and fail over across KiosAPI/CrowLLM | Legacy 19-alias parity, semantic prompt classification, quota ledger, full gateway API |
| **v2 parity** | New-API plus the Python sidecar/front classifier, likely `RetryTimes=0` | Prompt-aware routing, the 19 legacy aliases, quota/cooldown/telemetry, compatibility routes, rollback-tested replacement | Nothing may be cut over until every gate below passes |

v1 is a **prototype namespace**, not a smaller version of the old gateway. A
client that needs a legacy alias or the old `auto` semantics is a v2 client.

The reason for the split is architectural: New-API is a channel/model router. It
has no prompt classifier, no phase/tool-intent routing, no quota ledger and no
old gateway compatibility surface. Those live in the Python sidecar, which is
not running and cannot pass its strict audit against the current New-API config.

## v1 scope and acceptance

### In scope

- Native New-API model-first routing with the existing `RetryTimes=3`.
- The verified direct model `agnes-3.0-flash`; `router/*` aliases remain useful
  as single-provider smoke routes but are not reliable fallbacks.
- Two-provider failover where the same alias exists on more than one channel.
- Streaming, OpenAI tool calls, usage accounting, and an OpenCode tool loop.
- Keeping LiteLLM on `:8000` untouched; v1 clients opt in by changing only their
  base URL/token/model.

### Out of scope

- Do not expose the 19 legacy names in v1. In particular, do not map legacy
  `auto` to New-API's literal `auto` model: the old `auto` is a semantic router,
  while New-API's `auto` is an upstream model on HCNSEC/IAMHC.
- Do not claim old prompt classification, coding/chat profiles or quota-aware
  ordering.
- Do not bulk-import the 204-provider catalog.
- Do not change `RetryTimes`, clear channel overrides, enable 2FA, deploy the
  sidecar, or cut over traffic as part of v1.

### v1 acceptance gates

- [x] `router/fast` and `router/balanced` return usable streamed completions
  (`content:"ok"`, `finish_reason:"stop"`, usage present).
- [x] `agnes-3.0-flash` returns 200 with either channel 10 (KiosAPI) or channel 11
  (CrowLLM canonical) disabled; both channels were restored to `status:1`.
- [x] A tool-shaped request returns a valid `tool_calls` payload (`router/tools`
  and `agnes-3.0-flash`).
- [x] An OpenCode 1.19.5 tool loop completes through New-API with `tool_use` and
  the expected file content.
- [x] The old gateway remains healthy and unchanged throughout.
- [x] v1 rollback is client-only: point the client back at the old gateway.

The `router/*` aliases do **not** satisfy the failover gate; both are Groq-only
in practice. The passing fallback route is the direct model `agnes-3.0-flash`.
See "Live evidence" below.

## v2 scope and hard gates

v2 is the full replacement path. The recommended architecture remains an
unmodified New-API plus the quota-aware Python sidecar. A separate front
classifier is possible but must be explicitly chosen; it is not the current
design. The language/resource decision is recorded in
`sidecar-language-decision.md`: Python stays, with a 128 MiB canary guardrail.

v2 must deliver or explicitly document every one of these:

- the 19 legacy aliases and their step/profile hints;
- prompt classification and tool/phase-aware routing;
- free-only filtering, quota reservations, cooldowns and telemetry ordering;
- streaming completion and stream-failure behavior;
- the old gateway's client-facing routes, or a reviewed compatibility exception
  for each missing route;
- real-provider failover, OpenCode tool loops, restart/persistence and rollback.

The sidecar cannot start against the current install until these are resolved:

| Blocker | Current state | Required for v2 |
| --- | --- | --- |
| `RetryTimes` | `3` | `0` for strict one-attempt accounting |
| Forced streaming | Channels 3 and 4 have `param_override={"stream":true}` | Clear before a non-stream sidecar canary |
| Channel-key proof | Admin API hides keys; proof helper not installed | Operator-controlled `NEWAPI_CHANNEL_KEY_PROOF_COMMAND` |
| 2FA | No enrollments (`two_fas` rows: 0) | Authorized proof path for channel audit |
| Compatibility | Sidecar exposes 15 routes; 17 of the old 30 paths absent; model detail, provider list, TPS, cost estimation, token counting, and model management are partial | Route coverage or explicit exceptions |
| Legacy aliases | Sidecar source accepts them; not running | Deployed sidecar and live validation |

Changing `RetryTimes` to `0` and clearing the stream overrides are **v2
configuration changes**, not v1 cleanup. They would disable or alter the native
multi-channel behavior proved in v1, so they need explicit approval.

### V2 implementation checkpoint - 2026-11-03

- Added deterministic classification for the legacy `auto` alias in
  `newapi_runtime.py`. It inspects the latest user message; an explicit
  `X-Route-Step` remains authoritative.
- `implement` / `refactor` / code requests -> `auto-coding` / `implement`.
- `debug` / `traceback` / failing tests -> `auto-coding` / `debug`.
- `verify` / `verification` -> `auto-coding` / `verify`.
- General and tool-bearing requests -> `auto-chat` / `chat`, preserving
  `Work.tools`; an explicit code intent wins over the tools bit.
- All other aliases and explicit profiles keep their existing profile/step
  behavior. This is a deterministic baseline, **not** semantic-router parity.
- Added read-only `GET /v1/models/{model_id}` (old basic envelope for any
  non-empty id, aliases marked `deprecated`) and `GET /v1/providers` (sorted
  configured route-provider IDs). Neither reserves quota or dispatches.
- Added `GET /v1/tps-stats` over measured successful streaming ledger samples,
  plus `GET /v1/model/list`, `GET /v1/model/status`, and
  `POST /v1/model/parse-command`. The management routes use the 22 aliases and
  never switch a session; TPS rejects the gateway's `export` side effect.
- Added `POST /v1/cost-estimate` using the cheapest known configured route
  price. Unknown pricing stays unknown; no route is selected or dispatched.
- Added explicit free classes (`free_unlimited`, `free_limited`, `free_credits`,
  `paid`, `unknown`) and separate notional credit prices. Free routes keep zero
  customer prices; one-time credits still deplete and rank below unlimited
  routes. Diagnostics and operator previews expose the class. See
  `free-tier-classes.md`.
- Added `POST /v1/token-count` as a conservative UTF-8 byte upper bound with
  `estimated:true` and `method:"utf8_byte_upper_bound"`; no model tokenizer is
  bundled. It never dispatches.
- Retained Python after measuring the sidecar at **26,984 KiB** RSS following
  50 discovery requests. See `sidecar-language-decision.md`; the proposed
  service cap is 128 MiB, pending live canary measurement.
- Provider-free tests: **89 targeted**, **324 total**, one existing
  `asyncio_mode` warning. No VPS change, deployment, or cutover.

## Live snapshot (2026-11-03)

Read-only inventory of the VPS-40 install:

- New-API `v1.0.0-rc.36`, active on `:8001`; LiteLLM gateway active on `:8000`.
- 9 channels, 110 abilities, all channels `status:1`, `RetryTimes=3`.
- 7 usable providers: Groq, OpenRouter, HCNSEC, IAMHC, Bluesminds, CrowLLM,
  KiosAPI. Channel 5 is a deliberate bad-key test channel and is not a provider.
- `two_fas` has 0 rows; 2FA is disabled.
- Channels 3 and 4 still carry `param_override={"stream":true}`.

| Channel | Id | Exposes | Notes |
| --- | ---: | --- | --- |
| `groq-free` | 3 | `router/fast`, `router/balanced`, `router/tools`, direct Groq models | Stream override |
| `openrouter-free` | 4 | `router/fast`, `router/balanced`, `router/code`, `router/reasoning`, direct OpenRouter models | Stream override |
| `badkey-test-865` | 5 | `health-probe-bad-model` | Deliberate failure probe |
| `gw:hcnsec` | 6 | `auto` | Upstream literal `auto` |
| `gw:iamhc` | 7 | `auto` | Upstream literal `auto` |
| `gw:bluesminds` | 8 | `nvidia/ising-calibration-1.5-31b` | Verified earlier |
| `gw:crowllm` | 9 | 73 chat models | Includes `agnes-3.0-flash:free` |
| `gw:kiosapi` | 10 | `agnes-3.0-flash` + 3 models | Verified |
| `gw:crowllm-canonical` | 11 | `agnes-3.0-flash` -> `agnes-3.0-flash:free` | Priority 40 |

### Live evidence

Already proven:

- `auto` returned 200 with `gw:hcnsec` disabled, via `gw:iamhc`; both re-enabled.
- `agnes-3.0-flash` returned 200 with `gw:kiosapi` disabled, via
  `gw:crowllm-canonical`; both re-enabled.
- Parsed streaming calls to `router/fast` and `router/balanced` returned
  `content:"ok"`, `finish_reason:"stop"` and usage. `router/fast` used Groq
  `openai/gpt-oss-20b`; `router/balanced` used Groq `openai/gpt-oss-120b`.
- `router/tools` returned a valid `lookup_status` tool call with
  `{"service":"gateway"}`, `finish_reason:"tool_calls"` and usage.
- `router/reasoning` returned 404 because its mapped model is paid-only:
  `nemotron-3-super-120b-a12b`.
- The `router/fast`/`router/balanced` failover probe found both routes are
  Groq-only: disabling Groq produced 429 (fast) and a paid-only 404 (balanced);
  disabling OpenRouter still succeeded through Groq. Do not present them as
  fallback routes.
- `agnes-3.0-flash` succeeded in both directions: without channel 10 it used
  CrowLLM canonical; without channel 11 it used KiosAPI. It also returned a
  valid `lookup_status` tool call.
- OpenCode 1.19.5 completed a local tool loop through a temporary SSH tunnel and
  temporary config: exit 0, one `tool_use` event, output contained
  `hello-newapi`. No permanent OpenCode config or key was written; the VPS does
  not have OpenCode installed.

Excluded from v1:

- `router/reasoning` until a free upstream mapping exists.
- `router/tools` and `router/code` as v1 defaults: each exists on only one
  channel today, so they do not demonstrate cross-provider fallback.

### Note on the two virtual-model files

The live VPS file is authoritative for the old chains. The checked-in
`config/virtual_models.yaml` in this repo is not byte-identical to the live file;
the live file has been regenerated/reordered. The 19 alias names and their v2
profile/step mapping are stable, but do not copy chain order from the local file
as if it were the VPS state.

## v1 candidate routes

These are New-API route names, not legacy aliases.

| New-API route | Channels | Providers | v1 role | Evidence |
| --- | --- | --- | --- | --- |
| `router/fast` | 3 + 4 | Groq + OpenRouter | Single-provider chat smoke; **not fallback** | Groq 200; Groq disabled -> 429 |
| `router/balanced` | 3 + 4 | Groq + OpenRouter | Single-provider chat smoke; **not fallback** | Groq 200; Groq disabled -> paid-only 404 |
| `router/tools` | 3 | Groq | Tool-call smoke; no fallback | 200 `tool_calls` |
| `router/code` | 4 | OpenRouter | Single-provider code smoke; content untested | status-only 200 |
| `router/reasoning` | 4 | OpenRouter | Excluded | 404, paid-only upstream |
| `agnes-3.0-flash` | 10 + 11 | KiosAPI + CrowLLM | **Primary v1 fallback/model route** | both directions 200; tool call 200 |
| `auto` | 6 + 7 | HCNSEC + IAMHC | Reserved; do **not** present as legacy `auto` | Failover proven, semantic mismatch |

The safest v1 demo is the direct model `agnes-3.0-flash`: it is the only route
with a proven cross-provider fallback and a proven tool call. `router/tools` is a
tool smoke for one provider; `router/fast` and `router/balanced` are not fallback
examples. No legacy alias is needed for v1.

## Full legacy alias matrix

v1 status is "no" for every row. The v2 column uses the approved sidecar
profile/step mapping; it preserves request acceptance and capability intent, not
the old chain identity. The "current live head" column is from the VPS file
snapshot, abbreviated to the first one or two entries.

| Legacy alias | Current live chain head | v1 | v2 profile / step | Why v1 is not ready |
| --- | --- | --- | --- | --- |
| `coding-free` | `crowllm minimax-m2.5:free`, `septorlabs minimax-m2.5-free` | no | `auto-coding` / `implement` | Only CrowLLM is live; SeptorLabs was quota-limited |
| `p0-minimax-m3` | same CrowLLM/SeptorLabs minimax chain | no | `auto-coding` / `implement` | One live provider; no verified fallback |
| `glm5-elite` | `cliproxyapi gemini-2.5-flash` | no | `auto-coding` / `implement` | No GLM channel; cliproxyapi/Gemini path not live |
| `coding-elite` | `cliproxyapi gemini-3-flash`, `gemini-2.5-flash` | no | `auto-coding` / `implement` | No cliproxyapi channel; old chain leaders dead |
| `coding-smart` | `cliproxyapi gemini-3-flash`, `gemini-2.5-flash` | no | `auto-coding` / `implement` | No cliproxyapi channel; no equivalent verified model pair |
| `coding-fast` | `cliproxyapi gemini-2.5-flash`, `gemini-3-flash` | no | `auto-coding` / `implement` | No cliproxyapi channel; `router/fast` is not the same model set |
| `title-fast` | `nvidia nemotron-content-safety-reasoning-4b` | no | `auto-chat` / `chat` | NVIDIA models tested 404/410 |
| `chat-elite` | `gemini-3-pro-preview`, `gcli gemini-3.1-pro-preview` | no | `auto-chat` / `chat` | Gemini/GCLI paths not live |
| `chat-smart` | `g4fpool gemini-3.8-flash`, `cliproxyapi gemini-3-flash` | no | `auto-chat` / `chat` | g4f/cliproxyapi not available in New-API |
| `chat-fast` | `g4fpool qwen3.8-27b`, `cliproxyapi gemini-2.5-flash` | no | `auto-chat` / `chat` | g4f/cliproxyapi not available; `router/fast` is an approximation only |
| `auto` | semantic router, safe default `cliproxyapi gemini-2.5-flash` | **no** | `auto-chat` / `chat` | Name collision with literal New-API `auto`; v2 classifier required |
| `chat-rp` | `nvidia mistral-small-4-119b`, `mistral-large-3-675b` | no | `auto-chat` / `chat` | NVIDIA models unavailable; no roleplay-specific verified route |
| `agent-oracle` | `nvidia llama-3.3-nemotron-super-49b-v1` | no | `auto-coding` / `plan` | NVIDIA models unavailable |
| `agent-explore` | `nvidia llama-3.3-nemotron-super-49b-v1` | no | `auto-coding` / `explore` | NVIDIA models unavailable |
| `agent-librarian` | `nvidia llama-3.3-nemotron-super-49b-v1` | no | `auto-coding` / `explore` | NVIDIA models unavailable |
| `agent-build` | `nvidia llama-3.3-nemotron-super-49b-v1` | no | `auto-coding` / `implement` | NVIDIA models unavailable |
| `agent-metis` | `nvidia llama-3.3-nemotron-super-49b-v1` | no | `auto-coding` / `plan` | NVIDIA models unavailable |
| `agent-momus` | `nvidia llama-3.3-nemotron-super-49b-v1` | no | `auto-coding` / `final-review` | NVIDIA models unavailable |
| `safe-coding` | `privaite claude-opus-4-8` | no | `auto-coding` / `implement` | No PrivAiTe channel; PII-scrubbing behavior absent |

Do not "fix" this table by mapping every alias to the nearest live model. That
would accept requests but silently change what each name means. The v2 router is
the place for semantic compatibility.

## Migration checklist

### v1 — native model-first prototype

1. [x] Inventory channels, abilities, aliases and current gates.
2. [x] Status-test the five `router/*` aliases; exclude `router/reasoning`.
3. [x] Parse real `router/fast` and `router/balanced` SSE completions; content,
   finish and usage were present.
4. [x] Run a tool-call request through `router/tools` and `agnes-3.0-flash`;
   both returned valid `tool_calls`.
5. [x] Test channel 3/4 failover. Both `router/*` routes are Groq-only; record
   the 429/paid-only-404 results and do not use them as fallbacks.
6. [x] Test `agnes-3.0-flash` with channels 10 and 11 disabled in turn; both
   directions returned 200 and were restored to `status:1`.
7. [x] Run one OpenCode 1.19.5 tool loop against New-API; exit 0 and one
   `tool_use` event. The temporary config/tunnel were removed.
8. [x] Record results here and in `../STATE.md`. Keep the old gateway untouched.
9. [ ] Decide whether the narrow v1 is useful enough as a permanent tier or is
   only a stepping stone to v2.

v1 rollback: change the client's base URL/model back to the old gateway. No
service change, no data deletion, no credential change.

### v2 — full behavior

1. [ ] Choose the v2 architecture explicitly: sidecar + New-API (recommended) or
   a separate front classifier. Do not mix native `RetryTimes=3` with strict
   sidecar accounting.
2. [ ] Obtain explicit approval for the v2 configuration changes:
   `RetryTimes=0`, clearing channel stream overrides, and the channel-key proof
   path (2FA enrollment only with operator authorization).
3. [ ] Install the sidecar loopback-only with a separate unit and DB.
4. [ ] Pass the strict audit: unique private aliases, single-key channels, exact
   model mapping, credential fingerprint, endpoint identity, relay token.
5. [ ] Serve the 19 legacy aliases and prompt/phase classification; verify that
   aliases with the same profile/step are documented as equivalent. The
   deterministic `auto` baseline is implemented; semantic-router parity, full
   phase inference and running-sidecar validation remain open.
6. [ ] Cover quota, cooldown, telemetry ordering, stream failure and free-only
   enforcement with provider-free tests.
7. [ ] Close or explicitly document the remaining missing gateway paths (17 at
   this checkpoint); update `gateway-compatibility.md`.
8. [ ] Run bounded live provider tests, an OpenCode tool loop, restart and
   persistence checks, and a full rollback drill.
9. [ ] Obtain explicit cutover approval. Keep LiteLLM and its configuration
   intact until retirement is separately approved.

v2 rollback: stop the sidecar, point clients back to the old gateway, leave
New-API running, and restore the previous sidecar DB/config from backup if
needed. Never delete the old gateway as part of the switch.

## Non-negotiable safety boundary

- The LiteLLM gateway on `:8000` stays active until an explicit cutover approval.
- No credential rotation, secret printing, or secret commit.
- No deletion of provider/catalog data; dead entries stay disabled.
- Free-only, 1 GB VPS budget, pinned versions; no Docker, Go rewrite, New-API
  fork or paid service.
- A passing local suite is not production proof. Each stage needs its own live
  gate and rollback check.

## References

- `../../HANDOFF-newapi-sidecar.md` — safety boundary and current handoff.
- `provider-import-plan.md` — 9-channel inventory and model-first rationale.
- `gateway-compatibility.md` — 30-path old-gateway surface vs sidecar.
- `alias-mapping-proposal.md` — v2 19-alias profile/step mapping.
- `newapi-design.md` — strict sidecar accounting and audit contract.
- `../../docs/new-api-deploy.md` — VPS layout, version and rollback.
- `../STATE.md` — chronological checkpoints.
