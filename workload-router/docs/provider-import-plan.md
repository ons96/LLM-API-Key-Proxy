# Provider import and model-first routing

Date: 2026-11-03. Goal: let a caller pick a model name and have New-API serve it
from any provider that has that model, falling back when one fails.

## Inventory

- `/home/ubuntu/llm-provider-manager/llm_providers.db`: 204 providers, 6,827
  models, 164 fallback edges, 18 virtual models.
- `config/router_config.yaml`: 60 provider entries, 44 enabled.
- `.env`: 197 non-empty keys/base URLs.
- 137 provider rows are enabled; 36 survived the base-URL/key filter for the
  latest probe batch.
- 3,231 unique model ids; 391 are listed by at least two of those providers.

## Live findings (2026-11-03)

The catalog and its `last_verified` dates are stale, the telemetry is about 80
days old, and `provider_status.db` reports every provider down. Only live calls
are trustworthy.

| Provider | Live result |
| --- | --- |
| HCNSEC | `/models` 200, `auto` chat 200 |
| IAMHC | `/models` 200, `auto` chat 200 |
| Bluesminds | `/models` 200, `nvidia/ising-calibration-1.5-31b` chat 200 |
| Groq | existing channel, `openai/gpt-oss-20b` 200 |
| OpenRouter | existing channel, `cohere/north-mini-code:free` 200 |
| CrowLLM | `/models` 200 (114), `qwen-3.6-27b:free` chat 200 |
| KiosAPI | `/models` 200 (18), `agnes-3.0-flash` chat 200 |
| Cerebras | 403 (no free usage) |
| Together | 403 (no free usage) |
| Supacoder | 403 |
| FreeTheAI | 530 |
| GCLI | 401 invalid key |
| NVIDIA | `/models` 200, tested models 404/410 |
| Mistral | `/models` 200, chat 429 |
| Gemini | 2.5 Flash retired (404); 3.8 Flash 503 high-demand |
| Pooled | `/models` 200, first model chat 503 quota |

In the latest 15-provider chat batch, these advertised models but failed a
16-token chat: Agnes AI (429 daily quota), Atessa (429 credits exhausted),
Blaze Free (502), Dext (400/403), FreeModel (401 balance), Koyeb (500 malformed
upstream), Navy (403 free plan paused), Nous (404 requires credits), OpenCode Zen
(429 free limit), P0 (429 pool paused), PaxSenix (401 key required), SeptorLabs
(429 50/day), TokenRouter (403 credit limit). Re-probe after quotas reset; do
not create channels on `/models` alone.

## Model-first design

Use New-API as the router, not the sidecar.

- One channel per provider, with that provider's model list. Several channels
  may expose the same model name; New-API picks by priority/weight and retries
  another channel on failure.
- Keep `RetryTimes` above zero. Setting it to `0` (the sidecar's strict
  accounting requirement) disables the cross-channel fallback this goal needs.
- Where providers use different upstream ids for the same model, give each
  channel a `model_mapping` from the canonical name to its upstream id. Start
  with the high-overlap families (`gemini-*`, `gpt-*`, `glm-*`, `kimi-*`).
- Import dead or unverified providers as disabled channels (`status: 2`) so the
  catalog is retained without being used. Do not delete catalog rows.
- The sidecar's strict one-alias-per-channel audit is incompatible with
  multi-provider fallback. This is the v1 native design only; v2 would require
  `RetryTimes=0` and is not active. See `newapi-v1-v2-roadmap.md` for the staged
  scope and gates.

## Pilot applied

Created through the admin API:

| Channel | Id | Model | Status |
| --- | ---: | --- | --- |
| `gw:hcnsec` | 6 | `auto` | enabled |
| `gw:iamhc` | 7 | `auto` | enabled |
| `gw:bluesminds` | 8 | `nvidia/ising-calibration-1.5-31b` | enabled |
| `gw:crowllm` | 9 | `qwen-3.6-27b:free` (+72 chat models; 7 embedding ids removed) | enabled |
| `gw:kiosapi` | 10 | `agnes-3.0-flash` (+3 models) | enabled |
| `gw:crowllm-canonical` | 11 | `agnes-3.0-flash` -> `agnes-3.0-flash:free` | enabled |

New-API served `auto`, the Bluesminds model, `qwen-3.6-27b:free`, and
`agnes-3.0-flash` with HTTP 200 through relay token 21. Both `auto` channels
remain enabled. The explicit failover test passed:
`POST /api/channel/6/status` with `{"status": 2}` disabled `gw:hcnsec`, and
`auto` still returned HTTP 200 with "ok" through `gw:iamhc`; `gw:hcnsec` was
then re-enabled and confirmed `status: 1`. The status route is
`POST /api/channel/:id/status`; the generic `PUT /api/channel/` deliberately
rejects a `status` field.

The canonical channel adds a second failover path: `gw:crowllm-canonical`
(id 11, priority 40) serves `agnes-3.0-flash` and maps it to CrowLLM's
`agnes-3.0-flash:free`; New-API's abilities table now lists both channel 10 and
channel 11 for that model. The cross-provider canonical fallback test passed: with `gw:kiosapi` (id 10)
disabled, `agnes-3.0-flash` still returned HTTP 200 with "ok" through
`gw:crowllm-canonical` (id 11); channel 10 was then re-enabled and confirmed
`status: 1`.

Parsed v1 gates on 2026-11-03: `agnes-3.0-flash` passed both-direction
KiosAPI/CrowLLM failover and a tool call; OpenCode 1.19.5 passed a tool loop.
`router/fast` and `router/balanced` turned out to be Groq-only, so they are not
fallback examples. See `newapi-v1-v2-roadmap.md`.

## Next

- Follow the staged plan in `newapi-v1-v2-roadmap.md`. v1 is limited to verified
  native routes; the 19 legacy aliases and prompt classification are v2.
- Probe each remaining provider's `/models`, then one 15-token chat per candidate
  model before enabling its channel.
- Add verified providers in small batches; keep the rest disabled.
- Add canonical mappings for more overlapping ids, then consider broader
  catalog import.
