# Provider sweep - 2026-11-17

Run from the VPS gateway checkout, 2026-11-17T22:51:47Z, against `config/router_config.yaml`.

**Result: 5 verified, 19 unverified, 0 dead** over 24 routes (24 providers, 22 blocklisted skipped).

## Coverage

Run with `--max-models 1`, so this measures **provider availability, one model each**, not every configured model. The config holds 357 routes; a full per-model pass is `python scripts/probe_providers.py --max-models 0 --sweep-report <path>` (opt-in, ~357 requests).

## The rule this run applied

`dead` is reserved for transport-level failure only - DNS, connection refused, TLS, reset. A 401/403 is our key, 402 is our quota, 404 is our mapping, 5xx is their application, a timeout is slowness. None of those is evidence an API is gone, so none of them was archived.

## Routes

| Provider | Verdict | Reason | Disposition |
|---|---|---|---|
| agnes_ai | unverified | model_not_found | ignore (provider not worth keeping) |
| bluesminds | unverified | InternalServerError | ignore (provider not worth keeping) |
| cerebras | unverified | model_not_found | ignore (provider not worth keeping) |
| crowllm | unverified | InternalServerError | ignore (provider not worth keeping) |
| dlab-community | unverified | auth_error | ignore (provider not worth keeping) |
| freemodel | unverified | quota_exhausted | wait for reset |
| freemodel-anthropic | unverified | quota_exhausted | wait for reset |
| g4fpool | unverified | quota_exhausted | wait for reset |
| gcli | unverified | auth_error | ignore (provider not worth keeping) |
| gemini | unverified | model_not_found | ignore (provider not worth keeping) |
| groq | unverified | model_not_found | ignore (provider not worth keeping) |
| larprouter | unverified | APIStatusError | ignore (provider not worth keeping) |
| mistral | unverified | auth_error | ignore (provider not worth keeping) |
| nous-inference | unverified | model_not_found | fix mapping |
| nvidia | unverified | APIStatusError | ignore (provider not worth keeping) |
| openrouter | unverified | quota_exhausted | wait for reset |
| respan | unverified | model_not_found | ignore (provider not worth keeping) |
| septorlabs | unverified | rate_limited | fine, retry later |
| together | unverified | quota_exhausted | wait for reset |
| coffeevps | verified | - | keep |
| freegg | verified | - | keep |
| jasper | verified | - | keep |
| kiosapi | verified | - | keep |
| pollinations | verified | - | keep |

## Decisions

Owner decision 2026-11-17: of the failing providers, only **nous-inference** is worth fixing. `groq`, `gemini`, `cerebras`, `openrouter` and `mistral` are not good enough to spend effort enabling, so their stale mappings and rejected keys stay as they are - they remain in the catalog as `unverified` capacity and are never deleted.

## Next actions for the fixable one

1. Fetch `nous-inference`'s live `/v1/models` and replace the six stale ids in its `free_tier_models`.
2. Re-probe just that provider: `python scripts/probe_providers.py --provider nous-inference --max-models 0`.
3. Leave everything else alone until the next weekly re-probe of `dead` + `unverified`.
