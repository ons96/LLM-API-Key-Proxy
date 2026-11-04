# Gateway to sidecar compatibility inventory

Read-only inventory taken 2026-11-03 against the live VPS-40 gateway and the
sidecar source on `feat/newapi-sidecar`. No credentials were used: the gateway's
`/openapi.json` answered unauthenticated, and the virtual-model names came from
`config/virtual_models.yaml` keys only (chains were not copied).

This documents the gate "inventory the old gateway's actual clients, virtual
aliases and API routes; the new HTTP surface is not yet a full compatibility
replacement". Both halves of the gap are real and larger than previously
recorded.

## Method

- `curl http://127.0.0.1:8000/openapi.json` on VPS-40 -> HTTP 200, 30 paths.
- `virtual_models` keys from `/home/ubuntu/LLM-API-Key-Proxy/config/virtual_models.yaml`
  -> 19 aliases.
- Sidecar surface read from `newapi_server.py` (`do_GET` / `do_POST`) and
  `PROFILE_ALIASES` in `newapi_runtime.py`.
- Client traffic was not inspected, so route *usage* is unknown; only route
  *availability* is established.

## HTTP surface

Sidecar exposes fourteen paths: `GET /healthz` (unauthenticated), `GET /v1/models`,
`GET /v1/models/{model_id}`, `GET /v1/providers`, `GET /v1/tps-stats`,
`GET /v1/model/list`, `GET /v1/model/status`,
`GET /v1/router/diagnostics`, `POST /v1/router/route`,
`POST /v1/model/parse-command`, `POST /v1/chat/completions`,
`POST /v1/cost-estimate`, `POST /v1/responses` and `POST /v1/embeddings`
(all Bearer-authenticated, 1 MiB body cap).

| Gateway path | Sidecar | Status |
| --- | --- | --- |
| `/v1/chat/completions` | same path | **covered** (plus SSE, `X-Session-Id`, `X-Route-Step`) |
| `/v1/models` | same path | **partial** — returns the 22 accepted aliases, not the gateway's upstream model list |
| `/health` | `/healthz` | **renamed** — monitors must change path |
| `/` | — | missing |
| `/v1/models/{model_id}` | same path | **partial** — returns the old basic `{id, object, created, owned_by}` envelope for any non-empty id; known aliases carry `deprecated`; no upstream model-info stats |
| `/v1/responses` | same path | **partial** — text/messages, function tools, text-only streaming; no `previous_response_id` or non-function tools |
| `/v1/embeddings` | same path | **partial** — only aliases named in `SIDECAR_EMBEDDING_ALIASES`; text input only |
| `/v1/token-count` | — | missing |
| `/v1/cost-estimate` | same path | **partial** — cheapest known configured route price; unknown pricing returns `cost:null` and `source:"unknown"` |
| `/v1/providers` | same path | **partial** — sorted provider IDs referenced by the configured ledger routes, not the gateway's provider-plugin catalog |
| `/v1/tps-stats` | same path | **partial** — aggregates measured successful streaming samples from the sidecar ledger; `export=true` is rejected |
| `/v1/model-info/stats` | — | missing |
| `/v1/model/list` | same path | **partial** — 22 sidecar aliases and generated descriptions, not the gateway's exact model catalog |
| `/v1/model/status` | same path | **partial** — reports `auto-chat` as the default and the 22 aliases; no per-session current model |
| `/v1/model/switch` | — | missing |
| `/v1/model/detect-intent` | — | missing |
| `/v1/model/parse-command` | same path | **partial** — validates sidecar aliases and parses `/model`, `/use`, and shortcuts; it never switches a session |
| `/v1/model/session/{session_id}` | — | missing |
| `/api/providers/best` | — | missing |
| `/api/providers/health` | — | missing |
| `/api/providers/status` | — | missing |
| `/api/providers/status/{provider_name}` | — | missing |
| `/api/providers/history` | — | missing |
| `/api/providers/history/{provider_name}` | — | missing |
| `/api/providers/export/csv` | — | missing |
| `/api/cliproxyapi/config` | — | missing |
| `/api/cliproxyapi/health` | — | missing |
| `/api/cliproxyapi/models` | — | missing |
| `/api/cliproxyapi/status` | — | missing |
| `/api/cliproxyapi/backend/{backend_name}` | — | missing |

Totals: 11 gateway paths are served at the same path (1 covered, 10 partial),
1 is renamed, and 18 of the 30 gateway paths are still absent.

## Model aliases

Originally zero overlap: the sidecar accepted three profile names and rejected
everything else with HTTP 400, so every client pinned to a gateway alias would
have broken at cutover. The 2026-11-03 alias shim closed this: the sidecar now
accepts 22 names (3 profiles plus the 19 gateway aliases), advertises the legacy
ones with `"deprecated": true`, and still rejects unknown names. Mapping is
profile-and-step, not the gateway's exact chains; see
`docs/alias-mapping-proposal.md` for what is and is not preserved.

## What still differs at cutover

1. OpenCode profiles or scripts pinned to `coding-smart`, `coding-elite`, `auto`
   and the other 16 aliases are accepted, but resolve to a profile/step
   equivalent rather than the gateway's exact chain.
2. Model discovery UIs -> 22 aliases only, no upstream model list; `/v1/model/list`
   advertises the same alias set rather than the gateway's eight-name catalog.
3. Responses API clients (`/v1/responses`) -> 404. Now served for the text and
   function-tool subset; `previous_response_id`, image/audio input and
   streaming with tools still return a bounded 400.
4. Embedding/RAG callers -> 404. Now served only when the operator lists an
   audited alias in `SIDECAR_EMBEDDING_ALIASES`; otherwise 400.
5. Dashboards using `/api/providers/*` -> 404; `/v1/providers` returns only
   route-provider IDs, and `/v1/tps-stats` has ledger samples only.
6. Uptime probes on `/health` -> 404 unless moved to `/healthz`.
7. Token-counting tooling -> 404; cost estimation is available only from
   ledger pricing, not the gateway's model-info service.

## Options (recorded, not decided)

- **A. Alias translation in the sidecar.** Map each gateway alias to a profile or
  explicit route, and serve the full alias list from `/v1/models`. Smallest
  change, keeps every client config working; needs a decision because the public
  profile contract is currently documented as `auto-coding` / `auto-code` /
  `auto-chat`.
- **B. Permanent dual-path.** Gateway keeps the non-chat routes, sidecar serves
  chat only, on a separate port. No alias work, but two services stay live and
  the gate's "not a full replacement" caveat becomes permanent.
- **C. Front proxy split.** Caddy routes `/v1/chat/completions` and `/v1/models`
  to the sidecar and the rest to the gateway. Preserves URLs, but splits
  accounting: some spend bypasses the ledger.

## Unknowns

- Whether any live client actually calls the 18 absent routes.
- Gateway `/v1/models` response shape versus the sidecar's 22-alias list.
- Whether TPS consumers expect the gateway's `export=true` file side effect;
  the sidecar rejects it to remain read-only.
- Whether model-detail or provider-list clients depend on upstream metadata; the
  sidecar returns only the basic alias envelope and configured route providers.
- Whether gateway auth and sidecar Bearer auth ever differ in practice (gateway
  `/v1/models` returns 401 unauthenticated; `/health` and `/docs` are open).
- Route-level client usage requires a request log review, which has not been
  done.
