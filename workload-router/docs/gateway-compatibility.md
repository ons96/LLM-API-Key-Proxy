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

Sidecar exposes seven paths: `GET /healthz` (unauthenticated), `GET /v1/models`,
`GET /v1/router/diagnostics`, `POST /v1/router/route`, `POST /v1/chat/completions`,
`POST /v1/responses` and `POST /v1/embeddings` (all Bearer-authenticated,
1 MiB body cap).

| Gateway path | Sidecar | Status |
| --- | --- | --- |
| `/v1/chat/completions` | same path | **covered** (plus SSE, `X-Session-Id`, `X-Route-Step`) |
| `/v1/models` | same path | **partial** — returns only the 3 profile aliases, not models |
| `/health` | `/healthz` | **renamed** — monitors must change path |
| `/` | — | missing |
| `/v1/models/{model_id}` | — | missing |
| `/v1/responses` | same path | **partial** — text/messages, function tools, text-only streaming; no `previous_response_id` or non-function tools |
| `/v1/embeddings` | same path | **partial** — only aliases named in `SIDECAR_EMBEDDING_ALIASES`; text input only |
| `/v1/token-count` | — | missing |
| `/v1/cost-estimate` | — | missing |
| `/v1/providers` | — | missing |
| `/v1/tps-stats` | — | missing |
| `/v1/model-info/stats` | — | missing |
| `/v1/model/list` | — | missing |
| `/v1/model/status` | — | missing |
| `/v1/model/switch` | — | missing |
| `/v1/model/detect-intent` | — | missing |
| `/v1/model/parse-command` | — | missing |
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

Totals: 2 exact path matches, 1 renamed, 2 partially covered new routes;
25 of the 30 gateway paths are still absent.

## Model aliases

Originally zero overlap: the sidecar accepted three profile names and rejected
everything else with HTTP 400, so every client pinned to a gateway alias would
have broken at cutover. The 2026-11-03 alias shim closed this: the sidecar now
accepts 22 names (3 profiles plus the 19 gateway aliases), advertises the legacy
ones with `"deprecated": true`, and still rejects unknown names. Mapping is
profile-and-step, not the gateway's exact chains; see
`docs/alias-mapping-proposal.md` for what is and is not preserved.

## What a naive cutover would break

1. OpenCode profiles or scripts pinned to `coding-smart`, `coding-elite`, `auto`
   and the other 16 aliases -> 400 on every request.
2. Model discovery UIs -> 3 profile names only, no model list.
3. Responses API clients (`/v1/responses`) -> 404. Now served for the text and
   function-tool subset; `previous_response_id`, image/audio input and
   streaming with tools still return a bounded 400.
4. Embedding/RAG callers -> 404. Now served only when the operator lists an
   audited alias in `SIDECAR_EMBEDDING_ALIASES`; otherwise 400.
5. Dashboards and monitors using `/v1/providers`, `/api/providers/*`,
   `/v1/tps-stats` -> 404.
6. Uptime probes on `/health` -> 404 unless moved to `/healthz`.
7. Token counting and cost estimation tooling -> 404.

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

- Whether any live client actually calls the 27 absent routes.
- Gateway `/v1/models` response shape versus the sidecar's 3-item list.
- Whether gateway auth and sidecar Bearer auth ever differ in practice (gateway
  `/v1/models` returns 401 unauthenticated; `/health` and `/docs` are open).
- Route-level client usage requires a request log review, which has not been
  done.
