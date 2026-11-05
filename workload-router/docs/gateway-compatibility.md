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
- Client traffic was not part of the initial route inventory; the 2026-11-05
  installed-client audit below records observed local OpenCode usage.

## HTTP surface

Sidecar exposes twenty-eight paths: `GET /`, `GET /health` and `GET /healthz`
(unauthenticated), `GET /v1/models`, `GET /v1/models/{model_id}`,
`GET /v1/providers`, `GET /v1/tps-stats`, `GET /v1/model/list`,
`GET /v1/model/status`, `GET /v1/model-info/stats`,
`GET /v1/model/session/{session_id}`, `GET /v1/router/diagnostics`,
`GET /api/providers/status`, `GET /api/providers/status/{provider_name}`,
`GET /api/providers/best`, `GET /api/providers/health`,
`GET /api/providers/history`, `GET /api/providers/history/{provider_name}`,
`GET /api/providers/export/csv`, `POST /v1/router/route`,
`POST /v1/model/parse-command`, `POST /v1/model/switch`,
`POST /v1/model/detect-intent`, `POST /v1/chat/completions`,
`POST /v1/cost-estimate`, `POST /v1/token-count`, `POST /v1/responses` and
`POST /v1/embeddings` (all except `/`, `/health` and `/healthz` are
Bearer-authenticated, 1 MiB body cap).

| Gateway path | Sidecar | Status |
| --- | --- | --- |
| `/v1/chat/completions` | same path | **covered** (plus SSE, `X-Session-Id`, `X-Route-Step`) |
| `/v1/models` | same path | **partial** — returns the 22 accepted aliases (19 legacy names marked deprecated), not the gateway's upstream model list; real-binary discovery is covered |
| `/health` | same path | **partial** — unauthenticated liveness alias; body is sidecar-specific, not the gateway's exact payload |
| `/` | same path | **partial** — unauthenticated status object; not the gateway's exact body |
| `/v1/models/{model_id}` | same path | **partial** — returns the old basic `{id, object, created, owned_by}` envelope for any non-empty id; known aliases carry `deprecated`; no upstream model-info stats |
| `/v1/responses` | same path | **partial** — text/messages, function tools, text-only streaming; no `previous_response_id` or non-function tools |
| `/v1/embeddings` | same path | **partial** — only aliases named in `SIDECAR_EMBEDDING_ALIASES`; text input only |
| `/v1/token-count` | same path | **partial** — returns a conservative UTF-8 byte upper bound with `estimated:true` and `method:"utf8_byte_upper_bound"`; not an exact provider tokenizer count |
| `/v1/cost-estimate` | same path | **partial** — cheapest known configured route price with `free_class`/`free`; free routes return zero and `source:"free_tier"`; unknown pricing returns `cost:null` and `source:"unknown"` |
| `/v1/providers` | same path | **partial** — sorted provider IDs referenced by the configured ledger routes, not the gateway's provider-plugin catalog |
| `/v1/tps-stats` | same path | **partial** — aggregates measured successful streaming samples from the sidecar ledger; `export=true` is rejected |
| `/v1/model-info/stats` | same path | **partial** — ledger summary with `ready:false`; no upstream model-info registry |
| `/v1/model/list` | same path | **partial** — 22 sidecar aliases and generated descriptions, not the gateway's exact model catalog |
| `/v1/model/status` | same path | **partial** — reports `auto-chat` as the default and the 22 aliases; per-session lookup is separate |
| `/v1/model/switch` | same path | **partial** — process-local bounded session state; not shared across workers or restarts |
| `/v1/model/detect-intent` | same path | **partial** — deterministic regex heuristic; suggests sidecar aliases, not semantic-router parity |
| `/v1/model/parse-command` | same path | **partial** — validates sidecar aliases and parses `/model`, `/use`, and shortcuts; it never switches a session |
| `/v1/model/session/{session_id}` | same path | **partial** — process-local current alias with a 24-hour TTL |
| `/api/providers/best` | same path | **partial** — read-only snapshot; can return a stale best with `stale:true` |
| `/api/providers/health` | same path | **partial** — snapshot health; always `running:false`, not a live probe |
| `/api/providers/status` | same path | **partial** — latest rows from `provider_status.db`; authenticated and stale-labelled |
| `/api/providers/status/{provider_name}` | same path | **partial** — latest row for one provider; 404 when absent |
| `/api/providers/history` | same path | **partial** — 1–168 hours from the read-only snapshot |
| `/api/providers/history/{provider_name}` | same path | **partial** — 1–168 hours for one provider; 404 when absent |
| `/api/providers/export/csv` | same path | **partial** — legacy health columns from the snapshot |
| `/api/cliproxyapi/config` | — | missing |
| `/api/cliproxyapi/health` | — | missing |
| `/api/cliproxyapi/models` | — | missing |
| `/api/cliproxyapi/status` | — | missing |
| `/api/cliproxyapi/backend/{backend_name}` | — | missing |

Totals: 25 gateway paths are served at the same path (1 covered, 24 partial),
0 are renamed, and 5 of the 30 gateway paths are still absent.

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
5. Dashboards using `/api/providers/*` -> now served from a read-only
   `provider_status.db` snapshot, authenticated, with `stale:true` once the
   gateway stops updating it; `/v1/providers` still returns only route-provider
   IDs, and `/v1/tps-stats` has ledger samples only.
6. Uptime probes on `/health` now receive an unauthenticated liveness response
   from the same alias as `/healthz`; the JSON body is sidecar-specific.
7. Token-counting tooling -> served as a conservative byte bound, not an exact
   provider tokenizer; cost estimation is available only from ledger pricing,
   not the gateway's model-info service.
8. Model-info stats -> a ledger summary, not the old upstream metadata registry;
   `ready` and `upstream_model_info` are false.
9. Session switches and intent -> process-local bounded state and a regex
   heuristic, not shared session state or a semantic router. The five
   CLIProxyAPI routes remain absent until that external service is configured.

## Client usage audit - 2026-11-05

Read-only local audit of the installed OpenCode 1.19.5 config and state DB; no
credential value or prompt text was read into this repository.

- `~/.config/opencode/opencode.json` is mode 0600 and has 117 provider entries.
  `custom` and `vps-gateway` both use `http://localhost:8000/v1` and each list
  15 of the 19 legacy aliases; `agentic-gateway` uses the same base URL but has
  an empty model list. The `summary` agent is pinned to
  `vps-gateway/coding-fast`.
- A read-only query of `~/.local/share/opencode/opencode.db` (1,578 sessions,
  81,094 messages) found `vps-gateway` assistant traffic for `coding-elite`
  (189 messages), `coding-fast` (97) and `coding-smart` (17); the latest
  `coding-fast` use was 2026-10-27. Session rows also show `agent-oracle` and
  `auto`. `agentic-gateway` had no messages.
- OpenCode's recent log strings include `http://127.0.0.1:8000/health`,
  `http://127.0.0.1:8000/v1/models` and `/v1/chat/completions`; they contain no
  `/v1/embeddings`, `/api/cliproxyapi`, `/v1/token-count`, `/v1/cost-estimate`
  or `/api/providers` strings. Two `/v1/responses` strings had no captured
  gateway URL and are not counted as calls. This is log-string evidence, not a
  complete request trace.
- Therefore the required installed-client path is the legacy alias over chat,
  plus liveness and model discovery. `/health` is now a same-path,
  unauthenticated liveness alias. Model discovery remains the 22-alias list by
  design; it does not reproduce the gateway's upstream model catalog.

## Options and decision

**Decision (2026-11-05): Option A, alias translation.** The installed client
uses the legacy aliases over the chat path, so translation is the smallest
change that preserves existing configuration. Permanent dual-path and the
front-proxy split are not selected for this canary: they would either leave two
services in the request path or split accounting. Option A was already
implemented; the remaining partial and absent routes are explicit exceptions,
not silently emulated.

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

## Continuation review - 2026-11-05

Counts were rechecked against `newapi_server.py`: the sidecar exposes 28 paths,
the recorded gateway surface is 30, so 25 are same-path (1 covered, 24 partial),
0 are renamed and 5 are absent (`/api/cliproxyapi/*`). No empty shim was added
for this review.

Client-visible exceptions to settle before a canary:

1. **Health probes:** `/health` is now a same-path, unauthenticated liveness
   alias with the same body as `/healthz`; it exposes no provider or config
   data. Monitors that require the gateway's exact body still need review.
2. **Model discovery:** `/v1/models` and `/v1/model/list` expose the 22 sidecar
   aliases, not the gateway's upstream catalog. Option A (alias translation) is
   selected; a client that selects concrete upstream models still will not find
   them. That is an accepted limit for the installed client, not parity.
3. **Responses and embeddings:** only the documented text/function response
   subset and operator-allowlisted embedding aliases are served.
   `previous_response_id`, multimodal input, streaming tools and non-allowlisted
   embeddings still fail closed.
4. **Sessions and intent:** `/v1/model/switch`, session lookup and intent
   detection are process-local. They do not share state across workers,
   restarts or a multi-instance canary; treat them as advisory, not durable
   routing state.
5. **Provider status and TPS:** `/api/providers/*` is an authenticated,
   stale-labelled read-only snapshot; `/v1/providers` lists ledger route
   providers only; `/v1/tps-stats` reports ledger streaming samples and rejects
   `export=true`.
6. **CLIProxyAPI:** five routes remain absent until that external service and
   its credentials are configured. Empty 200 responses would still be
   incompatible, so do not add stubs.

Route usage has been reviewed for the installed OpenCode client: it uses the
legacy aliases over `/v1/chat/completions`, with `/health` and `/v1/models` also
referenced. The other partial and absent routes have no observed local client
demand, but external monitors or scripts may still use them. Record accepted
exceptions per client before a canary; a 200 response is not semantic parity.

## Real-binary verification - 2026-11-05

The provider-free verifier now asserts that `/health` and `/healthz` both return
200 without authentication and the same liveness body. Authenticated
`/v1/models` must return all 22 accepted aliases, with all 19 legacy names
marked deprecated; every legacy alias is then sent through the actual
entrypoint with a response-model assertion, and `coding-elite` is repeated
after restart. The `/tmp/opencode/newapi-proof-imi83fh6` run passed
(`passed:true`, scripted-local-only): discovery returned 22 models and 19
legacy names, all legacy aliases returned 200, and the restart alias preserved
its model. OpenCode 1.19.5 completed one read-tool loop, the paid route was
excluded, and benchmark grades were 0/5 on route one and 5/5 on route two. Full
suite: 402 passed; release verifier: 402 passed, health/route 200, RSS
26272 KB. This is client-boundary evidence, not a live provider or canary
result.

## Unknowns

- Whether any client other than the installed OpenCode instance calls the 5
  absent CLIProxyAPI routes.
- Gateway `/v1/models` response shape versus the sidecar's 22-alias list.
- Whether TPS consumers expect the gateway's `export=true` file side effect;
  the sidecar rejects it to remain read-only.
- Whether model-detail or provider-list clients depend on upstream metadata; the
  sidecar returns only the basic alias envelope and configured route providers.
- Whether gateway auth and sidecar Bearer auth ever differ in practice (gateway
  `/v1/models` returns 401 unauthenticated; `/health` and `/docs` are open).
- Only the installed OpenCode configuration, state DB and local logs were
  reviewed; gateway-side request logs and other clients remain unreviewed.
