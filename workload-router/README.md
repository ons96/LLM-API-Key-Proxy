> [!IMPORTANT]
> Vendored from the OmniRush `ai-workload-router` P0 core (provenance: OmniRush
> architecture + implementation). This is a PARALLEL, GATED component. It is NOT
> wired into the production gateway path and does not alter virtual_models.yaml
> fallback behavior. Enable only behind a feature flag after dry-run validation.
> See docs/workload-router-architecture.md and task-board #1070.

# AI Workload Router

This repository records the resource-light router work for the Oracle VPS at
`40.233.101.233`. The current release is running as a side-by-side,
loopback-only canary; it has not replaced the existing gateway.

The gateway changes currently deployed are:

- runtime enforcement of the mapping-based dead-provider policy;
- deployment-aware fallback ranking using concrete model telemetry;
- removal of an unused stream-timeout request field that was stripped before
  reaching provider adapters.

The production implementation remains in the VPS repository
`/home/ubuntu/LLM-API-Key-Proxy`. This repository contains portable review and
validation artifacts for the router project.

The latest VPS verification confirmed that the systemd service and socket are
active, the unauthenticated endpoint responds with HTTP 401, and the gateway
commits are `cf7a7fe`, `04a1986`, and `e6b86fa`.

The authenticated model-list probe still needs a follow-up because it exceeded
the ten-second request budget while startup health checks were active.

Run the local validation with:

```text
python3 tools/validate_session.py
```

Run the complete local test suite with:

```text
python3 -m pytest -q
```

Start the local server with:

```text
python3 router_server.py
```

## Local P0 prototype

The current VPS canary runs the failure-hardening release `failures-20261008`, served
by `ai-workload-router-canary.service` on `127.0.0.1:8123`. Ordinary chat uses
the existing gateway's `chat-smart` model. Tool and code phases use the existing
gateway's verified `coding-smart` or `coding-fast` virtual models; the old
gateway remains active on port 8000. See `docs/DEPLOYMENT.md` for tested checks
and rollback procedure.

The standalone deterministic core is in `router_core.py`. It can be exercised
without provider credentials through the route-inspection server:

```text
python3 router_server.py
```

Send a JSON request to `POST http://127.0.0.1:8080/v1/router/route`:

```json
{"messages": [{"role": "user", "content": "inspect the files"}], "tools": true}
```

The response identifies the selected deployment and capability. This prototype
does not call an upstream provider yet.

Operational or semantic results can be reported separately with
`POST /v1/router/outcome`, for example:

```json
{"deployment": "tool", "outcome": "operational_failure", "error_class": "timeout"}
```

Only `operational_failure` activates the deployment cooldown.

The local boundary rejects request bodies larger than 1 MiB before JSON
parsing, keeping the default process memory behavior bounded.

Deployment profiles are in `config/deployments.json`. The current server keeps
the safe built-in catalog by default; configuration loading is validated and
falls back to those defaults on missing or malformed files. Credentials and
provider URLs are intentionally absent from this catalog.

Each deployment also carries a non-secret provider label and configuration
fingerprint. These identify model/provider/configuration in telemetry without
storing endpoint URLs or credentials.

Quota class and remaining quota are soft scoring inputs. An exhausted quota is
penalized, but quota metadata never removes every candidate by itself.

`stall_detector.py` tracks repeated bounded action identifiers and failed
verification signals without retaining raw tool history. It supports one-way
escalation up to a ceiling and resets after meaningful progress.

Offline shadow replay is available through `trace_replay.py`. It consumes
JSONL metadata such as `examples/sample-traces.jsonl`, makes no provider calls,
and reports selected deployments for comparison with a fixed baseline.
`ReplayResult.as_dict()` or `write_report()` can persist aggregate metrics such
as routed count, deployment distribution, and baseline match rate without
writing prompts.

Run the provider-free VPS resource smoke test with:

```bash
python3 tools/resource_smoke.py
```

It starts the server on localhost, checks health and routing, and reports the
child process RSS. The target VPS budget is 1 GB; this is a startup check, not
a load benchmark. `run_router.sh` is the deployment-safe entrypoint and the
server supports `ROUTER_HOST` and `ROUTER_PORT`.

Set `ROUTER_STATE_DB` to a writable SQLite path for durable decisions, session
preferences, outcomes, and cache observations. The default `:memory:` mode is
intended for tests and ephemeral local inspection.

`provider_adapter.py` defines the transport boundary for real providers. It
forwards the complete OpenAI-compatible message/tool/structured-output request
while replacing only the client alias with the selected deployment model and
removing router-only telemetry fields. Its executor retries only eligible
deployments in the same capability group and marks operational failures for
cooldown; it does not perform semantic retries.

Provider HTTP failures are reduced to bounded telemetry classes such as
`timeout`, `rate_limit`, `upstream_failure`, `provider_rejection`, and
`network_error`. Failed attempts are recorded without persisting upstream error
text, while later same-capability attempts can still succeed. A malformed native
SSE event closes the stream without being reported as a successful completion.

The chat boundary validates `response_format` objects and supports OpenAI-style
SSE streaming. Adapters exposing `stream()` pass provider SSE payloads through
without buffering; completion-only adapters use a compatibility stream after
the provider completion returns.

Supported client model aliases are `auto`, `auto-code`, and `auto-chat`; an
unknown alias is rejected rather than silently ignored.

`POST /v1/chat/completions` returns an OpenAI-compatible response when adapters
are installed by the embedding process. The standalone server intentionally
installs no provider adapters, so it returns a bounded 503 rather than making
an unconfigured upstream call.

Health endpoints are available at `GET /healthz` and
`GET /v1/router/diagnostics`. They expose only service status and aggregate
counts; they never return prompt text.
Diagnostics also report aggregate successful routes, operational/semantic
failures, and cache hits.

`GET /v1/models` exposes the stable client aliases `auto`, `auto-chat`, and
`auto-code` without revealing provider credentials or upstream endpoint data.
JSON request endpoints reject non-object bodies with a bounded 400 response.

Successful route responses include `X-Router-Deployment` and
`X-Router-Capability` headers. JSON responses are marked `no-store` and include
content-type hardening headers.
