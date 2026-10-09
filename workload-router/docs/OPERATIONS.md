# Operations checklist

Run locally before installation:

```bash
python3 -m pytest -q
python3 tools/validate_session.py
python3 tools/resource_smoke.py
python3 router_server.py
python3 tools/concurrency_smoke.py --requests 100 --workers 8
```

The same provider-free release gate is available as
`python3 tools/verify_release.py`. Run it before copying a release to VPS 40;
it does not contact providers or perform remote deployment.

For VPS 40, use a versioned release directory and a loopback-only canary before
any gateway cutover. The current canary is documented in
`docs/DEPLOYMENT.md`; it runs as the `ubuntu` user on port 8123 and does not
replace the existing gateway on port 8000. The current canary uses the
gateway's `chat-smart` model for ordinary chat and gateway-backed
`coding-smart`/`coding-fast` profiles for tool and code phases. Keep provider
environment files mode 0600. The checked-in install
and rollback scripts are templates for a later
privileged service install; do not run them over the existing gateway without a
reviewed cutover plan.

Do not run installation until provider credentials, firewall policy, health
checks, and an existing-gateway fallback have been reviewed. The scripts are
local deployment tooling; this repository does not execute them remotely.

Native provider streaming is selected only when the configured adapter exposes
`stream()`. If a provider emits malformed SSE after headers are sent, the
connection is terminated rather than converted into a misleading successful
completion. Test provider-specific streaming behavior before enabling it for
agent sessions.

The current canary validation covered ordinary chat, tool calls, JSON output,
implementation, verification, debug, and native SSE. It did not perform a
reverse-proxy cutover or enable a version-specific OpenCode plugin. Keep the
existing gateway as the fallback while those client compatibility checks are
completed.

Clients can discover only the stable router aliases through `GET /v1/models`;
the endpoint does not expose upstream model or credential metadata.

`GET /v1/router/groups` exposes configured group IDs and ordered
deployment/provider/model labels for operator inspection. If
`ROUTER_PROVIDER_DB` is set, these groups are loaded from the provider-manager
SQLite metadata database; API keys remain environment-only. Set
`ROUTER_PROVIDER_GROUPS` to a comma-separated allowlist and leave
`ROUTER_FREE_ONLY=1` enabled when using free-tier metadata.

`POST /v1/router/route` records a routing decision only; it does not count as a
successful model completion. Completion success is recorded by the chat path or
by the explicit `/v1/router/outcome` endpoint after the caller verifies the
result. This keeps routing telemetry separate from semantic task outcomes.

Provider HTTP failures are classified into bounded operational categories and
recorded without upstream response text. A native stream that becomes malformed
after headers are sent is closed and recorded as an operational failure; it is
not converted into a false successful completion.

The local `tests/test_provider_faults.py` harness exercises these categories with
a scripted provider and does not consume live quota. Keep live provider fault
checks bounded. A real OpenCode 1.18.35 request through the loopback canary was
able to reach the router, but the existing gateway's upstream chain exhausted
or cooled down on the large tool-context request; the router intentionally does
not strip tools or silently downgrade that request. Repeat the tool-loop test
after upstream availability recovers.

Fresh requests restart at the top chain priority. Use the continuation signal
(`X-Router-Continuation: true` or `continuation: true`) only for raw API/tool
continuations that should retain the preferred deployment. A short rate-limit
wait is permitted only when fresh cache evidence makes it cheaper than a cold
fallback; all other timeout, quota, and upstream failures use bounded
same-group failover.
