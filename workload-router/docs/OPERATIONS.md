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

`POST /v1/router/route` records a routing decision only; it does not count as a
successful model completion. Completion success is recorded by the chat path or
by the explicit `/v1/router/outcome` endpoint after the caller verifies the
result. This keeps routing telemetry separate from semantic task outcomes.

Provider HTTP failures are classified into bounded operational categories and
recorded without upstream response text. A native stream that becomes malformed
after headers are sent is closed and recorded as an operational failure; it is
not converted into a false successful completion.
