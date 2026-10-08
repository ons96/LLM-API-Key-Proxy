# Deployment Target

## Intended runtime

The router is intended to run on Oracle VPS 40 at `40.233.101.233`, with the
existing 1 GB RAM, zram, weak CPU, and no GPU constraints. This local repository
is the development and test workspace. It does not deploy changes to the VPS
repository automatically.

## Runtime requirements

- Python standard library for the current prototype.
- No mandatory paid services, hosted classifier, model download, or GPU.
- SQLite for durable local state.
- Bounded request bodies, bounded in-memory state, and asynchronous telemetry
  as the implementation grows.
- Provider credentials supplied only through VPS environment/service
  configuration; never through the deployment catalog or source files.

## Safe deployment sequence

1. Run the complete local test and validation suite.
2. Build a clean artifact from this repository.
3. Run a VPS resource test with representative concurrency.
4. Install behind the existing gateway only after compatibility testing.
5. Keep a router-unavailable fallback to the existing gateway behavior.
6. Verify health, memory, CPU, latency, cooldowns, and SQLite persistence.

Development tests and shadow replay must remain provider-free and must not
require access to VPS 40.

## VPS 40 canary record

The first deployment is intentionally side-by-side and does not replace the
existing gateway. Release `cc65db6` is installed at:

```text
/home/ubuntu/ai-workload-router/releases/cc65db6
```

The user service `ai-workload-router-canary.service` binds only to
`127.0.0.1:8123`. Its state file is under
`/home/ubuntu/ai-workload-router/state/`, and its environment file is mode 0600
at `/home/ubuntu/ai-workload-router/router.env`. The service is enabled for the
`ubuntu` user and has a 160 MiB memory limit. General chat is currently
configured through the existing gateway using its `chat-smart` virtual model.
The credential is sourced only on the VPS into the mode-0600 canary environment
file; it is not present in this repository. No existing service, reverse proxy,
provider credential, or gateway port was changed.

Canary checks completed:

- `/healthz` and `/v1/router/route` returned HTTP 200.
- `/v1/models` returned the three stable router aliases.
- 100 route requests with 8 workers all succeeded.
- Resource smoke reported approximately 25 MiB RSS.
- A persisted decision and cache observation survived a service restart.
- Existing `llm-gateway.service` remained active on port 8000.
- Non-streaming chat and upstream SSE both returned HTTP 200 through
  `chat-smart`.
- `auto-code` and tool requests now use configured gateway-backed code profiles;
  provider-backed tool-call responses and native SSE both returned HTTP 200.

To stop or roll back only the canary, run as `ubuntu`:

```bash
systemctl --user disable --now ai-workload-router-canary.service
rm -f ~/.config/systemd/user/default.target.wants/ai-workload-router-canary.service
```

The release and state directories can then be archived or removed after
inspection. Re-enabling the existing gateway is not part of this rollback
because it was never stopped or modified. A future cutover must first test
provider adapters and reverse-proxy routing while retaining this fallback.

The adapter preserves message history, tool definitions, tool calls,
structured-output fields, and standard provider request options. The canary now
has code-capable profiles, but OpenCode agent sessions remain a separate
compatibility check until their deployed headers and tool-loop behavior are
verified.

The forwarding implementation is covered locally by the adapter and HTTP
boundary tests. Update the canary release before testing code profiles; do not
copy the local SQLite state or any local environment file.

The structured-forwarding release is currently active as
`releases/failures-20261008`. Health, model discovery, route, ordinary chat, tool-call,
structured-output, code-phase, debug-phase, and native SSE checks passed after
restart. An explicit operational-failure outcome also caused the affected
deployment to be cooldown-excluded, and the canary was restarted afterward.

The runtime catalog maps `tool`, `implement`, and `debug` to the gateway's
`coding-smart` model and maps `verify` to `coding-fast`. These mappings are
VPS-only and contain no credentials in the release or repository.

The VPS does not currently have an `opencode` executable, so a real OpenCode
client session was not available. The exact signal-header path was exercised
against the canary instead: repeated `X-Router-Action` values raised bounded
escalation from 0 to 2 while the implementation session remained on its
selected deployment.

For a durable runtime, set `ROUTER_STATE_DB` to a writable SQLite file outside
version control, such as `config/runtime/router-state.sqlite3`. Create its
parent directory before startup and back up the file according to the host's
operational policy. Keep the default in-memory mode for tests.
