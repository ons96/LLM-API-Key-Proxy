# New-API sidecar implementation boundary

The quota-aware sidecar extends this merged component, not the legacy gateway.
It uses the existing ProviderRequest, OpenAICompatibleAdapter, failover executor,
RouterHandler response helpers, phase classification and RouterState.

New code owns a separate SQLite catalog, per-key quota ledger, persistent
cooldowns, graded telemetry and cost scorer. No tables are added to the canonical
provider-manager database or a live New-API database. The updater changes channels
only through an explicitly configured admin API. OpenCode updates are additive,
backed up and symlink-aware.

The final deployment target is VPS-40. The New-API gateway plus quota-aware
sidecar will replace the current LLM gateway only after the replacement is
tested and confirmed working. Parallel operation is the validation stage, not
the final architecture. No deployment or cutover is authorized by this local
implementation step.

Until those gates pass, the normal router_server.py entry point and gateway
dry-run/cutover controls keep their current behavior. The New-API entry point
is opt-in and loopback-only by default. It never silently connects an
unconfigured route or falls back directly to an upstream, bypassing New-API.

Strict routes use one private model alias per provider/model/key, a single-key
channel, and downstream RetryTimes=0. Otherwise retries or channel selection
would hide quota-consuming attempts. Coding affinity and provider cache evidence
are scoring hints; quota, price, context and capability filters take precedence.
Free routes are classified as `free_unlimited`, `free_limited` or `free_credits`;
actual prices stay zero, while one-time credits carry a separate notional tariff
used only for depletion and ranking. See `docs/free-tier-classes.md`.
Chat requests do not reuse coding session affinity.

Production startup requires an authorized channel-key proof callback whenever
New-API hides a channel key. Set `NEWAPI_CHANNEL_KEY_PROOF_COMMAND` to an
operator-controlled executable. For each channel audit, the sidecar writes only
the numeric channel ID plus a newline to the helper's stdin; the helper must
write one proof token to stdout within 30 seconds. The command receives no
credential in argv, and its implementation owns the interactive or pre-approved
step-up authentication flow. If the setting is absent, startup fails closed;
the disposable verifier's generated 2FA enrollment is not used by production.

The full discovery, per-component decisions, workload, accounting limitations and
licensing gate are recorded in the llm-gateway migration repository at
docs/sidecar-decisions.md. Production deployment remains a separate explicit
step. Local tests are provider-free; live verification is a distinct operator
command and must report actual results.

Replacement gates include local regression/accounting tests, a real New-API
instance, real upstream providers, complete OpenCode tool loops, quota/error
failover and stream checks, and VPS-40 resource/restart/persistence checks.
Client compatibility and a tested rollback must pass before cutover. Preserve
the existing gateway and its configuration as rollback; do not delete them as
part of the switch.
