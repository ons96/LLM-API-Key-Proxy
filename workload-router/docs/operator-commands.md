# Sidecar operator commands

`newapi_operator.py` composes the existing catalog, ledger and audit
primitives into five repeatable commands. Read-only by default; the mutating
paths require an explicit flag. No credential ever appears in argv.

```sh
export SIDECAR_DB=/var/lib/newapi-sidecar/sidecar.db
export NEWAPI_PROVIDER_DB=/home/ubuntu/llm-provider-manager/llm_providers.db
# audit and priorities --apply only:
export NEWAPI_BASE_URL=http://127.0.0.1:8001/v1
export NEWAPI_ADMIN_TOKEN=...        # operator env file, 0600
export NEWAPI_ADMIN_USER_ID=...
export NEWAPI_API_KEY=...            # relay token
export NEWAPI_RELAY_TOKEN_ID=...
# only when New-API hides channel keys:
export NEWAPI_CHANNEL_KEY_PROOF_COMMAND=/opt/newapi-sidecar/bin/channel-proof --mode=totp
```

## `audit`

Runs the same strict checks `serve()` performs at startup: `RetryTimes=0`,
unique private aliases, enabled single-key channels, exact model mapping,
credential fingerprint and endpoint identity, plus the relay-token audit.

```sh
python3 newapi_operator.py audit
```

Output is `routes`, `channels` and a 64-character `audit_digest`. Useful as a
pre-flight before restarting the sidecar, and as a diffable record after catalog
changes. It is read-only.

## `routes`

Seeds the ledger from an explicit JSON file. Preview is the default; `--apply`
writes only the sidecar database and never calls New-API.

```sh
python3 newapi_operator.py routes --file /etc/newapi-sidecar/routes.json
python3 newapi_operator.py routes --file /etc/newapi-sidecar/routes.json --apply
```

Each list entry is a full `Route` definition (`route_id`, `provider_id`,
`model_id`, `key_id`, `alias`, `channel_id`, prices, limits,
`key_fingerprint`, `channel_base_url`). Unknown fields, duplicate ids and
invalid values fail closed before any write. A `free_credits` route keeps
`input_price` and `output_price` at zero and sets `credit_input_price`,
`credit_output_price` and `credit_balance`; those credit prices are notional
ranking inputs, not customer charges. See `free-tier-classes.md` for the full
class mapping.

The preview now includes each route's `free_class` and `quota_type`, so an
operator can catch a paid-looking free route before applying it.

## `catalog`

Fetches each provider's `/models` into the sidecar ledger, then previews the
OpenCode additions. This sends no completions and never removes an existing
OpenCode model entry.

```sh
python3 newapi_operator.py catalog --providers groq,cerebras \
    --opencode-config ~/.config/opencode/opencode.json            # preview
python3 newapi_operator.py catalog --providers groq,cerebras \
    --opencode-config ~/.config/opencode/opencode.json --apply-opencode
```

Provider keys are read from the env var named in the provider metadata row
(`env_var`); providers marked `no_api_key_required` send no Authorization
header. A missing env var fails with the variable name, never its value. An
empty or malformed catalog is rejected and the previous catalog is preserved.
`--apply-opencode` writes a `0600` backup and replaces the file atomically.

## `priorities`

Reorders New-API channel priorities to match an explicit best-first route order.
Preview is the default.

```sh
python3 newapi_operator.py priorities --order route-a,route-b,route-c           # preview
python3 newapi_operator.py priorities --order route-a,route-b,route-c --apply
```

The order must list every ledger route exactly once; duplicates, omissions and
unknown ids fail closed before any request. `--apply` re-audits first and then
updates priorities only — never keys, models or mappings.

The order is explicit rather than derived. Automatic ordering from persisted
benchmark grades and telemetry is still open work; deriving it would encode a
policy decision that belongs in review, not in a CLI default.

## `boot-check`

Validates an OpenCode configuration read-only, before pointing a client at the
canary: file readable and bounded, JSON object, `provider` map present, and each
named provider carrying at least one model.

```sh
python3 newapi_operator.py boot-check --opencode-config ~/.config/opencode/opencode.json \
    --providers groq,cerebras
```

`ok` is false when a named provider is missing or has an empty model map.

## Still unverified

These commands are covered by provider-free tests only. A real three-provider
refresh, a real OpenCode boot on the current client version, and a live priority
update have not been run against VPS-40, because the sidecar cannot start there
until the channel-key proof helper is configured on the host.
