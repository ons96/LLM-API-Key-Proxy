# Sidecar operator commands

`newapi_operator.py` composes the existing catalog, ledger and audit
primitives into six repeatable commands. Read-only by default; the mutating
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

## `classify`

Compares the sidecar ledger with the provider classifications already stored in
`llm_providers.db`. It is read-only and never rewrites a route, because an
explicit route may intentionally narrow or override the provider default.

```sh
python3 newapi_operator.py classify \
    --provider-db /home/ubuntu/llm-provider-manager/llm_providers.db
```

The provider DB is the source for `free_one_time`, `free_daily`,
`free_unlimited`, `checkin_*`, and `provider_type`. Each route result reports
`match`, `mismatch`, `unclassified`, or `missing_provider_metadata`, plus the
current/suggested quota type and free class. Credit routes also report missing
balances or notional credit tariffs. A mismatch is a review item, not an
automatic migration; use the route JSON (`routes --apply`) after deciding.

`quota_amount`, `quota_unit`, and `quota_reset` are often absent in the live DB.
Do not infer a credit balance that is not present.

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

`ok` is false when a named provider is missing or has an empty model map; the
command now exits 1 in that case so shell-based boot gates cannot silently pass.
A valid configuration exits 0.

## Disposable end-to-end operator and client check

```sh
python3 tools/verify_newapi.py --binary /home/osees/new-api/bin/new-api \
    --artifacts /tmp/opencode --opencode "$(command -v opencode)"
```

The `--opencode` flag is optional. Without it, the same verifier still exercises
operator commands and startup; with it, the installed client must pass as well.
The local proof uses fresh owner-only state and a network namespace containing
only loopback, not the existing OpenCode config, secret files, or provider accounts.

Verified against the pinned New-API binary and OpenCode 1.19.5:

- route-file preview/apply preserves configured identities;
- three provider `/models` catalogs refresh through the operator CLI, preview
  never changes OpenCode, and apply preserves its symlink and curated metadata;
- an empty catalog leaves last-good models and config intact;
- missing-provider boot checks and invalid priority orders return failure;
- authorized audit and priority-only updates preserve other channel fields and
  dispatch no completions;
- OpenCode loads the three refreshed catalogs, then completes exactly one
  read-only tool loop through the actual sidecar and New-API. Both attempts are
  accounted on the same deployment; the provider responses are scripted.

Client HOME/XDG paths, model cache, config and generated file-backed key are
isolated. Updates, plugins, external skills and network model fetching are
unavailable/disabled. The only allowed tool is reading the disposable fixture.
This proves client/transport orchestration, not real model quality, live free
quota, or production readiness.

## Still unverified

No operator command, sidecar canary or client loop from this milestone was run
against VPS-40. Real free-provider refresh/usage, compatibility exceptions,
resource/load and rollback remain gated; do not change the live native v1 retry
policy or current gateway to make the local tests pass.
