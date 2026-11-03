# Review package: New-API sidecar increment

Everything below is uncommitted working-tree change on `feat/newapi-sidecar`.
VPS-40 was not modified except for the earlier `new-api.service` memory cap; the
LiteLLM gateway is untouched and still serving.

## What changed and why

| File | Purpose |
| --- | --- |
| `newapi_runtime.py` | Legacy alias table (19 names), Responses translation, embeddings forwarding, ledger-accounted `Runtime.embed` |
| `newapi_server.py` | `/v1/responses`, `/v1/embeddings`, startup embedding-alias validation, JSON 405s, SIGTERM shutdown, bounded server banner |
| `newapi_operator.py` | `audit`, `routes`, `catalog`, `priorities`, `boot-check`; read-only by default |
| `tools/newapi_channel_proof.py` | Operator-owned 2FA proof helper (interactive or TOTP seed file) |
| `tests/test_legacy_aliases.py` | All 19 aliases, inventory size, step override, unknown rejection |
| `tests/test_channel_proof_helper.py` | Helper modes, seed handling, failure paths |
| `tests/test_newapi_operator.py` | Each operator command, including preview-vs-apply |
| `tests/test_newapi_server.py` | Responses, embeddings, 405, existing chat/stream/auth coverage |
| `docs/gateway-compatibility.md` | 30 gateway paths vs current sidecar surface |
| `docs/alias-mapping-proposal.md` | Why legacy names map to profiles, and what is lost |
| `docs/operator-commands.md` | How to run the four operator commands |
| `docs/channel-proof-helper.md` | Proof helper contract and TOTP caveat |
| `docs/responses-and-embeddings.md` | Supported subset and rejection list |

## Reproduce the checks

```sh
cd workload-router
PYTHONPATH=. uv run --with pytest pytest tests -o addopts= --quiet
git diff --check
```

Expected: the provider-free suite passes, with only the pre-existing
`Unknown config option: asyncio_mode` warning from the repository's pytest
config. No network, credentials or VPS access is required.

## Verified

- Full provider-free suite: **307 passed** (300 before the route-seeding
  command), with the one pre-existing `asyncio_mode` warning.
- HTTP behaviour is exercised against a real loopback server with a fake
  adapter: auth, streaming, body limits, worker cap, alias echo, ledger settle.
- Live New-API relay calls to Groq (`openai/gpt-oss-20b`) and OpenRouter
  (`cohere/north-mini-code:free`) both returned 200 at `max_tokens=15`.
- `python3 -m py_compile` on both edited modules.

## Not verified

- A sidecar startup on VPS-40: strict audit is blocked on New-API
  `RetryTimes=3`, forced-stream channel overrides, and disabled 2FA.
- Channel-key proof on the live install; 2FA enrollment hit New-API's login
  rate limiter before it could be completed.
- Streaming Responses against a real OpenAI SDK client.
- Embedding vectors from a real provider, and whether the relay token's group
  may call the embedding channel.
- Whether one TOTP window authorizes all channel proofs in a single startup.
- A second sidecar restart with backup codes (single-use caveat).

## Open decisions

- Priority ordering is explicit in the CLI, not derived from telemetry.
- Embedding aliases are opt-in through `SIDECAR_EMBEDDING_ALIASES` rather than
  inferred, so a misconfigured ledger cannot silently accept embeddings.
- The Responses shim is stateless; `previous_response_id` stays unsupported
  instead of pretending to resume.
