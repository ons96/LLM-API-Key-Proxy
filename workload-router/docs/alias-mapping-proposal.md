# Alias mapping proposal (not implemented)

Status: **implemented 2026-11-03** (approved option 1). The table below is live
as `LEGACY_ALIASES` in `newapi_runtime.py`, and `/v1/models` lists all 22 names
with the legacy ones marked `"deprecated": true`.

Scope: this is the **v2** sidecar contract. The v1 native New-API prototype does
not expose these legacy names; see `newapi-v1-v2-roadmap.md`.

It exists because the compatibility inventory found a hard cutover blocker: the
sidecar accepted three model names and the live gateway serves nineteen, with
zero overlap. See `gateway-compatibility.md`.

## Problem

`newapi_runtime.prepare()` rejects any model other than `auto-coding`,
`auto-code` and `auto-chat`. The live gateway publishes nineteen virtual models.
A client pinned to `coding-smart` would receive HTTP 400 after a naive cutover,
and `/v1/models` would advertise three names instead of nineteen.

## Proposed mapping

Accept every legacy name and translate it to a sidecar profile plus a step hint.
The original name is echoed back in the response `model` field, so clients see
what they asked for.

| Legacy alias | Profile | Step hint |
| --- | --- | --- |
| `auto` | `auto-chat` | `chat` |
| `chat-fast`, `chat-smart`, `chat-elite`, `chat-rp` | `auto-chat` | `chat` |
| `title-fast` | `auto-chat` | `chat` |
| `coding-fast`, `coding-smart`, `coding-elite`, `coding-free` | `auto-coding` | `implement` |
| `safe-coding` | `auto-coding` | `implement` |
| `glm5-elite`, `p0-minimax-m3` | `auto-coding` | `implement` |
| `agent-build` | `auto-coding` | `implement` |
| `agent-explore`, `agent-librarian` | `auto-coding` | `explore` |
| `agent-metis` | `auto-coding` | `plan` |
| `agent-momus` | `auto-coding` | `final-review` |
| `agent-oracle` | `auto-coding` | `plan` |
| `auto-coding`, `auto-code`, `auto-chat` | unchanged | unchanged |

For legacy `auto`, the v2 sidecar now applies a deterministic prompt classifier:
code/verify/debug intent selects `auto-coding` with the matching step, while
general and tool-bearing requests stay on `auto-chat` / `chat`. The table's
`auto` row remains the fallback for general requests; explicit `X-Route-Step`
still wins. See `newapi-v1-v2-roadmap.md`.

`STEPS` already contains all of these (`route_policy.py:15`): `chat`, `explore`,
`plan`, `implement`, `verify`, `debug`, `deep-debug`, `final-review`,
`post-failure-debug`.

## What this preserves and what it does not

**Preserves:** request acceptance, the `model` echo, client configuration
compatibility, and a sensible capability/step hint.

**Does not preserve:** the gateway's actual fallback chains. `coding-elite`
currently means `glm-5.2 -> gpt-5.5-mini -> qwen3.7-plus -> gemini-3.5-flash`;
under this mapping it means "coding profile, implement step" and routes over the
sidecar's own ledger ordering. Two aliases mapped to the same profile and step
become indistinguishable, so `coding-fast` and `coding-elite` would rank
identically.

That difference is the reason this is a proposal rather than an implementation:
accepting the name is cheap, but silently changing what the name *means* is a
product decision.

## Decision taken

Option 1, the compatibility shim, was approved and is implemented. Options 2
(chain identity per alias) and 3 (chat-only sidecar, gateway keeps the nineteen
names) remain available if the semantics turn out to matter.

Two mappings are still provisional and can be changed without touching anything
else: `agent-oracle` maps to `plan` (could be `final-review`) and
`agent-librarian` maps to `explore` (could be `plan`).

## Implementation (applied)

- `LEGACY_ALIASES: dict[str, tuple[str, str]]` and `MODEL_ALIASES` (the 22-name
  union) live in `newapi_runtime.py`.
- `prepare()` resolves `model` through `MODEL_ALIASES`, applies the legacy step
  hint, and still lets an explicit `X-Route-Step` override it. `Prepared.model`
  keeps the requested name, so responses echo what the client asked for.
- `do_GET /v1/models` returns the union, marking legacy names with
  `"deprecated": true`.
- `tests/test_legacy_aliases.py` covers every legacy alias, the inventory size,
  step override, public-profile defaults and unknown-name rejection;
  `tests/test_newapi_server.py` asserts the 22-name discovery payload.
- No change to quota accounting: the profile/step pair is what ranks routes, so
  aliases that map to the same pair become indistinguishable (see above).
