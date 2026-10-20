# RESULT.md

Run date: 2026-10-20. Branch: `work/gateway-catalog`. Host: VPS-40
(40.233.101.233). Read `GATES.md` for the command and raw output behind every
claim here.

---

## 1. What is live right now

The gateway on `:8000` serves in free-only mode and enforces it.

| Probe | Result |
|---|---|
| `GET /health` | 200 |
| `GET /v1/models` without a key | 401 |
| `GET /v1/models` with a key | 2116 ids (was 2888), `g4f/*` absent |
| `POST /v1/chat/completions` with `g4f/auto` | **402** `paid_model_rejected`, tier 4, `provider_type=paid` |
| `systemctl is-active llm-gateway` | active |
| rollback time (measured before cutover) | 41.9 s |

`FREE_ONLY_MODE=true` in the host `.env`. Reverting is that one line plus a
restart.

## 2. What shipped

| Area | Artifact |
|---|---|
| Catalog pipeline | `llm-provider-manager/gateway_catalog/` (model, schema, classify, diff, build, probe, alerts), `scripts/build_gateway_catalog.py` -- 204 providers in 0.9 s |
| Free-only invariant | `src/proxy_app/routing/free_policy.py`, `/v1/models` filter, typed 402 |
| Quota ledger | `src/proxy_app/routing/quota_ledger.py` -- per-window headroom before the 429 |
| Chain ordering | `src/proxy_app/routing/chain_order.py` -- tiers 0/1/2, then 3; never 4 |
| Decision header + fallback | `src/proxy_app/routing/decision.py` |
| Verification harness | `scripts/harness_check.py`, `scripts/gateway_smoke_stream.py`, `scripts/telemetry_report.py` |
| Tests | 94 new across 4 files; 726 passing in the full gateway suite (2 pre-existing failures); 18 in the catalog package |

## 3. Bugs found and fixed

1. **Transient connection errors recorded as day-long quota exhaustion.** A brief
   network blip took a healthy provider offline for 24 h. Only quota-shaped
   failures consume budget now.
2. **Provider id spelling.** The catalog writes `blaze-free`, the gateway config
   writes `blaze_free`, so the tier lookup missed and fell back to "free". Ids are
   normalised before every lookup.
3. **Tier 3 excluded outright.** The brief makes one-time-credit providers a last
   resort; excluding them emptied every chain that depended on one.

Two failures remain in the suite (`tests/test_probe_providers_332.py::TestSelectTargets`).
They fail identically at the branch base `e7f3917`, so they are recorded as
pre-existing rather than fixed here -- see DECISIONS D3.

Not a bug, but the most expensive finding: **the host `.env` had
`FREE_ONLY_MODE=false`**, which disabled the listing filter while the code and
config both looked correct (DECISIONS D10). The endpoint now states when it is
serving unfiltered.

## 4. Open items

| Item | State |
|---|---|
| Push alerts | Local digest only. No `NTFY` topic is configured, so nothing can be delivered or verified. Needs an owner decision on destination (DECISIONS D14). |
| `new-api` | Left running. Zero requests in 24 h, but four watchdog/healthcheck/backup timers manage it, so stopping it means disabling monitoring too (DECISIONS D12). |
| `cliproxyapi` | Left running on :8317. Carrying live traffic (DECISIONS D11). |
| `zeroclaw` | Stopped and disabled. Reversible: `systemctl enable --now zeroclaw`. |
| `test_probe_providers_332` | 2 failures (`TestSelectTargets`), reproduced identically at the branch base `e7f3917`. Pre-existing, untouched (DECISIONS D3). |

## 5. Where to start next

1. Configure an alert destination, then exercise the catalog alert end to end.
2. Run `scripts/harness_check.py` against the live host on a schedule and diff the
   result, so a silent config regression like D10 is caught in minutes.
3. Decide on `new-api`: either retire it and its four timers together, or accept
   it as a fallback and document the intent.
