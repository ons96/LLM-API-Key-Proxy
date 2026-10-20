# INVESTIGATION.md -- Phase 0 measurements

Run date: 2026-10-20. Every number below was produced by the command shown next
to it on this date. Nothing here is copied from a README or a previous report.

Raw artifacts: `docs/gateway-run-20261020/`.

---

## 1. Existing test suites

### LLM-API-Key-Proxy (baseline, before any change in this run)

```
cd ~/CodingProjects/LLM-API-Key-Proxy && venv/bin/python -m pytest -q
```

Result: `4 failed, 630 passed in 155.26s` (log:
`docs/gateway-run-20261020/m0-pytest-gateway.log`, exit 1).

| test | cause | disposition |
|---|---|---|
| test_probe_providers_332.py::TestSelectTargets::test_only_filter | needs `GROQ_API_KEY` in the test env | environment failure, pre-existing, left failing |
| test_probe_providers_332.py::TestSelectTargets::test_max_models_cap | same | environment failure, pre-existing, left failing |
| test_free_only_mode.py::TestFreeOnlyMode::test_free_tier_models_defined_for_providers | the running host config enables provider `crax` with no `free_tier_models` | fixed in config: `crax` now declares `free_tier: true` + `free_tier_models: []` (fail closed) |
| test_router_core.py::TestDeadProviderFastSkip::test_repo_policy_keeps_gemini_blocked | the running host config removed gemini from `dead_providers.yaml` | fixed the guard: it now asserts the captured policy (gemini unblocked, `g4f_` prefix still blocked) |

The 2 GROQ failures are the same 2 the brief predicted; the count is 2, not 3.

Pre-existing flake found while repairing: `TestMoEMode::test_expert_candidate_selection`
fails when `tests/test_router_core.py` is run alone, and passes in the full suite.
Verified pre-existing by stashing this run's edits and reproducing on the pristine
tree. It is an order dependency (module-level `_dead_providers_cache`), not a
regression from this run. Left as-is; recorded in DECISIONS.md.

### Other suites (all green)

```
cd ~/CodingProjects/LLM-API-Key-Proxy/workload-router && python3 -m pytest -q
  -> 128 passed in 10.39s
cd ~/CodingProjects/llm-provider-manager && python3 -m pytest -q
  -> 172 passed in 7.26s
cd ~/CodingProjects/free-llm-provider-pipeline && python3 -m pytest -q
  -> 60 passed in 0.49s
```

---

## 2. The live gateway, measured

### Model surface

```
ssh ubuntu@40.233.101.233 'cd ~/LLM-API-Key-Proxy && set -a && . ./.env && set +a &&
  curl -s -H "Authorization: Bearer $VPS_GATEWAY_API_KEY" http://127.0.0.1:8000/v1/models'
```

- unauthenticated `GET /v1/models` -> `401` (healthy)
- authenticated -> `http=200 bytes=2365490`
- `total_ids 2894`, `nonalias 2874`, `providers 28`, `aliases 20`
- aliases: agent-build, agent-explore, agent-librarian, agent-metis, agent-momus,
  agent-oracle, auto, chat-elite, chat-fast, chat-rp, chat-smart, coding-elite,
  coding-fast, coding-free, coding-smart, glm5-elite, p0-minimax-m3, safe-coding,
  swiftrouter, title-fast
- model ids ending in `:free`: 140 (4.9%)

The brief said ~3046 ids over ~30 providers; today it is 2894 over 28. The
surface shrank.

### Streaming smoke, 10 concurrent

```
scp scripts/gateway_smoke_stream.py ubuntu@40.233.101.233:/tmp/
ssh ... 'cd ~/LLM-API-Key-Proxy && set -a && . ./.env && set +a && \
  python3 /tmp/gateway_smoke_stream.py --aliases auto,coding-fast,chat-fast \
  --concurrent 10 --concurrent-alias auto --out /tmp/m0-smoke.json'
```

```
auto             status=200 ttft=3115.7ms chars=93  decision=None
coding-fast      status=200 ttft=1707.8ms chars=54  decision=None
chat-fast        status=200 ttft=1104.2ms chars=74  decision=None
concurrent n=10 wall=1883.1ms statuses=[200] 5xx=0 ttft_p50=1375.7 ttft_p95=1790.6
```

- zero 5xx under 10 concurrent streams
- `X-Gateway-Decision` is absent on every response: the observability header the
  brief requires does not exist yet
- RSS of the gateway process (pid 1078689, `ps -eo pid,rss,args`) = 173576 KB;
  systemd `MemoryCurrent` = 176848896 B (168.7 MiB) against `MemoryMax` = 512 MiB

### Failure rate per provider (telemetry DB)

```
scp scripts/telemetry_report.py ubuntu@40.233.101.233:/tmp/
ssh ... 'python3 /tmp/telemetry_report.py --db /tmp/llm_proxy_telemetry.db --top 15'
```

Window: 2026-07-22 02:52:51 .. 2026-10-20 22:34:28, 27832 calls.

| provider | calls | ok | fail | fail% |
|---|---|---|---|---|
| blaze-free | 4355 | 2338 | 2017 | 46.3 |
| freetheai | 3619 | 0 | 3619 | 100.0 |
| cliproxyapi | 2733 | 419 | 2314 | 84.7 |
| cerebras | 2163 | 57 | 2106 | 97.4 |
| supacoder | 2033 | 0 | 2033 | 100.0 |
| groq | 1887 | 615 | 1272 | 67.4 |
| nvidia | 1394 | 621 | 773 | 55.5 |
| gemini | 1330 | 358 | 972 | 73.1 |
| kilocode | 1324 | 1 | 1323 | 99.9 |
| futureppo | 1282 | 0 | 1282 | 100.0 |
| pooled | 769 | 0 | 769 | 100.0 |
| iamhc | 682 | 1 | 681 | 99.9 |
| zenllm-free | 634 | 0 | 634 | 100.0 |
| tokenrouter | 513 | 0 | 513 | 100.0 |
| g4f_groq | 383 | 0 | 383 | 100.0 |

Error taxonomy: provider_error 13245, transient 6514, rate_limit 1776,
invalid_request 1246, auth_error 312.

Seventeen of the providers in the top-15 table are unusable in practice: 8 are at
100% failure and 4 more are above 95%. Chains ordered by quality alone will walk
through all of them before reaching a working provider.

`time_to_first_token_ms` is NULL for every row: the TTFT column is never
populated, so no per-alias TTFT history exists to compare against. The numbers in
the smoke table above are the only TTFT evidence in this repo, and they are
single-shot.

---

## 3. The four-way provider/model diff

Produced by `llm-provider-manager/scripts/four_way_diff.py`, run on the host (the
provider keys live there), artifact copied back:

```
cd ~/CodingProjects/llm-provider-manager
scp scripts/four_way_diff.py llm_providers.db ubuntu@40.233.101.233:/tmp/
ssh ubuntu@40.233.101.233 'cd /tmp && python3 four_way_diff.py \
  --db /tmp/llm_providers.db --gateway /home/ubuntu/LLM-API-Key-Proxy \
  --env-file /home/ubuntu/LLM-API-Key-Proxy/.env \
  --capture-live /tmp/live-models.json'          # (c) one GET per provider
ssh ubuntu@40.233.101.233 'cd /tmp && python3 four_way_diff.py ... \
  --live-snapshot /tmp/live-models.json --probe --max-probes 30 \
  --out /tmp/four_way_diff.json --md /tmp/four_way_diff.md'
scp ubuntu@40.233.101.233:/tmp/{four_way_diff.json,four_way_diff.md,live-models.json} \
  ~/CodingProjects/LLM-API-Key-Proxy/docs/gateway-run-20261020/
```

Totals: `providers=203, db_models=4592, config_models=488, live_models=2383,
answered_models=2, providers_live_ok=23, providers_with_any_gap=33`.

The four-way diff is the problem statement, and it is worse than the brief
described:

| provider | type | db | cfg | live | ans | db-only | live-only | cfg-only |
|---|---|---|---|---|---|---|---|---|
| openrouter | fully_free | 365 | 3 | 468 | 1 | 124 | 227 | 0 |
| nous-inference | free_tier_limited | 0 | 8 | 426 | 0 | 0 | 426 | 8 |
| koyeb | free_one_time | 200 | 0 | 396 | 0 | 43 | 239 | 0 |
| koyeb-new | free_tier_limited | 0 | 0 | 396 | 0 | 0 | 396 | 0 |
| g4f | fully_free | 178 | 0 | 223 | 0 | 166 | 211 | 0 |
| g4f_nvidia | fully_free | 20 | 20 | 81 | 0 | 18 | 79 | 0 |
| nvidia | fully_free | 126 | 34 | 81 | 0 | 60 | 15 | 7 |
| g4f_gemini | fully_free | 54 | 3 | 61 | 0 | 16 | 23 | 0 |
| mistral | fully_free | 65 | 3 | 53 | 1 | 28 | 16 | 0 |

Read the way that matters:

- The DB holds 4592 model rows; the gateway config references 488; the providers
  advertise 2383 today. No two of those numbers agree.
- 2 of 30 bounded 1-token probes succeeded (mistral/codestral-2508 and
  openrouter/minimax/minimax-m3.1-flash:free). Everything else failed with
  402/403/401/404/429/400/503. "Advertised" and "answers" are different sets, and
  the gap is enormous.
- 402 responses from all six g4f variants mean those are paywalled now, not free.
- `nous-inference` advertises 426 models, has none in the DB and 8 in config.

Probe policy: `free_one_time` and `paid` providers are skipped by the tool
(`NO_PROBE_TYPES`), so the 35 one-time-credit providers were never chat-probed.

---

## 4. Commercial shapes, with evidence

Evidence column is the DB's `quota_evidence` (provenance tag) as it stands today;
M1 replaces `source:X` tags with real URLs and confidence levels.

| shape | providers (DB `provider_type`) | count | evidence | confidence |
|---|---|---|---|---|
| A. permanently free, rate-limited only | `fully_free`: groq, cerebras, nvidia, mistral, gemini, openrouter, together, g4f, g4f_*, cohere, cloudflare, aihubmix, antigravity, deepseek | 35 | `source:vps`, `source:opencode.json`, `source:user_report` | medium |
| B. recurring free allowance | `free_daily` (atessa, blazeai, anthropic, api-kr777-top, choosec) + `free_tier_limited` (agnes_ai, dlab-community, euromodels, gcli, iflow, kiosapi, crowllm) | 19 + 16 | `source:probe-2026-08-11`, `source:vps`, `source:web_search` | medium |
| C. one-time signup credit | `free_one_time`: agentrouter, blaze, bluesminds, iamhc, blaze-free, aerolink, aibz, anmix + 27 more | 35 | `source:opencode-both (local + vps155)`, `source:user` | medium |
| D. paid only | `paid`: aitools, claude-carter, kilocloud, ktai, ktai-paid, kuae-cloud-coding-plan, openai + 5 more | 12 | `source:local` | medium |
| internal (not a real provider) | router aliases, localhost daemons | 86 | `source:*` | high |

Corroborated live today: `septorlabs` is shape B at exactly 50 requests/day, from
the upstream 429 body (section 5). `g4f*` is shape D despite the DB's
`fully_free` tag -- six of six variants returned 402 (payment required) today.
That is a classification defect M1 has to correct.

Unknown shape: 0 providers are typed `unknown`, but 0 providers have a real
`quota_amount` either. Every row has `quota_amount IS NULL`. So "shape" is
asserted, "how much" is unknown for all 203.

---

## 5. What 429 responses actually contain

```
ssh ubuntu@40.233.101.233 'sudo journalctl -u llm-gateway.service -n 4000 | \
  grep -iE "429|retry-after|rate.?limit"'
```

Observed field availability:

- **Body, textual.** The only 429 evidence the gateway records is the message
  string: `litellm.RateLimitError: ... - Free-model daily request limit reached
  (50 requests/day). Use qwen3-coder.` and `free tier queue is saturated; please
  retry shortly`. Both are prose; neither carries a machine-readable window or
  reset timestamp.
- **Headers, not captured.** `api_calls` has no column for response headers, and
  the gateway logs no `Retry-After`. The header parsers that exist
  (`parse_quota_error` in `src/rotator_library/providers/*`) read
  `retry_after`/`quotaResetDelay` from the error body, not from headers.
- **Config.** `config/cooldown_policy.yaml` (added on this branch) has 10
  providers, `default wait_on_429=False`. The live journal confirms:
  `CooldownPolicy loaded ... 10 providers configured, default wait_on_429=False`.
- **Volume.** 1776 `rate_limit` failures in the 27.8k-call window.

Conclusion for M3: quota accounting cannot be header-driven today. It must be
header-*first* with a body/heuristic fallback, and it must record `source`
(`header` | `config` | `observed_429` | `assumed`) so the weaker numbers are
visibly weaker. A single provider (septorlabs) has a known daily number; the rest
have none.

---

## 6. Runtime footprint on the real host (already-running processes only)

Nothing was built, installed, started or restarted for this measurement.

```
ssh ubuntu@40.233.101.233 'free -m; ps -eo pid,rss,etime,comm --sort=-rss | head -12;
  systemctl show llm-gateway.service -p MemoryCurrent -p MemoryMax -p ExecMainStartTimestamp;
  systemctl show new-api -p MemoryCurrent ...; systemctl show cliproxyapi ...; systemctl show zeroclaw ...'
```

Host: 954 MB total, 76 MB free, 330 MB buff/cache, **255 MB available**,
5119 MB swap (539 MB used).

| process | RSS / MemoryCurrent | uptime from systemd |
|---|---|---|
| llm-gateway.service | 176848896 B = 168.7 MiB (ps RSS 173576 KB) | started 2026-10-20 22:00:09 UTC, NRestarts=0 |
| new-api | 68415488 B = 65.2 MiB | started 2026-10-16 06:08:25 UTC |
| cliproxyapi | 26767360 B = 25.5 MiB | started 2026-10-16 06:08:29 UTC |
| zeroclaw | 23842816 B = 22.7 MiB | started 2026-10-20 06:14:24 UTC |
| tailscaled | 26.2 MiB | 72 days |
| fail2ban-server | 21.6 MiB | 5 days |
| caddy | 19.2 MiB | 47 days |
| Runner.Listener (GitHub runner) | 18.3 MiB | 19 days |

RSS under the 10-stream smoke could not be attributed per-process from inside the
same call: `rss_kb_before/after` returned `null` because the sampler matched on
`proxy_app.main` while ps shows the argv as `src/proxy_app/main.py`. Single-shot
RSS after the smoke was 173576 KB, versus 176848896 B from systemd at the same
time -- consistent, so the smoke did not grow the process measurably. M5 redoes
this with a working sampler.

Budget note: 168.7 MiB of the stated 250 MB practical budget is already used by
the gateway before any new code. The four `python`/`uvicorn` side daemons on
loopback ports (8091, 8100, 8101, 8081, 8090, 8095, 8123) are not accounted for
in the brief and are not systemd units; they are listed in QUESTIONS.md.

---

## 7. Should new-api, cliproxyapi and zeroclaw keep running?

**No, for new-api and zeroclaw. Not yet evidence-backed for cliproxyapi.**

- **new-api (65.2 MiB, up 4 days).** The cutover to new-api never happened
  (`llm-gateway/` is abandoned migration artifacts). The live gateway does not
  route through it: `router_config.yaml` has no `new-api` provider entry, and no
  `new-api` model ids appear in the 2894 ids served. It is 65 MiB of the 255 MB
  available for a service nothing references. What is lost by stopping it: the
  abandoned channel inventory and its UI, both of which are already exported
  under `llm-gateway/`. Stop it in M6 after re-checking request counts.
- **zeroclaw (22.7 MiB, listening on 127.0.0.1:8080).** No config reference in
  `router_config.yaml`. Not in the serving path.
- **cliproxyapi (25.5 MiB).** This one *is* referenced: it is both a provider
  (`cliproxyapi`, 2733 calls, 84.7% failure) and an internal API surface
  (`src/proxy_app/cliproxyapi_api.py`). Stopping it breaks a configured provider.
  Leave it running unless M6's zero-request window proves it idle.

CLEANUP POLICY in the brief requires zero requests during the observation window
AND no config reference. Only new-api and zeroclaw are candidates; M6 measures
before stopping anything.

---

## 8. Things that are true but were not asked

- `free_only_mode: false` in the captured live `router_config.yaml`. The whole
  free invariant is currently off.
- 12 providers typed `paid` are `enabled: 1` in the DB, and the gateway config
  enables `openai` with no `free_tier_models`.
- The DB's `models.free_tier` is populated for 6818 rows but never derived from
  price; it is carried from whichever import created the row.
- `config/cooldown_policy.yaml` exists on this branch but not on the host: the
  host process loaded it from the repo checkout, so it is deployed code that is
  not in git at the host's HEAD. That is part of the drift.
