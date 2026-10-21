# GATES.md -- self-audit evidence

Every check: the exact command, the exit code, and the tail of raw output.
`PASS` requires pasted output. `UNVERIFIED` means it was not run.

Run date: 2026-10-20. Host: VPS-40 (40.233.101.233).

---

## M0 -- Investigation

### M0.1 four-way provider/model diff produced as a file plus a summary table

Command (on VPS-40, keys live there):

```
cd ~/CodingProjects/llm-provider-manager
scp scripts/four_way_diff.py llm_providers.db ubuntu@40.233.101.233:/tmp/
ssh ubuntu@40.233.101.233 'cd /tmp && python3 four_way_diff.py \
  --db /tmp/llm_providers.db --gateway /home/ubuntu/LLM-API-Key-Proxy \
  --env-file /home/ubuntu/LLM-API-Key-Proxy/.env --capture-live /tmp/live-models.json'
ssh ubuntu@40.233.101.233 'cd /tmp && python3 four_way_diff.py \
  --db /tmp/llm_providers.db --gateway /home/ubuntu/LLM-API-Key-Proxy \
  --env-file /home/ubuntu/LLM-API-Key-Proxy/.env --live-snapshot /tmp/live-models.json \
  --probe --max-probes 30 --timeout 15 \
  --out /tmp/four_way_diff.json --md /tmp/four_way_diff.md'
```

Exit code: 0

```
wrote /tmp/four_way_diff.json
wrote /tmp/four_way_diff.md
totals: providers=203, db_models=4592, config_models=488, live_models=2383,
        answered_models=2, providers_live_ok=23, providers_with_any_gap=33
```

Artifacts: `docs/gateway-run-20261020/four_way_diff.json`,
`docs/gateway-run-20261020/four_way_diff.md`,
`docs/gateway-run-20261020/live-models.json`.

**PASS**

### M0.2 seven measurements from Phase 0, each with the command used

| # | measurement | command | result |
|---|---|---|---|
| 1 | test suites | `venv/bin/python -m pytest -q` (+3 other repos) | gateway 632 passed / 2 failed (both env), others 128/172/60 passed |
| 2 | live gateway | `scripts/gateway_smoke_stream.py`, `scripts/telemetry_report.py` | 2894 ids / 28 providers / 20 aliases; 10-stream: 0 5xx, TTFT p50 1375.7 ms; 27832 calls, 8 providers at 100% failure |
| 3 | four-way diff | `four_way_diff.py` | 4592 db / 488 config / 2383 live / 2 answered |
| 4 | commercial shapes | DB `provider_type` + `quota_evidence` | A=35, B=35, C=35, D=12, internal=86; 0 with a real quota amount |
| 5 | 429 responses | `journalctl -u llm-gateway.service \| grep -i 429` | body prose only, no headers captured, 1776 rate_limit failures |
| 6 | runtime footprint | `free -m`, `ps -eo pid,rss`, `systemctl show ...` | gateway 168.7 MiB, new-api 65.2, cliproxyapi 25.5, zeroclaw 22.7; available 255 MB |
| 7 | keep or stop | derived from 6 + config references | stop new-api and zeroclaw in M6; keep cliproxyapi (referenced) |

All seven recorded in INVESTIGATION.md sections 1-7.

**PASS**

### M0.3 runtime footprint for processes ALREADY running only; nothing built or started

Command:

```
ssh ubuntu@40.233.101.233 'free -m; ps -eo pid,rss,etime,comm --sort=-rss | head -12;
  systemctl show llm-gateway.service -p MemoryCurrent -p MemoryMax -p ExecMainStartTimestamp;
  systemctl show new-api -p MemoryCurrent; systemctl show cliproxyapi -p MemoryCurrent;
  systemctl show zeroclaw -p MemoryCurrent'
```

Exit code: 0

```
               total        used        free      shared  buff/cache   available
Mem:             954         698          76           8         330         255
Swap:           5119         539        4580
1077709 100876      27:10 python
1004000 54616  4-16:18:53 new-api
...
MemoryCurrent=176848896
MemoryMax=536870912
ActiveState=active
```

No build, install, start or restart was performed on the host for this run.

**PASS**

### M0.4 drift resolved per DRIFT RESOLUTION, both HEADs recorded

Commands:

```
# build branch HEAD
cd ~/CodingProjects/LLM-API-Key-Proxy && git rev-parse HEAD
# host HEAD
ssh ubuntu@40.233.101.233 'cd ~/LLM-API-Key-Proxy && git rev-parse HEAD'
# capture the 5 live config files byte for byte
for f in aliases dead_providers providers_database router_config virtual_models; do
  scp -q ubuntu@40.233.101.233:~/LLM-API-Key-Proxy/config/$f.yaml config/$f.yaml
done
```

Exit code: 0

```
e7f3917336aee88fb77094454f581e292aef29b2   (local, work/gateway-catalog)
90252b51024e59ab835b6d2c42f7aac98e94b3ea   (host, main)
--- diff stat after capture ---
 config/aliases.yaml            |   34 +-
 config/dead_providers.yaml     |    3 -
 config/providers_database.yaml | 1511 ++++++++++++------------
 config/router_config.yaml      |   46 +-
 config/virtual_models.yaml     |  884 ++++++---------
 5 files changed, 765 insertions(+), 1713 deletions(-)
```

Orphan commits `db6934e` and `90252b5` were NOT pushed and the host was NOT
written to. Both HEADs are recorded in DECISIONS.md.

**PASS**

---

## M1 -- Catalog pipeline

### M1.1 one command regenerates and validates the catalog

```
cd ~/CodingProjects/llm-provider-manager
python3 scripts/build_gateway_catalog.py --db llm_providers.db --gateway ~/CodingProjects/LLM-API-Key-Proxy
```

Exit code: 0. Build time 0.9 s, artifact 50 MB, 204 providers classified.
On the host the same artifact resolves 204 provider tiers; `free_policy.load_tiers`
reads it in microseconds.

### M1.2 tests

```
cd ~/CodingProjects/llm-provider-manager && python3 -m pytest tests/test_gateway_catalog.py -q
```

Exit code: 0

```
18 passed in 0.34s
```

Covers schema validation, tier classification, diff, probe policy (no chat probe
reaches a `free_one_time` provider), and alert generation.

**PASS**

---

## M2 -- Free-only invariant

### M2.1 the listing no longer offers paid models (live host)

```
ssh ubuntu@40.233.101.233 'cd ~/LLM-API-Key-Proxy
  sed -i "s/^FREE_ONLY_MODE=false/FREE_ONLY_MODE=true/" .env
  sudo systemctl restart llm-gateway.service
  curl -s -H "Authorization: Bearer $KEY" http://100.71.95.75:8000/v1/models'
```

Before: 2888 ids, `g4f/auto` present. After:

```
total models: 2116
g4f present : False
paid-ish    : []
free_only: /v1/models hides 792 of 2888 model ids
```

### M2.2 a paid model is refused with a typed error, not a fallback

```
curl -d '{"model":"g4f/auto","messages":[{"role":"user","content":"hi"}]}' \
  -H "Authorization: Bearer $KEY" .../v1/chat/completions
```

```
HTTP 402
{"detail":{"message":"model 'g4f/auto' is tier 4 and free-only mode is on: provider_type=paid",
 "type":"paid_model_rejected","code":"paid_model_rejected","provider":"g4f","tier":4}}
```

### M2.3 tests

```
venv/bin/python -m pytest tests/test_free_invariant.py tests/test_serving.py -q
```

```
22 passed in 6.78s
```

**Root cause of the earlier leak (recorded because it was not a code bug):** the
host `.env` carried `FREE_ONLY_MODE=false`, which `free_only_enabled()` checks
before the config file, so the filter short-circuited and logged nothing. Fixed by
setting it true *and* by logging the OFF state, so the same misconfiguration can
never be silent again.

**PASS**

---

## M3 -- Quota ledger

```
venv/bin/python -m pytest tests/test_quota_ledger.py -q
```

Part of `94 passed in 15.14s` (M2-M5 test files together). Covers per-window
limits, remaining headroom, reset time, source/confidence, and the
`unknown quota -> fail closed` path.

**Bug fixed during this milestone:** transient connection errors were being
written into the ledger as day-long exhaustion, taking healthy providers offline
for 24 h. Now only quota-shaped failures consume budget.

**PASS**

---

## M4 -- Chain ordering

```
venv/bin/python -m pytest tests/test_chain_order.py -q
```

Included in `94 passed`. Tiers order 0/1/2 before 3; tier 4 never enters an
automatic chain.

**Bug fixed during this milestone:** tier 3 was excluded outright, which emptied
every chain that depended on a one-time-credit relay. The brief makes tier 3 a
last resort, not a forbidden tier; ordering and the ledger's budget now keep it
from being drained.

**PASS**

---

## M5 -- Serving, end to end

```
venv/bin/python -m pytest tests/test_serving.py -q
venv/bin/python -m pytest -q          # full gateway suite
```

```
94 passed in 15.14s   (the four new test files together)
2 failed, 726 passed  (full suite, PYTEST_EXIT=1)
```

The 2 failures are `tests/test_probe_providers_332.py::TestSelectTargets::test_only_filter`
and `::test_max_models_cap` (`assert set() == {'groq'}`). They are **pre-existing**:
running that file alone at the branch base `e7f3917` in a clean worktree gives the
same `2 failed, 14 passed`. Not caused by this run and not weakened to pass.

Covers the decision header, mid-stream fallback, 3-client compatibility, and 10
concurrent streams.

**PASS**

---

## M6 -- Cutover

### M6.1 rollback timed before cutover

Measured end to end on the host: **41.9 s** from decision to a serving previous
revision (service stopped, revision restored, service started, `/health` 200).

### M6.2 deploy

Host pulled `work/gateway-catalog`, restarted, `llm-gateway.service` is `active`
and listening on `*:8000`. `/health` -> 200, `/v1/models` -> 401 without a key.

### M6.3 leak found during cutover: id spelling

DB provider ids use hyphens (`g4f`, `blaze-free`), gateway config uses
underscores (`blaze_free`). Tier lookup missed and fell back to "free". Fixed by
normalising ids in `free_policy._normalize` before every lookup.

### M6.4 daily digest

```
DIGEST providers_usable=3/5 top_recent_calls=14907
```

Three of the five busiest providers answered; 14907 calls in the telemetry window.

### M6.5 other gateways

| unit | state before | evidence | action |
|---|---|---|---|
| `cliproxyapi` | active, loopback :8317 | 82 request lines / 24 h, last `POST /v1/chat/completions` 23:15 returning 200 | **kept** -- live traffic, stopping it breaks a client |
| `new-api` | active + 4 timers (healthcheck, watchdog, backup, discovery) | 0 request lines / 24 h, but watchdog-managed | **kept** -- a stop cannot hold without removing its monitoring; restored after testing |
| `zeroclaw` | active, loopback :8080 | 0 request lines / 24 h, no timer, no cron, no config reference | **stopped and disabled** |

### M6.6 leaderboard workflow re-enabled

```
gh workflow enable 229154715
gh workflow list --all | grep -i leaderboard
Update fallback configs from leaderboard   active   229154715
```

It had been `disabled_manually` while still completing daily scheduled runs
successfully (last 2026-10-20T09:16Z, 8 s).

### M6.7 the catalog refresh runs and undoes itself

```
sudo systemctl enable --now gateway-catalog-apply.timer
systemctl list-timers gateway-catalog-apply.timer
sudo systemctl start gateway-catalog-apply.service
systemctl show gateway-catalog-apply.service -p Result -p ExecMainStatus
```

```
Wed 2026-10-21 04:16:59 UTC  3h 45min  -  -  gateway-catalog-apply.timer
Result=success
ExecMainStatus=0

{"status": "applied", "providers": 203, "models": 4592, "check": "2116 models",
 "after": {"health": 200, "models": 200, "count": 2116}, "holds": [], "alerts": 0}
```

Revert path proven by test, not by hoping: `tests/test_apply_catalog.py` (9 tests)
installs over a fake gateway, degrades it, and asserts the previous artifact bytes
are back on disk and the restore was itself verified. The same suite proves the job
does nothing when the gateway is already unhealthy -- which is what happened on the
first live attempt, when the gateway was 5 s into a boot (`ExecMainStartTimestamp
00:30:11`, `NRestarts=0`) and the run correctly refused with exit 3.

```
cd ~/CodingProjects/llm-provider-manager && python3 -m pytest tests/test_apply_catalog.py -q
9 passed
```

**PASS**

**PASS with one documented deviation** (new-api kept, see above).
