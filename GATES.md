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

Not started yet.
