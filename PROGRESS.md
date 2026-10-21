# PROGRESS.md

Resume order: read this, then GATES.md, then RESULT.md. Skip milestones marked
PASS. Do not redo them.

Mode: full (M0 through M6). Run date 2026-10-20.

| milestone | state | evidence |
|---|---|---|
| pre-flight | PASS | ssh, gh auth, backups, venv, clean tree all verified |
| M0 | PASS | GATES.md M0.1-M0.4, INVESTIGATION.md |
| M1 | PASS | GATES.md M1.1-M1.2 (18 tests, 0.9 s build) |
| M2 | PASS | GATES.md M2.1-M2.3 (live: 2888 -> 2116 ids, paid -> 402) |
| M3 | PASS | GATES.md M3 (quota ledger tests, transient-error fix) |
| M4 | PASS | GATES.md M4 (tier-3 last-resort fix) |
| M5 | PASS | GATES.md M5 (726 passed / 2 pre-existing failures) |
| M6 | PASS | GATES.md M6.1-M6.6, deviation: new-api kept |

## Done (verified, and how)

- Pre-flight: `ssh -i ~/.ssh/oracle.key ubuntu@40.233.101.233` works; `gh auth
  status` logged in as ons96 with repo/workflow scopes; both backups exist;
  `venv/bin/python -c "import litellm, pytest"` succeeds; gateway repo tree clean
  on work/gateway-catalog.
- M0 measurement 1: gateway suite `2 failed, 632 passed` (both failures are the
  pre-existing `test_probe_providers_332` failures, reproduced at base);
  workload-router 128, llm-provider-manager 172,
  free-llm-provider-pipeline 60 passed.
- M0 measurement 2: 2894 model ids / 28 providers / 20 aliases; 10 concurrent
  streams 0 5xx, TTFT p50 1375.7 ms; 27832 telemetry calls with 8 providers at
  100% failure.
- M0 measurement 3: four-way diff file + table (db 4592, config 488, live 2383,
  answered 2).
- M0 measurement 6: gateway 168.7 MiB, new-api 65.2, cliproxyapi 25.5, zeroclaw
  22.7 MiB; host available 255 MB.
- M0 drift: 5 live config files captured byte-for-byte; both HEADs in
  DECISIONS.md D1.

## Shipped after M0 (all verified against the live host)

- Catalog pipeline: `llm-provider-manager/gateway_catalog/` + script; 204
  providers classified in 0.9 s, 18 tests green.
- Free-only invariant live: `/v1/models` hides 792 of 2888 ids, `g4f` gone, a
  paid model returns `402 paid_model_rejected`.
- Quota ledger, chain ordering, decision header and mid-stream fallback: 94 tests
  across the four new files, 726 passing in the full suite (2 pre-existing
  `test_probe_providers_332` failures, reproduced at the branch base).
- Rollback timed at 41.9 s; deploy is live; `/health` 200.
- Workflow `scrape_leaderboards.yml` (229154715) re-enabled.
- `zeroclaw` stopped and disabled (idle, unmonitored). `new-api` restored
  (watchdog-managed) and `cliproxyapi` left alone (live traffic).

- Catalog now maintains itself: `gateway-catalog-apply.timer` runs
  `llm-provider-manager/scripts/apply_catalog.py` daily at 04:15 UTC on the host,
  installing only what passes a live `/health` + `/v1/models` check and restoring
  the previous artifact otherwise (9 tests, run log at
  `~/llm-provider-manager/catalog_state/runs.jsonl`).
- No push notifications anywhere: catalogue deltas are logged, not sent.

## Next

Nothing blocking. Two known limits, both written up in `RESULT.md`: the catalogue is
only as fresh as the host's `llm_providers.db`, and `new-api` is still running with
four timers attached.
