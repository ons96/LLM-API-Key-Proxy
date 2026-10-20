# PROGRESS.md

Resume order: read this, then GATES.md, then RESULT.md. Skip milestones marked
PASS. Do not redo them.

Mode: full (M0 through M6). Run date 2026-10-20.

| milestone | state | evidence |
|---|---|---|
| pre-flight | PASS | ssh, gh auth, backups, venv, clean tree all verified |
| M0 | PASS | GATES.md M0.1-M0.4, INVESTIGATION.md |
| M1 | in progress | catalog pipeline in llm-provider-manager |
| M2 | not started | free-only invariant |
| M3 | not started | quota ledger |
| M4 | not started | ordering |
| M5 | not started | serving end to end |
| M6 | not started | cutover |

## Done (verified, and how)

- Pre-flight: `ssh -i ~/.ssh/oracle.key ubuntu@40.233.101.233` works; `gh auth
  status` logged in as ons96 with repo/workflow scopes; both backups exist;
  `venv/bin/python -c "import litellm, pytest"` succeeds; gateway repo tree clean
  on work/gateway-catalog.
- M0 measurement 1: gateway suite `2 failed, 632 passed` (both failures are the
  known GROQ env failures); workload-router 128, llm-provider-manager 172,
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

## Next

M1: one command regenerates and validates `catalog.json` in
llm-provider-manager, with real evidence URLs, daily diff, >50% hold, and a test
proving no chat probe reaches a one-time-credit provider.
