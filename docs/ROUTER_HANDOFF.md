# AI Workload Router Handoff

Last updated: 2026-10-07

## Active repository

- Host: Oracle VPS 40.233.101.233
- Repository: `/home/ubuntu/LLM-API-Key-Proxy`
- Branch: `main`
- Remote: `origin/main`
- Latest pushed commit: `9fa6b7e test: align chain policy expectations`
- Service: `llm-gateway.service`
- Socket: `llm-gateway.socket`
- API: local OpenAI-compatible gateway on port 8000

## Work completed

The following changes are pushed and deployed:

- Runtime fallback policy now understands mapping-based dead-provider rules.
- Provider, prefix, exact-model, and model-prefix exclusions are enforced.
- Fallback ranking can use concrete provider/model telemetry rather than provider names only.
- The rotator client passes the concrete model to the dynamic ranker.
- Provider health errors redact API keys, bearer tokens, and token-like values.
- VPS provider metadata resolution finds `/home/ubuntu/llm-provider-manager/llm_providers.db`.
- Crowllm capability limits are recorded in `config/provider_caps.yaml`.
- Chain-policy tests were updated to match the active `config/dead_providers.yaml` policy.

## Verification already run

- `python3 -m unittest discover -s tests -p "test_chain_policy.py" -v`: 12 passed.
- YAML parsing for `config/*.yaml`: passed.
- Python compilation for changed router, telemetry, health, ranker, and reorder modules: passed.
- `python3 src/rotator_library/dynamic_chain.py`: self-test passed.
- Gateway service and socket are active.
- Unauthenticated `/v1/models` returns HTTP 401.
- Git secret scanning passed on pushes.

## Important current state

The VPS working tree should contain only this intentionally untracked file:

`docs/provider-free-tier-notes-2026-10-05.md`

Do not delete or commit that document without reviewing its historical provider claims and ASCII requirements.

The pre-rebase configuration work is preserved in `stash@{0}` with message:

`preserve config review before upstream reconciliation`

## Known issue discovered during handoff

`scripts/reorder_chains.py --dry-run` is not safely read-only. A run on 2026-10-07 changed `config/virtual_models.yaml` by adding reorder metadata and reducing fallback chains, even though it reported `dry-run`. The file was restored from Git immediately.

Fix this before using reorder automation:

1. Trace every write path in `reorder_config()` and helper functions.
2. Ensure dry-run does not call atomic YAML writes or mutate the loaded document in a way that is later persisted.
3. Add a test that snapshots file bytes, runs dry-run, and asserts identical bytes.
4. Run the test against a temporary config, then run a VPS dry-run and verify `git status`.

## Telemetry alignment issue

- Gateway components use `TELEMETRY_DB_PATH`, defaulting to `/dev/shm/telemetry.db`.
- The reorder script defaults to durable `data/telemetry.db`.
- The scheduled reorder unit explicitly uses `/dev/shm/telemetry.db`.
- Current `/dev/shm/telemetry.db` data is stale and no current 24-hour rows were available.
- Reordering was therefore not performed from stale data.

Next telemetry work:

1. Choose one path, preferably durable storage with a size/rotation policy.
2. Configure both gateway and reorder service to use it.
3. Verify schema, permissions, WAL behavior, and restart persistence.
4. Collect fresh rows before enabling automatic reorder.
5. Preserve chain pins and policy exclusions during reorder.

## Safe next implementation order

1. Fix and test reorder dry-run immutability.
2. Add a small telemetry path/configuration check.
3. Make telemetry durable and configure the systemd units consistently.
4. Add measured deterministic routing overhead and fallback-selection tests.
5. Add session/phase/cache stickiness only after the baseline path is measured.
6. Do not enable semantic/JEV classification on every request; keep it ambiguity-only.

## Operational cautions

- Do not blindly restore the old broad provider/model config from the stash.
- Do not reorder using stale `/dev/shm` data.
- Do not print journal lines containing upstream provider errors; prior logs exposed a suspended Google credential.
- The Google credential was not rotated. Rotate it manually before re-enabling that provider.
- Preserve free-only behavior unless a deliberate policy change is reviewed.
- Use `git diff`, `git status`, and a temporary config before changing live YAML.

## Resume commands

```sh
ssh -i ~/.ssh/oracle.key ubuntu@40.233.101.233
cd /home/ubuntu/LLM-API-Key-Proxy
git status --short --branch
git log --oneline -10
python3 -m unittest discover -s tests -p 'test_chain_policy.py' -v
systemctl is-active llm-gateway.service llm-gateway.socket
```
