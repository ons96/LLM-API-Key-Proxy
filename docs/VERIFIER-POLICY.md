# Verifier policy (#1422)

The verifier is the authority for "does this change pass?". It is deliberately
un-editable by the meta-loop.

## What the verifier is

`scripts/verify.py` runs three pinned stages and its **exit code is the sole
pass criterion**:

1. the existing pytest suite (the modules CI already gates on),
2. the golden routing replay (`scripts/replay_routing.py` against
   `scripts/golden/routing_replay.json`),
3. the quality-floor check (`scripts/quality_floor.py`).

A non-zero exit from any stage means the change does not pass. Nothing else
(an LLM opinion, a label, a comment) can turn a failing verifier into a pass.

## Verifier changes require human review

Changes to the verifier or its tests require human review. The protected paths
are listed in `.github/CODEOWNERS`:

- `tests/`
- `scripts/replay_routing.py`, `scripts/golden/`, `scripts/quality_floor.py`
- `scripts/verify.py`, `scripts/check_protected_paths.py`
- `src/proxy_app/telemetry/`
- `.github/workflows/`, `.github/actions/`, `.github/CODEOWNERS`

The `protected-paths` job in `.github/workflows/verifier.yml` fails any pull
request that touches a protected path unless a human (not the PR author, not a
bot) has approved the pull request. The auto-merge workflows skip such pull
requests. With GitHub branch protection requiring the `Verifier / protected-paths`
and `Verifier / verifier` checks, a pull request that edits the evaluator
cannot auto-merge.

## Golden replay

`scripts/replay_routing.py` replays the pinned traces in
`scripts/golden/routing_replay.json` through the workload-router routing core
and diffs the outcome against the pinned expectations. The script never
regenerates expectations; changing the golden file is a human-reviewed change
because the golden file is protected.

## Quality floor

`scripts/quality_floor.py` verifies that every virtual model keeps a valid
intelligence floor (`I_floor` in `config/tier_config.yaml`) and that every
chain keeps at least one candidate meeting that floor whenever benchmark data
exists. Chains without any benchmark row are reported as unverified, not
failed: unknown quality is not low quality.
