# DECISIONS.md

Every judgment call, the alternative rejected, and the evidence behind it.
Anything where this run disagreed with the brief is called out explicitly.

---

## D1. Drift captured byte-for-byte, host untouched (build mode)

**Decision.** The 5 modified live config files were scp'd down into
`config/` on `work/gateway-catalog` and committed as "running config pulled from
the host". No host write.

**Alternative rejected.** Reconstructing the config from git + notes.

**Evidence.** The host's `git status` shows exactly 5 modified files and the
branch had not touched them since the common base (`git diff --stat 83aeeb0 HEAD
-- config/` shows only cooldown_policy.yaml added), so the capture is pure drift,
not a merge.

**HEADs.** local build branch `e7f3917` (work/gateway-catalog); host `90252b5`
(main, with orphans `db6934e` and `90252b5` above the common base `83aeeb0`).
Recorded because a later run cannot reconstruct them without the host.

## D2. Two drift-induced test failures fixed, not quarantined

**Decision.** Fixed both, in the direction the running config points.

- `test_repo_policy_keeps_gemini_blocked`: the live config removed gemini from
  `dead_providers.yaml`. The guard was rewritten to assert the captured reality
  (gemini unblocked, `g4f_` prefix still blocked) instead of deleting it. The
  guard still loads the real file, so it still catches a broken loader.
- `test_free_tier_models_defined_for_providers`: `crax` is enabled in the live
  config with no `free_tier_models`. Added `free_tier: true` and
  `free_tier_models: []`.

**Alternative rejected.** Adding `crax` to the test's exclusion list (weakens the
invariant) or disabling `crax` (changes serving behaviour for a provider the user
turned on).

**Evidence.** `venv/bin/python -m pytest -q` after the fix -> `2 failed, 632
passed`, and the 2 are the predicted GROQ environment failures.

Empty `free_tier_models` is deliberate: fail closed. Free-only mode treats an
empty list as "no known free models", which is the honest state for a provider we
have never verified.

## D3. The 2 GROQ failures are left failing

**Decision.** Not touched.

**Evidence.** They need `GROQ_API_KEY` in the test environment and the brief
identifies them as environment failures, not regressions. Making them pass by
injecting a fake key would hide a real signal.

## D4. Tier vocabulary kept distinct from the existing `tier_classifier`

**Decision.** The quota/commercial tier is called `quota_tier` everywhere new
(provider tier 0-4). The pre-existing `src/proxy_app/routing/tier_classifier.py`
`Tier` enum is task *difficulty* (trivial/edit/refactor/architecture) and is left
alone.

**Alternative rejected.** Reusing the name `Tier` or extending `TIER_FLOORS`.

**Evidence.** `tests/test_tier_classifier.py` covers `Tier`, `TIER_FLOORS`,
`classify_request`; mixing the two concepts would make both untestable.

## D5. Four-way diff run on the host, tooling lives in the repo

**Decision.** `llm-provider-manager/scripts/four_way_diff.py` is the artifact;
it was executed on VPS-40 because that is where the provider keys are.

**Alternative rejected.** Running it from the laptop (no provider keys locally).

**Evidence.** Pre-flight found no `VPS_GATEWAY_API_KEY` or provider keys in the
local environment; the tunnel to the host was also down (Tailscale
`100.71.95.75:22` timing out, restart counter 2965).

## D6. Probe policy applied during M0, not only M1

**Decision.** The four-way diff never chat-probes a `free_one_time` or `paid`
provider.

**Evidence.** 35 providers are typed `free_one_time`; probing them is exactly the
grant-burning failure mode the brief calls out. The 30-probe budget therefore hit
only recurring/unlimited providers.

## D7. Build mode cannot fix the tunnel, so measurements run over ssh

**Decision.** Do not repair `vps-gateway-tunnel.service` (it needs a live
Tailscale path, which is outside this repo). Measure on the host over ssh.

**Evidence.** `journalctl --user -u vps-gateway-tunnel.service` shows
`ssh: connect to host 100.71.95.75 port 22: Connection timed out`, restart counter
2965. Direct ssh to `40.233.101.233` works. Logged in QUESTIONS.md.

## D8. The 2 orphan commits stay unpublished (build mode)

**Decision.** `db6934e` and `90252b5` exist only on the host. They are not
cherry-picked, not fetched into the build branch, and not pushed.

**Evidence.** Brief: "Do not push the 2 orphan commits." Their content reaches
main only via M6's `vps-truth-20261020` branch.

## D9. `free_only_mode` stays false until M2 changes runtime behaviour

**Decision.** No config flip in M0. The invariant work lands in M2 together with
its tests and its `/v1/models` enforcement, so there is never a window where the
flag is on but the code does not enforce it.

**Evidence.** Live capture: `free_only_mode: false`. Turning it on alone is
worse than useless: `_get_candidates` only filters on per-candidate
`free_tier_only`, and `/v1/models` ignores the flag entirely.
