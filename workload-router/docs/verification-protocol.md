# Verification protocol (harness-agnostic)

Status: normative companion to `DESIGN-v2-verified-routing.md` §3. Nothing in
the router may grade a request unless this protocol produced the verdict.

## 1. Who grades

The **client** grades, because only the client has the checkout, the toolchain
and the test runner. The router issues `X-Router-Attempt-Id` and stores the
verdict it is given. A router that never receives verdicts can only learn
"HTTP 200", which is not correctness.

## 2. Definition of done

A patch is `pass` only if **all** of these hold, in this order:

| Step | Check | Cost |
|---|---|---|
| V0 | patch applies to the base commit, parses/compiles/typechecks | seconds |
| V4 | diff invariants (no test edits, no skips, no new deps, file allowlist) | ms |
| V2 | the repo's **existing** test command passes | seconds–minutes |

Lint (V1) is advisory: it may fail a patch only if the repo's CI already treats
it as blocking. Never invent a gate the project does not have.

## 3. Threat model — why the pristine worktree exists

| Attack | What it looks like | Defence |
|---|---|---|
| weaken tests | edits assertions, deletes cases | tests come from base commit |
| skip tests | adds `skip`/`xfail`/`.only()`/`@Disabled` | V4 regex + tests from base |
| edit fixtures | changes golden files/snapshots | snapshot paths in denylist |
| edit CI | `.github/workflows/**`, `tox.ini`, `pytest.ini` | denylist |
| disable lint | changes lint config to `off` | config files in denylist |
| game the runner | adds a `test` script that always exits 0 | run the command from the base commit's `package.json`/`Makefile`, not the patched one |
| over-broad patch | rewrites unrelated files to make one test pass | file allowlist per task, or `fail_scope` |

The rule is: **verification inputs come from the base commit; only the
non-test source diff comes from the model.**

## 4. Reference implementation

`git worktree` gives you a second checkout of the same object store for free
(no copy, no clone, ~1 MB of pointers). Apply the model's non-test diff there
and run the project's own command.

```sh
#!/bin/sh
# verify-pristine.sh <repo> <base-ref>
# exit 0 = pass, 3 = fail_scope, 4 = fail_tamper, 5 = fail_test, 2 = unknown
set -eu
repo=$1; base=${2:-HEAD}
wt=$(mktemp -d "${TMPDIR:-/tmp}/verify.XXXXXX")
git -C "$repo" worktree add --detach "$wt" "$base" >/dev/null
trap 'git -C "$repo" worktree remove --force "$wt" >/dev/null 2>&1 || true' EXIT

# New files must be visible to diff: intent-to-add, do not stage content.
git -C "$repo" add -N . >/dev/null 2>&1 || true

if ! git -C "$repo" diff --binary |
  git -C "$wt" apply --binary \
    --exclude='tests/**' --exclude='test/**' \
    --exclude='**/tests/**' --exclude='**/test/**' \
    --exclude='**/*_test.go' --exclude='**/*.test.*' --exclude='**/*.spec.*' \
    --exclude='**/__snapshots__/**' --exclude='**/snapshots/**' \
    --exclude='**/conftest.py' --exclude='.github/workflows/**' \
    --exclude='**/pytest.ini' --exclude='**/tox.ini' --exclude='**/jest.config.*'
then
  echo "VERDICT=fail_scope reason=patch_does_not_apply"; exit 3
fi

if git -C "$repo" diff | grep -Eq '^\+[^+].*(pytest\.mark\.skip|xfail|@Disabled|\.only\(|it\.skip\(|t\.Skip\()'; then
  echo "VERDICT=fail_tamper reason=skip_inserted"; exit 4
fi

cd "$wt"
if [ -f go.mod ]; then            go test ./...
elif [ -f Cargo.toml ]; then      cargo test --quiet
elif [ -f pyproject.toml ] || [ -f pytest.ini ]; then python3 -m pytest -q
elif [ -f package.json ]; then    npm test --silent
elif [ -f Makefile ]; then        make check
else echo "VERDICT=unknown reason=no_runner"; exit 2
fi
rc=$?
[ $rc -eq 0 ] && echo "VERDICT=pass" || { echo "VERDICT=fail_test"; exit 5; }
```

Notes:

- Time-box it (`timeout 300 ...`) and report `timeout` as a **failure with the
  elapsed budget**, never as "no data" (see §7 of the design doc).
- Cache by `hash(diff + base_sha + runner_version)`.
- The trap removes the worktree even on failure; never leave stale ones.

Residual risk: this stops test tampering, not weak coverage. A patch can satisfy
a thin test suite while being wrong. That is what the weekly human sample of
10 graded passes and `verdict=wrong_but_green` are for.

## 5. Reporting

```json
POST /v1/router/outcome
{"attempt_id":"...", "grade": true,
 "verdict":"pass|fail_build|fail_lint|fail_test|fail_scope|fail_tamper|timeout|unknown",
 "oracle":"V0|V2|V4|V5", "duration_ms": 4120}
```

`verdict` is a closed enum. Never send stderr, file paths, or prompt text —
telemetry must stay prompt-free (see `STATE.md`).

## 6. When there is no oracle

Prose, design docs, research summaries: there is no compile-or-test gate.

- Do **not** mark them `pass`. Send `verdict=unknown, grade=<omitted>`.
- The router then keeps its prior for that `(route, step)` cell instead of
  updating a success count. Unknown is not success.
- Optional: a V5 grader from a different model family, sampled (e.g. 5% of
  calls) for calibration only. Its verdict never gates a route, because a model
  grading a model is not independent evidence.

## 7. Harness mapping

The protocol is runner-agnostic; the harness only supplies three things:

1. run `verify-pristine.sh` after the edit and before declaring done,
2. post the verdict,
3. never retry the same model on a semantic failure (that is the router's call).

OpenCode: plugin hook on tool results (`bash`/`edit`) → script → POST.
Others (pi, omp, jcode): same script via their shell tool, or the gateway does
it if the client supports `X-Router-Attempt-Id` round-tripping.

If a harness cannot run the script, it must report `unknown` rather than
guessing. A router fed guesses is worse than a router fed nothing.
