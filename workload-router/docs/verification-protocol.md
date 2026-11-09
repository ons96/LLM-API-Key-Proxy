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

---

## 8. Component spec — what the Oracle must be

Two implementations are allowed; start with (a):

- **(a) In-harness script + plugin.** The shell script above, called from the
  harness's tool-result hook. ~150 lines, no dependencies, no daemon.
- **(b) Local sidecar** `POST /verify {attempt_id, repo, base_sha}` → verdict.
  Only build this when a second harness needs it; it buys portability, not
  capability, and costs you a process to keep alive.

### 8.1 Inputs (all local, none sensitive)

| Input | Source | Notes |
|---|---|---|
| `attempt_id` | `X-Router-Attempt-Id` response header | required for grading |
| `repo` | harness cwd | absolute path |
| `base_sha` | `git rev-parse HEAD` **before** the edit | the pre-model tree |
| `diff` | `git diff` + `git add -N` | never leaves the machine |
| `profile` | auto-detected, cached by config hash | see 8.2 |

### 8.2 Repo profile (auto-detect, do not hand-configure per repo)

| Signal | Runner | Selective run |
|---|---|---|
| `go.mod` | `go test ./...` | already package-scoped; use `./...` for changed packages |
| `Cargo.toml` | `cargo test` (or `cargo nextest run`) | nextest filter by crate |
| `pyproject.toml` / `pytest.ini` | `python3 -m pytest -q` | `pytest-testmon --testmon` runs only tests affected by the diff |
| `package.json` | `npm test --silent` | `jest --onlyChanged`, `vitest related`, `nx affected` |
| `Makefile` | `make check` | whatever the target does |

`MUST`: the runner command comes from the **base** commit's manifest, never from
the patched one (otherwise a model can rewrite `scripts.test` to `exit 0`).

`SHOULD`: selective runs first, full suite only on escalation or shared-module
diffs. This is the difference between a 3 s oracle and a 3 min oracle, and it
decides whether the cascade is viable at all.

### 8.3 Where it runs

On the **client**: laptop, desktop, or phone. Never on the 1 GB VPS — it has no
checkout, no toolchain, and no CPU to spare. If the client is a thin/remote one
that cannot run it, the answer is `verdict=unknown`, not a downgraded guess.

Constraints: no network, no dependency installation from the patch, CPU/time
capped, worktree removed afterwards.

### 8.4 Latency budget

| Tier | Target | Hard cap |
|---|---|---|
| V0 build/typecheck | 5 s | 30 s |
| V4 diff invariants | 100 ms | 1 s |
| V2 selected tests | 30 s | 180 s |
| V2 full suite (escalation only) | — | 300 s |

If V2 exceeds its cap, report `timeout` with the elapsed budget. Do not report
`pass`, and do not silently skip the tier.

### 8.5 Config (one file, per repo, optional)

```yaml
oracle:
  tiers: [V0, V4, V2]
  timeout_ms: {V0: 30000, V4: 1000, V2: 180000}
  test_select: auto          # auto | testmon | jest-changed | package | none
  allowlist: []              # empty = whole tree; set to constrain scope
  denylist:                  # merged with the defaults in section 3
    - "tests/**"
    - "**/__snapshots__/**"
  cache_dir: ~/.cache/oracle
  env: {CI: "1"}             # keep test runs non-interactive
```

### 8.6 Features, ranked

**Must have (without these it is not an oracle):**
pristine worktree, base-commit test inputs, skip/xfail detection, timeout,
closed verdict enum, idempotent cache keyed by `hash(diff+base+runner)`.

**Should have (these are what make it fast enough to cascade on):**
selective test execution, output truncated to the last ~50 failing lines, base
build cache reuse (`ccache`, `target/`, `node_modules`, `.pytest_cache`),
per-tier early exit, "fast repo" flag when the full suite is under 30 s.

**Won't do (deliberately):**
run an LLM judge in the gate, install dependencies from the patch, reach the
network, verify prose.

### 8.7 Acceptance tests for the Oracle itself

Build these before wiring it to the router; they are the reason to trust it:

1. A patch that edits a test to make it pass → `fail_scope` (test file not
   applied) or `fail_tamper`.
2. A patch that adds `@pytest.mark.skip` → `fail_tamper`.
3. A correct patch → `pass`.
4. A patch with a syntax error → `fail_build`.
5. A patch that rewrites `scripts.test` to `exit 0` → runner still comes from
   base, result unchanged.
6. A repo with no runner → `unknown`, exit 2.
7. The same diff twice → identical verdict, second run served from cache.
8. A deliberately flaky test → verdict is stable across 3 runs, or the suite is
   flagged flaky and excluded from grading.

