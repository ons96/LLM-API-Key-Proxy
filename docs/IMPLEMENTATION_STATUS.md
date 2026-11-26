# Implementation Status

## Checkpoint

- Updated: 2026-11-24 (UTC).
- Completed implementation: Stage 1, hermetic financial-safety test verifier.
- Completed implementation: Stage 2 kernel and review corrections, verified below.
  Follow-up independent review remains separate from production enablement.
- Open review findings being closed: the two Medium test-coverage gaps from the
  first independent Stage 2 review (T02 masking, P07 expired/revoked duplicate
  IDs). See "Review follow-up" below. No kernel/production code changed.
- Current authorization: owner requested fixes for all Stage 2 review findings,
  then explicitly authorized a local checkpoint commit for fresh review.
  Stage 3, pushing and production integration are NOT authorized.
- No production financial policy, router, provider adapter or runtime behavior
  changed. No real lane has been qualified or enabled.
- Repository: /home/osees/CodingProjects/LLM-API-Key-Proxy.
- Branch: feat/financial-admission-kernel.
- Historical design baseline: 5acacda126d6f9385f8ca17f1e4d032a93442f36.
- Corrective-task entry HEAD: 06f8f3a5affc5ad696a21d186c7db395195a4f8a (clean).
- Pre-checkpoint HEAD: bcffc999fb4b79aa99f3b5c5700ca9cc58b99321.
- Implementation verification HEAD: 74b38de01c91c73fd3fc3d4bbd803d2b9fa54ce5.
- Local checkpoint: the commit containing this record, on the branch above.
  Its parent is the pre-checkpoint HEAD; identify the checkpoint with git log.
- The initial implementation was already committed at 71bd2ce. Another process
  committed b8808ff and 74b38de during corrections, including edits made here.
  It then committed the remaining correction files as bcffc99 during checkpoint
  preparation. The 243-source manifest was unchanged across that commit.
  The correcting agent did NOT commit during implementation, push, rewrite
  history, deploy or operate services. The later local checkpoint is now
  explicitly owner-authorized; it is not independent acceptance or a release.
  External commit authorization/push/deployment are not verified. Preserve the
  commits; do not automatically roll them back. Verification is bound to hashes
  and a stable source snapshot, not the former incorrect 'uncommitted main' claim.

## Stage 2 review corrections and verified snapshot

The owner requested these corrections after ACCEPT WITH REQUIRED FIXES. The
correcting agent authored the original contract and now these fixes; this is
self-review, NOT fresh independent implementation acceptance.

### Exact corrections

- src/financial_safety.py: reject any model ID containing '://'. No monetary,
  funding, trust, validation-precedence or runtime behavior was relaxed.
- docs/FINANCIAL_ADMISSION_CONTRACT.md: reviewed, stricter section 6 clarification
  resolves the regex/prose contradiction; lifecycle wording acknowledges the
  implementation. Required DTOs/APIs, zero-money invariant and Stage 3 gates
  are unchanged. The owner authorized fixing this finding, not enabling lanes.
- tests/test_financial_safety.py: expanded missing variants and isolated masked
  tests. Its shared evaluate harness now exercises BOTH public APIs for every
  matrix evaluation and checks every denial's zero dispatch/fallback calls,
  exact exception, safe metadata, determinism and unchanged synthetic inputs.
- S06 installs failing spies BEFORE fresh import, not only afterward. It checks
  allowed imports statically and dynamically, spies on file/environment/DB/
  logging/process/network/clock APIs, and runs eligible plus all denial-code
  branches. Its 24 separate synthetic canaries prove the spies block attempted
  side effects BEFORE execution. The actual policy is never patched/replaced.
  CPython open_code delegates to _io.open; only get_data's trusted loader reads
  are allowed there, still under the original filesystem guard.
- README.md documents the URL restriction and expanded test guarantee.
- This status corrects branch/commit/provenance claims. PROJECT_STATUS.md and
  archive moves were external work; they were preserved, not undone or expanded.

Files changed relative to corrective-task entry 06f8f3a: the five above. The
verifier, conftest, isolation self-tests, audit/roadmap, production code/configs,
package/deployment files and provider-manager data were not changed by fixes.
External commits captured some edits; use the entry-to-worktree diff AND the
remaining uncommitted diff, not only HEAD-to-worktree, to review the whole fix.

### Required-variant traceability

All 47 section 8 IDs retain named tests; this map explains the former gaps,
not merely the presence of function names. Original required rows remain.

| IDs | Corrective coverage |
| --- | --- |
| A05 | All 10 operations with distinct exact profiles; 3 separate N/A controls; namespace and length boundaries |
| F05/F08/F09 | Both cache fields plus tool/search/other; each promotional charge and exhaustion/expiry False/None; both relay flags False/None |
| I07/I08 | Every target/record/revision ID, registry ID elements and both digests; malformed types/lengths/characters/case; matching URL approval and unrelated URL records before/after match |
| P01 | Each input DTO subclass, separately with exact outer policy; raw/missing nested DTOs; enum/string/class/foreign-enum cases; scalar subclasses and prohibited conversion callbacks; container subclasses |
| P07/P09 | Distinct expired/revoked duplicate cases; each malformed collection element/container; collection over-limits and exact limits; schema/active types |
| T02 | Every invalid timestamp kind at now, verified_at, valid_from and valid_until; all ordering comparisons; valid epoch extremes |
| N01-N03/F04 | Original eight-field numeric matrix; more currency/unit/proof kinds; tiny positives under restrictive Decimal contexts without flags/rounding |
| S01/S02/S04/S05 | Shared invariants applied to every denied matrix variant, not 18 representative outcomes; sensitive malformed canaries; all six DTOs frozen/slotted/required and exact annotations/signatures |
| S03/S06 | Fixed truthiness error across all denial codes; pre-import and evaluation spies plus negative self-check canaries |

A01-A04/A06, F01-F04/F06-F07, I01-I06, P02-P06/P08/P10, T01/T03 and S07-S08
remain directly asserted. The shared harness adds safety assertions without
replacing individual expected-code assertions or weakening the actual policy.

### Final verification

The exact three commands in the Stage 2 verification section were executed
through the unchanged hermetic bootstrap, using the existing venv and offline
uv. Only registered disposable private-network fixtures were permitted.

- financial: 754 passed in 3.82s (709 kernel + 45 isolation tests), exit 0;
  unexpected_denials=0, expected_denials=44.
- legacy: 50 passed in 5.50s, exit 0; unexpected_denials=0, expected_denials=0.
- sidecar: 403 passed in 28.57s, exit 0; unexpected_denials=0, expected_denials=0.
- No required tests skipped/xfail; no raw-pytest fallback or guard weakening.
- HEAD was 74b38de01c91c73fd3fc3d4bbd803d2b9fa54ce5 before AND after all runs.
  All 243 source inputs copied by the verifier were hashed before the runs and
  checked after EACH suite. No source changed during that verification interval.

Source manifest SHA-256 (sorted path-to-hash JSON):
93734ec058568315f972ada3c4d0855696bce845f667565cdb4006233052215b

Artifact SHA-256 values:
- src/financial_safety.py:
  b0bb7451a044a3dd016f9eaf24e4b8675de89ce9fd30e77f593b1af681931e54
- tests/test_financial_safety.py:
  bc34a239c15ecdf7a1734f37455e38c0d269259a71d1d489210fe05c1f492884
- tests/support/financial_test_bootstrap.py (unchanged):
  f77cc16e93d8d3df31e8aa90aef137d5ee27d8c7a5daa058c93da7e5e1a23da2
- tests/conftest.py (unchanged):
  6a4f96f82397dd4dc707e301f7122b375deccd42a325a88498c205e5f2bd155a
- tests/test_financial_test_isolation.py (unchanged):
  4d950fa2a5b84016f67bb479bb7123df105851c478345038a5568e41d58cb633

Development failures were visible: URL regressions first produced 10 failures
and 504 passes with zero unexpected isolation denials. The corrected policy
resolved those. Stronger S06 initially rejected CPython's own loader delegation;
its loader-only allowance was corrected, not its application spies removed.
An intermediate 724-case suite passed before the final expansions above.

Static checks on the final diff passed: in-memory compilation/tabnanny, ASCII
and whitespace checks, Python 88-column checks, child-source compilation and
`git diff --check 06f8f3a`. All 78 test functions map to the 47 required IDs; this
naming check supplements, not replaces, the variant assertions above. The full
five-file entry-to-worktree diff was read and self-reviewed. No code changes
followed the final suite runs; recorded code/guard/audit hashes still match.

Remaining limitations: no installed ruff/black/mypy/pyright/coverage tool and no
installation/download was attempted. In-memory syntax/tabnanny, ASCII/new-line
checks, public-type/signature assertions, manual type review and diff checks are
not measured coverage or a full static type/lint check. Real account/evidence,
billing contracts, snapshot trust, production dispatch, privacy/auth/quota gates
and external commit/push authorization remain unverified. Stage 3 is not started.

## Stage 2 review follow-up: closing the two Medium test gaps

Second-pass fixes for the two Medium findings in the first independent Stage 2
review. Test-only: `src/financial_safety.py`, the contract, the verifier,
conftest, isolation self-tests and production code are unchanged. This is again
the correcting agent's own work, NOT independent acceptance.

### T02 - timestamp type/range checks were masked by the ordering check

Finding: every malformed timestamp was paired with values that also broke
`verified_at <= valid_from < valid_until`. A per-field type/range regression
could therefore still produce `INVALID_POLICY` via the ordering check, so the
test proved only that the context was rejected, not that the type check ran.

Fix: `TIMESTAMP_CASES` now uses malformed values that are numerically inside
that field's own ordering window (for example `verified_at=900.0`, `"900"`,
`IntSubclass(900)`), so ordering still passes and only the exact-type/range
check can deny. Booleans and negatives are supplied with ordering-valid
companions. Values that cannot be isolated at all - a negative `valid_from` or
`valid_until`, an out-of-range bound that no companion can satisfy, and
`valid_until=False` (since `False == 0` makes `valid_from < valid_until`
unsatisfiable) - are kept in a separate `NON_ISOLATING_TIMESTAMP_CASES` test
whose comment states the ordering check co-denies. They are retained for
regression value but are not cited as proof of the per-field check.

### P07 - expired/revoked duplicate approval IDs were not covered

Finding: duplicate IDs were tested only for active records; expired/revoked
variants existed only for duplicate *targets* with different IDs. An
implementation that skipped ID uniqueness for expired/revoked records was
untested.

Fix: `test_p07_duplicate_approval_ids_are_invalid` is parameterized over
`duplicate_id`, `duplicate_expired_id` and `duplicate_revoked_id`, each using a
distinct valid target. All require `INVALID_POLICY` through the shared harness,
which also asserts no dispatch, no fallback, safe metadata and determinism.

### Mutation evidence

The stronger tests were checked against deliberately weakened copies of the
kernel in `/tmp` copies (no repository file was modified):

- coerced `verified_at` (`int(value)` when not a bool): the mutant returned
  `eligible` for the float, string and `IntSubclass` variants; only the T02
  isolating test failed (3 failed, 752 passed);
- approval-ID uniqueness skipped for records valid only until 1100, or whose ID
  is revoked: both duplicate-ID cases failed. The expired duplicate returned
  `eligible`; the revoked duplicate returned `APPROVAL_REVOKED` rather than the
  required `INVALID_POLICY`, because an isolated uniqueness-only mutation cannot
  reach eligibility for a revoked ID (2 failed, 753 passed).

Both mutants are now rejected by the tests above. These mutants are synthetic
proof that the tests bite; they are not evidence about production behaviour.

### Verification (fresh, this session)

- financial: 756 passed in 4.09s, exit 0; unexpected_denials=0,
  expected_denials=44 (711 kernel + 45 isolation tests; two net new cases
  versus the 754-test checkpoint).
- legacy: 50 passed in 7.68s, exit 0; unexpected_denials=0.
- sidecar: 403 passed in 29.69s, exit 0; unexpected_denials=0.
- A 287-file tracked manifest of `src`, `tests`, `workload-router` and
  `pytest.ini` was hashed before the runs and after each suite; it did not
  change. Only `tests/test_financial_safety.py` and this file differ from the
  review checkpoint.
- A third `/tmp` mutant that accepts `valid_until=False` and skips ordering for
  it is now rejected by the new non-isolating case (1 failed, 755 passed).
- Static checks: in-memory compile, ASCII and 88-column checks on the edited
  test file. No ruff/black/mypy/pyright/coverage tool was installed or invoked,
  so no lint, type-check or coverage result is claimed.

Still unverified: independent acceptance, real account/evidence proof, billing
contracts, snapshot trust, production dispatch and the auth/quota/privacy/
transport gates. Stage 3 is not started and remains separately owner-gated.

## Owner-authorized local review checkpoint

The owner requested a local commit after secret scanning so a fresh session can
review a fixed revision. Another process committed the remaining four correction
files (README, contract, implementation status and policy tests) as bcffc99;
this local checkpoint records the owner's authorization and accurate provenance.
The kernel URL fix is already in 74b38de. Review all corrections from 06f8f3a,
not only the last commit, plus the complete kernel and unchanged verifier.

No implementation changes were made for checkpointing. The 243-source manifest
and policy/test hashes still match the verified snapshot above. Commit checks
include the hermetic financial/legacy/sidecar suites, staged-diff secret scanning
with redacted output, whitespace checks and exact staged-file scope.
Checkpoint reruns: financial 754 passed in 3.61s; legacy 50 passed in 5.76s;
sidecar 403 passed in 28.32s. All exited 0 with zero unexpected isolation denials;
financial retained 44 expected denials. No push,
service operation, signing-key access or deployment is authorized or performed.
Fresh independent review remains required; Stage 3 remains off.

## Stage 1 exact changes

- tests/support/financial_test_bootstrap.py (new): stdlib-only runner for fixed
  suites (three in Stage 1; Stage 2 added the fourth, `financial`). Requires initial Python -I -S, disables bytecode, clears inherited
  environment and launches fresh user/network/PID/mount namespaces through the
  existing unshare command. Copies Python source, not provider configs or data,
  into a readonly chroot; mounts existing interpreter/dependencies readonly.
  Host homes, credentials and persistent storage are absent. Only tmpfs scratch
  is writable; proc is private/readonly and setup capabilities are dropped.
- The runner supplies an empty synthetic providers_database.yaml to prevent a
  legacy host-specific catalog fallback. It suppresses dotenv loading before
  pytest/conftest/application imports, forces LiteLLM's local model-cost map and
  disables plugin autoload/cache writes. Nothing is installed or synchronized.
- Python audit guard blocks credential/default-storage reads, unsafe writes,
  unregistered connects/DNS/UDP/addressed Unix sockets, shell/native/direct
  process launches and unsupported child invocations. Validated SQLite file
  URIs and descriptor/symlink targets must remain disposable or readonly inputs.
- Successfully activated TCPServer fixtures on 127.0.0.1:0 are explicitly
  registered by the bootstrap adapter. Raw listeners require register_endpoint.
  Closed parent listeners are removed. Anonymous socketpairs are permitted.
- Python subprocesses are re-bootstrapped with -I -S, constrained environment,
  closed inherited descriptors and current registered endpoints. They inherit
  the kernel sandbox. Unexpected denials are sticky, including swallowed child
  errors and sys.exit(0); intentional negative self-tests acknowledge denials.
- tests/test_financial_test_isolation.py (new): 45 negative/positive self-tests,
  including threads, lower-level sockets, URI/FD/symlink controls, child startup,
  inherited-environment overrides, original-Popen bypass and closed listeners.
- tests/conftest.py: skip application telemetry imports for isolation self-tests;
  under the guarded legacy runner only, redirect both import spellings of
  RateLimitTracker's default paths before collection. Existing explicit fixture
  paths and business logic are unchanged.
- README.md: supported invocation, suite inventory, registration and limitations.
- docs/IMPLEMENTATION_STATUS.md (new): this continuity artifact.

## Stage 1 exact verification and results

Run from repository root, using the existing Python 3.12 venv:

```sh
uv --no-config --no-cache run --offline --no-project --no-env-file --python venv/bin/python -- venv/bin/python -I -S tests/support/financial_test_bootstrap.py --suite isolation
uv --no-config --no-cache run --offline --no-project --no-env-file --python venv/bin/python -- venv/bin/python -I -S tests/support/financial_test_bootstrap.py --suite legacy
uv --no-config --no-cache run --offline --no-project --no-env-file --python venv/bin/python -- venv/bin/python -I -S tests/support/financial_test_bootstrap.py --suite sidecar
```

Final runs:
- isolation: 45 passed in 0.76s; 0 unexpected / 44 expected denials.
- legacy: 50 passed in 1.87s; 0 unexpected denials. Selection: routing integrity
  #369, alias streaming, counter persistence #377, transient-429, task telemetry.
- sidecar: 403 passed in 25.23s; 0 unexpected denials, no warning in this run.
  Entire workload-router/tests, including fake-backed tests named live; no live
  verifier, provider completion, real New-API or installed client was executed.

Development failures were not hidden: initial isolation 4 failed/29 passed
  (pipe FDs); legacy 50 passed but runner failed on one catalog fallback denial;
  initial sidecar 17 failed/386 passed (URI/symlink/synthetic-secret/socket-FD
  compatibility), then 2 failed/401 passed (validated synthetic file FDs).
  Verifier-only corrections resolved these; no production behavior was patched.

Static verification: `python3 -I -S -B` checked the three Python files using
in-memory AST/compile/tabnanny, ASCII decoding and an 88-column bound. New
Markdown and the inserted README block were ASCII/whitespace checked.
`git diff --check` passed; complete tracked and new-file diffs were reviewed.
No existing ruff/black/mypy/coverage tool was available; none was downloaded.
Syntax, ASCII, line-width and whitespace checks are not a substitute for a
full linter/type checker or measured coverage.

## Preservation and limitations

- Pre-existing untracked docs/PROJECT_AUDIT.md and docs/DEVELOPMENT_ROADMAP.md
  remain untouched. Recorded SHA-256 values:
  f4f6170418954fa3295717f6dac036e80d94048b3df24bc51d269fee2dfca15f
  e25a1a791534ccfc1ef44f36457e652375bd21ec1bafae9d2a6d2173e12e8dd3
- Stage 1 changed/created only the five files named in its changes section.
  Tests used disposable tmpfs and owned temporary mount directories, cleaned
  on exit. No real credentials/logs/default databases were opened; no live
  quota, external network, dependency install, service or account mutation.
- Linux x86_64/Python 3.12/existing venv/unshare/ip/user namespaces are required.
  Setup failure has no permissive fallback. Initial -I -S is mandatory; a later
  error cannot undo startup hooks from an incorrectly invoked interpreter.
- Trusted tests and installed dependencies, not adversarial/native code, are
  the supported boundary. Python internals can defeat cooperative audit/endpoint
  checks; kernel containment still separates host resources. Children inherit
  launch-time endpoint snapshots, not atomic cross-process revocation leases.
- Full legacy suite, production lifespan, installed/packaged releases, root CI,
  pinned external transport and account billing contracts remain unverified.
  No coverage percentage or complete financial-safety guarantee is claimed.

## Stage 2 exact changes

Only the six files permitted by the contract were created or edited:

- src/financial_safety.py (new): stdlib-only (`dataclasses`, `decimal`, `enum`,
  `re`) immutable DTOs and enums from contract section 5, plus
  `evaluate_admission` and `require_admission`. Strict exact-type validation,
  ASCII identity matching, exact Decimal zero tests, deterministic denial
  precedence, safe denial exception and truthiness rejection. No I/O, logging,
  clock read, provider/account/credential access or runtime wiring.
- tests/test_financial_safety.py (new): synthetic acceptance matrix covering all
  47 case IDs (A/F/I/P/T/N/S groups) with parameterized fields and variants.
  Denial spies are in-memory functions/lists only. A guarded Python child proves
  a fresh import adds no application modules and that evaluation does not touch
  replaced file, environment, clock, socket, process or logging APIs.
- tests/support/financial_test_bootstrap.py: added the `financial` suite choice
  and a branch collecting both the isolation self-tests and the new matrix.
  Existing guards, suites, endpoint policy and child handling are unchanged.
- tests/conftest.py: extended the application-free autouse-fixture exclusion to
  the financial tests. No application import and no other fixture change.
- README.md: documented the `financial` invocation and its narrow meaning.
- docs/IMPLEMENTATION_STATUS.md: this milestone record.

## Stage 2 implementation verification

Commands run from the repository root, unchanged for all three suites:

```sh
uv --no-config --no-cache run --offline --no-project --no-env-file --python venv/bin/python -- venv/bin/python -I -S tests/support/financial_test_bootstrap.py --suite financial
uv --no-config --no-cache run --offline --no-project --no-env-file --python venv/bin/python -- venv/bin/python -I -S tests/support/financial_test_bootstrap.py --suite legacy
uv --no-config --no-cache run --offline --no-project --no-env-file --python venv/bin/python -- venv/bin/python -I -S tests/support/financial_test_bootstrap.py --suite sidecar
```

Results reported by the original implementer (historical, not current fixes):
- financial: 389 passed in 2.31s; 0 unexpected / 44 expected isolation denials.
  Includes all 45 isolation self-tests and the original 344 kernel cases.
  No skips/xfails reported. Independent review subsequently found missing field
  variants; these counts did NOT prove complete acceptance-matrix coverage.
- legacy: 50 passed in 7.53s; 0 unexpected denials.
- sidecar: 403 passed in 29.95s; 0 unexpected denials.
- No raw pytest, weakened isolation or permissive fallback was used. All exit
  codes were 0 only through the hermetic runner above.

Static verification (offline, no installs): `python3 -I -S -B` AST/compile and
ASCII/88-column/whitespace checks over the new and edited Python files, plus
`git diff --check`. No ruff, black, mypy, pyright or coverage tool is available
in the venv; none was downloaded. Those checks were not run and are not claimed.

Self-review findings and limitations:
- The initial implementer claimed sections 5-9 were complete with no
  contradiction. Independent review found a model-regex/URL-prohibition
  contradiction and incomplete regression coverage. The corrective pass above
  supersedes that claim; section 6 now explicitly rejects any '://' model ID.
  This is a stricter clarification, not a relaxation of financial requirements.
  Duplicate approval IDs and targets (including expired/revoked records) still
  invalidate the whole context before lookup.
- Monetary tests use exact Decimal comparison only; no float conversion,
  quantization, tolerance or rounding. Non-finite and negative bounds are
  malformed policy, not nonzero charges. Denials never claim an observed charge.
- Identity matching is unnormalized ASCII and exact; case/alias changes and
  changed physical targets deny. The kernel cannot prove that an operator's
  approval is honest, that `now`/`expected_revision` are current, or that claimed
  evidence exists; the contract assigns those to the trusted caller/registry.
- Synthetic approvals exist only in tests. No flag, zero price, environment
  value or metadata can create approval, and no real provider lane is enabled.
- S06's file/env/clock/network/process/logging replacements are in-memory spies
  in a guarded child; they are not a hostile-code sandbox claim.
- Installations, deployments, provider calls, credentials, quotas, database
  mutation and external network remain out of scope and were not exercised.

## Remaining financial-safety work

Stage 1 does not fix the audit's production findings: disabled/inconsistent
legacy free-only gates; MoE/embedding/auxiliary dispatch differences; sidecar
positive-money credit/embedding/cache cases; fail-open credential filtering;
caller transport controls/opaque retries; missing account-bound funding proof.
Protected safe-coding, legacy auth/mounted routes and sensitive logging also
remain unresolved. Passing historical suites does not certify these behaviors.

## Stage 2 specification checkpoint - 2026-11-24 (superseded)

This checkpoint recorded the specification before implementation. The Stage 2
implementation and verification sections above now supersede it.

- Created docs/FINANCIAL_ADMISSION_CONTRACT.md, contract version 1. Defined the
  financial-only invariant, owner-controlled immutable approval context,
  resolved target/profile/transport binding, account and anonymous identities,
  separate actual-money/allowance semantics, complete zero-charge bounds,
  unconditional boundary assertions, strict types and denial precedence.
- Fixed the public DTOs/two APIs and 47 acceptance-case IDs with required field
  parameterization. Included malformed/untrusted/expired/revoked/changed targets,
  tiny positive charges, cache/tool/search fees, denial exception and truthiness
  behavior, no-side-effect checks, and deterministic multi-failure precedence.
- Specified exactly six future implementation files, verifier-suite extension,
  test commands, stop conditions, self-review findings, later integration gates,
  independent strong-model review requirements and a ready-to-use authorization
  prompt. No speculative router, ledger or evidence-collection implementation.
- Only the new contract and this status file changed during this checkpoint.
  Stage 1 code/README and both pre-existing audit documents were hash-checked
  unchanged. HEAD/branch/index unchanged; all existing uncommitted work preserved.
- Validation: `python3 -I -S -B` document/preservation checks exited 0:
  12 contract sections, 47 unique case IDs, 17 decision codes, valid status links,
  ASCII/whitespace/fence checks and six unchanged pre-existing artifact hashes.
  `git diff --check` passed; the complete contract/status text was reviewed.
  No pytest suites were rerun: current authorization prohibits network calls,
  including disposable HTTP fixtures. The 45/50/403 results above remain Stage 1
  results, not new specification tests.
- No credentials, provider-manager data, account evidence, network calls,
  dependency installation, services, deployment, commit or push were accessed
  or performed. No kernel/test/production files were created or modified.
- Review is same-author specification self-review, not independent review of
  an implementation that does not exist. Real account proof and runtime
  enforcement remain unresolved, as do all listed production bypasses.

## Next gate (requires separate approval)

Finish review of the verified Stage 2 correction diff and return it for a
fresh implementation review. The correcting agent's self-review is NOT
independent acceptance. Stage 3 (evidence/snapshot trust, physical-dispatch wiring
or runtime enforcement) remains separately owner-gated. Neither this record nor
passing synthetic tests qualifies any real account or enables a provider lane.
No additional router, quota engine, credential access or production integration
is authorized by the Stage 2 correction request.

## Mechanical traceability check (automated; not independent review)

Run 2026-11-24 on commit 71bd2ce. Reproducible, and explicitly not the
independent approval required above: it proves naming coverage, not that each
assertion tests what the matrix names.

- Contract section 8 matrix rows: 47; unique IDs: 47.
- Case IDs with no matching `test_<id>_*` function: 0.
- Test functions not mapped to a matrix ID: 0 (54 functions = 47 cases plus
  seven cases with a second test).
- Case IDs with more than one test: F09, I01, I04, I07, I08, P07, T02.
- `DecisionCode` values: 17; never named in the policy tests: 0; absent from
  the contract: 0.
- Contract prohibitions: no `xfail`, `pytest.skip`, `skipif` or policy
  monkeypatch in tests/test_financial_safety.py. The only `socket` reference is
  the S06 failing-spy replacement, which S06 requires.
- Documented `--suite financial` invocation re-run before recording this:
  389 passed, expected_denials=44, unexpected_denials=0, exit 0.

The mapping is by test-function name (`test_a01_...` is case A01). An ID with a
test is not proof that the test is correct; assertion-level review remains open
for the independent reviewer.

### Assertion-level pass (2026-11-24)

Checked the "each field / each class / each case" claims against the actual
parametrizations rather than the test names:

- `MONEY_FIELDS` 8, `BOUNDARY_FIELDS` 4, `ELIGIBLE_FUNDING` 3: F02-F04 and N01
  cover all eight money fields; F07 covers 4 boundary fields x 3 funding classes
  x 2 falsey values = 24; A01 covers all three eligible classes.
- `PRECEDENCE_CASES` covers the eleven section 7 combinations; `DENIAL_CASES`
  backs S01.
- One real gap found and fixed: I01 claimed eight independent identity changes
  (provider, model, account, authentication, fingerprint, operation, profile,
  transport) but parametrized only seven, with no standalone fingerprint change.
  Keyed fingerprint change is now its own case and must deny with
  APPROVAL_NOT_FOUND. Suite after the fix: 390 passed.

This was still not independent review. It checked selected named dimensions,
but the later independent pass found timestamp/type/import-time gaps. The
corrective verification above supersedes its completeness implications.

### Precedence re-derivation (2026-11-24)

Mechanical comparison of `evaluate_admission` in src/financial_safety.py against
the table in contract section 7: both yield the same 17-entry order from
INVALID_INPUT through ELIGIBLE (consecutive duplicates collapsed), matched
programmatically by AST line order. Every P10 case was additionally read against
section 7 and expects the first-failure code; the eleven combinations cover
invalid-input/invalid-policy, inactive/revision-mismatch, mismatch/absent,
revoked/expired, expired/untrusted, unknown-funding/nonzero,
incomplete/unknown-bound, unknown-bound/nonzero, nonzero/unproven,
unproven/uncontrolled.

The implemented order and the asserted first failures are correct. This does not
judge whether section 7 is the right policy, and it is not independent sign-off.

### Root markdown fold (2026-11-24)

Six single-reference handoffs moved from the root to docs/archive/:
FINAL_COMPLETION_REPORT, FINAL_TYPE_SAFETY_FIXES, PERMANENT_HTTPS_SETUP,
VPS_STATUS, HTTPS_SETUP_GUIDE, LSP_FIXES_SUMMARY. Root markdown 26 -> 20 and the
relative-link scan still reports 0 broken. HANDOFF_OPUS_2026-06-24.md and
ROUTER_DOCUMENTATION.md stay at the root because SPEC.md links them directly.
