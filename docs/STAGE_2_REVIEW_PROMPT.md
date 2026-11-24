# Independent Stage 2 acceptance review

Independently review the corrected Stage 2 financial-admission kernel.
This is a read-only acceptance review, not implementation work.

Repository:
  /home/osees/CodingProjects/LLM-API-Key-Proxy

Branch:
  feat/financial-admission-kernel

Pinned review commit:
  e64bc5d1f5df2e34102533e207e97bb780434498

Full correction range:
  06f8f3a..e64bc5d

First explicitly accept or decline this independent review. Disclose if you
participated in writing the kernel, corrective tests or contract.

## 1. Establish the review snapshot

Confirm repository root, branch, HEAD and working-tree status.
The checkpoint was clean when handed off.

This prompt file was created AFTER that checkpoint. If its only working-tree
change is an untracked docs/STAGE_2_REVIEW_PROMPT.md, report that expected
review-instruction artifact separately; it is not implementation drift and
must not be included in the reviewed code diff.

If HEAD or other files differ, report the discrepancy. Do not check out, reset,
stash, clean or overwrite anything. Do not silently review a moving target.

Review the entire correction range, not merely the last commit. Some fixes
were committed by another process before the final checkpoint. Also read the
complete kernel and acceptance tests, including unchanged portions.

## 2. Read the normative context

Read:
- Root AGENTS.md and SPEC.md; CLAUDE.md and STATE.md if present.
- /home/osees/CodingProjects/AI_USAGE_GUIDE.md
- docs/FINANCIAL_ADMISSION_CONTRACT.md, especially sections 5-9.
- docs/ARCHITECTURE-DECISION-2026-11-22.md
- docs/IMPLEMENTATION_STATUS.md

Treat IMPLEMENTATION_STATUS.md and prior passing-test claims as unverified
evidence, not proof of correctness.

## 3. Review scope

Inspect:
- src/financial_safety.py
- tests/test_financial_safety.py
- docs/FINANCIAL_ADMISSION_CONTRACT.md
- docs/IMPLEMENTATION_STATUS.md
- README.md

Verify preservation of:
- tests/support/financial_test_bootstrap.py
- tests/conftest.py
- tests/test_financial_test_isolation.py

Do not expand into Stage 3 or repair production routing.

## 4. Required checks

Check:
- Exact public APIs, enum values, required fields, runtime types, frozen/slotted
  DTOs, no permissive defaults and no implicit coercion.
- Exact finite nonnegative Decimal validation; zero-only admission; no rounding,
  tolerance, conversion, netting or credit-based monetary exceptions.
- Complete proof requirements, account/auth/credential/model/operation/profile/
  transport binding, trusted-context lookup and deterministic denial precedence.
- Fail-closed handling of malformed, missing, unknown, expired, revoked,
  untrusted, conflicting or changed inputs.
- FinancialAdmissionDenied directly inherits Exception, not ValueError.
- No denial can reach dispatch or a ValueError-triggered fallback.
- Decision truthiness raises TypeError and metadata/messages remain safe.
- Every mandatory section 8 case ID and every named parameterized field/variant
  is substantively covered, not merely present in test names.
- The shared test harness does not mask wrong expected outcomes or make tests
  tautological.
- URL-shaped model IDs containing :// are rejected, while permitted namespace/
  model:tag identities and restricted N/A control models still work.
- The stricter contract clarification resolves the original contradiction
  without weakening financial requirements.
- Import-time spies are installed before the fresh kernel import.
- Loader-read allowances remain narrow; synthetic canaries genuinely block
  side effects before execution.
- No application I/O, environment/database/clock access, logging, process/network
  activity, argument mutation or mutable application state exists in the kernel.
- Isolation guards, existing suites and application-free fixture exclusions
  were not weakened.
- Scope and commit/provenance records are accurate.

Distinguish financial eligibility from quota reservation, authentication,
privacy, capability checks, transport enforcement and actual external billing
proof. Passing this kernel does not qualify a real account or provider lane.

## 5. Verification constraints

Offline only. No edits, installations, downloads, real credentials, provider
requests, live quota consumption, host services, deployment, commits or pushes.

You may run the following documented hermetic suites. Only their explicitly
registered disposable endpoints inside the private sandbox are permitted.
Arbitrary host/loopback endpoints are not permitted.

Run from the repository root:

```sh
uv --no-config --no-cache run --offline --no-project --no-env-file --python venv/bin/python -- venv/bin/python -I -S tests/support/financial_test_bootstrap.py --suite financial

uv --no-config --no-cache run --offline --no-project --no-env-file --python venv/bin/python -- venv/bin/python -I -S tests/support/financial_test_bootstrap.py --suite legacy

uv --no-config --no-cache run --offline --no-project --no-env-file --python venv/bin/python -- venv/bin/python -I -S tests/support/financial_test_bootstrap.py --suite sidecar
```

Latest expected counts:
- financial: 754 passed, including 709 kernel and 45 isolation tests.
- legacy: 50 passed.
- sidecar: 403 passed.
- Every suite: exit 0 and zero unexpected isolation denials.
- Financial suite: 44 expected isolation denials.
- No skipped or xfailed required cases.

Do not weaken guards, alter the policy or use raw pytest as a fallback.
If verification is unavailable or prohibited, explicitly mark it unverified.
Do not install missing lint/type/coverage tools or infer coverage percentages.

## 6. Required final report

Return:
- Acceptance/decline statement and independence disclosure.
- Exact reviewed commit and snapshot status.
- Numbered findings ordered by severity, with file/line references, impact
  and concrete required fixes.
- Mandatory matrix IDs/variants not fully verified, with reasons.
- Commands actually run and exact results; distinguish reruns from old claims.
- Verdict: ACCEPT, ACCEPT WITH REQUIRED FIXES, or REJECT.
- Whether Stage 2 is ready for the next separately authorized milestone.

If accepted, say that Stage 2 is accepted only as a pure policy component.
Production integration remains off and Stage 3 remains unauthorized.

Stop after the report. Do not implement fixes or begin another milestone.
