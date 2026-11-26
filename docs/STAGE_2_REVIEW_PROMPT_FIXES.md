# Independent review of the Stage 2 test-gap fixes

Read-only acceptance review. Do not implement fixes and do not start Stage 3.

Repository: /home/osees/CodingProjects/LLM-API-Key-Proxy
Branch: feat/financial-admission-kernel
Review commit: 2a64c2473b61d1101f6aabe2a2620a25f04efff0
Range to review: e64bc5d..2a64c24

This supersedes the earlier revision of this file, which pinned 093d7f1 and
expected 755 financial tests. A previous independent review of 093d7f1
returned ACCEPT WITH REQUIRED FIXES, so reviewing that commit again will
reproduce the finding it records. Review 2a64c24 instead.

First accept or decline this review, and disclose whether you contributed to
the kernel, these tests, or the contract. A previous agent did this work; it is
not independent acceptance of itself.

## Why this review exists

An earlier independent review returned ACCEPT WITH REQUIRED FIXES with two
Medium findings, both in the acceptance tests rather than the kernel. A second
independent review of 093d7f1 confirmed the P07 fix but found a remaining T02
gap (valid_until=False) and an overstated mutation claim. Commit 2a64c24 closes
both. Read the earlier review request at docs/STAGE_2_REVIEW_PROMPT.md for
context and the matrix definitions at docs/FINANCIAL_ADMISSION_CONTRACT.md
section 8.

## Snapshot

The checkpoint handed to you is 2a64c24 with a clean working tree. This prompt
file was written after that commit, so if the only difference from 2a64c24 is
this file, report that as the expected review-instruction artifact and exclude
it from the code diff. Report any other drift instead of reviewing a moving
target.

## What changed

Manifest changes in e64bc5d..2a64c24 are five paths, not two:

- tests/test_financial_safety.py: T02 and P07 tests, plus constants.
- docs/IMPLEMENTATION_STATUS.md: the "Stage 2 review follow-up" section.

Those two are the substantive changes and are the ones to review as code and
record. The remaining three are non-code or review artifacts:

- README.md: ASCII status/arrow conversion only (`[ok]`/`[warn]`, `->`).
- docs/STAGE_2_REVIEW_PROMPT.md: the earlier review request.
- docs/STAGE_2_REVIEW_PROMPT_FIXES.md: this file.

src/financial_safety.py, docs/FINANCIAL_ADMISSION_CONTRACT.md,
tests/support/financial_test_bootstrap.py, tests/conftest.py and
tests/test_financial_test_isolation.py are unchanged from e64bc5d. Verify that
claim yourself with git diff, do not take it on trust.

## Claims to check

1. T02 masking is fixed. Malformed timestamps are now paired so that the
   ordering rule verified_at <= valid_from < valid_until still passes, leaving
   only the exact-type/range check able to deny. Confirm this is actually true
   for each parameterized case rather than only for the ones that look obvious.
2. Values that cannot be isolated at all are split into
   test_t02_out_of_range_timestamp_denies_with_ordering_companion. Confirm the
   split is honest: it must not be a way to make weak coverage look complete.
   It now includes verified_at out-of-range, negative valid_from, out-of-range
   valid_from, negative valid_until and valid_until=False. Check the reasoning
   for each, especially valid_until=False (False == 0, so no nonnegative
   valid_from can satisfy valid_from < valid_until).
3. P07 now covers duplicate approval IDs for active, expired and revoked
   records, each with a distinct valid target, all expecting INVALID_POLICY.
4. Three mutants are reported as caught. Reproduce or contradict them. Do not
   edit the repository to do it; work on copies in /tmp.
   - coercing verified_at: 3 failed, 752 passed (float, string, IntSubclass);
   - skipping ID uniqueness for expired/revoked records: 2 failed, 753 passed.
     The expired duplicate returns eligible; the revoked duplicate returns
     APPROVAL_REVOKED, not eligible. Confirm that this is a property of the
     kernel and not of the mutation, and that the test still fails either way;
   - accepting valid_until=False and bypassing ordering for it: 1 failed,
     755 passed, on the newly added non-isolating case.
5. The kernel still satisfies contract sections 5-9 in full. Previously
   reviewed, but re-read the complete kernel including unchanged parts.

## Judgement calls for you to overrule if wrong

- The agent decided which timestamp variants were isolable and which were not.
- The agent declared the kernel correct from reading it plus mutation testing.
  That is not external billing proof and not independent verification.
- No coverage, lint or type-check tool was available, so no percentage or
  lint/type result is claimed anywhere.
- The T02 and P07 closures above are the correcting agent's own work. They need
  this independent pass before Stage 2 counts as accepted.

## Verification constraints

Offline only. No edits, installs, downloads, credentials, provider requests,
host services, commits or pushes. Run only these, from the repository root:

    uv --no-config --no-cache run --offline --no-project --no-env-file --python venv/bin/python -- venv/bin/python -I -S tests/support/financial_test_bootstrap.py --suite financial
    uv --no-config --no-cache run --offline --no-project --no-env-file --python venv/bin/python -- venv/bin/python -I -S tests/support/financial_test_bootstrap.py --suite legacy
    uv --no-config --no-cache run --offline --no-project --no-env-file --python venv/bin/python -- venv/bin/python -I -S tests/support/financial_test_bootstrap.py --suite sidecar

Expected on this commit: financial 756 passed, legacy 50, sidecar 403, every
suite exit 0, zero unexpected isolation denials, financial 44 expected denials,
no skipped or xfailed required cases. Treat these numbers as claims to confirm,
not as evidence. Do not weaken a guard or fall back to raw pytest.

## Required report

- Accept or decline, with independence disclosure.
- Exact reviewed commit and snapshot status, including any discrepancy.
- Numbered findings by severity with file and line references.
- Any matrix ID whose required variants are still not really proven, and why.
- Commands actually run with exact results, separating reruns from claims.
- Verdict: ACCEPT, ACCEPT WITH REQUIRED FIXES, or REJECT.
- Whether Stage 2 may be treated as accepted as a pure policy component.

State plainly that a green suite does not qualify a real account or provider
lane, and that financial eligibility is separate from quota, authentication,
privacy, capability, transport and actual billing proof.

Stop after the report. Stage 3 remains unauthorized.
