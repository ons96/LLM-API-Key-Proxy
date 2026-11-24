# Financial Admission Contract

## 1. Status, scope and decision

- Contract version: 1; specified 2026-11-24; Stage 2 review amendment below.
- Owner of enablement/implementation approval: repository owner.
- Historical specification baseline: main, 5acacda126d6f9385f8ca17f1e4d032a93442f36.
- Status: kernel implemented, under corrective review; NOT runtime integration.
- Current authorization: the owner's request to fix the Stage 2 review findings.
  It does not authorize Stage 3, provider calls, deployment, commit or push.
- Architecture is unchanged: workload-router is the brain, New-API is transport,
  and the legacy gateway remains rollback. This is not another router.

Decision: retain the pure, stdlib-only financial eligibility evaluator.
It consumes a resolved target and a trusted, immutable policy context. It does
not discover providers, read credentials, infer account funding or make calls.
A faster capable coding model can implement this bounded contract; a fresh
strong-model review must follow before accepting its implementation.

Normative words MUST, MUST NOT and SHOULD below are implementation requirements.
This contract supersedes the historical plan only for
Stage 2's kernel boundary. The audit/roadmap still govern later integration.

## 2. Financial invariant and important distinctions

A selected physical provider/model/account/operation must have a proven maximum
actual monetary obligation of zero. Unknown, conditional or ambiguous financial
eligibility fails closed. Monetary obligation includes debt, delayed billing,
overage, purchases/top-ups, paid prepaid-value consumption and operation fees;
it is not limited to an immediate credit-card transaction.

Distinguish three quantities:

| Quantity | Interpretation | Can prove zero-money eligibility alone? |
| --- | --- | --- |
| Actual monetary obligation | Money charged/owed or paid value consumed | Must be established as zero |
| Free-quota consumption | Requests/tokens/other renewable allowance | No |
| Promotional-credit consumption | Value granted without monetary purchase | No |

Positive promotional-credit consumption can coexist with zero actual money.
Purchased credits, paid subscriptions and paid reservations are not promotional
credits. Retail/reference prices and opportunity costs are neither actual
account charges nor authorization evidence. A positive reference price does
not disqualify a separately proven non-billable arrangement; a positive actual
charge does. The kernel MUST NOT subtract credits, discounts, rebates, refunds
or negative costs from positive actual charges to make them appear zero.

This kernel answers FINANCIAL ELIGIBILITY ONLY. In particular:

- Exhausted/unknown allowance can be financially non-billable if a verified
  hard boundary prevents all monetary transitions. It still MUST NOT dispatch
  without the separate required quota admission/reservation.
- A successful financial decision does not authenticate a request, validate
  capabilities/context, establish privacy scrubbing, reserve allowance or
  certify the final transport. Those gates remain independently mandatory.
- No successful completion, HTTP status or billing estimate establishes an
  account's hard no-charge boundary. Observed debit is not measured here.

## 3. Trust boundary and evidence requirements

### Trusted context, not caller-supplied evidence

The evaluator takes its approval records FROM the PolicyContext, not from a
separate caller-provided approval/price object. The context must eventually be
built by an owner-controlled evidence/snapshot workflow, not request JSON,
provider discovery, model rankings, a pricing library or a marketing whitelist.
There is no public parser, self-approval API, database loader or environment
fallback in Stage 2.

Membership in approved_sources means the operator has accepted that evidence
registry/process as a source of account-bound approval. A source string or
fingerprint is not itself proof. Future compilation must verify the underlying
non-secret evidence, review contradictions and produce an immutable context.
No real account or evidence source is approved by this document.

The kernel validates the supplied context's structure and assertions. It cannot
prove external contractual truth, detect every remote billing-settings change,
or distinguish a privileged caller's fabricated context from a genuine one.
If this trust boundary cannot be maintained at dispatch, DO NOT wire the kernel.
Signing/transporting snapshots, evidence collection and secret management belong
to later separately approved work, not this implementation.

### Required meaning of a qualified approval

The approved arrangement must bind the exact target and establish all of:

1. The account/access principal and final credential/authentication are known.
2. All actual charging modes for the permitted dispatch profile are covered.
3. Every applicable monetary component has a zero upper bound; explicitly
   inapplicable components have an established zero bound, not a missing value.
4. Exhaustion and expiry cannot cause charges; paid recovery/substitution,
   debt, automatic purchases/top-ups and paid prepaid-value consumption are
   blocked independently of the currently observed allowance balance.
5. The transport reaches that one bound target and cannot silently redirect,
   substitute credentials/models, discover another destination or retry below
   the application's per-physical-attempt control.
6. Evidence is verified, currently valid, from an approved source and not
   revoked; policy revision agrees with the current trusted runtime revision.

Account settings such as 'auto-top-up off' alone do not establish item 4.
A capped budget or positive remaining credit balance is not an unconditional
zero-money bound: concurrent users, other keys, late usage and expiry can race.
All keys sharing an account/pool must be considered by the external evidence
workflow. The safe default is denial, not a balance-based exception.

Version 1 supports a single verified physical-target closure. Opaque pools or
unbounded hidden routing are denied, even if the public relay advertises free
access. A later reviewed extension may represent a proven finite safe closure;
do not implement it now or use a wildcard/single 'pool' account to imitate it.

## 4. Resolved target and dispatch-profile semantics

FinancialTarget identifies what the final physical attempt would actually do,
not the gateway alias or the initially requested provider label.

- Translated Responses using upstream chat are CHAT_COMPLETION; native upstream
  Responses are RESPONSE. The public HTTP path cannot grant either approval.
- Search basic/advanced/research are different operations; do not inherit basic
  search eligibility or credit tariffs into research.
- Stream/tool/cache/batch/endpoint-option variants must use the appropriately
  validated dispatch_profile_id. Profile identifiers are exact binding labels,
  not a classifier and not caller-controlled permission strings.
- A profile represents a closed permitted payload/operation shape. Monetary
  bounds cover EVERY payload within that profile, including permitted volumes,
  cache behavior, tools, failures/cancellation and applicable ancillary fees.
  This is not a token-count estimate for one cheap-looking request.
- New charge-affecting options or provider behavior require reviewed coverage
  and a new/updated profile approval. Unknown coverage means denial.
- Model identifiers must represent the proved physical mapping. A mutable
  upstream alias is not a resolved physical identity unless its complete
  monetary/target behavior is independently established. The kernel cannot
  infer alias resolution from the spelling of a model identifier. In version 1,
  the proved mapping must still be a singleton; do not silently introduce the
  deferred multi-target/opaque-closure extension.

transport_binding is an opaque digest of the validated final transport contract:
endpoint/route, model mapping, credential/header ownership, permitted payload
controls, SDK/relay behavior and redirect/retry/substitution restrictions.
It is computed outside Stage 2 from non-secret normalized descriptors. It is
NOT a raw URL, a prompt hash, a credential value or proof of financial terms.
Canonicalization and actual binding verification are later integration work.

Authentication can be keyed or explicitly anonymous. Anonymous means verified
absence of upstream credentials, including SDK/environment fallback credentials;
it does not mean an unknown key/account. Its nonempty account_id identifies the
specific proved public access principal, not all accounts or an opaque pool.
Anonymous/public approvals still need complete singleton-target evidence.

## 5. Minimal public data types

Implement the following names in src/financial_safety.py. Use Enum (not implicit
string coercion) and dataclasses frozen=True, slots=True. No mutable container
fields, mutable defaults or constructor-side I/O. Data constructors do not
perform validation or raise ValueError; public evaluation validates the complete
input. All fields are required: unknown values must be passed explicitly where
None or UNKNOWN is permitted. Do not add convenience approval defaults.

### Enums

- AuthenticationKind: KEYED='keyed', ANONYMOUS='anonymous'.
- Operation: CHAT_COMPLETION='chat_completion', RESPONSE='response',
  EMBEDDING='embedding', SEARCH_BASIC='search_basic',
  SEARCH_ADVANCED='search_advanced', SEARCH_RESEARCH='search_research',
  MODEL_DISCOVERY='model_discovery', TOKEN_COUNT='token_count',
  CREDENTIAL_REFRESH='credential_refresh', TRANSPORT_AUDIT='transport_audit'.
- FundingClass: NON_BILLABLE='non_billable', FREE_QUOTA='free_quota',
  PROMOTIONAL_CREDITS='promotional_credits', BILLABLE='billable',
  UNKNOWN='unknown'. This is the SELECTED ARRANGEMENT, not the provider-manager
  provider_type or quota taxonomy. Never map those flags automatically here.
- DecisionCode: the codes in section 7, with lowercase snake-case values.

### FinancialTarget

| Field | Type | Meaning |
| --- | --- | --- |
| provider_id | str | Exact physical provider identity |
| model_id | str | Exact proved physical model, or explicit control-operation N/A |
| account_id | str | Opaque account/public-principal identity; not a raw user ID |
| authentication | AuthenticationKind | Keyed or explicitly anonymous |
| credential_fingerprint | str or None | Exact keyed credential fingerprint; None ONLY for anonymous |
| operation | Operation | Actual upstream operation |
| dispatch_profile_id | str | Exact validated operation/payload profile |
| transport_binding | str | Digest of final controlled transport identity |

The reserved model_id '__not_applicable__' is permitted ONLY for MODEL_DISCOVERY,
CREDENTIAL_REFRESH and TRANSPORT_AUDIT. It is not a wildcard for generation,
embeddings, search or token counting. These control operations require their own
approved non-billable contracts; HTTP GET or an endpoint's name is insufficient.

### MonetaryBounds

| Field | Type | Requirement |
| --- | --- | --- |
| currency | str | Exactly 'USD' in version 1 |
| unit | str | Exactly 'currency_per_attempt' |
| coverage_complete | bool or None | True means complete coverage of the profile |
| input_charge | Decimal or None | Actual monetary upper bound for input |
| output_charge | Decimal or None | Actual monetary upper bound for output |
| cache_read_charge | Decimal or None | Actual monetary upper bound for cache reads |
| cache_write_charge | Decimal or None | Actual monetary upper bound for cache creation |
| request_charge | Decimal or None | Actual fixed/per-request charges |
| tool_charge | Decimal or None | Provider-side tool/ancillary execution charges |
| search_charge | Decimal or None | Actual search/research charges |
| other_charge | Decimal or None | All remaining actual monetary charges |

Each component is a bound for ONE PHYSICAL ATTEMPT over the entire approved
profile, not a per-token tariff or reported usage. All must be exactly zero.
Fields cover disjoint monetary components; no arithmetic/netting is required.
Unknown/unlisted charges make coverage_complete unknown/false and deny.
Version 1 intentionally rejects other currencies/units instead of guessing or
converting. This is a conservative format limitation, not evidence that an
otherwise proved non-USD zero-money arrangement is intrinsically unsafe.

### NoChargeBoundary

Four bool-or-None fields, all required and all must be True:

- exhaustion_zero: exhaustion/overrun cannot create monetary obligations.
- expiry_zero: allowance/evidence access expiry cannot create obligations.
- no_monetary_funding: no debt, purchases/top-ups, paid reservations or purchased
  prepaid/subscription value can be used by this arrangement/operation.
- substitution_zero: no paid recovery, fallback or billing-mode transition is
  reachable in the approved profile.

These are owner-accepted proof assertions, not operator preference toggles.

### FinancialApproval

Required fields:
- approval_id: str; target: FinancialTarget; funding: FundingClass.
- money: MonetaryBounds; boundary: NoChargeBoundary.
- verified_at: int; valid_from: int; valid_until: int (UTC epoch seconds).
- source_id: str; evidence_ref: str (opaque approved-registry reference).
- single_target_bound: bool or None; no_hidden_dispatch: bool or None.

### PolicyContext

Required fields:
- schema_version: int, exactly 1.
- revision: str, current owner-controlled immutable policy-generation ID.
- active: bool; False denies all financial admission, never permits paid mode.
- approved_sources: frozenset[str], at most 64 items.
- revoked_approval_ids: frozenset[str], at most 1024 items.
- approvals: tuple[FinancialApproval, ...], at most 1024 items.

Only one record per exact FinancialTarget and per approval_id is allowed.
Duplicate IDs OR targets invalidate the whole context, including duplicates
that are expired/revoked. Archive historical/diagnostic rows elsewhere. An empty
approved_sources or approvals collection is structurally valid but grants no
implicit eligibility. Unknown revoked IDs are allowed for retained revocations.
revision is an opaque generation ID, not a cryptographic signature. Stage 3 must
protect integrity and replace the generation whenever its contents/trust change.

### AdmissionDecision and FinancialAdmissionDenied

AdmissionDecision has required immutable fields:
- financially_eligible: bool; code: DecisionCode.
- policy_revision: str or None; approval_id: str or None.
- max_monetary_charge: Decimal or None.

On eligibility, code=ELIGIBLE and max_monetary_charge=Decimal('0'). On every denial,
financially_eligible=False and max_monetary_charge=None: a denial is not a claim
that the attempted/observed charge was zero. No account, credential, transport,
proof body, prompt or raw error belongs in the decision.

Metadata: INVALID_INPUT/INVALID_POLICY return policy_revision=None and
approval_id=None. Other codes return the structurally validated policy revision;
approval_id is populated only after an exact record match. No raw invalid data
is echoed. Decision __bool__ MUST raise TypeError with a fixed safe message;
callers must inspect financially_eligible explicitly. A denied DTO must never
become permission merely because Python objects are normally truthy.

FinancialAdmissionDenied directly subclasses Exception, not ValueError. It has
a decision attribute and a fixed safe message including only the DecisionCode
value. Its Exception args/string/repr must not contain inputs or record contents.
Existing adapter code catches ValueError and invokes LiteLLM fallback; financial
denial must not activate that path. Boolean misuse is a programming error, not
an alternative financial-admission API.

## 6. Strict validation and interfaces

Required public signatures:

- evaluate_admission(target: FinancialTarget, policy: PolicyContext, *,
  now: int, expected_revision: str) -> AdmissionDecision
- require_admission(target: FinancialTarget, policy: PolicyContext, *,
  now: int, expected_revision: str) -> AdmissionDecision

require_admission calls the same evaluator and raises FinancialAdmissionDenied
for EVERY denied decision. It returns the eligible decision unchanged otherwise.
Do not add validate_policy_snapshot, JSON/YAML parsers, pricing conversion,
account discovery, quota reservation or a transport wrapper in this milestone.

Validation rules:

- Use exact runtime types for DTOs, enums, containers, strings, ints and Decimal;
  reject subclasses/coercible objects as well as raw dictionaries/lists.
- Identifier strings are ASCII and unnormalized. provider_id, account_id,
  dispatch_profile_id, approval_id, source_id, evidence_ref and revision match
  `[A-Za-z0-9][A-Za-z0-9._:-]{0,127}`. model_id matches
  `[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}`, except the restricted N/A sentinel.
  No whitespace, wildcard, normalization, empty/missing or URL-based identity.
  Review amendment: the model regex is necessary but not sufficient. A model
  ID MUST NOT contain '://', regardless of scheme spelling or case. This closes
  the regex/prose contradiction without weakening the original URL prohibition.
  Namespace/model:tag spellings remain allowed; physical mapping still requires
  the separately trusted singleton proof described in section 4.
- credential_fingerprint and transport_binding are exactly 64 lowercase hex
  characters. Keyed requires a fingerprint; anonymous requires exactly None.
  Account identity is mandatory in both cases. UNKNOWN auth is invalid.
- now/verified_at/valid_from/valid_until are exact ints (not bool/float/string),
  between 0 and 253402300799 inclusive. Require
  verified_at <= valid_from < valid_until. Validity is [valid_from, valid_until):
  lower boundary included; upper boundary excluded. No grace period or TTL
  inferred from a quota reset. There is no implicit time.time() or clock read.
- bool-or-None fields accept exact bool or None only; eligibility requires True,
  not truthy values. active is exact bool, schema_version is exact int 1.
- Monetary fields accept exact finite nonnegative Decimal or None only.
  Decimal('-0') equals zero and is valid; any positive value, however tiny,
  denies. Negative/non-finite numbers and all float/int/bool/string inputs are
  malformed. Never use float(), quantize(), epsilon, tolerance or context-based
  rounding to test zero. No conversion/parsing happens in this module.
- Validate ALL record shapes and uniqueness before lookup. Malformed unrelated
  records invalidate the context. A well-shaped record with unknown funding,
  missing bounds or unproven assertions is denied when selected; its presence
  does not grant eligibility and is not equivalent to a validated startup
  catalog. Stage 3 must separately reject enabled ineligible catalogs.
- Do not mutate arguments, call callbacks, read application files/env/DB/clock,
  log, perform network/process operations or import application modules. Normal
  Python source/stdlib reads by the import loader are permitted trusted inputs,
  not permission for module-body I/O. Repeated calls with identical inputs must
  produce equal decisions. No module-level mutable application state. Keep
  runtime imports to dataclasses, enum, decimal, re and typing; private helpers
  and immutable validation constants are sufficient.

expected_revision must come from the CURRENT trusted policy manager at the
future dispatch boundary, not automatically from policy.revision or a ranking
snapshot. now must also be supplied there from a trustworthy current clock.
The pure kernel cannot detect a caller supplying stale time/revision together.

## 7. Deterministic denial precedence

Implement the following ordered checks. The first failing check supplies code:

| Order | Check/failure | DecisionCode |
| --- | --- | --- |
| 1 | Invalid target, now or expected_revision | INVALID_INPUT |
| 2 | Invalid policy/record shape, schema, bounds format, time ordering, duplicates or collection limits | INVALID_POLICY |
| 3 | policy.active is False | POLICY_INACTIVE |
| 4 | policy.revision != expected_revision | POLICY_REVISION_MISMATCH |
| 5 | No exact target match in policy.approvals | APPROVAL_NOT_FOUND |
| 6 | Matching approval_id is revoked | APPROVAL_REVOKED |
| 7 | now < valid_from | APPROVAL_NOT_YET_VALID |
| 8 | now >= valid_until | APPROVAL_EXPIRED |
| 9 | source_id not in approved_sources | UNTRUSTED_EVIDENCE |
| 10a | funding is UNKNOWN | UNKNOWN_FUNDING |
| 10b | funding is BILLABLE | BILLABLE_FUNDING |
| 11 | coverage_complete is not True | INCOMPLETE_CHARGE_COVERAGE |
| 12 | Any monetary component is None | UNKNOWN_MONETARY_BOUND |
| 13 | Any monetary component != exact zero | NONZERO_MONETARY_BOUND |
| 14 | Any NoChargeBoundary assertion is not True | UNPROVEN_NO_CHARGE_BOUNDARY |
| 15 | single_target_bound or no_hidden_dispatch is not True | UNCONTROLLED_TRANSPORT |
| 16 | All checks pass | ELIGIBLE |

Failures must return decisions rather than escaping as ValueError/AttributeError
from ordinary malformed inputs. Do not blanket-catch BaseException or convert
unexpected programming failures into eligibility. Explicit type/shape checks
make comparison/validation safe; no user-defined equality/string conversions.

## 8. Acceptance matrix for the implementer

Create explicit synthetic fixtures: fixed now=1500, verified_at=900,
valid_from=1000, valid_until=2000, revision='fixture-policy-1', approved source
'fixture-authority', distinct account/provider/model/profile identifiers,
keyed fingerprint 'a' repeated 64 times, transport binding 'b' repeated 64 times,
all eight Decimal('0') charges, coverage/boundary/transport assertions True.
These are test data, not copied provider rows or real account approvals.

Each row below is mandatory. Parameterize every named field/variant, not just a
representative one. Include the case IDs in tests/comments for review traceability.
Unless otherwise stated, make one change to an otherwise eligible fixture.

| ID | Synthetic case | Expected outcome |
| --- | --- | --- |
| A01 | Each NON_BILLABLE, FREE_QUOTA, PROMOTIONAL_CREDITS arrangement with complete hard proof | ELIGIBLE |
| A02 | Exact valid_from and one second before valid_until | ELIGIBLE |
| A03 | Each monetary field Decimal('-0') | ELIGIBLE |
| A04 | Anonymous, fingerprint None, explicit public principal, complete singleton proof | ELIGIBLE |
| A05 | Each Operation with an exact matched profile/target; allowed N/A control models separately | ELIGIBLE |
| A06 | require_admission then in-memory spy call | One synthetic call, exact target |
| F01 | funding UNKNOWN with zero charges; funding BILLABLE with zero charges | UNKNOWN_FUNDING / BILLABLE_FUNDING |
| F02 | Each monetary field None | UNKNOWN_MONETARY_BOUND |
| F03 | Each monetary field Decimal('0.01') | NONZERO_MONETARY_BOUND |
| F04 | Each monetary field Decimal('1e-1000') | NONZERO_MONETARY_BOUND |
| F05 | Positive actual cache/tool/search/other bounds alongside free funding | NONZERO_MONETARY_BOUND |
| F06 | Each coverage_complete False/None | INCOMPLETE_CHARGE_COVERAGE |
| F07 | Each boundary field False/None under each eligible funding class | UNPROVEN_NO_CHARGE_BOUNDARY |
| F08 | Promotional arrangement, positive actual charge or unknown exhaustion/expiry boundary | NONZERO_MONETARY_BOUND / UNPROVEN_NO_CHARGE_BOUNDARY |
| F09 | Each single_target_bound/no_hidden_dispatch False/None, including apparently free anonymous relay | UNCONTROLLED_TRANSPORT |
| I01 | Independently change provider/model/account/auth/fingerprint/operation/profile/transport to a different valid identity; update dependent auth/fingerprint or N/A fields together when required for validity | APPROVAL_NOT_FOUND |
| I02 | Two keys, same provider/model, different account or fingerprint; only first approved | Second APPROVAL_NOT_FOUND |
| I03 | Changed physical target after initial eligible evaluation | Re-evaluation APPROVAL_NOT_FOUND; spy unchanged |
| I04 | Raw alias/selected operation converted to a different physical model/operation | APPROVAL_NOT_FOUND unless separately approved |
| I05 | Keyed fingerprint missing; anonymous fingerprint present; anonymous account missing | INVALID_INPUT |
| I06 | N/A model sentinel on generation/embedding/search/token count | INVALID_INPUT |
| I07 | Empty, wildcard, whitespace, non-ASCII, excessive-length IDs; uppercase/invalid digests | INVALID_INPUT for target; INVALID_POLICY inside a record |
| I08 | Change the case of a valid identifier, or a valid policy generation ID; no trimming/case normalization | APPROVAL_NOT_FOUND / POLICY_REVISION_MISMATCH |
| P01 | Missing/raw-dict policy; mutable lists/sets; DTO/enum/scalar subclasses | INVALID_POLICY or INVALID_INPUT by position |
| P02 | Empty approvals tuple; otherwise valid empty approved_sources | APPROVAL_NOT_FOUND / UNTRUSTED_EVIDENCE |
| P03 | False active flag | POLICY_INACTIVE |
| P04 | Valid but changed expected_revision | POLICY_REVISION_MISMATCH |
| P05 | Revoke matching ID; remove it from approvals; remove its source from approved_sources | APPROVAL_REVOKED / APPROVAL_NOT_FOUND / UNTRUSTED_EVIDENCE |
| P06 | Each missing/empty source_id/evidence_ref | INVALID_POLICY |
| P07 | Duplicate approval IDs; duplicate exact targets, including expired/revoked duplicates | INVALID_POLICY |
| P08 | Malformed unrelated approval plus otherwise eligible match | INVALID_POLICY |
| P09 | schema version 0/2/True; invalid active; exceed each collection bound; malformed collection element | INVALID_POLICY |
| P10 | Multiple simultaneous failures: invalid input+invalid policy; invalid policy+inactive; inactive+revision mismatch; revision mismatch+absent target; revoked+expired; expired+untrusted source; unknown funding+positive money; incomplete coverage+missing bound; missing+positive bounds; positive money+unproven boundary; unproven boundary+uncontrolled transport | Exact first-failure code from section 7 |
| T01 | now one second before valid_from; exactly valid_until; after valid_until | APPROVAL_NOT_YET_VALID / APPROVAL_EXPIRED / APPROVAL_EXPIRED |
| T02 | Each negative/bool/float/string/None/out-of-range timestamp; invalid verified/valid ordering | INVALID_INPUT for now, otherwise INVALID_POLICY |
| T03 | Expiry, revocation or current-revision change after ranking | Fresh evaluation denies; in-memory spy unchanged |
| N01 | Each monetary field int 0, float 0.0, False, '0', negative Decimal, NaN, sNaN, infinities or Decimal subclass | INVALID_POLICY |
| N02 | Missing/unsupported currency/unit, including a per-token tariff unit | INVALID_POLICY |
| N03 | Each proof/coverage/transport boolean field 0/1/'true'/object | INVALID_POLICY |
| S01 | require_admission for every denial row | FinancialAdmissionDenied with same decision/code; zero spy calls |
| S02 | ValueError-only adapter-fallback harness around require_admission | FinancialAdmissionDenied propagates; zero fallback calls |
| S03 | bool() on both eligible and denied decisions | TypeError; no implicit permission |
| S04 | Decision/exception metadata and messages, including malformed synthetic inputs | No raw target/credential/proof/prompt/error content; denial max charge None |
| S05 | Equal repeated evaluations; original objects unchanged; attempts to mutate frozen DTOs | Deterministic equality; ordinary mutations rejected |
| S06 | Fresh guarded import with only permitted runtime imports, followed by evaluation with application I/O/logging/process/clock APIs replaced by failing spies | No application imports or module-body/evaluation side effects; normal trusted loader reads permitted |
| S07 | A01 fixture with positive reference price in test-only ancillary data, not in approval | Eligibility unchanged; reference prices are not evaluator inputs |
| S08 | Different/unknown/exhausted test-only allowance observations, hard zero-money proof unchanged | Financial decision unchanged; no claim that dispatch quota passed |

A denial 'spy' is only an in-memory function/list, NOT HTTP or a provider adapter.
It proves require_admission's local behavior, not coverage of production paths.
Do not build a second quota engine to implement S08 or infer numeric balances.
No xfail/skip/monkeypatch of the policy to make these cases pass. Preserve the
existing sidecar tests that reflect old behavior until a separately authorized
runtime milestone changes that behavior; their success is not financial proof.

## 9. Bounded implementation handoff and verification

After separate implementation approval, the exact permitted file scope is:

1. src/financial_safety.py (new): types and the two pure public APIs above.
2. tests/test_financial_safety.py (new): fixtures, all matrix cases and spy checks.
3. tests/support/financial_test_bootstrap.py: only add suite choice 'financial'
   and a _run_tests branch collecting BOTH isolation and financial test modules.
   Preserve all namespace/FS/network/process guards and existing suite choices.
4. tests/conftest.py: extend the application-free telemetry-fixture exclusion to
   test_financial_safety.py as well as test_financial_test_isolation.py. Do not
   import financial_safety/application modules earlier to perform that exclusion.
5. README.md: document the new financial suite, explicitly not runtime safety.
6. docs/IMPLEMENTATION_STATUS.md: exact implemented changes/results/limitations.

Do not edit this normative contract to relax a failing test. If a genuine
contradiction is discovered, stop implementation, record the issue in the status
artifact and request a reviewed contract amendment. Do not touch existing user
changes, production configs, routers/adapters/ledgers, provider-manager data,
credentials, CI/deployment/package installation or services.

Future verification commands (NOT executed by this specification session):

```sh
uv --no-config --no-cache run --offline --no-project --no-env-file --python venv/bin/python -- venv/bin/python -I -S tests/support/financial_test_bootstrap.py --suite financial
uv --no-config --no-cache run --offline --no-project --no-env-file --python venv/bin/python -- venv/bin/python -I -S tests/support/financial_test_bootstrap.py --suite legacy
uv --no-config --no-cache run --offline --no-project --no-env-file --python venv/bin/python -- venv/bin/python -I -S tests/support/financial_test_bootstrap.py --suite sidecar
```

financial must collect/run all existing isolation self-tests and the new matrix;
exit 0, zero unexpected isolation denials, no skipped required cases. Existing
legacy/sidecar baselines are 50/403 passing tests from Stage 1, not fresh results
here. Tests may use only the verifier's registered disposable fixture endpoints;
no real providers or unregistered/host loopback. Do not run raw pytest as fallback.

Use existing free/local tooling for formatting/lint/type/coverage when available;
no network downloads. If unavailable, record that fact, run in-memory syntax,
whitespace/ASCII/line checks and manually inspect types. Do not claim lint/type
or measured coverage success from syntax checks. Inspect the complete diff,
including untracked new files, and prove pre-existing files were preserved.

Completion requires a bounded diff, passing commands and a traceable test matrix,
not merely a module that imports or a successful positive fixture. No commit,
push or progression into another milestone without its own authorization.

## 10. Self-review decisions and future integration gates

Specification self-review (same author, NOT an independent implementation review):

| Risk | Decision in this contract | Remaining gate |
| --- | --- | --- |
| Free flag/zero price masquerades as account proof | Trusted account-bound approvals; no metadata parser | Owner evidence collection and snapshot integrity |
| Finite credits race/expire into bills | Unconditional hard boundary; allowance admission separate | Pool-aware reservations and verified provider behavior |
| Tiny positive charge rounded to zero | Exact finite Decimal, no conversion/netting/tolerance | Review every monetary validation branch |
| Missing 'not applicable' charge becomes free | Explicit proved zero per field plus complete coverage | Complete external charging-mode inventory |
| Caller changes key/model/options after ranking | Exact target/profile/transport binding; revision/time explicit | Guard after final identity/payload preparation per physical attempt |
| Forged approval ID or evidence object | No separate approval argument; lookup only in trusted context | Do not expose context construction to clients/discovery |
| Ambiguous/conflicting approvals | Duplicate IDs/targets invalidate whole context | Merge corroborating evidence outside kernel, archive history |
| Money safety confused with privacy/quota/auth | financially_eligible is explicitly narrow; no truthiness | Separate mandatory gates and safe-alias rejection |
| Denial accidentally invokes LiteLLM fallback | Dedicated non-ValueError exception | Cross-path dispatch/fallback tests and disabled unsupported paths |
| Opaque relay/no-key pool assumed safe | Explicit auth/principal and singleton proof; opaque closure denies | Independently verified target closure; otherwise remain disabled |
| Stale context/time presented together | Kernel states caller trust dependency explicitly | Current policy manager, expiry, revocation and TOCTOU semantics |
| Strong specification mistaken for deployed safety | Synthetic kernel under review; no runtime wiring | Separate implementation review, integration, release/account gates |

Fresh strong-model review after implementation must inspect the FULL diff and
matrix, especially type coercion, zero arithmetic, proof completeness, record
lookup, mutation, denial precedence, exception behavior, isolation preservation
and missing tests. It must distinguish same-author self-review from independent
review and local fixture proof from external billing evidence.

Later implementation must authorize every physical generation/search/discovery/
refresh/control attempt or explicitly disable that path. Apply money admission
before unapproved startup/background activity and again after final identity
selection, payload preparation and reservation, immediately before dispatch.
Generator creation is not always physical dispatch: streaming/lazy SDK calls
must be guarded when their physical attempt starts. No hidden attempts or target
changes may inherit an earlier financial decision.

The production paths/catalog/compiler/packaging and their existing defects stay
untouched in Stage 2. No real lane is enabled merely because this kernel passes.

## 11. Suggested next authorization and model split

Ready-to-use next prompt, ONLY if the owner separately approves implementation:

'Implement Stage 2 exactly as docs/FINANCIAL_ADMISSION_CONTRACT.md sections 5-9.
Preserve all existing changes. Implement the pure kernel and synthetic matrix;
add only the bounded financial-suite/fixture documentation changes listed there.
No runtime wiring, provider metadata/credential access, dependency download,
external network, live quota, deployment, commit or push. Registered disposable
fixture endpoints inside the verifier are permitted for its existing self-tests.
Run the documented checks, review the complete diff, update implementation status
and stop. Do not relax the contract or proceed to Stage 3.'

A capable faster/free coding model with moderate reasoning can implement this
fixed slice; do not use a weak model to reinterpret financial semantics. Return
to a strong high-reasoning model for the independent post-implementation review,
any contract contradiction, evidence/snapshot trust design, and physical-dispatch
integration. Syntax, routine test execution and concise status updates can use
a cheaper model. Changing models does not grant implementation authorization.

## 12. Local evidence and reference boundaries

Inputs: docs/PROJECT_AUDIT.md, docs/DEVELOPMENT_ROADMAP.md,
docs/IMPLEMENTATION_STATUS.md, docs/ARCHITECTURE-DECISION-2026-11-22.md,
and /home/osees/p0-engineering-plan-reference.md. Root CLAUDE.md is absent.

Historical source checks supporting the original specification boundary:
- src/proxy_app/router_core.py:1497-1510 and 2775-2788 catch ValueError into
  direct LiteLLM fallback.
- workload-router/route_policy.py:87-90 and
  workload-router/quota_ledger.py:222-245 separate nominal money/credits without
  comprehensive account proof.
- At specification time, tests/support/financial_test_bootstrap.py had no
  financial suite and tests/conftest.py excluded only isolation self-tests.
  Stage 2 has since added the bounded suite/exclusion; see implementation status.

No credentials, live accounts, provider-manager rows, provider requests or network
calls were used in the original specification session. That session validated
only documents/preservation and did not rerun Stage 1. Current corrective-test
results are recorded separately in docs/IMPLEMENTATION_STATUS.md; they approve
no real account, evidence source or runtime integration.
