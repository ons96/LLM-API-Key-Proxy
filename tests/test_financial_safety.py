"""Stage 2 acceptance tests for the pure financial-admission kernel.

Every fixture here is synthetic. No provider, credential, quota, transport or
network resource is ever created or contacted.
"""

from __future__ import annotations

import copy
import dataclasses
import decimal
import enum
import inspect
import subprocess
import sys

import pytest

from financial_safety import (
    AdmissionDecision,
    AuthenticationKind,
    DecisionCode,
    FinancialAdmissionDenied,
    FinancialApproval,
    FinancialTarget,
    FundingClass,
    MonetaryBounds,
    NoChargeBoundary,
    Operation,
    PolicyContext,
    evaluate_admission,
    require_admission,
)

REVISION = "fixture-policy-1"
OTHER_REVISION = "fixture-policy-2"
SOURCE = "fixture-authority"
NOW = 1500
FINGERPRINT = "a" * 64
TRANSPORT = "b" * 64
APPROVAL_ID = "fixture-approval"

MONEY_FIELDS = (
    "input_charge",
    "output_charge",
    "cache_read_charge",
    "cache_write_charge",
    "request_charge",
    "tool_charge",
    "search_charge",
    "other_charge",
)

BOUNDARY_FIELDS = (
    "exhaustion_zero",
    "expiry_zero",
    "no_monetary_funding",
    "substitution_zero",
)

ELIGIBLE_FUNDING = (
    FundingClass.NON_BILLABLE,
    FundingClass.FREE_QUOTA,
    FundingClass.PROMOTIONAL_CREDITS,
)

TARGET = FinancialTarget(
    provider_id="fixture-provider",
    model_id="fixture-model",
    account_id="fixture-account",
    authentication=AuthenticationKind.KEYED,
    credential_fingerprint=FINGERPRINT,
    operation=Operation.CHAT_COMPLETION,
    dispatch_profile_id="fixture-profile",
    transport_binding=TRANSPORT,
)

MONEY = MonetaryBounds(
    currency="USD",
    unit="currency_per_attempt",
    coverage_complete=True,
    **{name: decimal.Decimal(0) for name in MONEY_FIELDS},
)

BOUNDARY = NoChargeBoundary(
    exhaustion_zero=True,
    expiry_zero=True,
    no_monetary_funding=True,
    substitution_zero=True,
)

APPROVAL = FinancialApproval(
    approval_id=APPROVAL_ID,
    target=TARGET,
    funding=FundingClass.NON_BILLABLE,
    money=MONEY,
    boundary=BOUNDARY,
    verified_at=900,
    valid_from=1000,
    valid_until=2000,
    source_id=SOURCE,
    evidence_ref="fixture-evidence",
    single_target_bound=True,
    no_hidden_dispatch=True,
)

POLICY = PolicyContext(
    schema_version=1,
    revision=REVISION,
    active=True,
    approved_sources=frozenset({SOURCE}),
    revoked_approval_ids=frozenset(),
    approvals=(APPROVAL,),
)

OPERATION_CASES = (
    ("chat", Operation.CHAT_COMPLETION, "fixture-model"),
    ("response", Operation.RESPONSE, "fixture-model"),
    ("embedding", Operation.EMBEDDING, "fixture-model"),
    ("search_basic", Operation.SEARCH_BASIC, "fixture-model"),
    ("search_advanced", Operation.SEARCH_ADVANCED, "fixture-model"),
    ("search_research", Operation.SEARCH_RESEARCH, "fixture-model"),
    ("model_discovery", Operation.MODEL_DISCOVERY, "fixture-model"),
    ("token_count", Operation.TOKEN_COUNT, "fixture-model"),
    ("credential_refresh", Operation.CREDENTIAL_REFRESH, "fixture-model"),
    ("transport_audit", Operation.TRANSPORT_AUDIT, "fixture-model"),
    ("discovery_na", Operation.MODEL_DISCOVERY, "__not_applicable__"),
    ("refresh_na", Operation.CREDENTIAL_REFRESH, "__not_applicable__"),
    ("audit_na", Operation.TRANSPORT_AUDIT, "__not_applicable__"),
)


class DecimalSubclass(decimal.Decimal):
    pass


class StrSubclass(str):
    pass


class IntSubclass(int):
    pass


class TupleSubclass(tuple):
    pass


class FrozenSetSubclass(frozenset):
    pass


class ApprovalSubclass(FinancialApproval):
    pass


class BoundsSubclass(MonetaryBounds):
    pass


class BoundarySubclass(NoChargeBoundary):
    pass


class ForeignEnum(str, enum.Enum):
    KEYED = "keyed"
    CHAT = "chat_completion"
    NON_BILLABLE = "non_billable"


class Coercible:
    """A malformed scalar whose conversion/comparison must never be invoked."""

    def __eq__(self, other: object) -> bool:
        raise AssertionError("unexpected equality callback")

    def __bool__(self) -> bool:
        raise AssertionError("unexpected truthiness callback")

    def __str__(self) -> str:
        raise AssertionError("unexpected string callback")

    def __int__(self) -> int:
        raise AssertionError("unexpected integer callback")


class TargetSubclass(FinancialTarget):
    pass


class PolicySubclass(PolicyContext):
    pass


def target(**overrides):
    return dataclasses.replace(TARGET, **overrides)


def money(**overrides):
    return dataclasses.replace(MONEY, **overrides)


def boundary(**overrides):
    return dataclasses.replace(BOUNDARY, **overrides)


def approval(**overrides):
    return dataclasses.replace(APPROVAL, **overrides)


def policy(**overrides):
    return dataclasses.replace(POLICY, **overrides)


def evaluate(
    target_value=TARGET, policy_value=POLICY, now=NOW, expected=REVISION
) -> AdmissionDecision:
    """Exercise S01/S02/S04/S05 for every parameterized matrix evaluation.

    Individual tests still assert their independently expected outcomes. Repr
    snapshots use synthetic test data only: opaque invalid objects and NaNs
    cannot reliably be compared through deepcopy/equality.
    """
    inputs = (target_value, policy_value, now, expected)
    snapshot = repr(inputs)
    decision = evaluate_admission(
        target_value, policy_value, now=now, expected_revision=expected
    )
    assert type(decision) is AdmissionDecision
    assert type(decision.financially_eligible) is bool
    assert type(decision.code) is DecisionCode
    assert decision == evaluate_admission(
        target_value, policy_value, now=now, expected_revision=expected
    )
    if decision.financially_eligible:
        assert decision == require_admission(
            target_value, policy_value, now=now, expected_revision=expected
        )
    else:
        spy = DispatchSpy()
        fallback_calls = []
        with pytest.raises(FinancialAdmissionDenied) as caught:
            try:
                admitted = require_admission(
                    target_value, policy_value, now=now, expected_revision=expected
                )
                spy.dispatch(target_value, admitted)
            except ValueError:
                fallback_calls.append("unexpected fallback")
                raise
        error = caught.value
        assert type(error) is FinancialAdmissionDenied
        assert error.decision == decision
        message = "financial admission denied: " + decision.code.value
        assert error.args == (message,)
        assert str(error) == message
        assert repr(error) == f"FinancialAdmissionDenied({message!r})"
        assert spy.calls == fallback_calls == []
        assert decision.max_monetary_charge is None
        if decision.code in (DecisionCode.INVALID_INPUT, DecisionCode.INVALID_POLICY):
            assert decision.policy_revision is decision.approval_id is None
        else:
            assert decision.policy_revision == policy_value.revision
            before_match = (
                DecisionCode.POLICY_INACTIVE,
                DecisionCode.POLICY_REVISION_MISMATCH,
                DecisionCode.APPROVAL_NOT_FOUND,
            )
            if decision.code in before_match:
                assert decision.approval_id is None
            else:
                matched = next(
                    item for item in policy_value.approvals
                    if item.target == target_value
                )
                assert decision.approval_id == matched.approval_id
    assert repr(inputs) == snapshot
    return decision


def assert_code(decision, code):
    assert decision.financially_eligible is False
    assert decision.code is code
    assert decision.max_monetary_charge is None


def assert_eligible(decision, revision=REVISION, approval_id=APPROVAL_ID):
    assert decision.financially_eligible is True
    assert decision.code is DecisionCode.ELIGIBLE
    assert decision.policy_revision == revision
    assert decision.approval_id == approval_id
    assert decision.max_monetary_charge == decimal.Decimal(0)


class DispatchSpy:
    def __init__(self):
        self.calls = []

    def dispatch(self, target_value, decision):
        self.calls.append((target_value, decision))


# ---------------------------------------------------------------------------
# A - positive cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "funding", ELIGIBLE_FUNDING, ids=[item.value for item in ELIGIBLE_FUNDING]
)
def test_a01_eligible_for_each_non_billable_funding_class(funding):
    decision = evaluate(policy_value=policy(approvals=(approval(funding=funding),)))
    assert_eligible(decision)


@pytest.mark.parametrize(
    "now", (1000, 1999), ids=("valid_from", "valid_until_minus_one")
)
def test_a02_validity_window_is_half_open(now):
    assert_eligible(evaluate(now=now))


@pytest.mark.parametrize("field", MONEY_FIELDS)
def test_a03_negative_zero_is_an_exact_zero(field):
    zero = decimal.Decimal("-0")
    decision = evaluate(
        policy_value=policy(approvals=(approval(money=money(**{field: zero})),))
    )
    assert_eligible(decision)


def test_a04_anonymous_target_with_explicit_principal_is_eligible():
    anonymous = target(
        account_id="fixture-public-principal",
        authentication=AuthenticationKind.ANONYMOUS, credential_fingerprint=None
    )
    assert_eligible(
        evaluate(
            anonymous,
            policy(approvals=(approval(target=anonymous),)),
        )
    )


@pytest.mark.parametrize(
    "operation,model_id", [(case[1], case[2]) for case in OPERATION_CASES],
    ids=[case[0] for case in OPERATION_CASES],
)
def test_a05_every_operation_matches_its_exact_profile(operation, model_id):
    exact = target(
        operation=operation, model_id=model_id,
        dispatch_profile_id="fixture-profile-" + operation.value,
    )
    assert_eligible(
        evaluate(exact, policy(approvals=(approval(target=exact),)))
    )


def test_a06_require_admission_reaches_bounded_dispatch_exactly_once():
    spy = DispatchSpy()
    decision = require_admission(TARGET, POLICY, now=NOW, expected_revision=REVISION)
    spy.dispatch(TARGET, decision)
    assert len(spy.calls) == 1
    recorded_target, recorded_decision = spy.calls[0]
    assert recorded_target is TARGET
    assert recorded_decision is decision
    assert_eligible(decision)


# ---------------------------------------------------------------------------
# F - funding, coverage, bounds, boundary and transport failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "funding,code",
    (
        (FundingClass.UNKNOWN, DecisionCode.UNKNOWN_FUNDING),
        (FundingClass.BILLABLE, DecisionCode.BILLABLE_FUNDING),
    ),
    ids=("unknown", "billable"),
)
def test_f01_unknown_or_billable_funding_denies_with_zero_charges(funding, code):
    decision = evaluate(policy_value=policy(approvals=(approval(funding=funding),)))
    assert_code(decision, code)


@pytest.mark.parametrize("field", MONEY_FIELDS)
def test_f02_each_missing_bound_component_is_unknown(field):
    decision = evaluate(
        policy_value=policy(approvals=(approval(money=money(**{field: None})),))
    )
    assert_code(decision, DecisionCode.UNKNOWN_MONETARY_BOUND)


@pytest.mark.parametrize("field", MONEY_FIELDS)
def test_f03_each_positive_component_is_nonzero(field):
    decision = evaluate(
        policy_value=policy(
            approvals=(approval(money=money(**{field: decimal.Decimal("0.01")})),)
        )
    )
    assert_code(decision, DecisionCode.NONZERO_MONETARY_BOUND)


@pytest.mark.parametrize("field", MONEY_FIELDS)
def test_f04_tiny_positive_component_is_nonzero(field):
    decision = evaluate(
        policy_value=policy(
            approvals=(approval(money=money(**{field: decimal.Decimal("1e-1000")})),)
        )
    )
    assert_code(decision, DecisionCode.NONZERO_MONETARY_BOUND)


@pytest.mark.parametrize(
    "field",
    ("cache_read_charge", "cache_write_charge", "tool_charge",
     "search_charge", "other_charge"),
)
def test_f05_positive_ancillary_component_with_free_quota_is_nonzero(field):
    decision = evaluate(
        policy_value=policy(
            approvals=(
                approval(
                    funding=FundingClass.FREE_QUOTA,
                    money=money(**{field: decimal.Decimal("0.01")}),
                ),
            )
        )
    )
    assert_code(decision, DecisionCode.NONZERO_MONETARY_BOUND)


@pytest.mark.parametrize("coverage", (False, None), ids=("false", "none"))
def test_f06_incomplete_coverage_denies(coverage):
    decision = evaluate(
        policy_value=policy(
            approvals=(approval(money=money(coverage_complete=coverage)),)
        )
    )
    assert_code(decision, DecisionCode.INCOMPLETE_CHARGE_COVERAGE)


@pytest.mark.parametrize(
    "funding", ELIGIBLE_FUNDING, ids=[item.value for item in ELIGIBLE_FUNDING]
)
@pytest.mark.parametrize("field", BOUNDARY_FIELDS)
@pytest.mark.parametrize("value", (False, None), ids=("false", "none"))
def test_f07_each_unproven_boundary_denies_under_each_funding_class(
    funding, field, value
):
    decision = evaluate(
        policy_value=policy(
            approvals=(
                approval(funding=funding, boundary=boundary(**{field: value})),
            )
        )
    )
    assert_code(decision, DecisionCode.UNPROVEN_NO_CHARGE_BOUNDARY)


@pytest.mark.parametrize(
    "case_id,overrides,code",
    tuple(
        (
            "positive_" + field,
            {"money": money(**{field: decimal.Decimal("0.01")})},
            DecisionCode.NONZERO_MONETARY_BOUND,
        )
        for field in MONEY_FIELDS
    ) + tuple(
        (
            field + "_" + str(value),
            {"boundary": boundary(**{field: value})},
            DecisionCode.UNPROVEN_NO_CHARGE_BOUNDARY,
        )
        for field in ("exhaustion_zero", "expiry_zero")
        for value in (False, None)
    ),
    ids=lambda item: item if type(item) is str else None,
)
def test_f08_promotional_credits_never_mask_a_real_charge(case_id, overrides, code):
    assert case_id
    decision = evaluate(
        policy_value=policy(
            approvals=(
                approval(funding=FundingClass.PROMOTIONAL_CREDITS, **overrides),
            )
        )
    )
    assert_code(decision, code)


@pytest.mark.parametrize("field", ("single_target_bound", "no_hidden_dispatch"))
@pytest.mark.parametrize("value", (False, None), ids=("false", "none"))
def test_f09_uncontrolled_transport_denies(field, value):
    decision = evaluate(
        policy_value=policy(approvals=(approval(**{field: value}),))
    )
    assert_code(decision, DecisionCode.UNCONTROLLED_TRANSPORT)


@pytest.mark.parametrize("field", ("single_target_bound", "no_hidden_dispatch"))
@pytest.mark.parametrize("value", (False, None), ids=("false", "none"))
def test_f09_apparently_free_anonymous_relay_is_still_uncontrolled(field, value):
    anonymous = target(
        account_id="fixture-public-principal",
        authentication=AuthenticationKind.ANONYMOUS, credential_fingerprint=None
    )
    decision = evaluate(
        anonymous,
        policy(
            approvals=(
                approval(
                    target=anonymous,
                    funding=FundingClass.FREE_QUOTA,
                    **{field: value},
                ),
            )
        ),
    )
    assert_code(decision, DecisionCode.UNCONTROLLED_TRANSPORT)


# ---------------------------------------------------------------------------
# I - identity binding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    (
        {"provider_id": "other-provider"},
        {"model_id": "other-model"},
        {"account_id": "other-account"},
        {"credential_fingerprint": "c" * 64},
        {
            "authentication": AuthenticationKind.ANONYMOUS,
            "credential_fingerprint": None,
        },
        {"operation": Operation.RESPONSE},
        {"dispatch_profile_id": "other-profile"},
        {"transport_binding": "c" * 64},
    ),
    ids=("provider", "model", "account", "fingerprint", "authentication",
         "operation", "profile", "transport"),
)
def test_i01_any_identity_change_is_not_found(overrides):
    decision = evaluate(target(**overrides))
    assert_code(decision, DecisionCode.APPROVAL_NOT_FOUND)


@pytest.mark.parametrize(
    "overrides",
    ({"account_id": "fixture-account-2"}, {"credential_fingerprint": "c" * 64}),
    ids=("account", "fingerprint"),
)
def test_i02_second_key_is_not_approved(overrides):
    changed = target(**overrides)
    decision = evaluate(changed, POLICY)
    assert_code(decision, DecisionCode.APPROVAL_NOT_FOUND)


def test_i03_changed_physical_target_after_ranking_is_not_found():
    spy = DispatchSpy()
    first = require_admission(TARGET, POLICY, now=NOW, expected_revision=REVISION)
    spy.dispatch(TARGET, first)
    changed = target(model_id="other-model")
    with pytest.raises(FinancialAdmissionDenied) as caught:
        require_admission(changed, POLICY, now=NOW, expected_revision=REVISION)
    assert caught.value.decision.code is DecisionCode.APPROVAL_NOT_FOUND
    assert len(spy.calls) == 1
    assert spy.calls[0][0] is TARGET


@pytest.mark.parametrize(
    "overrides",
    ({"model_id": "fixture-model-alias"}, {"operation": Operation.RESPONSE}),
    ids=("model_alias", "operation"),
)
def test_i04_converted_physical_identity_is_not_found(overrides):
    changed = target(**overrides)
    assert_code(evaluate(changed), DecisionCode.APPROVAL_NOT_FOUND)


@pytest.mark.parametrize(
    "overrides",
    ({"model_id": "fixture-model-alias"}, {"operation": Operation.RESPONSE}),
    ids=("model_alias", "operation"),
)
def test_i04_converted_physical_identity_is_eligible_when_separately_approved(
    overrides,
):
    changed = target(**overrides)
    second = approval(approval_id="fixture-approval-alias", target=changed)
    decision = evaluate(changed, policy(approvals=(APPROVAL, second)))
    assert_eligible(decision, approval_id="fixture-approval-alias")


@pytest.mark.parametrize(
    "overrides",
    (
        {"credential_fingerprint": None},
        {"authentication": AuthenticationKind.ANONYMOUS},
        {
            "authentication": AuthenticationKind.ANONYMOUS,
            "credential_fingerprint": None,
            "account_id": "",
        },
    ),
    ids=("keyed_without_digest", "anonymous_with_digest", "anonymous_without_account"),
)
def test_i05_credential_and_principal_shape_is_enforced(overrides):
    assert_code(evaluate(target(**overrides)), DecisionCode.INVALID_INPUT)


@pytest.mark.parametrize(
    "operation",
    (
        Operation.CHAT_COMPLETION,
        Operation.RESPONSE,
        Operation.EMBEDDING,
        Operation.SEARCH_BASIC,
        Operation.SEARCH_ADVANCED,
        Operation.SEARCH_RESEARCH,
        Operation.TOKEN_COUNT,
    ),
    ids=("chat", "response", "embedding", "search_basic", "search_advanced",
         "search_research", "token_count"),
)
def test_i06_not_applicable_model_is_rejected_outside_control_operations(operation):
    decision = evaluate(target(operation=operation, model_id="__not_applicable__"))
    assert_code(decision, DecisionCode.INVALID_INPUT)


TARGET_IDENTITY_FIELDS = (
    "provider_id", "model_id", "account_id", "dispatch_profile_id",
)
BAD_IDENTIFIERS = (
    ("empty", ""), ("missing", None), ("wildcard", "fixture-*"),
    ("space", "fixture value"), ("tab", "fixture\tvalue"),
    ("newline", "fixture\nvalue"), ("non_ascii", "fixtur\u00e9"),
    ("wrong_type", 3), ("subclass", StrSubclass("fixture")),
)
URL_MODEL_IDS = (
    "https://provider.invalid/model", "http://provider.invalid/model",
    "HTTPS://provider.invalid/model", "file:///fixture", "custom://fixture",
)
TARGET_IDENTITY_OVERRIDES = tuple(
    (field + "_" + case, {field: value})
    for field in TARGET_IDENTITY_FIELDS
    for case, value in BAD_IDENTIFIERS + (
        ("too_long", "x" * (257 if field == "model_id" else 129)),
    )
) + tuple(
    (field + "_" + case, {field: value})
    for field in ("credential_fingerprint", "transport_binding")
    for case, value in (
        ("uppercase", "A" * 64), ("short", "a" * 63), ("long", "a" * 65),
        ("non_hex", "g" * 64), ("empty", ""), ("missing", None),
        ("wrong_type", 3), ("subclass", StrSubclass("a" * 64)),
    )
) + tuple(
    ("model_url_" + str(index), {"model_id": value})
    for index, value in enumerate(URL_MODEL_IDS)
)


@pytest.mark.parametrize(
    "overrides", [case[1] for case in TARGET_IDENTITY_OVERRIDES],
    ids=[case[0] for case in TARGET_IDENTITY_OVERRIDES],
)
def test_i07_malformed_target_identity_is_invalid_input(overrides):
    assert_code(evaluate(target(**overrides)), DecisionCode.INVALID_INPUT)


@pytest.mark.parametrize(
    "overrides", [case[1] for case in TARGET_IDENTITY_OVERRIDES],
    ids=[case[0] for case in TARGET_IDENTITY_OVERRIDES],
)
def test_i07_malformed_record_identity_is_invalid_policy(overrides):
    malformed = approval(target=target(**overrides))
    decision = evaluate(policy_value=policy(approvals=(malformed,)))
    assert_code(decision, DecisionCode.INVALID_POLICY)


@pytest.mark.parametrize(
    "overrides",
    ({field: getattr(TARGET, field).upper()} for field in TARGET_IDENTITY_FIELDS),
    ids=TARGET_IDENTITY_FIELDS,
)
def test_i08_identifier_case_change_is_not_found(overrides):
    assert_code(evaluate(target(**overrides)), DecisionCode.APPROVAL_NOT_FOUND)


def test_i08_policy_generation_case_change_is_revision_mismatch():
    decision = evaluate(policy_value=policy(revision="Fixture-Policy-1"))
    assert_code(decision, DecisionCode.POLICY_REVISION_MISMATCH)


# ---------------------------------------------------------------------------
# P - policy shape, generation and registry failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case_id,policy_value,code",
    (
        ("missing_policy", None, DecisionCode.INVALID_POLICY),
        ("raw_dict", {"schema_version": 1}, DecisionCode.INVALID_POLICY),
        (
            "mutable_sources",
            dataclasses.replace(POLICY, approved_sources={SOURCE}),
            DecisionCode.INVALID_POLICY,
        ),
        (
            "mutable_approvals",
            dataclasses.replace(POLICY, approvals=list(POLICY.approvals)),
            DecisionCode.INVALID_POLICY,
        ),
        (
            "policy_subclass",
            PolicySubclass(
                schema_version=1,
                revision=REVISION,
                active=True,
                approved_sources=POLICY.approved_sources,
                revoked_approval_ids=POLICY.revoked_approval_ids,
                approvals=POLICY.approvals,
            ),
            DecisionCode.INVALID_POLICY,
        ),
        (
            "target_subclass",
            policy(
                approvals=(
                    approval(target=TargetSubclass(**dataclasses.asdict(TARGET))),
                ),
            ),
            DecisionCode.INVALID_POLICY,
        ),
        (
            "scalar_subclass_target",
            POLICY,
            DecisionCode.INVALID_INPUT,
        ),
    ),
    ids=(
        "missing_policy",
        "raw_dict",
        "mutable_sources",
        "mutable_approvals",
        "policy_subclass",
        "target_subclass",
        "scalar_subclass_target",
    ),
)
def test_p01_wrong_runtime_shapes_are_rejected(case_id, policy_value, code):
    if case_id == "scalar_subclass_target":
        decision = evaluate(target(provider_id=StrSubclass("fixture-provider")))
    else:
        decision = evaluate(policy_value=policy_value)
    assert_code(decision, code)


def test_p02_empty_registries_grant_nothing():
    assert_code(
        evaluate(policy_value=policy(approvals=())),
        DecisionCode.APPROVAL_NOT_FOUND,
    )
    assert_code(
        evaluate(policy_value=policy(approved_sources=frozenset())),
        DecisionCode.UNTRUSTED_EVIDENCE,
    )


def test_p03_inactive_policy_denies():
    decision = evaluate(policy_value=policy(active=False))
    assert_code(decision, DecisionCode.POLICY_INACTIVE)


def test_p04_stale_expected_revision_denies():
    decision = evaluate(expected=OTHER_REVISION)
    assert_code(decision, DecisionCode.POLICY_REVISION_MISMATCH)


def test_p05_revocation_removal_and_source_removal_deny():
    revoked = policy(revoked_approval_ids=frozenset({APPROVAL_ID}))
    assert_code(evaluate(policy_value=revoked), DecisionCode.APPROVAL_REVOKED)
    assert_code(
        evaluate(policy_value=policy(approvals=())),
        DecisionCode.APPROVAL_NOT_FOUND,
    )
    assert_code(
        evaluate(policy_value=policy(approved_sources=frozenset())),
        DecisionCode.UNTRUSTED_EVIDENCE,
    )


@pytest.mark.parametrize(
    "overrides",
    ({"source_id": ""}, {"evidence_ref": ""}, {"source_id": None},
     {"evidence_ref": None}),
    ids=("empty_source", "empty_evidence", "missing_source", "missing_evidence"),
)
def test_p06_missing_evidence_identity_is_invalid_policy(overrides):
    malformed = approval(**overrides)
    assert_code(
        evaluate(policy_value=policy(approvals=(malformed,))),
        DecisionCode.INVALID_POLICY,
    )


def test_p07_duplicate_approval_ids_are_invalid():
    second = approval(
        target=target(model_id="other-model"), approval_id=APPROVAL_ID
    )
    assert_code(
        evaluate(policy_value=policy(approvals=(APPROVAL, second))),
        DecisionCode.INVALID_POLICY,
    )


@pytest.mark.parametrize(
    "second,revoked",
    (
        (approval(approval_id="fixture-approval-2"), frozenset()),
        (approval(approval_id="fixture-approval-2", valid_until=1100), frozenset()),
        (approval(approval_id="fixture-approval-2"),
         frozenset({"fixture-approval-2"})),
    ),
    ids=("duplicate_target", "duplicate_expired_target", "duplicate_revoked_target"),
)
def test_p07_duplicate_targets_are_invalid_even_when_inactive(second, revoked):
    assert_code(
        evaluate(
            policy_value=policy(
                approvals=(APPROVAL, second), revoked_approval_ids=revoked
            )
        ),
        DecisionCode.INVALID_POLICY,
    )


def test_p08_malformed_unrelated_record_invalidates_the_context():
    unrelated = approval(
        approval_id="fixture-approval-2",
        target=target(model_id="other-model"),
        money=money(currency="EUR"),
    )
    assert_code(
        evaluate(policy_value=policy(approvals=(APPROVAL, unrelated))),
        DecisionCode.INVALID_POLICY,
    )


@pytest.mark.parametrize(
    "policy_value",
    (
        policy(schema_version=0),
        policy(schema_version=2),
        policy(schema_version=True),
        policy(active=0),
        policy(active=1),
        policy(active="yes"),
        policy(active=None),
        policy(approved_sources=frozenset(f"source-{i}" for i in range(65))),
        policy(revoked_approval_ids=frozenset(f"revoked-{i}" for i in range(1025))),
        policy(
            approvals=tuple(
                approval(
                    approval_id=f"fixture-approval-{i}",
                    target=target(provider_id=f"fixture-provider-{i}"),
                )
                for i in range(1025)
            )
        ),
        dataclasses.replace(POLICY, approved_sources=frozenset({SOURCE, 3})),
        dataclasses.replace(POLICY, approvals=(3,)),
        policy(revoked_approval_ids=frozenset({3})),
        policy(schema_version=None),
        policy(schema_version=1.0),
        policy(schema_version="1"),
        policy(schema_version=IntSubclass(1)),
        policy(active=object()),
    ),
    ids=(
        "schema_zero",
        "schema_two",
        "schema_true",
        "active_zero",
        "active_one",
        "active_string",
        "active_none",
        "sources_exceeded",
        "revoked_exceeded",
        "approvals_exceeded",
        "source_element_type",
        "approval_element_type",
        "revoked_element_type",
        "schema_none",
        "schema_float",
        "schema_string",
        "schema_subclass",
        "active_object",
    ),
)
def test_p09_schema_collection_and_element_limits(policy_value):
    assert_code(evaluate(policy_value=policy_value), DecisionCode.INVALID_POLICY)


PRECEDENCE_CASES = (
    (
        "invalid_input_before_invalid_policy",
        target(provider_id=""),
        None,
        NOW,
        REVISION,
        DecisionCode.INVALID_INPUT,
    ),
    (
        "invalid_policy_before_inactive",
        TARGET,
        policy(active=False, schema_version=2),
        NOW,
        REVISION,
        DecisionCode.INVALID_POLICY,
    ),
    (
        "inactive_before_mismatch",
        TARGET,
        policy(active=False),
        NOW,
        OTHER_REVISION,
        DecisionCode.POLICY_INACTIVE,
    ),
    (
        "mismatch_before_absent",
        target(model_id="other-model"),
        POLICY,
        NOW,
        OTHER_REVISION,
        DecisionCode.POLICY_REVISION_MISMATCH,
    ),
    (
        "revoked_before_expired",
        TARGET,
        policy(revoked_approval_ids=frozenset({APPROVAL_ID})),
        2000,
        REVISION,
        DecisionCode.APPROVAL_REVOKED,
    ),
    (
        "expired_before_untrusted",
        TARGET,
        policy(approved_sources=frozenset()),
        2000,
        REVISION,
        DecisionCode.APPROVAL_EXPIRED,
    ),
    (
        "unknown_funding_before_nonzero",
        TARGET,
        policy(
            approvals=(
                approval(
                    funding=FundingClass.UNKNOWN,
                    money=money(input_charge=decimal.Decimal("0.01")),
                ),
            )
        ),
        NOW,
        REVISION,
        DecisionCode.UNKNOWN_FUNDING,
    ),
    (
        "incomplete_before_unknown_bound",
        TARGET,
        policy(
            approvals=(
                approval(money=money(coverage_complete=False, input_charge=None)),
            )
        ),
        NOW,
        REVISION,
        DecisionCode.INCOMPLETE_CHARGE_COVERAGE,
    ),
    (
        "unknown_bound_before_nonzero",
        TARGET,
        policy(
            approvals=(
                approval(
                    money=money(
                        input_charge=None,
                        output_charge=decimal.Decimal("0.01"),
                    ),
                ),
            )
        ),
        NOW,
        REVISION,
        DecisionCode.UNKNOWN_MONETARY_BOUND,
    ),
    (
        "nonzero_before_unproven",
        TARGET,
        policy(
            approvals=(
                approval(
                    money=money(input_charge=decimal.Decimal("0.01")),
                    boundary=boundary(exhaustion_zero=False),
                ),
            )
        ),
        NOW,
        REVISION,
        DecisionCode.NONZERO_MONETARY_BOUND,
    ),
    (
        "unproven_before_uncontrolled",
        TARGET,
        policy(
            approvals=(
                approval(
                    boundary=boundary(expiry_zero=False),
                    single_target_bound=False,
                ),
            )
        ),
        NOW,
        REVISION,
        DecisionCode.UNPROVEN_NO_CHARGE_BOUNDARY,
    ),
)


@pytest.mark.parametrize(
    "case_id,target_value,policy_value,now,expected,code",
    PRECEDENCE_CASES,
    ids=[case[0] for case in PRECEDENCE_CASES],
)
def test_p10_denial_precedence_is_deterministic(
    case_id, target_value, policy_value, now, expected, code
):
    assert case_id
    decision = evaluate(target_value, policy_value, now, expected)
    assert_code(decision, code)


# ---------------------------------------------------------------------------
# T - time and lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "now,code",
    (
        (999, DecisionCode.APPROVAL_NOT_YET_VALID),
        (2000, DecisionCode.APPROVAL_EXPIRED),
        (2001, DecisionCode.APPROVAL_EXPIRED),
    ),
    ids=("before_valid_from", "at_valid_until", "after_valid_until"),
)
def test_t01_validity_window_boundaries(now, code):
    assert_code(evaluate(now=now), code)


@pytest.mark.parametrize(
    "now",
    (-1, True, False, 1500.0, "1500", None, 253402300800, IntSubclass(NOW)),
    ids=("negative", "bool_true", "bool_false", "float", "string", "none",
         "out_of_range", "int_subclass"),
)
def test_t02_malformed_now_is_invalid_input(now):
    assert_code(evaluate(now=now), DecisionCode.INVALID_INPUT)


@pytest.mark.parametrize(
    "overrides",
    (
        {"verified_at": 1001},
        {"verified_at": 2000},
        {"valid_from": 2000, "valid_until": 2000},
        {"valid_from": 2001},
    ),
    ids=("verified_after_from", "verified_at_until", "empty_window",
         "from_after_until"),
)
def test_t02_bad_policy_time_ordering_is_invalid_policy(overrides):
    malformed = approval(**overrides)
    assert_code(
        evaluate(policy_value=policy(approvals=(malformed,))),
        DecisionCode.INVALID_POLICY,
    )


def test_t03_ranking_is_not_cached_and_lifecycle_change_denies():
    spy = DispatchSpy()
    first = evaluate()
    spy.dispatch(TARGET, first)
    assert_code(evaluate(now=2000), DecisionCode.APPROVAL_EXPIRED)
    assert_code(
        evaluate(
            policy_value=policy(revoked_approval_ids=frozenset({APPROVAL_ID}))
        ),
        DecisionCode.APPROVAL_REVOKED,
    )
    mismatch = evaluate(expected=OTHER_REVISION)
    assert_code(mismatch, DecisionCode.POLICY_REVISION_MISMATCH)
    assert len(spy.calls) == 1
    assert spy.calls[0][0] is TARGET


# ---------------------------------------------------------------------------
# N - malformed bound values
# ---------------------------------------------------------------------------


INVALID_MONEY_VALUES = (
    ("int_zero", 0),
    ("float_zero", 0.0),
    ("false", False),
    ("string_zero", "0"),
    ("negative", decimal.Decimal("-1")),
    ("nan", decimal.Decimal("NaN")),
    ("snan", decimal.Decimal("sNaN")),
    ("infinity", decimal.Decimal("Infinity")),
    ("negative_infinity", decimal.Decimal("-Infinity")),
    ("decimal_subclass", DecimalSubclass("0")),
)


@pytest.mark.parametrize("field", MONEY_FIELDS)
@pytest.mark.parametrize(
    "value", [case[1] for case in INVALID_MONEY_VALUES],
    ids=[case[0] for case in INVALID_MONEY_VALUES],
)
def test_n01_malformed_monetary_values_are_invalid_policy(field, value):
    malformed = approval(money=money(**{field: value}))
    assert_code(
        evaluate(policy_value=policy(approvals=(malformed,))),
        DecisionCode.INVALID_POLICY,
    )


@pytest.mark.parametrize(
    "overrides",
    (
        {"currency": "EUR"},
        {"currency": None},
        {"currency": ""},
        {"currency": "usd"},
        {"currency": 0},
        {"currency": StrSubclass("USD")},
        {"unit": "currency_per_token"},
        {"unit": None},
        {"unit": ""},
        {"unit": 0},
        {"unit": StrSubclass("currency_per_attempt")},
    ),
    ids=("currency_unsupported", "currency_missing", "currency_empty",
         "currency_case", "currency_int", "currency_subclass", "unit_per_token",
         "unit_missing", "unit_empty", "unit_int", "unit_subclass"),
)
def test_n02_currency_and_unit_must_match_the_contract(overrides):
    malformed = approval(money=money(**overrides))
    assert_code(
        evaluate(policy_value=policy(approvals=(malformed,))),
        DecisionCode.INVALID_POLICY,
    )


INVALID_BOOLEANS = (
    ("zero", 0),
    ("one", 1),
    ("string", "true"),
    ("object", object()),
    ("coercible", Coercible()),
)

BOOLEAN_LOCATIONS = (
    ("money", "coverage_complete"),
    ("boundary", "exhaustion_zero"),
    ("boundary", "expiry_zero"),
    ("boundary", "no_monetary_funding"),
    ("boundary", "substitution_zero"),
    ("approval", "single_target_bound"),
    ("approval", "no_hidden_dispatch"),
)


@pytest.mark.parametrize("location,field", BOOLEAN_LOCATIONS)
@pytest.mark.parametrize(
    "value", [case[1] for case in INVALID_BOOLEANS],
    ids=[case[0] for case in INVALID_BOOLEANS],
)
def test_n03_non_boolean_proof_fields_are_invalid_policy(location, field, value):
    if location == "money":
        malformed = approval(money=money(**{field: value}))
    elif location == "boundary":
        malformed = approval(boundary=boundary(**{field: value}))
    else:
        malformed = approval(**{field: value})
    assert_code(
        evaluate(policy_value=policy(approvals=(malformed,))),
        DecisionCode.INVALID_POLICY,
    )


# ---------------------------------------------------------------------------
# S - exception, metadata, determinism and side-effect isolation
# ---------------------------------------------------------------------------


DENIAL_CASES = (
    (
        "invalid_input_target",
        target(provider_id=""),
        POLICY,
        NOW,
        REVISION,
        DecisionCode.INVALID_INPUT,
    ),
    (
        "invalid_input_now",
        TARGET,
        POLICY,
        1500.0,
        REVISION,
        DecisionCode.INVALID_INPUT,
    ),
    (
        "invalid_input_revision",
        TARGET,
        POLICY,
        NOW,
        "",
        DecisionCode.INVALID_INPUT,
    ),
    ("invalid_policy", TARGET, None, NOW, REVISION, DecisionCode.INVALID_POLICY),
    (
        "policy_inactive",
        TARGET,
        policy(active=False),
        NOW,
        REVISION,
        DecisionCode.POLICY_INACTIVE,
    ),
    (
        "revision_mismatch",
        TARGET,
        POLICY,
        NOW,
        OTHER_REVISION,
        DecisionCode.POLICY_REVISION_MISMATCH,
    ),
    (
        "approval_not_found",
        target(model_id="other-model"),
        POLICY,
        NOW,
        REVISION,
        DecisionCode.APPROVAL_NOT_FOUND,
    ),
    (
        "approval_revoked",
        TARGET,
        policy(revoked_approval_ids=frozenset({APPROVAL_ID})),
        NOW,
        REVISION,
        DecisionCode.APPROVAL_REVOKED,
    ),
    (
        "not_yet_valid",
        TARGET,
        POLICY,
        999,
        REVISION,
        DecisionCode.APPROVAL_NOT_YET_VALID,
    ),
    ("expired", TARGET, POLICY, 2000, REVISION, DecisionCode.APPROVAL_EXPIRED),
    (
        "untrusted",
        TARGET,
        policy(approved_sources=frozenset()),
        NOW,
        REVISION,
        DecisionCode.UNTRUSTED_EVIDENCE,
    ),
    (
        "unknown_funding",
        TARGET,
        policy(approvals=(approval(funding=FundingClass.UNKNOWN),)),
        NOW,
        REVISION,
        DecisionCode.UNKNOWN_FUNDING,
    ),
    (
        "billable_funding",
        TARGET,
        policy(approvals=(approval(funding=FundingClass.BILLABLE),)),
        NOW,
        REVISION,
        DecisionCode.BILLABLE_FUNDING,
    ),
    (
        "incomplete_coverage",
        TARGET,
        policy(approvals=(approval(money=money(coverage_complete=False)),)),
        NOW,
        REVISION,
        DecisionCode.INCOMPLETE_CHARGE_COVERAGE,
    ),
    (
        "unknown_bound",
        TARGET,
        policy(approvals=(approval(money=money(tool_charge=None)),)),
        NOW,
        REVISION,
        DecisionCode.UNKNOWN_MONETARY_BOUND,
    ),
    (
        "nonzero_bound",
        TARGET,
        policy(
            approvals=(approval(money=money(other_charge=decimal.Decimal("0.01"))),)
        ),
        NOW,
        REVISION,
        DecisionCode.NONZERO_MONETARY_BOUND,
    ),
    (
        "unproven_boundary",
        TARGET,
        policy(approvals=(approval(boundary=boundary(exhaustion_zero=None)),)),
        NOW,
        REVISION,
        DecisionCode.UNPROVEN_NO_CHARGE_BOUNDARY,
    ),
    (
        "uncontrolled_transport",
        TARGET,
        policy(approvals=(approval(no_hidden_dispatch=None),)),
        NOW,
        REVISION,
        DecisionCode.UNCONTROLLED_TRANSPORT,
    ),
)


@pytest.mark.parametrize(
    "case_id,target_value,policy_value,now,expected,code",
    DENIAL_CASES,
    ids=[case[0] for case in DENIAL_CASES],
)
def test_s01_require_admission_raises_for_every_denial(
    case_id, target_value, policy_value, now, expected, code
):
    assert case_id
    spy = DispatchSpy()
    with pytest.raises(FinancialAdmissionDenied) as caught:
        decision = require_admission(
            target_value, policy_value, now=now, expected_revision=expected
        )
        spy.dispatch(target_value, decision)
    denial = caught.value
    assert denial.decision.code is code
    assert denial.decision.financially_eligible is False
    assert denial.decision.max_monetary_charge is None
    assert denial.decision == evaluate(target_value, policy_value, now, expected)
    assert spy.calls == []


@pytest.mark.parametrize(
    "case_id,target_value,policy_value,now,expected,code",
    DENIAL_CASES,
    ids=[case[0] for case in DENIAL_CASES],
)
def test_s02_safe_default_adapter_harness_is_never_entered(
    case_id, target_value, policy_value, now, expected, code
):
    assert case_id
    fallback_calls = []

    def safe_default_adapter(decision):
        fallback_calls.append(decision)

    with pytest.raises(FinancialAdmissionDenied) as caught:
        try:
            require_admission(
                target_value, policy_value, now=now, expected_revision=expected
            )
        except ValueError:
            safe_default_adapter(None)
            raise
    assert not isinstance(caught.value, ValueError)
    assert caught.value.decision.code is code
    assert fallback_calls == []


def test_s03_decision_truthiness_is_a_programming_error():
    eligible = evaluate()
    denied = evaluate(now=2000)
    with pytest.raises(TypeError):
        bool(eligible)
    with pytest.raises(TypeError):
        bool(denied)
    with pytest.raises(TypeError):
        if eligible:
            pass


def test_s04_metadata_and_messages_are_safe():
    invalid = evaluate(target(provider_id=""))
    assert invalid.policy_revision is None
    assert invalid.approval_id is None
    assert invalid.max_monetary_charge is None

    invalid_policy = evaluate(policy_value=None)
    assert invalid_policy.policy_revision is None
    assert invalid_policy.approval_id is None

    expired = evaluate(now=2000)
    assert expired.policy_revision == REVISION
    assert expired.approval_id == APPROVAL_ID
    assert expired.max_monetary_charge is None

    not_found = evaluate(target(model_id="other-model"))
    assert not_found.policy_revision == REVISION
    assert not_found.approval_id is None

    with pytest.raises(FinancialAdmissionDenied) as caught:
        require_admission(
            TARGET,
            policy(revoked_approval_ids=frozenset({APPROVAL_ID})),
            now=NOW,
            expected_revision=REVISION,
        )
    message = str(caught.value)
    assert message == "financial admission denied: approval_revoked"
    assert caught.value.args == (message,)
    combined = message + repr(caught.value)
    for raw in (
        FINGERPRINT,
        TRANSPORT,
        "fixture-provider",
        "fixture-account",
        SOURCE,
        "fixture-evidence",
        APPROVAL_ID,
    ):
        assert raw not in combined


def test_s05_evaluation_is_deterministic_and_inputs_stay_immutable():
    snapshot = copy.deepcopy((TARGET, POLICY, MONEY, BOUNDARY, APPROVAL))
    assert evaluate() == evaluate()
    assert (TARGET, POLICY, MONEY, BOUNDARY, APPROVAL) == snapshot
    with pytest.raises(dataclasses.FrozenInstanceError):
        TARGET.provider_id = "changed"
    with pytest.raises(dataclasses.FrozenInstanceError):
        POLICY.active = False
    with pytest.raises(dataclasses.FrozenInstanceError):
        MONEY.input_charge = decimal.Decimal(1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        evaluate().approval_id = "changed"


CHILD_SOURCE = r'''
import ast
import builtins
import dataclasses
import decimal
import enum
import io
import _io
import logging
import os
import pathlib
import re
import sqlite3
import subprocess
import sys
import time
import typing
import socket

source_path = "/repo/src/financial_safety.py"
source = pathlib.Path(source_path).read_text()
allowed = {"dataclasses", "decimal", "enum", "re", "typing"}
for node in ast.walk(ast.parse(source)):
    if isinstance(node, ast.Import):
        assert all(alias.name in allowed for alias in node.names)
    elif isinstance(node, ast.ImportFrom):
        assert node.level == 0 and node.module in allowed

spy_calls = []
loader_depth = 0
original_open_code = _io.open_code
original_io_open = _io.open
original_import = builtins.__import__
original_print = builtins.print
logger = logging.getLogger("synthetic-spy")


def fail(*args, **kwargs):
    spy_calls.append("blocked")
    raise AssertionError("unexpected application side effect")


def loader_open_code(path):
    # Only the real loader may read source, not module-body open_code calls.
    # Guard.path's filesystem checks still execute for every loader read.
    global loader_depth
    caller = sys._getframe(1)
    if (caller.f_globals.get("__name__") not in {
        "importlib._bootstrap_external", "_frozen_importlib_external",
    } or caller.f_code.co_name != "get_data"):
        return fail()
    loader_depth += 1
    try:
        return original_open_code(path)
    finally:
        loader_depth -= 1


def guarded_metadata(original):
    def guarded(*args, **kwargs):
        if loader_depth:
            return original(*args, **kwargs)
        return fail()
    return guarded


def audited_import(name, globals=None, locals=None, fromlist=(), level=0):
    caller = sys._getframe(1)
    if caller.f_globals.get("__name__") in {
        "financial_safety", "_financial_side_effect_probe",
    } and (level or name not in allowed):
        return fail()
    return original_import(name, globals, locals, fromlist, level)


class ForbiddenEnvironment:
    __getitem__ = __contains__ = __iter__ = __len__ = fail
    get = keys = values = items = copy = fail
    __setitem__ = __delitem__ = fail


# Spies precede the fresh import. No policy function is patched or replaced.
# Only normal import-loader reads remain allowed, under the existing guard.
builtins.__import__ = audited_import
builtins.open = builtins.print = fail
io.open = fail
# CPython open_code delegates dynamically to _io.open. Keep that loader-only
# read under Guard.path; direct _io.open calls still fail outside loader_depth.
_io.open = guarded_metadata(original_io_open)
io.open_code = _io.open_code = loader_open_code
os.environ = ForbiddenEnvironment()
for name in ("getenv", "putenv", "unsetenv", "open", "system", "fork",
             "forkpty", "execv", "execve", "posix_spawn", "posix_spawnp"):
    if hasattr(os, name):
        setattr(os, name, fail)
for name in ("stat", "lstat", "fstat", "readlink", "listdir", "scandir",
             "getcwd"):
    setattr(os, name, guarded_metadata(getattr(os, name)))
for name in ("time", "monotonic", "monotonic_ns", "perf_counter",
             "perf_counter_ns", "process_time", "thread_time", "sleep"):
    setattr(time, name, fail)
for name in ("socket", "socketpair", "create_connection", "getaddrinfo",
             "gethostbyname", "gethostbyaddr", "gethostname"):
    setattr(socket, name, fail)
for name in ("Popen", "run", "call", "check_call", "check_output"):
    setattr(subprocess, name, fail)
for name in ("getLogger", "log", "debug", "info", "warning", "error",
             "exception", "critical", "basicConfig"):
    setattr(logging, name, fail)
for name in ("log", "debug", "info", "warning", "error", "exception",
             "critical", "_log", "handle"):
    setattr(logging.Logger, name, fail)
for name in ("open", "read_text", "read_bytes", "write_text", "write_bytes",
             "stat", "exists", "resolve", "iterdir", "glob", "rglob"):
    setattr(pathlib.Path, name, fail)
sqlite3.connect = fail

before = set(sys.modules)
import financial_safety as fs
assert set(sys.modules) - before == {"financial_safety"}
assert not any(
    name.split(".")[0] in {"proxy_app", "rotator_library"}
    or name.startswith(("src.proxy_app", "src.rotator_library"))
    for name in sys.modules
)
assert spy_calls == []

fingerprint = "a" * 64
transport = "b" * 64
target = fs.FinancialTarget(
    provider_id="fixture-provider",
    model_id="fixture-model",
    account_id="fixture-account",
    authentication=fs.AuthenticationKind.KEYED,
    credential_fingerprint=fingerprint,
    operation=fs.Operation.CHAT_COMPLETION,
    dispatch_profile_id="fixture-profile",
    transport_binding=transport,
)
money = fs.MonetaryBounds(
    currency="USD",
    unit="currency_per_attempt",
    coverage_complete=True,
    input_charge=decimal.Decimal(0),
    output_charge=decimal.Decimal(0),
    cache_read_charge=decimal.Decimal(0),
    cache_write_charge=decimal.Decimal(0),
    request_charge=decimal.Decimal(0),
    tool_charge=decimal.Decimal(0),
    search_charge=decimal.Decimal(0),
    other_charge=decimal.Decimal(0),
)
boundary = fs.NoChargeBoundary(
    exhaustion_zero=True,
    expiry_zero=True,
    no_monetary_funding=True,
    substitution_zero=True,
)
approval = fs.FinancialApproval(
    approval_id="fixture-approval",
    target=target,
    funding=fs.FundingClass.FREE_QUOTA,
    money=money,
    boundary=boundary,
    verified_at=900,
    valid_from=1000,
    valid_until=2000,
    source_id="fixture-authority",
    evidence_ref="fixture-evidence",
    single_target_bound=True,
    no_hidden_dispatch=True,
)
policy = fs.PolicyContext(
    schema_version=1,
    revision="fixture-policy-1",
    active=True,
    approved_sources=frozenset({"fixture-authority"}),
    revoked_approval_ids=frozenset(),
    approvals=(approval,),
)
decision = fs.evaluate_admission(
    target, policy, now=1500, expected_revision="fixture-policy-1"
)
assert decision.financially_eligible is True
assert decision.code is fs.DecisionCode.ELIGIBLE
assert fs.require_admission(
    target, policy, now=1500, expected_revision="fixture-policy-1"
) == decision

for changed in (
    dataclasses.replace(approval, funding=fs.FundingClass.UNKNOWN),
    dataclasses.replace(approval, funding=fs.FundingClass.BILLABLE),
    dataclasses.replace(approval, money=dataclasses.replace(
        money, coverage_complete=None)),
    dataclasses.replace(approval, money=dataclasses.replace(money, input_charge=None)),
    dataclasses.replace(approval, money=dataclasses.replace(
        money, input_charge=decimal.Decimal("0.01"))),
    dataclasses.replace(approval, boundary=dataclasses.replace(
        boundary, exhaustion_zero=None)),
    dataclasses.replace(approval, single_target_bound=None),
):
    current = dataclasses.replace(policy, approvals=(changed,))
    denied = fs.evaluate_admission(
        target, current, now=1500, expected_revision="fixture-policy-1")
    assert denied.financially_eligible is False
    try:
        fs.require_admission(
            target, current, now=1500, expected_revision="fixture-policy-1")
    except fs.FinancialAdmissionDenied as error:
        assert error.decision == denied
    else:
        raise AssertionError("denial was ignored")
assert spy_calls == []

# Also exercise validation, lookup and lifecycle denials while spies are active.
for current_target, current_policy, now, revision in (
    (dataclasses.replace(target, provider_id=""), policy, 1500, "fixture-policy-1"),
    (target, policy, False, "fixture-policy-1"),
    (target, policy, 1500, ""),
    (target, None, 1500, "fixture-policy-1"),
    (target, dataclasses.replace(policy, active=False), 1500, "fixture-policy-1"),
    (target, policy, 1500, "fixture-policy-2"),
    (target, dataclasses.replace(policy, approvals=()), 1500, "fixture-policy-1"),
    (target, dataclasses.replace(policy,
        revoked_approval_ids=frozenset({"fixture-approval"})),
     1500, "fixture-policy-1"),
    (target, policy, 999, "fixture-policy-1"),
    (target, policy, 2000, "fixture-policy-1"),
    (target, dataclasses.replace(policy, approved_sources=frozenset()),
     1500, "fixture-policy-1"),
):
    denied = fs.evaluate_admission(
        current_target, current_policy, now=now, expected_revision=revision)
    assert denied.financially_eligible is False
    try:
        fs.require_admission(
            current_target, current_policy, now=now, expected_revision=revision)
    except fs.FinancialAdmissionDenied as error:
        assert error.decision == denied
    else:
        raise AssertionError("denial was ignored")
assert spy_calls == []

# Prove the spies can actually reject synthetic module-body calls, including
# already-loaded forbidden imports. These are separate test canaries, never a
# replacement/patch of financial_safety. Every operation is blocked BEFORE I/O.
canaries = (
    "open(source_path)", "io.open(source_path)", "_io.open_code(source_path)",
    "pathlib.Path(source_path).read_text()", "os.open(source_path, 0)",
    "os.listdir('/repo')", "os.environ.get('SYNTHETIC')",
    "os.environ['SYNTHETIC']", "os.getenv('SYNTHETIC')",
    "time.time()", "time.monotonic()", "time.perf_counter()",
    "time.process_time()", "time.thread_time()", "socket.socket()",
    "socket.getaddrinfo('synthetic.invalid', 443)",
    "subprocess.Popen(['synthetic-denied'])", "os.system('synthetic-denied')",
    "sqlite3.connect(':memory:')", "logging.getLogger('synthetic')",
    "logging.warning('synthetic')", "logger.warning('synthetic')",
    "print('synthetic')", "__import__('os')",
)
for payload in canaries:
    before_count = len(spy_calls)
    namespace = dict(globals(), __name__="_financial_side_effect_probe")
    try:
        exec(compile(payload, "<side-effect-canary>", "exec"), namespace)
    except AssertionError as error:
        assert str(error) == "unexpected application side effect"
    else:
        raise AssertionError("spy failed to reject canary")
    assert len(spy_calls) == before_count + 1
original_print("s06-ok")
'''


def test_s06_fresh_import_and_evaluation_have_no_side_effects():
    completed = subprocess.run(
        [sys.executable, "-c", CHILD_SOURCE],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "s06-ok"


def test_s07_ancillary_reference_price_does_not_change_eligibility():
    reference_price_per_million = decimal.Decimal("0.0000004")
    assert reference_price_per_million > 0
    assert_eligible(evaluate())


@pytest.mark.parametrize(
    "observation",
    (
        {"allowance": "different"},
        {"allowance": "unknown"},
        {"allowance": "exhausted"},
    ),
    ids=("different", "unknown", "exhausted"),
)
def test_s08_allowance_observations_do_not_change_the_financial_decision(observation):
    assert observation["allowance"]
    assert_eligible(evaluate())


# Supplemental exhaustive field variants. S01/S02/S04/S05 are also enforced
# by evaluate() above for every denied evaluation in these and the original rows.


@pytest.mark.parametrize("field", ("approval_id", "source_id", "evidence_ref"))
@pytest.mark.parametrize(
    "value", [item[1] for item in BAD_IDENTIFIERS] + ["x" * 129],
    ids=[item[0] for item in BAD_IDENTIFIERS] + ["too_long"],
)
def test_i07_each_record_identifier_is_strict(field, value):
    assert_code(
        evaluate(policy_value=policy(approvals=(approval(**{field: value}),))),
        DecisionCode.INVALID_POLICY,
    )


@pytest.mark.parametrize(
    "value", [item[1] for item in BAD_IDENTIFIERS] + ["x" * 129],
    ids=[item[0] for item in BAD_IDENTIFIERS] + ["too_long"],
)
def test_i07_policy_and_expected_revision_identity_are_strict(value):
    assert_code(evaluate(expected=value), DecisionCode.INVALID_INPUT)
    assert_code(evaluate(policy_value=policy(revision=value)),
                DecisionCode.INVALID_POLICY)


@pytest.mark.parametrize("field", ("approved_sources", "revoked_approval_ids"))
@pytest.mark.parametrize(
    "value", [item[1] for item in BAD_IDENTIFIERS] + ["x" * 129],
    ids=[item[0] for item in BAD_IDENTIFIERS] + ["too_long"],
)
def test_p09_each_registry_identifier_element_is_strict(field, value):
    assert_code(evaluate(policy_value=policy(**{field: frozenset({value})})),
                DecisionCode.INVALID_POLICY)


@pytest.mark.parametrize("field", ("approved_sources", "revoked_approval_ids"))
@pytest.mark.parametrize(
    "container", (None, [], set(), (), {}, FrozenSetSubclass()),
    ids=("none", "list", "set", "tuple", "dict", "frozenset_subclass"),
)
def test_p01_registry_containers_require_exact_frozensets(field, container):
    assert_code(evaluate(policy_value=policy(**{field: container})),
                DecisionCode.INVALID_POLICY)


@pytest.mark.parametrize(
    "container", (None, [], set(), frozenset(), {}, TupleSubclass((APPROVAL,))),
    ids=("none", "list", "set", "frozenset", "dict", "tuple_subclass"),
)
def test_p01_approval_container_requires_exact_tuple(container):
    assert_code(evaluate(policy_value=policy(approvals=container)),
                DecisionCode.INVALID_POLICY)


@pytest.mark.parametrize(
    "value", (None, {}, TARGET, MONEY, object()),
    ids=("none", "dict", "wrong_target_dto", "wrong_money_dto", "object"),
)
def test_p09_approval_elements_require_exact_approval_dtos(value):
    assert_code(evaluate(policy_value=policy(approvals=(value,))),
                DecisionCode.INVALID_POLICY)


@pytest.mark.parametrize(
    "kind", ("target", "approval", "money", "boundary", "policy"),
)
def test_p01_each_input_dto_subclass_is_rejected_without_masking(kind):
    originals = {
        "target": (TargetSubclass, TARGET),
        "approval": (ApprovalSubclass, APPROVAL),
        "money": (BoundsSubclass, MONEY),
        "boundary": (BoundarySubclass, BOUNDARY),
        "policy": (PolicySubclass, POLICY),
    }
    cls, original = originals[kind]
    subclass = cls(**{
        field.name: getattr(original, field.name)
        for field in dataclasses.fields(original)
    })
    if kind == "target":
        assert_code(evaluate(subclass), DecisionCode.INVALID_INPUT)
        malformed = approval(target=subclass)
    elif kind == "policy":
        assert_code(evaluate(policy_value=subclass), DecisionCode.INVALID_POLICY)
        return
    elif kind == "approval":
        malformed = subclass
    else:
        malformed = approval(**{kind: subclass})
    # The outer policy and every OTHER component are exact valid runtime types.
    assert_code(evaluate(policy_value=policy(approvals=(malformed,))),
                DecisionCode.INVALID_POLICY)


@pytest.mark.parametrize("kind", ("target", "approval", "money", "boundary"))
@pytest.mark.parametrize("shape", ("dict", "none"))
def test_p01_nested_dtos_cannot_be_raw_dicts_or_missing(kind, shape):
    original = {"target": TARGET, "approval": APPROVAL,
                "money": MONEY, "boundary": BOUNDARY}[kind]
    value = dataclasses.asdict(original) if shape == "dict" else None
    if kind == "target":
        assert_code(evaluate(value), DecisionCode.INVALID_INPUT)
        malformed = approval(target=value)
    elif kind == "approval":
        malformed = value
    else:
        malformed = approval(**{kind: value})
    assert_code(evaluate(policy_value=policy(approvals=(malformed,))),
                DecisionCode.INVALID_POLICY)


@pytest.mark.parametrize(
    "field,value",
    (
        ("authentication", "keyed"), ("authentication", ForeignEnum.KEYED),
        ("authentication", None), ("authentication", AuthenticationKind),
        ("operation", "chat_completion"), ("operation", ForeignEnum.CHAT),
        ("operation", None), ("operation", Operation),
    ),
    ids=("auth_string", "auth_foreign_enum", "auth_none", "auth_class",
         "op_string", "op_foreign_enum", "op_none", "op_class"),
)
def test_p01_target_enums_are_not_coerced(field, value):
    changed = target(**{field: value})
    assert_code(evaluate(changed), DecisionCode.INVALID_INPUT)
    assert_code(evaluate(policy_value=policy(approvals=(approval(target=changed),))),
                DecisionCode.INVALID_POLICY)


@pytest.mark.parametrize(
    "value", ("non_billable", ForeignEnum.NON_BILLABLE, None, FundingClass),
    ids=("string", "foreign_enum", "none", "enum_class"),
)
def test_p01_funding_enum_is_not_coerced(value):
    assert_code(evaluate(policy_value=policy(approvals=(approval(funding=value),))),
                DecisionCode.INVALID_POLICY)


@pytest.mark.parametrize(
    "field", TARGET_IDENTITY_FIELDS + ("credential_fingerprint", "transport_binding"),
)
def test_p01_target_scalar_callbacks_are_never_invoked(field):
    assert_code(evaluate(target(**{field: Coercible()})), DecisionCode.INVALID_INPUT)


@pytest.mark.parametrize("field", MONEY_FIELDS)
def test_p01_money_scalar_callbacks_are_never_invoked(field):
    assert_code(evaluate(policy_value=policy(
        approvals=(approval(money=money(**{field: Coercible()})),),
    )), DecisionCode.INVALID_POLICY)


@pytest.mark.parametrize("field", ("verified_at", "valid_from", "valid_until"))
@pytest.mark.parametrize(
    "value", (-1, True, False, 1500.0, "1500", None, 253402300800,
              IntSubclass(NOW)),
    ids=("negative", "true", "false", "float", "string", "none",
         "out_of_range", "int_subclass"),
)
def test_t02_each_approval_timestamp_has_exact_type_and_range(field, value):
    assert_code(evaluate(policy_value=policy(approvals=(approval(**{field: value}),))),
                DecisionCode.INVALID_POLICY)


@pytest.mark.parametrize(
    "start,end,now", ((0, 1, 0), (253402300798, 253402300799, 253402300798)),
    ids=("lower_epoch_bound", "upper_epoch_bound"),
)
def test_a02_timestamp_extremes_remain_valid(start, end, now):
    current = policy(approvals=(approval(
        verified_at=start, valid_from=start, valid_until=end,
    ),))
    assert_eligible(evaluate(policy_value=current, now=now))


@pytest.mark.parametrize("field", MONEY_FIELDS)
def test_f04_tiny_positive_bounds_are_not_rounded_by_decimal_context(field):
    value = decimal.Decimal("1e-1000")
    with decimal.localcontext() as context:
        context.prec = 1
        context.Emin = context.Emax = 0
        for signal in (decimal.Inexact, decimal.Rounded, decimal.Underflow):
            context.traps[signal] = True
        flags = context.flags.copy()
        assert_code(evaluate(policy_value=policy(
            approvals=(approval(money=money(**{field: value})),),
        )), DecisionCode.NONZERO_MONETARY_BOUND)
        assert context.flags == flags


@pytest.mark.parametrize("field", TARGET_IDENTITY_FIELDS)
def test_a05_namespace_and_maximum_length_ids_are_allowed(field):
    value = "x" * (256 if field == "model_id" else 128)
    exact = target(**{field: value})
    assert_eligible(evaluate(exact, policy(approvals=(approval(target=exact),))))
    if field == "model_id":
        exact = target(model_id="fixture-namespace/model:tag")
        assert_eligible(evaluate(exact, policy(approvals=(approval(target=exact),))))


@pytest.mark.parametrize("field", ("approved_sources", "revoked_approval_ids",
                                  "approvals"))
def test_p09_collection_limit_itself_is_structurally_valid(field):
    if field == "approved_sources":
        value = frozenset({SOURCE} | {f"source-{i}" for i in range(63)})
    elif field == "revoked_approval_ids":
        value = frozenset(f"revoked-{i}" for i in range(1024))
    else:
        value = (APPROVAL,) + tuple(
            approval(approval_id=f"approval-{i}", target=target(model_id=f"model-{i}"))
            for i in range(1023)
        )
    assert_eligible(evaluate(policy_value=policy(**{field: value})))


@pytest.mark.parametrize(
    "location", ("target", "policy", "expected_revision", "money", "evidence"),
)
def test_s04_malformed_sensitive_canaries_are_not_echoed(location):
    canary = "synthetic prompt credential proof private error!"
    if location == "target":
        decision = evaluate(target(account_id=canary))
    elif location == "policy":
        decision = evaluate(policy_value=policy(revision=canary))
    elif location == "expected_revision":
        decision = evaluate(expected=canary)
    else:
        changed = (approval(money=money(input_charge=canary))
                   if location == "money" else approval(evidence_ref=canary))
        decision = evaluate(policy_value=policy(approvals=(changed,)))
    assert_code(decision, DecisionCode.INVALID_INPUT if location in
                ("target", "expected_revision") else DecisionCode.INVALID_POLICY)
    assert canary not in repr(decision)


@pytest.mark.parametrize(
    "instance", (TARGET, MONEY, BOUNDARY, APPROVAL, POLICY,
                 AdmissionDecision(True, DecisionCode.ELIGIBLE, REVISION,
                                   APPROVAL_ID, decimal.Decimal(0))),
    ids=("target", "money", "boundary", "approval", "policy", "decision"),
)
def test_s05_all_public_dtos_are_frozen_slotted_and_have_required_fields(instance):
    cls = type(instance)
    assert cls.__dataclass_params__.frozen
    assert not hasattr(instance, "__dict__")
    values = {field.name: getattr(instance, field.name)
              for field in dataclasses.fields(instance)}
    for field in dataclasses.fields(instance):
        assert field.default is field.default_factory is dataclasses.MISSING
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(instance, field.name, object())
        with pytest.raises(TypeError):
            cls(**{name: value for name, value in values.items() if name != field.name})
    with pytest.raises((AttributeError, TypeError)):
        instance.extra_field = "rejected"


@pytest.mark.parametrize("function", (evaluate_admission, require_admission))
def test_s05_public_signatures_have_no_defaults_or_escape_hatches(function):
    signature = inspect.signature(function)
    parameters = signature.parameters
    assert tuple(parameters) == ("target", "policy", "now", "expected_revision")
    for name, parameter in parameters.items():
        assert parameter.default is inspect.Parameter.empty
        assert parameter.kind is (inspect.Parameter.KEYWORD_ONLY if name in
                                  ("now", "expected_revision") else
                                  inspect.Parameter.POSITIONAL_OR_KEYWORD)
    assert signature.return_annotation is AdmissionDecision
    assert FinancialAdmissionDenied.__bases__ == (Exception,)


@pytest.mark.parametrize(
    "case_id,target_value,policy_value,now,expected,code",
    DENIAL_CASES, ids=[case[0] for case in DENIAL_CASES],
)
def test_s03_every_denial_code_has_fixed_truthiness_rejection(
    case_id, target_value, policy_value, now, expected, code,
):
    assert case_id
    decision = evaluate(target_value, policy_value, now, expected)
    assert_code(decision, code)
    with pytest.raises(TypeError) as caught:
        bool(decision)
    assert str(caught.value) == (
        "AdmissionDecision has no truth value; inspect financially_eligible"
    )


@pytest.mark.parametrize("model_id", URL_MODEL_IDS)
def test_i07_url_model_is_rejected_even_with_matching_zero_money_approval(model_id):
    exact = target(model_id=model_id)
    context = policy(approvals=(approval(target=exact),))
    assert_code(evaluate(exact, context), DecisionCode.INVALID_INPUT)


@pytest.mark.parametrize("model_id", URL_MODEL_IDS)
@pytest.mark.parametrize("position", (0, 1), ids=("before_match", "after_match"))
def test_p08_unrelated_url_record_invalidates_the_whole_context(model_id, position):
    unrelated = approval(approval_id="unrelated", target=target(model_id=model_id))
    records = [APPROVAL]
    records.insert(position, unrelated)
    assert_code(evaluate(policy_value=policy(approvals=tuple(records))),
                DecisionCode.INVALID_POLICY)


@pytest.mark.parametrize(
    "instance", (TARGET, MONEY, BOUNDARY, APPROVAL, POLICY,
                 AdmissionDecision(True, DecisionCode.ELIGIBLE, REVISION,
                                   APPROVAL_ID, decimal.Decimal(0))),
    ids=("target", "money", "boundary", "approval", "policy", "decision"),
)
def test_s05_dto_annotations_match_the_normative_types(instance):
    schemas = {
        FinancialTarget: (
            str, str, str, AuthenticationKind, str | None, Operation, str, str,
        ),
        MonetaryBounds: (str, str, bool | None) + (decimal.Decimal | None,) * 8,
        NoChargeBoundary: (bool | None,) * 4,
        FinancialApproval: (
            str, FinancialTarget, FundingClass, MonetaryBounds, NoChargeBoundary,
            int, int, int, str, str, bool | None, bool | None,
        ),
        PolicyContext: (
            int, str, bool, frozenset[str], frozenset[str],
            tuple[FinancialApproval, ...],
        ),
        AdmissionDecision: (bool, DecisionCode, str | None, str | None,
                            decimal.Decimal | None),
    }
    assert tuple(field.type for field in dataclasses.fields(instance)) == (
        schemas[type(instance)]
    )


@pytest.mark.parametrize("function", (evaluate_admission, require_admission))
def test_s05_api_parameter_annotations_match_the_contract(function):
    parameters = inspect.signature(function).parameters
    assert tuple(parameter.annotation for parameter in parameters.values()) == (
        FinancialTarget, PolicyContext, int, str,
    )
