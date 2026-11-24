"""Stage 2 acceptance tests for the pure financial-admission kernel.

Every fixture here is synthetic. No provider, credential, quota, transport or
network resource is ever created or contacted.
"""

from __future__ import annotations

import copy
import dataclasses
import decimal
import subprocess
import sys

import pytest

from financial_safety import (
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
    ("model_discovery", Operation.MODEL_DISCOVERY, "__not_applicable__"),
    ("token_count", Operation.TOKEN_COUNT, "fixture-model"),
    ("credential_refresh", Operation.CREDENTIAL_REFRESH, "__not_applicable__"),
    ("transport_audit", Operation.TRANSPORT_AUDIT, "__not_applicable__"),
)


class DecimalSubclass(decimal.Decimal):
    pass


class StrSubclass(str):
    pass


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


def evaluate(target_value=TARGET, policy_value=POLICY, now=NOW, expected=REVISION):
    return evaluate_admission(
        target_value, policy_value, now=now, expected_revision=expected
    )


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
    exact = target(operation=operation, model_id=model_id)
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
    ("cache_read_charge", "tool_charge", "search_charge", "other_charge"),
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
    (
        (
            "positive_charge",
            {"money": money(input_charge=decimal.Decimal("0.01"))},
            DecisionCode.NONZERO_MONETARY_BOUND,
        ),
        (
            "exhaustion_unproven",
            {"boundary": boundary(exhaustion_zero=False)},
            DecisionCode.UNPROVEN_NO_CHARGE_BOUNDARY,
        ),
        (
            "expiry_unproven",
            {"boundary": boundary(expiry_zero=None)},
            DecisionCode.UNPROVEN_NO_CHARGE_BOUNDARY,
        ),
    ),
    ids=("positive_charge", "exhaustion_unproven", "expiry_unproven"),
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
def test_f09_apparently_free_anonymous_relay_is_still_uncontrolled(field):
    anonymous = target(
        authentication=AuthenticationKind.ANONYMOUS, credential_fingerprint=None
    )
    decision = evaluate(
        anonymous,
        policy(
            approvals=(
                approval(
                    target=anonymous,
                    funding=FundingClass.FREE_QUOTA,
                    **{field: False},
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
        {
            "authentication": AuthenticationKind.ANONYMOUS,
            "credential_fingerprint": None,
        },
        {"operation": Operation.RESPONSE},
        {"dispatch_profile_id": "other-profile"},
        {"transport_binding": "c" * 64},
    ),
    ids=("provider", "model", "account", "authentication", "operation", "profile",
         "transport"),
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


TARGET_IDENTITY_OVERRIDES = (
    ("provider_empty", {"provider_id": ""}),
    ("provider_wildcard", {"provider_id": "fixture-*"}),
    ("provider_whitespace", {"provider_id": "fixture provider"}),
    ("provider_non_ascii", {"provider_id": "fixtur\u00e9"}),
    ("provider_too_long", {"provider_id": "p" * 129}),
    ("fingerprint_uppercase", {"credential_fingerprint": "A" * 64}),
    ("fingerprint_short", {"credential_fingerprint": "a" * 63}),
    ("transport_uppercase", {"transport_binding": "B" * 64}),
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


def test_i08_identifier_case_change_is_not_found():
    assert_code(
        evaluate(target(provider_id="Fixture-Provider")),
        DecisionCode.APPROVAL_NOT_FOUND,
    )


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
            PolicySubclass(
                schema_version=1,
                revision=REVISION,
                active=True,
                approved_sources=POLICY.approved_sources,
                revoked_approval_ids=POLICY.revoked_approval_ids,
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
    "second",
    (
        approval(approval_id="fixture-approval-2"),
        approval(approval_id="fixture-approval-2", valid_until=1100),
        approval(approval_id="fixture-approval-2", valid_until=1100),
    ),
    ids=("duplicate_target", "duplicate_expired_target", "duplicate_revoked_target"),
)
def test_p07_duplicate_targets_are_invalid_even_when_inactive(second):
    revoked = frozenset({APPROVAL_ID, "fixture-approval-2"})
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
    (-1, True, 1500.0, "1500", None, 253402300800),
    ids=("negative", "bool", "float", "string", "none", "out_of_range"),
)
def test_t02_malformed_now_is_invalid_input(now):
    assert_code(evaluate(now=now), DecisionCode.INVALID_INPUT)


@pytest.mark.parametrize(
    "overrides",
    (
        {"verified_at": 1001},
        {"valid_from": 2000, "valid_until": 2000},
        {"valid_from": 2001},
    ),
    ids=("verified_after_from", "empty_window", "from_after_until"),
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
        {"unit": "currency_per_token"},
        {"unit": None},
    ),
    ids=("currency_unsupported", "currency_missing", "unit_per_token", "unit_missing"),
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
import dataclasses
import decimal
import enum
import re
import sys

before = set(sys.modules)
import financial_safety as fs
added = set(sys.modules) - before
assert added == {"financial_safety"}, sorted(added)
assert not any(
    name == "proxy_app"
    or name.startswith("proxy_app.")
    or name == "src.proxy_app"
    or name.startswith("src.proxy_app.")
    or name == "rotator_library"
    or name.startswith("rotator_library.")
    for name in sys.modules
)


def fail(*args, **kwargs):
    raise AssertionError("unexpected side effect")


import builtins
import logging
import os
import socket
import subprocess
import time

builtins.open = fail
os.getenv = fail
time.time = fail
socket.socket = fail
subprocess.Popen = fail
logging.getLogger = fail

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
print("s06-ok")
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
