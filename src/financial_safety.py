"""Pure financial-admission policy kernel.

This module answers one question only: does the selected physical target have a
proven maximum actual monetary obligation of zero for one physical attempt?

It performs no I/O, reads no configuration, discovers no providers or accounts,
and never calls an application, network, clock or logging API. Approvals are
trusted only when they arrive inside :class:`PolicyContext`; callers cannot pass
an approval object directly.

A decision of ``ELIGIBLE`` is financial eligibility only. Quota admission,
authentication, capability, privacy and transport gates remain separate.
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
import re


class AuthenticationKind(Enum):
    """How the physical target authenticates."""

    KEYED = "keyed"
    ANONYMOUS = "anonymous"


class Operation(Enum):
    """Physical operation the approval must bind."""

    CHAT_COMPLETION = "chat_completion"
    RESPONSE = "response"
    EMBEDDING = "embedding"
    SEARCH_BASIC = "search_basic"
    SEARCH_ADVANCED = "search_advanced"
    SEARCH_RESEARCH = "search_research"
    MODEL_DISCOVERY = "model_discovery"
    TOKEN_COUNT = "token_count"
    CREDENTIAL_REFRESH = "credential_refresh"
    TRANSPORT_AUDIT = "transport_audit"


class FundingClass(Enum):
    """Observed funding arrangement, not a price estimate."""

    NON_BILLABLE = "non_billable"
    FREE_QUOTA = "free_quota"
    PROMOTIONAL_CREDITS = "promotional_credits"
    BILLABLE = "billable"
    UNKNOWN = "unknown"


class DecisionCode(Enum):
    """Deterministic admission outcome."""

    INVALID_INPUT = "invalid_input"
    INVALID_POLICY = "invalid_policy"
    POLICY_INACTIVE = "policy_inactive"
    POLICY_REVISION_MISMATCH = "policy_revision_mismatch"
    APPROVAL_NOT_FOUND = "approval_not_found"
    APPROVAL_REVOKED = "approval_revoked"
    APPROVAL_NOT_YET_VALID = "approval_not_yet_valid"
    APPROVAL_EXPIRED = "approval_expired"
    UNTRUSTED_EVIDENCE = "untrusted_evidence"
    UNKNOWN_FUNDING = "unknown_funding"
    BILLABLE_FUNDING = "billable_funding"
    INCOMPLETE_CHARGE_COVERAGE = "incomplete_charge_coverage"
    UNKNOWN_MONETARY_BOUND = "unknown_monetary_bound"
    NONZERO_MONETARY_BOUND = "nonzero_monetary_bound"
    UNPROVEN_NO_CHARGE_BOUNDARY = "unproven_no_charge_boundary"
    UNCONTROLLED_TRANSPORT = "uncontrolled_transport"
    ELIGIBLE = "eligible"


@dataclass(frozen=True, slots=True)
class FinancialTarget:
    """One exact physical provider/model/account/operation target."""

    provider_id: str
    model_id: str
    account_id: str
    authentication: AuthenticationKind
    credential_fingerprint: str | None
    operation: Operation
    dispatch_profile_id: str
    transport_binding: str


@dataclass(frozen=True, slots=True)
class MonetaryBounds:
    """Upper monetary bounds for one physical attempt over the whole profile."""

    currency: str
    unit: str
    coverage_complete: bool | None
    input_charge: Decimal | None
    output_charge: Decimal | None
    cache_read_charge: Decimal | None
    cache_write_charge: Decimal | None
    request_charge: Decimal | None
    tool_charge: Decimal | None
    search_charge: Decimal | None
    other_charge: Decimal | None


@dataclass(frozen=True, slots=True)
class NoChargeBoundary:
    """Proofs that each exhaustion and expiry path stays non-charging."""

    exhaustion_zero: bool | None
    expiry_zero: bool | None
    no_monetary_funding: bool | None
    substitution_zero: bool | None


@dataclass(frozen=True, slots=True)
class FinancialApproval:
    """Operator-accepted evidence for exactly one target."""

    approval_id: str
    target: FinancialTarget
    funding: FundingClass
    money: MonetaryBounds
    boundary: NoChargeBoundary
    verified_at: int
    valid_from: int
    valid_until: int
    source_id: str
    evidence_ref: str
    single_target_bound: bool | None
    no_hidden_dispatch: bool | None


@dataclass(frozen=True, slots=True)
class PolicyContext:
    """Immutable snapshot of the trusted approval and evidence registries."""

    schema_version: int
    revision: str
    active: bool
    approved_sources: frozenset[str]
    revoked_approval_ids: frozenset[str]
    approvals: tuple[FinancialApproval, ...]


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    """Financial-admission result.

    A denial never claims an observed charge of zero, so
    ``max_monetary_charge`` is ``None`` on every denial.
    """

    financially_eligible: bool
    code: DecisionCode
    policy_revision: str | None
    approval_id: str | None
    max_monetary_charge: Decimal | None

    def __bool__(self) -> bool:
        raise TypeError(
            "AdmissionDecision has no truth value; inspect financially_eligible"
        )


class FinancialAdmissionDenied(Exception):
    """Raised by :func:`require_admission` for every denial."""

    def __init__(self, decision: AdmissionDecision) -> None:
        self.decision = decision
        super().__init__("financial admission denied: " + decision.code.value)


_SCHEMA_VERSION = 1
_MAX_EPOCH = 253402300799
_MAX_SOURCES = 64
_MAX_REVOKED = 1024
_MAX_APPROVALS = 1024
_ZERO = Decimal(0)
_NOT_APPLICABLE = "__not_applicable__"

_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_MODEL_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}")
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")

_CONTROL_OPERATIONS = frozenset(
    (
        Operation.MODEL_DISCOVERY,
        Operation.CREDENTIAL_REFRESH,
        Operation.TRANSPORT_AUDIT,
    )
)

_MONETARY_FIELDS = (
    "input_charge",
    "output_charge",
    "cache_read_charge",
    "cache_write_charge",
    "request_charge",
    "tool_charge",
    "search_charge",
    "other_charge",
)

_BOUNDARY_FIELDS = (
    "exhaustion_zero",
    "expiry_zero",
    "no_monetary_funding",
    "substitution_zero",
)


def _is_identifier(value: object) -> bool:
    return type(value) is str and _IDENTIFIER_RE.fullmatch(value) is not None


def _is_digest(value: object) -> bool:
    return type(value) is str and _DIGEST_RE.fullmatch(value) is not None


def _is_epoch(value: object) -> bool:
    return type(value) is int and 0 <= value <= _MAX_EPOCH


def _is_bool_or_none(value: object) -> bool:
    return value is None or type(value) is bool


def _is_money(value: object) -> bool:
    if value is None:
        return True
    if type(value) is not Decimal:
        return False
    if not value.is_finite():
        return False
    return value >= _ZERO


def _is_target(value: object) -> bool:
    if type(value) is not FinancialTarget:
        return False
    if not _is_identifier(value.provider_id):
        return False
    if not _is_identifier(value.account_id):
        return False
    if not _is_identifier(value.dispatch_profile_id):
        return False
    if not _is_digest(value.transport_binding):
        return False
    if type(value.authentication) is not AuthenticationKind:
        return False
    if type(value.operation) is not Operation:
        return False
    model_id = value.model_id
    if type(model_id) is not str:
        return False
    if model_id == _NOT_APPLICABLE:
        if value.operation not in _CONTROL_OPERATIONS:
            return False
    # The ID regex also admits scheme:// spellings. Those are destinations,
    # not opaque physical model identities; the contract forbids them separately.
    elif "://" in model_id or _MODEL_ID_RE.fullmatch(model_id) is None:
        return False
    if value.authentication is AuthenticationKind.KEYED:
        if not _is_digest(value.credential_fingerprint):
            return False
    else:
        if value.credential_fingerprint is not None:
            return False
    return True


def _target_key(value: FinancialTarget) -> tuple[object, ...]:
    return (
        value.provider_id,
        value.model_id,
        value.account_id,
        value.authentication,
        value.credential_fingerprint,
        value.operation,
        value.dispatch_profile_id,
        value.transport_binding,
    )


def _is_money_bounds(value: object) -> bool:
    if type(value) is not MonetaryBounds:
        return False
    if type(value.currency) is not str or value.currency != "USD":
        return False
    if type(value.unit) is not str or value.unit != "currency_per_attempt":
        return False
    if not _is_bool_or_none(value.coverage_complete):
        return False
    for name in _MONETARY_FIELDS:
        if not _is_money(getattr(value, name)):
            return False
    return True


def _is_boundary(value: object) -> bool:
    if type(value) is not NoChargeBoundary:
        return False
    for name in _BOUNDARY_FIELDS:
        if not _is_bool_or_none(getattr(value, name)):
            return False
    return True


def _is_approval(value: object) -> bool:
    if type(value) is not FinancialApproval:
        return False
    if not _is_identifier(value.approval_id):
        return False
    if not _is_target(value.target):
        return False
    if type(value.funding) is not FundingClass:
        return False
    if not _is_money_bounds(value.money):
        return False
    if not _is_boundary(value.boundary):
        return False
    if not _is_epoch(value.verified_at):
        return False
    if not _is_epoch(value.valid_from):
        return False
    if not _is_epoch(value.valid_until):
        return False
    if not value.verified_at <= value.valid_from < value.valid_until:
        return False
    if not _is_identifier(value.source_id):
        return False
    if not _is_identifier(value.evidence_ref):
        return False
    if not _is_bool_or_none(value.single_target_bound):
        return False
    if not _is_bool_or_none(value.no_hidden_dispatch):
        return False
    return True


def _is_policy(value: object) -> bool:
    if type(value) is not PolicyContext:
        return False
    if type(value.schema_version) is not int:
        return False
    if value.schema_version != _SCHEMA_VERSION:
        return False
    if not _is_identifier(value.revision):
        return False
    if type(value.active) is not bool:
        return False

    sources = value.approved_sources
    if type(sources) is not frozenset:
        return False
    if len(sources) > _MAX_SOURCES:
        return False
    for source in sources:
        if not _is_identifier(source):
            return False

    revoked = value.revoked_approval_ids
    if type(revoked) is not frozenset:
        return False
    if len(revoked) > _MAX_REVOKED:
        return False
    for revoked_id in revoked:
        if not _is_identifier(revoked_id):
            return False

    approvals = value.approvals
    if type(approvals) is not tuple:
        return False
    if len(approvals) > _MAX_APPROVALS:
        return False
    seen_ids: set[str] = set()
    seen_targets: set[tuple[object, ...]] = set()
    for approval in approvals:
        if not _is_approval(approval):
            return False
        if approval.approval_id in seen_ids:
            return False
        seen_ids.add(approval.approval_id)
        target_key = _target_key(approval.target)
        if target_key in seen_targets:
            return False
        seen_targets.add(target_key)
    return True


def _deny(
    code: DecisionCode,
    revision: str | None = None,
    approval_id: str | None = None,
) -> AdmissionDecision:
    return AdmissionDecision(False, code, revision, approval_id, None)


def evaluate_admission(
    target: FinancialTarget,
    policy: PolicyContext,
    *,
    now: int,
    expected_revision: str,
) -> AdmissionDecision:
    """Evaluate financial eligibility for one exact physical target."""

    if not _is_target(target):
        return _deny(DecisionCode.INVALID_INPUT)
    if not _is_epoch(now):
        return _deny(DecisionCode.INVALID_INPUT)
    if not _is_identifier(expected_revision):
        return _deny(DecisionCode.INVALID_INPUT)
    if not _is_policy(policy):
        return _deny(DecisionCode.INVALID_POLICY)

    if policy.active is not True:
        return _deny(DecisionCode.POLICY_INACTIVE, policy.revision)
    if policy.revision != expected_revision:
        return _deny(DecisionCode.POLICY_REVISION_MISMATCH, policy.revision)

    wanted = _target_key(target)
    matched: FinancialApproval | None = None
    for approval in policy.approvals:
        if _target_key(approval.target) == wanted:
            matched = approval
            break

    if matched is None:
        return _deny(DecisionCode.APPROVAL_NOT_FOUND, policy.revision)

    revision = policy.revision
    approval_id = matched.approval_id

    if approval_id in policy.revoked_approval_ids:
        return _deny(DecisionCode.APPROVAL_REVOKED, revision, approval_id)
    if now < matched.valid_from:
        return _deny(DecisionCode.APPROVAL_NOT_YET_VALID, revision, approval_id)
    if now >= matched.valid_until:
        return _deny(DecisionCode.APPROVAL_EXPIRED, revision, approval_id)
    if matched.source_id not in policy.approved_sources:
        return _deny(DecisionCode.UNTRUSTED_EVIDENCE, revision, approval_id)

    if matched.funding is FundingClass.UNKNOWN:
        return _deny(DecisionCode.UNKNOWN_FUNDING, revision, approval_id)
    if matched.funding is FundingClass.BILLABLE:
        return _deny(DecisionCode.BILLABLE_FUNDING, revision, approval_id)

    if matched.money.coverage_complete is not True:
        return _deny(DecisionCode.INCOMPLETE_CHARGE_COVERAGE, revision, approval_id)

    for name in _MONETARY_FIELDS:
        if getattr(matched.money, name) is None:
            return _deny(DecisionCode.UNKNOWN_MONETARY_BOUND, revision, approval_id)
    for name in _MONETARY_FIELDS:
        if getattr(matched.money, name) != _ZERO:
            return _deny(DecisionCode.NONZERO_MONETARY_BOUND, revision, approval_id)

    for name in _BOUNDARY_FIELDS:
        if getattr(matched.boundary, name) is not True:
            return _deny(
                DecisionCode.UNPROVEN_NO_CHARGE_BOUNDARY, revision, approval_id
            )

    if matched.single_target_bound is not True:
        return _deny(DecisionCode.UNCONTROLLED_TRANSPORT, revision, approval_id)
    if matched.no_hidden_dispatch is not True:
        return _deny(DecisionCode.UNCONTROLLED_TRANSPORT, revision, approval_id)

    return AdmissionDecision(True, DecisionCode.ELIGIBLE, revision, approval_id, _ZERO)


def require_admission(
    target: FinancialTarget,
    policy: PolicyContext,
    *,
    now: int,
    expected_revision: str,
) -> AdmissionDecision:
    """Return the eligible decision or raise :class:`FinancialAdmissionDenied`."""

    decision = evaluate_admission(
        target, policy, now=now, expected_revision=expected_revision
    )
    if not decision.financially_eligible:
        raise FinancialAdmissionDenied(decision)
    return decision
