import pytest

from router_core import Capability, Deployment, HealthRegistry, RequestFeatures, classify, estimate_context_tokens, normalize_phase, select, select_with_stickiness


def deployments():
    return [
        Deployment("general", Capability.FAST_GENERAL, 4096),
        Deployment("tool", Capability.FAST_TOOL, 4096, tools=True),
        Deployment("debug", Capability.CODE_DEBUG, 8192),
        Deployment("json", Capability.FAST_GENERAL, 4096, structured_output=True),
    ]


def test_classification_uses_phase_and_tools():
    assert classify(RequestFeatures("run the tests", phase="verify")) == Capability.CODE_VERIFY
    assert classify(RequestFeatures("inspect files", tools=True)) == Capability.FAST_TOOL
    assert classify(RequestFeatures("make the change", phase="implement")) == Capability.CODE_IMPLEMENT


def test_tool_request_filters_non_tool_deployments():
    assert select(RequestFeatures("inspect files", tools=True), deployments()).deployment_id == "tool"


def test_structured_output_is_hard_requirement():
    assert select(RequestFeatures("answer", structured_output=True), deployments()).deployment_id == "json"


def test_context_limit_is_hard_requirement():
    with pytest.raises(LookupError):
        select(RequestFeatures("answer", context_tokens=5000), deployments())


def test_selection_prefers_expected_fast_success():
    options = [
        Deployment("slow-reliable", Capability.FAST_GENERAL, 4096, latency_ms=100, success_rate=1.0),
        Deployment("fast-fragile", Capability.FAST_GENERAL, 4096, latency_ms=20, success_rate=0.1),
    ]
    assert select(RequestFeatures("answer"), options).deployment_id == "slow-reliable"


def test_quota_is_a_soft_score_penalty_not_a_hard_filter():
    options = [
        Deployment("exhausted", Capability.FAST_GENERAL, 4096, latency_ms=1, success_rate=1, quota_remaining=0),
        Deployment("available", Capability.FAST_GENERAL, 4096, latency_ms=20, success_rate=1, quota_remaining=10),
    ]
    assert select(RequestFeatures("answer"), options).deployment_id == "available"


def test_operational_cooldown_excludes_failed_deployment():
    health = HealthRegistry(cooldown_seconds=10)
    health.mark_failure("general", now=100)
    assert not health.is_healthy("general", now=105)
    assert health.is_healthy("general", now=110)


def test_stickiness_keeps_preferred_eligible_deployment():
    options = [
        Deployment("first", Capability.FAST_GENERAL, 4096, latency_ms=100),
        Deployment("second", Capability.FAST_GENERAL, 4096, latency_ms=10),
    ]
    assert select_with_stickiness(RequestFeatures("answer"), options, "first").deployment_id == "first"


def test_phase_normalization_is_bounded():
    assert normalize_phase("verification") == "verify"
    assert normalize_phase("DEEP-DEBUG") == "deep_debug"
    assert normalize_phase("arbitrary-user-text") == ""


def test_context_estimate_is_conservative_and_nonzero():
    assert estimate_context_tokens([]) == 1
    assert estimate_context_tokens([{"content": "12345"}]) == 2
