import pytest

from router_core import CacheHint, Capability, Deployment, HealthRegistry, RequestFeatures, classify, estimate_context_tokens, normalize_phase, route_plan, select, select_with_stickiness, warm_cache_wait_seconds


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


def test_success_clears_operational_cooldown():
    health = HealthRegistry(cooldown_seconds=10)
    health.mark_failure("general", now=100, error_class="timeout")
    assert not health.is_healthy("general", now=101)
    health.mark_success("general")
    assert health.is_healthy("general", now=101)
    assert health.error_class("general") is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_failure_hints_do_not_poison_cooldown(value):
    health = HealthRegistry(cooldown_seconds=10)
    health.mark_failure(
        "general",
        now=100,
        cooldown_seconds=value,
        retry_after_seconds=value,
        quota_reset_at=value,
    )
    assert health.retry_after("general", now=100) == 10
    assert health.is_healthy("general", now=110)


def test_stickiness_keeps_preferred_eligible_deployment():
    options = [
        Deployment("first", Capability.FAST_GENERAL, 4096, latency_ms=100),
        Deployment("second", Capability.FAST_GENERAL, 4096, latency_ms=10),
    ]
    assert select_with_stickiness(RequestFeatures("answer"), options, "first").deployment_id == "first"


def test_new_requests_start_at_chain_top_before_latency_score():
    options = [
        Deployment("top", Capability.FAST_GENERAL, 4096, latency_ms=100, success_rate=1, group="chat-fast", chain_priority=1),
        Deployment("fallback", Capability.FAST_GENERAL, 4096, latency_ms=1, success_rate=1, group="chat-fast", chain_priority=2),
    ]
    plan = route_plan(RequestFeatures("answer", model_group="chat-fast"), options)
    assert [item.deployment_id for item in plan.candidates] == ["top", "fallback"]
    assert plan.reason == "chain_top"


def test_implicit_classification_uses_metadata_group_when_capability_group_is_absent():
    options = [
        Deployment("chat-top", Capability.FAST_GENERAL, 4096, group="chat-fast", chain_priority=1),
        Deployment("chat-fallback", Capability.FAST_GENERAL, 4096, group="chat-fast", chain_priority=2),
    ]
    plan = route_plan(RequestFeatures("answer"), options)
    assert plan.candidates[0].deployment_id == "chat-top"


def test_explicit_group_does_not_cross_route_to_another_group():
    options = [Deployment("chat", Capability.FAST_GENERAL, 4096, group="chat-fast")]
    with pytest.raises(LookupError):
        route_plan(RequestFeatures("inspect", model_group="chat-fast", tools=True), options)


def test_continuation_stays_on_eligible_preferred_deployment():
    options = [
        Deployment("top", Capability.FAST_GENERAL, 4096, group="chat-fast", chain_priority=1),
        Deployment("fallback", Capability.FAST_GENERAL, 4096, group="chat-fast", chain_priority=2),
    ]
    features = RequestFeatures("answer", model_group="chat-fast", continuation=True)
    plan = route_plan(features, options, preferred_id="fallback")
    assert plan.candidates[0].deployment_id == "fallback"
    assert plan.reason == "continuation_sticky"


def test_warm_cache_overrides_chain_only_at_break_even():
    options = [
        Deployment("top", Capability.FAST_GENERAL, 4096, group="chat-fast", chain_priority=1),
        Deployment("warm", Capability.FAST_GENERAL, 4096, group="chat-fast", chain_priority=3),
    ]
    hint = CacheHint("warm", cached_tokens=100000, observed_at=1000, ttl_seconds=600)
    plan = route_plan(RequestFeatures("answer", model_group="chat-fast"), options, cache_hints={"warm": hint}, now=1100)
    assert plan.candidates[0].deployment_id == "warm"
    assert plan.reason == "warm_cache_override"
    assert plan.cache_saved_ms > 0


def test_stale_cache_does_not_change_chain_order():
    options = [
        Deployment("top", Capability.FAST_GENERAL, 4096, group="chat-fast", chain_priority=1),
        Deployment("warm", Capability.FAST_GENERAL, 4096, group="chat-fast", chain_priority=2),
    ]
    hint = CacheHint("warm", cached_tokens=5000, observed_at=100, ttl_seconds=10)
    plan = route_plan(RequestFeatures("answer", model_group="chat-fast"), options, cache_hints={"warm": hint}, now=1000)
    assert plan.candidates[0].deployment_id == "top"
    assert plan.reason == "chain_top"


def test_warm_cache_wait_is_limited_to_short_rate_limits():
    hint = CacheHint("warm", cached_tokens=100000, observed_at=1000, ttl_seconds=600)
    assert warm_cache_wait_seconds("rate_limit", 1.0, hint, now=1100) == 1.0
    assert warm_cache_wait_seconds("timeout", 1.0, hint, now=1100) == 0.0
    assert warm_cache_wait_seconds("rate_limit", 5.0, hint, now=1100) == 0.0


def test_phase_normalization_is_bounded():
    assert normalize_phase("verification") == "verify"
    assert normalize_phase("DEEP-DEBUG") == "deep_debug"
    assert normalize_phase("arbitrary-user-text") == ""


def test_context_estimate_is_conservative_and_nonzero():
    assert estimate_context_tokens([]) == 1
    assert estimate_context_tokens([{"content": "12345"}]) == 2
