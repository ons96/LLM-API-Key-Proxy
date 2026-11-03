from dataclasses import replace
import math
import random

import pytest

from quota_ledger import QuotaLedger
from route_policy import Work, Weights, compact_json_tools, input_token_bound, rank_routes, score_route
from test_quota_ledger import route


def test_exact_formula_free_time_and_paid_exclusion():
    ledger = QuotaLedger()
    candidate = route(quota_type="unlimited", latency_ms=1000, limits={})
    ledger.configure(candidate)
    work = Work("auto-chat", "chat", 10, 10)
    weights = Weights(chat_time_value=2, escalation_cost=3)
    score = score_route(candidate, work, ledger, weights=weights)
    assert score.score == (0 + 1 * 2 + 0.5 * 3) / 0.5
    assert math.isinf(score_route(replace(candidate, quota_type="paid"), work, ledger).score)
    faster = replace(candidate, route_id="fast", alias="sc.fast", channel_id=2, latency_ms=100)
    assert rank_routes([candidate, faster], work, ledger, explore=False)[0].route == faster


def test_credit_shadow_price_pressure_and_hard_gates():
    ledger = QuotaLedger()
    candidate = route(quota_type="credit_based", credit_balance=100, input_price=1_000_000,
                      output_price=1_000_000, shadow_price=2, limits={"rpm": 10})
    ledger.configure(candidate)
    work = Work("auto-coding", "implement", 10, 10, tools=True)
    score = score_route(candidate, work, ledger)
    assert score.effective_money == pytest.approx(40 + 0.01 * 0.2)
    assert math.isinf(score_route(replace(candidate, tool_call=None), work, ledger).score)
    assert math.isinf(score_route(replace(candidate, context_window=15), work, ledger).score)


def test_bandit_top_n_only_and_chat_has_no_affinity():
    ledger = QuotaLedger()
    routes = [route(route_id=str(i), alias=f"sc.{i}", channel_id=i + 1, quota_type="unlimited",
                    limits={}, latency_ms=100 + i * 1000) for i in range(5)]
    for candidate in routes:
        ledger.configure(candidate)
    work = Work("auto-chat", "chat", 10, 10)
    winners = {rank_routes(routes, work, ledger, weights=Weights(top_n=2), rng=random.Random(i))[0].route.route_id for i in range(30)}
    assert winners <= {"0", "1"}
    assert score_route(routes[0], work, ledger, preferred_id="1", lost_cached_tokens=10000).seconds == 0.1


def test_cache_switch_cost_cannot_bypass_cooldown():
    ledger = QuotaLedger()
    candidate = route(quota_type="unlimited", limits={})
    ledger.configure(candidate)
    work = Work("auto-coding", "debug", 100, 10)
    score = score_route(candidate, work, ledger, preferred_id="other", lost_cached_tokens=1000)
    assert score.seconds == pytest.approx(2 + (25 + 15) / 1000)
    ledger.cooldown(candidate, "rate_limit")
    assert not rank_routes([candidate], work, ledger, preferred_id=candidate.route_id)


def test_graded_telemetry_changes_fallback_priority():
    ledger = QuotaLedger()
    a = route(route_id="a", alias="sc.a", quota_type="unlimited", limits={}, latency_ms=1000)
    b = replace(a, route_id="b", alias="sc.b", channel_id=2, latency_ms=1500)
    for candidate in (a, b):
        ledger.configure(candidate)
    work = Work("auto-coding", "verify", 10, 10)
    assert rank_routes([a, b], work, ledger, explore=False)[0].route == a
    for candidate, passed in ((a, False), (b, True)):
        identity = ledger.reserve(candidate, 10, 10, profile=work.profile, step=work.step)
        ledger.start(identity)
        ledger.finish(identity, candidate, success=True, latency_ms=candidate.latency_ms, quality=passed)
    assert rank_routes([a, b], work, ledger, explore=False)[0].route == b


def test_semantic_compaction_keeps_strings_numbers_and_duplicate_keys():
    original = '{ "x" : 1234567890.123456789012345, "path" : "a  b\\\\c", "neg" : -0.0 }'
    duplicate = '{ "same" : 1, "same" : 2 }'
    payload = {"messages": [{"role": "user", "content": original},
                            {"role": "tool", "content": original, "tool_call_id": "x"},
                            {"role": "tool", "content": duplicate, "tool_call_id": "y"}],
               "tools": [{"type": "function", "function": {"name": "x", "parameters": {"type": "object"}}}]}
    result, baseline, compressed = compact_json_tools(payload, True)
    assert compressed < baseline
    assert result["messages"][0] == payload["messages"][0]
    assert result["tools"] == payload["tools"]
    assert result["messages"][2]["content"] == duplicate
    assert '1234567890.123456789012345' in result["messages"][1]["content"]
    assert '"neg":-0.0' in result["messages"][1]["content"]
    assert payload["messages"][1]["content"] == original
    assert input_token_bound(payload) == baseline


def test_multimodal_and_nonfinite_requests_rejected():
    with pytest.raises(ValueError):
        input_token_bound({"messages": [{"role": "user", "content": [{"type": "image_url"}]}]})
    with pytest.raises(ValueError):
        input_token_bound({"messages": [{"role": "user", "content": "hello"}], "temperature": float("nan")})
