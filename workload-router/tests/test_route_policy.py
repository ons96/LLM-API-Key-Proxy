from dataclasses import replace
import math
import random

import pytest

from quota_ledger import QuotaLedger
from route_policy import Work, Weights, compact_json_tools, free_only_audit, input_token_bound, quota_lambda_for, rank_routes, score_route
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


def test_declared_free_classes_rank_by_scarcity_without_customer_cost():
    ledger = QuotaLedger()
    unlimited = route(route_id="unlimited", alias="sc.unlimited", channel_id=1,
                      quota_type="unlimited", limits={}, latency_ms=1000)
    credits = route(route_id="credits", alias="sc.credits", channel_id=2,
                    quota_type="credit_based", credit_balance=100,
                    credit_input_price=1_000_000, credit_output_price=1_000_000,
                    limits={"rpm": 10}, latency_ms=100)
    for candidate in (unlimited, credits):
        ledger.configure(candidate)
    work = Work("auto-chat", "chat", 10, 10)
    ranked = rank_routes([credits, unlimited], work, ledger, explore=False)
    assert [item.route.free_class for item in ranked] == ["free_unlimited", "free_credits"]
    assert ranked[1].effective_money > ranked[0].effective_money
    assert credits.money(10, 10) == 0
    limited = replace(unlimited, route_id="limited", alias="sc.limited", channel_id=3,
                      quota_type="unlimited", limits={"rpm": 1})
    ledger.configure(limited)
    limited_score = score_route(limited, work, ledger)
    assert limited.free_class == "free_unlimited"
    assert limited_score.effective_money == pytest.approx(0.01)


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


def test_cost_vector_is_split_and_only_blended_for_the_score():
    ledger = QuotaLedger()
    credits = route(quota_type="credit_based", credit_balance=100, input_price=1_000_000,
                    output_price=1_000_000, shadow_price=2, limits={"rpm": 10})
    ledger.configure(credits)
    work = Work("auto-coding", "implement", 10, 10, tools=True)
    scored = score_route(credits, work, ledger)
    # Free credits spend no dollars, so their whole cost is the opportunity
    # value of the allowance they deplete.
    assert scored.cost_money == 0
    assert scored.cost_opportunity == pytest.approx(40 + 0.01 * 0.2)
    assert scored.effective_money == pytest.approx(40 + 0.01 * 0.2)
    assert scored.effective_money == pytest.approx(scored.cost_money + scored.cost_opportunity)

    paid = replace(credits, route_id="paid", provider_id="p2", model_id="m2",
                   alias="sc.paid", channel_id=2, quota_type="paid",
                   credit_balance=None, credit_input_price=None,
                   credit_output_price=None)
    ledger.configure(paid)
    cash = score_route(paid, work, ledger, free_only=False)
    assert cash.cost_money == pytest.approx(20)
    assert cash.cost_opportunity == pytest.approx(0.01 * 0.1)
    assert cash.effective_money == pytest.approx(cash.cost_money + cash.cost_opportunity)
    assert cash.score == pytest.approx(
        (cash.effective_money + cash.seconds * 0.003 + cash.failure_probability * 0.03)
        / (1 - cash.failure_probability))
    # The same paid route stays unreachable under the default free-only policy.
    assert math.isinf(score_route(paid, work, ledger).score)


def test_lambda_t_is_one_explicit_exchange_rate_for_both_profiles():
    ledger = QuotaLedger()
    candidate = route(quota_type="unlimited", limits={}, latency_ms=1000)
    ledger.configure(candidate)
    coding = Work("auto-coding", "implement", 10, 10)
    chat = Work("auto-chat", "chat", 10, 10)
    flat = Weights(escalation_cost=0)
    # Unset, revision 2's profile defaults still apply, so nothing moves.
    assert score_route(candidate, coding, ledger, weights=flat).score == pytest.approx(0.003 / 0.5)
    assert score_route(candidate, chat, ledger, weights=flat).score == pytest.approx(0.001 / 0.5)
    # One owner-set value overrides both profiles: lambda_T is a single knob.
    owner = Weights(escalation_cost=0, time_value=0.05)
    assert score_route(candidate, coding, ledger, weights=owner).score == pytest.approx(0.05 / 0.5)
    assert score_route(candidate, chat, ledger, weights=owner).score == pytest.approx(0.05 / 0.5)
    with pytest.raises(ValueError):
        Weights(time_value=-1)
    with pytest.raises(ValueError):
        Weights(time_value=float("inf"))


def test_quota_price_is_derived_from_the_time_exchange_rate():
    ledger = QuotaLedger()
    limited = route(quota_type="unlimited", limits={"rpm": 1}, latency_ms=1000)
    ledger.configure(limited)
    coding = Work("auto-coding", "implement", 10, 10)
    chat = Work("auto-chat", "chat", 10, 10)
    # One whole allowance still costs 0.01 on both profiles, but it is now
    # derived from lambda_T instead of being a second, unrelated constant.
    assert score_route(limited, coding, ledger).cost_opportunity == pytest.approx(0.01)
    assert score_route(limited, chat, ledger).cost_opportunity == pytest.approx(0.01)
    assert quota_lambda_for(Weights(), "auto-coding") == pytest.approx(0.003 * 10 / 3)
    assert quota_lambda_for(Weights(), "auto-chat") == pytest.approx(0.001 * 10)

    # Raising lambda_T makes the router faster-preferring and more willing to
    # spend quota in the same proportion, which is why it is one number.
    fast_clock = Weights(time_value=0.006)
    assert quota_lambda_for(fast_clock, "auto-coding") == pytest.approx(0.02)
    assert score_route(limited, coding, ledger,
                       weights=fast_clock).cost_opportunity == pytest.approx(0.02)

    # `allowance_seconds` is the plain knob: seconds of waiting one whole
    # allowance is worth. Zero means quota is free to burn.
    assert score_route(limited, coding, ledger,
                       weights=Weights(allowance_seconds=10 / 6)).cost_opportunity == pytest.approx(0.005)
    assert score_route(limited, coding, ledger,
                       weights=Weights(allowance_seconds=0)).cost_opportunity == 0
    assert score_route(limited, coding, ledger,
                       weights=Weights(quota_lambda=0.05)).cost_opportunity == pytest.approx(0.05)
    with pytest.raises(ValueError):
        Weights(allowance_seconds=-1)


def test_free_only_audit_reports_priced_routes_instead_of_raising():
    free = route(quota_type="unlimited", limits={})
    priced = replace(free, route_id="paid", alias="sc.paid", channel_id=2,
                     quota_type="paid", input_price=1, output_price=2)
    assert free_only_audit([free]) == {"free_only": True, "priced_routes": [], "excluded": 0}
    assert free_only_audit([free, priced]) == {"free_only": True,
                                               "priced_routes": ["paid"], "excluded": 1}
    assert free_only_audit([free, priced], False) == {"free_only": False,
                                                      "priced_routes": [], "excluded": 0}
