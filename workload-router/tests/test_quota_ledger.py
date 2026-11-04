from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from quota_ledger import QuotaLedger, Route


def route(**changes):
    base = Route("r", "p", "m", "key-label", "sc.private", 1,
                 quota_type="rate_limited", input_price=0, output_price=0,
                 context_window=32000, output_limit=4096, tool_call=True,
                 stream=True, limits={"rpm": 1, "rpd": 10, "tpm": 1000, "tpd": 10000})
    return replace(base, **changes)


def reserve(ledger, candidate, now=1000):
    return ledger.reserve(candidate, 100, 100, profile="auto-chat", step="chat", now=now)


def test_all_axes_settlement_and_no_started_release():
    ledger = QuotaLedger()
    candidate = route()
    ledger.configure(candidate)
    identity = reserve(ledger, candidate)
    assert identity and reserve(ledger, candidate) is None
    ledger.start(identity, now=1000)
    with pytest.raises(ValueError, match="started"):
        ledger.cancel(identity)
    ledger.finish(identity, candidate, success=True, usage={"prompt_tokens": 50, "completion_tokens": 20},
                  latency_ms=200, quality=True, now=1001)
    assert ledger.decision(identity)["usage_estimated"] == 0
    assert ledger.capacity(candidate, 1, 1, now=1061).feasible
    assert ledger.evidence(candidate, "chat")[:2] == (3, 2)
    with pytest.raises(ValueError):
        ledger.finish(identity, candidate, success=True, latency_ms=1)


def test_atomic_cross_connection_and_shared_provider_pool(tmp_path):
    database = tmp_path / "ledger.db"
    first, second = QuotaLedger(database), QuotaLedger(database)
    a = route(limits={"rpm": 10}, provider_limits={"rpm": 1})
    b = replace(a, route_id="b", model_id="other", alias="sc.other", channel_id=2)
    first.configure(a)
    first.configure(b)
    with ThreadPoolExecutor(2) as executor:
        results = list(executor.map(lambda item: reserve(*item), [(first, a), (second, b)]))
    assert sum(identity is not None for identity in results) == 1
    assert database.stat().st_mode & 0o777 == 0o600
    first.close()
    second.close()


def test_rolling_minute_and_daily_midnight():
    ledger = QuotaLedger()
    candidate = route(limits={"rpm": 1, "rpd": 1})
    ledger.configure(candidate)
    identity = reserve(ledger, candidate, now=86400)
    ledger.start(identity, now=86400)
    ledger.finish(identity, candidate, success=True, usage={"prompt_tokens": 0, "completion_tokens": 0},
                  latency_ms=1, now=86401)
    assert not ledger.capacity(candidate, 0, 0, now=86459).feasible
    assert ledger.capacity(candidate, 0, 0, now=86461).reason == "exhausted_rpd"
    assert ledger.capacity(candidate, 0, 0, now=172800).feasible


def test_crashed_started_reservation_stays_charged():
    ledger = QuotaLedger()
    candidate = route()
    ledger.configure(candidate)
    identity = reserve(ledger, candidate)
    ledger.start(identity, now=1000)
    assert ledger.capacity(candidate, 1, 1, now=100000).reason == "exhausted_rpm"


def test_cancel_before_network_and_unknown_usage_is_conservative():
    ledger = QuotaLedger()
    candidate = route()
    ledger.configure(candidate)
    identity = reserve(ledger, candidate)
    ledger.cancel(identity)
    identity = reserve(ledger, candidate)
    ledger.start(identity, now=1000)
    ledger.finish(identity, candidate, success=False, latency_ms=1, error_class="rate_limit", now=1001)
    decision = ledger.decision(identity)
    assert decision["input_tokens"] == decision["output_tokens"] == 100
    assert decision["usage_estimated"] == 1
    assert ledger.capacity(candidate, 1, 1, now=1002).reason == "exhausted_rpm"


def test_learning_only_tightens_and_auth_is_key_wide():
    ledger = QuotaLedger()
    candidate = route(limits={"rpm": 10})
    ledger.configure(candidate)
    ledger.learn_limits(candidate, {"rpm": 2, "untrusted": 1})
    ledger.learn_limits(candidate, {"rpm": 20})
    ledger.configure(candidate)
    assert ledger.db.execute("SELECT learned FROM quota_limits WHERE window_type='rpm'").fetchone()[0] == 2
    ledger.cooldown(candidate, "rate_limit", retry_after=120, now=1000)
    assert ledger.capacity(candidate, 1, 1, now=1100).retry_after == 20
    other = replace(candidate, route_id="other", model_id="other", alias="sc.other", channel_id=2)
    ledger.configure(other)
    assert ledger.capacity(other, 1, 1, now=1100).feasible
    ledger.cooldown(candidate, "provider_rejection", status_code=401, now=1100)
    assert not ledger.capacity(other, 1, 1, now=1101).feasible


def test_unknown_metadata_and_credit_budget():
    ledger = QuotaLedger()
    for candidate, reason in [(route(input_price=None), "unknown_pricing"),
                              (route(quota_type="unknown"), "unknown_quota_type"),
                              (route(limits={}), "unknown_limits"),
                              (route(quota_type="credit_based"), "unknown_credit_balance"),
                              (route(quota_type="credit_based", credit_balance=100,
                                     limits={"rpm": 10}), "unknown_credit_price")]:
        assert ledger.capacity(candidate, 1, 1).reason == reason
    candidate = route(quota_type="credit_based", credit_balance=0.001, input_price=10, output_price=10)
    ledger.configure(candidate)
    assert reserve(ledger, candidate) is None


def test_free_classes_and_zero_customer_price_for_credits():
    ledger = QuotaLedger()
    unlimited = route(quota_type="unlimited", limits={})
    limited = route(route_id="limited", alias="sc.limited", channel_id=2,
                    quota_type="rate_limited", limits={"rpm": 5})
    credits = route(route_id="credits", alias="sc.credits", channel_id=3,
                    quota_type="credit_based", credit_balance=100,
                    credit_input_price=1_000_000, credit_output_price=1_000_000,
                    limits={"rpm": 10})
    for candidate in (unlimited, limited, credits):
        ledger.configure(candidate)
    assert unlimited.free_class == "free_unlimited"
    assert limited.free_class == "free_limited"
    assert credits.free_class == "free_credits"
    assert route(quota_type="unlimited", limits={"rpm": 5}).free_class == "free_limited"
    assert credits.money(10, 10) == 0
    assert credits.credit_money(10, 10) == 20
    assert ledger.capacity(credits, 10, 10).feasible
    assert ledger.estimate_cost("sc.credits", 10, 10)["cost"] == 0
    assert ledger.estimate_cost("sc.credits", 10, 10)["source"] == "free_tier"


@pytest.mark.parametrize("changes", [dict(stream=1), dict(input_price="0"), dict(limits={"rpm": float("nan")}),
                                  dict(key_id="*"), dict(alias="a,b"), dict(prior_success=0)])
def test_invalid_route_metadata(changes):
    with pytest.raises(ValueError):
        route(**changes)
