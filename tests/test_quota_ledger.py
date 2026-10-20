"""M3: quota ledger.

Proves the four M3 acceptance checks:
  * header parsing for the providers that actually send headers
  * fault injection: short 429 -> bounded wait -> success
  * fault injection: daily exhaustion -> immediate fallback -> no attempt until
    the reset time
  * exhausted state survives a process restart
  * added latency p50 under 2 ms
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from proxy_app.routing.quota_ledger import (  # noqa: E402
    FALLBACK,
    PROCEED,
    WAIT,
    QuotaLedger,
    parse_rate_limit_headers,
)


@pytest.fixture
def ledger(tmp_path: Path) -> QuotaLedger:
    db = tmp_path / "ledger.db"
    led = QuotaLedger(db, short_wait_ceiling_s=60.0, max_total_wait_s=45.0)
    yield led
    led.close()


# --------------------------------------------------------------------------- #
# header parsing
# --------------------------------------------------------------------------- #


def test_parses_openai_style_headers():
    parsed = parse_rate_limit_headers(
        {
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "42",
            "x-ratelimit-reset-requests": "30",
        }
    )
    assert parsed["limit"] == 100
    assert parsed["remaining"] == 42
    assert parsed["window"] == "minute"
    assert parsed["reset_at"] > time.time()


def test_parses_vendor_spellings():
    parsed = parse_rate_limit_headers(
        {"RateLimit-Limit": "20", "RateLimit-Remaining": "0", "RateLimit-Reset": "5"}
    )
    assert parsed["limit"] == 20
    assert parsed["remaining"] == 0


def test_parses_retry_after_ms():
    parsed = parse_rate_limit_headers({"retry-after-ms": "1500"})
    assert parsed["retry_after_s"] == pytest.approx(1.5)


def test_parses_anthropic_reset_as_timestamp():
    future = time.time() + 120
    parsed = parse_rate_limit_headers(
        {"anthropic-ratelimit-requests-reset": str(future)}
    )
    assert parsed["reset_at"] == pytest.approx(future, rel=1e-6)


def test_no_headers_is_no_information():
    assert parse_rate_limit_headers({}) == {}
    assert parse_rate_limit_headers(None) == {}
    assert parse_rate_limit_headers({"content-type": "application/json"}) == {}


def test_day_window_from_long_reset():
    parsed = parse_rate_limit_headers({"retry-after": "3600"})
    assert parsed["window"] == "hour"
    parsed2 = parse_rate_limit_headers({"retry-after": str(12 * 3600)})
    assert parsed2["window"] == "day"


def test_observe_headers_marks_exhausted(ledger: QuotaLedger):
    learned = ledger.observe_headers(
        "groq",
        "llama-3.3-70b",
        {"x-ratelimit-limit-requests": "100", "x-ratelimit-remaining-requests": "0"},
    )
    assert learned is True
    state = ledger.snapshot("groq", "llama-3.3-70b")
    assert state["status"] == "exhausted"


# --------------------------------------------------------------------------- #
# fault injection 1: short 429 -> bounded wait -> success
# --------------------------------------------------------------------------- #


def test_short_429_waits_then_succeeds(ledger: QuotaLedger):
    now = 1_000_000.0
    ledger.record_429("openai", "gpt-5-mini", retry_after_s=20, now=now)

    decision = ledger.decide("openai", "gpt-5-mini", now=now + 1)
    assert decision.action == WAIT
    assert 0 < decision.wait_s <= 45.0
    assert decision.window in {"minute"}

    # Simulate the bounded wait expiring and the retry succeeding.
    retry_at = now + 1 + decision.wait_s
    ledger.record_ok("openai", "gpt-5-mini", now=retry_at)

    after = ledger.decide("openai", "gpt-5-mini", now=retry_at + 0.1)
    assert after.action == PROCEED


def test_wait_is_capped(ledger: QuotaLedger):
    now = 1_000_000.0
    ledger.record_429("groq", "m", retry_after_s=59, now=now)
    decision = ledger.decide("groq", "m", now=now)
    assert decision.action == WAIT
    assert decision.wait_s <= ledger.max_total_wait_s


# --------------------------------------------------------------------------- #
# fault injection 2: daily exhaustion -> immediate fallback, no retry
# --------------------------------------------------------------------------- #


def test_daily_exhaustion_falls_back_immediately(ledger: QuotaLedger):
    now = 1_000_000.0
    # A day-long reset must not become a wait.
    ledger.record_429("septorlabs", "qwen3-coder", retry_after_s=8 * 3600, now=now)
    decision = ledger.decide("septorlabs", "qwen3-coder", now=now + 1)
    assert decision.action == FALLBACK
    assert decision.wait_s == 0.0
    assert "exhausted" in decision.reason


def test_exhausted_provider_is_not_retried_until_reset(ledger: QuotaLedger):
    now = 1_000_000.0
    ledger.record_429("septorlabs", "qwen3-coder", retry_after_s=8 * 3600, now=now)

    candidates = [("septorlabs", "qwen3-coder"), ("groq", "llama-3.3-70b")]
    for offset in (0, 60, 3600, 4 * 3600):
        kept = ledger.filter_candidates(candidates, now=now + offset)
        assert ("septorlabs", "qwen3-coder") not in kept, (
            f"exhausted candidate retried at +{offset}s"
        )

    assert "septorlabs" in ledger.exhausted_providers(now=now + 3600)

    # Lazy reset: once the window has rolled over the key is usable again.
    after_reset = now + 8 * 3600 + 1
    assert ("septorlabs", "qwen3-coder") in ledger.filter_candidates(
        candidates, now=after_reset
    )
    assert "septorlabs" not in ledger.exhausted_providers(now=after_reset)


def test_unknown_reset_fails_towards_fallback(ledger: QuotaLedger):
    now = 1_000_000.0
    ledger.record_429("mystery", "m", retry_after_s=None, now=now)
    decision = ledger.decide("mystery", "m", now=now + 1)
    assert decision.action == FALLBACK, "an unknown reset must not become a wait"


# --------------------------------------------------------------------------- #
# one-time credits never wait
# --------------------------------------------------------------------------- #


def test_one_time_credit_never_waits(ledger: QuotaLedger):
    now = 1_000_000.0
    ledger.record_429("together", "qwen-2.5-72b", retry_after_s=10, now=now)
    waiting = ledger.decide("together", "qwen-2.5-72b", now=now + 1)
    assert waiting.action == WAIT  # a renewing provider would wait here

    one_time = ledger.decide("together", "qwen-2.5-72b", now=now + 1, is_one_time=True)
    assert one_time.action == FALLBACK
    assert one_time.wait_s == 0.0


# --------------------------------------------------------------------------- #
# restart persistence
# --------------------------------------------------------------------------- #


def test_exhausted_state_survives_restart(tmp_path: Path):
    db = tmp_path / "ledger.db"
    now = 1_000_000.0

    first = QuotaLedger(db)
    first.record_429("septorlabs", "qwen3-coder", retry_after_s=8 * 3600, now=now)
    assert first.decide("septorlabs", "qwen3-coder", now=now + 1).action == FALLBACK
    first.close()  # process would exit here

    second = QuotaLedger(db)  # a fresh process, same file
    decision = second.decide("septorlabs", "qwen3-coder", now=now + 60)
    assert decision.action == FALLBACK, "exhausted state did not survive the restart"
    assert second.exhausted_providers(now=now + 60) == {"septorlabs"}
    second.close()


def test_header_state_survives_restart(tmp_path: Path):
    db = tmp_path / "ledger.db"
    first = QuotaLedger(db)
    first.observe_headers(
        "groq",
        "llama-3.3-70b",
        {"x-ratelimit-limit-requests": "100", "x-ratelimit-remaining-requests": "3"},
    )
    first.close()

    second = QuotaLedger(db)
    state = second.snapshot("groq", "llama-3.3-70b")
    assert state["windows"]["minute"]["limit"] == 100
    assert state["windows"]["minute"]["remaining"] == 3
    second.close()


# --------------------------------------------------------------------------- #
# degrade, never fail a request
# --------------------------------------------------------------------------- #


def test_corrupt_ledger_degrades_to_proceed(tmp_path: Path):
    bad = tmp_path / "broken.db"
    bad.write_bytes(b"this is not a sqlite database")
    led = QuotaLedger(bad)
    decision = led.decide("groq", "m")
    assert decision.action == PROCEED
    assert decision.reason == "ledger disabled"
    led.close()


def test_disabled_ledger_is_a_noop(tmp_path: Path):
    led = QuotaLedger(tmp_path / "l.db", enabled=False)
    led.record_429("groq", "m", retry_after_s=10)
    assert led.decide("groq", "m").action == PROCEED


# --------------------------------------------------------------------------- #
# added latency
# --------------------------------------------------------------------------- #


def test_added_latency_p50_under_2ms(ledger: QuotaLedger):
    """Measure the decision path on a warm ledger."""
    now = 1_000_000.0
    ledger.record_429("groq", "m", retry_after_s=20, now=now)

    samples: list[float] = []
    for _ in range(2000):
        start = time.perf_counter()
        ledger.decide("groq", "m", now=now + 1)
        samples.append((time.perf_counter() - start) * 1000.0)

    p50 = statistics.median(samples)
    p95 = sorted(samples)[int(0.95 * len(samples)) - 1]
    print(f"\nledger decide(): p50={p50:.3f}ms p95={p95:.3f}ms n={len(samples)}")
    assert p50 < 2.0, f"ledger added {p50:.3f} ms at p50, budget is 2 ms"
