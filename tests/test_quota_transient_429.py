"""Transient throttling must never be charged as spent allowance (#378).

A 429 means "slow down". Crediting it as quota exhaustion marks a finite-credit
key burned, so a busy provider silently costs the user a one-time credit.
"""

import pytest

from src.proxy_app.rate_limiter import REASON_RATE_LIMIT, REASON_USAGE_CAP
from src.proxy_app.router_core import (
    ErrorCategory,
    RouterCore,
    is_quota_exhaustion,
    quota_exhaustion_text,
)


class HttpError(Exception):
    """Stand-in for provider errors, which carry the status on the exception."""

    def __init__(self, message, status_code=None, body=None, code=None):
        super().__init__(message)
        self.status_code = status_code
        if body is not None:
            self.body = body
        if code is not None:
            self.code = code


@pytest.mark.parametrize(
    "message",
    [
        "429 Too many requests",
        "Rate limit reached for requests",
        "Concurrency limit reached for this model",
        "Capacity temporarily exceeded, please try again",
        "1302 concurrency exceeded",
        "Upstream is busy, slow down",
    ],
)
def test_throttling_is_never_exhaustion(message):
    assert quota_exhaustion_text(message) is False
    assert is_quota_exhaustion(HttpError(message, status_code=429)) is False


@pytest.mark.parametrize(
    "message",
    [
        "You exceeded your current quota, please check your plan and billing details",
        "Daily limit reached",
        "insufficient_quota: your credit balance is too low",
        "billing hard limit reached for this key",
        "Credit allowance depleted for the day",
    ],
)
def test_real_exhaustion_is_still_detected(message):
    assert quota_exhaustion_text(message) is True
    assert is_quota_exhaustion(HttpError(message, status_code=429)) is True


def test_structured_body_outranks_the_message():
    err = HttpError(
        "upstream failed",
        status_code=429,
        body={"error": {"code": "insufficient_quota"}},
    )
    assert is_quota_exhaustion(err) is True


def test_ambiguous_error_makes_no_claim():
    assert quota_exhaustion_text("upstream returned 500") is False
    assert quota_exhaustion_text("") is False
    assert is_quota_exhaustion(HttpError("something went wrong")) is False


@pytest.mark.asyncio
async def test_cooldown_reason_follows_the_classification(monkeypatch):
    """A throttled key gets a cooldown, not a usage-cap block."""
    router = RouterCore()
    seen = []

    async def fake_hit(provider, model, api_key=None, retry_after=None, reason=None):
        seen.append(reason)

    monkeypatch.setattr(router.rate_limiter, "record_rate_limit_hit", fake_hit)
    monkeypatch.setattr(router, "_has_multiple_api_keys", lambda provider: False)

    await router._apply_error_cooldown(
        "groq", "llama-3.3-70b", ErrorCategory.RATE_LIMIT, 5, "429 too many requests"
    )
    assert seen == [REASON_RATE_LIMIT]

    await router._apply_error_cooldown(
        "groq", "llama-3.3-70b", ErrorCategory.RATE_LIMIT, 5,
        "You exceeded your current quota",
    )
    assert seen[-1] == REASON_USAGE_CAP
