"""M5: serving end to end.

Covers the parts that can be proven without a live upstream:
  * the X-Gateway-Decision header format, and that it is ASCII/parseable
  * the decision record is stored and drained per request
  * an upstream that dies mid-stream does not duplicate content and does not hang
  * a pre-content failure still falls back to the next provider
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from proxy_app.routing.decision import (  # noqa: E402
    HEADER_NAME,
    Decision,
    headers_for,
)
from proxy_app.router_core import ErrorCategory, ProviderCandidate, RouterCore  # noqa: E402


# --------------------------------------------------------------------------- #
# decision header
# --------------------------------------------------------------------------- #


def test_decision_header_is_parseable_and_ascii():
    decision = Decision(
        alias="auto",
        provider="groq",
        model="llama-3.3-70b",
        tier=0,
        attempt=0,
        fallbacks=2,
        wait_ms=150,
        reason="fallback",
    )
    header = decision.to_header()
    header.encode("ascii")  # must not raise
    assert "alias=auto" in header
    assert "provider=groq" in header
    assert "tier=0" in header
    assert "fallbacks=2" in header
    assert "wait_ms=150" in header
    assert "reason=fallback" in header
    # separator-free so intermediaries cannot mangle it
    assert " " not in header


def test_decision_header_never_injects_separators():
    decision = Decision(alias="a;b", provider="p q", model="m\nx", tier=None)
    header = decision.to_header()
    assert header.count(";") == 7, header
    assert "\n" not in header
    assert "tier=-1" in header


def test_headers_for_none_is_empty():
    assert headers_for(None) == {}
    assert HEADER_NAME in headers_for(Decision())


# --------------------------------------------------------------------------- #
# decision lifecycle
# --------------------------------------------------------------------------- #


def test_decision_recorded_and_drained():
    router = _bare_router()
    router.record_decision("req_1", provider="groq", model="m", tier=0, reason="ok")
    decision = router.get_decision("req_1")
    assert decision is not None
    assert decision.provider == "groq"
    assert decision.reason == "ok"
    assert router.get_decision("req_1") is None, "decision map should drain"


def test_decision_map_is_bounded():
    router = _bare_router()
    for i in range(900):
        router.record_decision(f"req_{i}", provider="groq", model="m")
    assert len(router._decisions) <= 512


# --------------------------------------------------------------------------- #
# streaming fallback
# --------------------------------------------------------------------------- #


def _bare_router() -> RouterCore:
    """A RouterCore with only the collaborators the stream path touches."""
    router = RouterCore.__new__(RouterCore)
    router._decisions = {}
    router.virtual_models = {}
    router.config = {"providers": {}}
    router._free_policy_cache = None
    return router


def _candidate(provider: str, model: str) -> ProviderCandidate:
    return ProviderCandidate(provider=provider, model=model)


def _install_fakes(router: RouterCore, streams: dict[str, list]) -> None:
    """Fake executor: each provider yields its scripted items, then maybe raises."""

    async def fake_stream(candidate, request, request_id):
        for item in streams[candidate.provider]:
            if isinstance(item, Exception):
                raise item
            yield item

    async def fake_classify(error):
        return ErrorCategory.TRANSIENT, None

    async def fake_cooldown(*args, **kwargs):
        return None

    class _NoopLedger:
        def record_429(self, *a, **k):
            return None

        def record_ok(self, *a, **k):
            return None

        def observe_headers(self, *a, **k):
            return False

        def decide(self, *a, **k):
            class _D:
                should_fallback = False
                reason = "ok"

            return _D()

    router._execute_single_candidate_stream = fake_stream  # type: ignore[assignment]
    router._classify_error = fake_classify  # type: ignore[assignment]
    router._apply_error_cooldown = fake_cooldown  # type: ignore[assignment]
    router._ledger = lambda: _NoopLedger()  # type: ignore[assignment]
    router._tier_of = lambda provider: 0  # type: ignore[assignment]


def _chunk(text: str) -> dict:
    return {"choices": [{"delta": {"content": text}}]}


def test_mid_stream_death_does_not_duplicate_or_retry():
    """Content already sent must not be replayed from the next provider."""
    router = _bare_router()
    boom = RuntimeError("upstream connection reset")
    _install_fakes(
        router,
        {
            "first": [_chunk("Hello "), _chunk("wor"), boom],
            "second": [_chunk("Hello "), _chunk("world")],
        },
    )

    async def run() -> list[dict]:
        out = []
        async for item in router._stream_with_fallback(
            [_candidate("first", "m1"), _candidate("second", "m2")],
            {"model": "auto", "stream": True},
            "req_stream_1",
        ):
            out.append(item)
        return out

    chunks = asyncio.run(asyncio.wait_for(run(), timeout=10))

    texts = [c["choices"][0]["delta"].get("content", "") for c in chunks]
    joined = "".join(texts)
    assert joined == "Hello wor", f"unexpected content: {joined!r}"
    assert joined.count("Hello") == 1, "content was duplicated by a mid-stream retry"
    decision = router.get_decision("req_stream_1")
    assert decision is not None and decision.reason == "all_failed"


def test_pre_content_failure_still_falls_back():
    router = _bare_router()
    _install_fakes(
        router,
        {
            "first": [RuntimeError("connect failed")],
            "second": [_chunk("Hello "), _chunk("world")],
        },
    )

    async def run() -> list[dict]:
        out = []
        async for item in router._stream_with_fallback(
            [_candidate("first", "m1"), _candidate("second", "m2")],
            {"model": "auto", "stream": True},
            "req_stream_2",
        ):
            out.append(item)
        return out

    chunks = asyncio.run(asyncio.wait_for(run(), timeout=10))
    joined = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert joined == "Hello world"


def test_all_providers_failing_before_content_raises_without_yielding():
    """Nothing was sent, so raising is correct; the SSE layer turns it into a
    terminal error chunk. What matters is that nothing partial is emitted."""
    router = _bare_router()
    _install_fakes(
        router,
        {"first": [RuntimeError("a")], "second": [RuntimeError("b")]},
    )

    async def run() -> list[dict]:
        out: list[dict] = []
        async for item in router._stream_with_fallback(
            [_candidate("first", "m1"), _candidate("second", "m2")],
            {"model": "auto", "stream": True},
            "req_stream_3",
        ):
            out.append(item)
        return out

    with pytest.raises(RuntimeError):
        asyncio.run(asyncio.wait_for(run(), timeout=10))
