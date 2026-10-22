"""Upstream provider error text must never reach an HTTP client.

When every fallback candidate fails, ``RouterCore.route_request`` re-raises the
raw provider exception. ``router_integration`` and the chat handler used to copy
``str(e)`` straight into ``HTTPException.detail`` (and into the SSE error
chunk), so upstream response bodies - which can carry internal URLs, provider
error payloads or credential material - were handed to the caller.

These tests pin the boundary: the client-visible detail is metadata only
(exception class + request id), while the full message stays in the server log
so the request id can be correlated with it.
"""

import logging
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proxy_app import main as main_module
from proxy_app.router_integration import RouterIntegration
from proxy_app.safe_errors import safe_error_detail

# Synthetic marker standing in for credential material an upstream error body
# can carry. Deliberately not shaped like a real key so secret scanners stay quiet.
SECRET = "UPSTREAM_SECRET_SENTINEL"
UPSTREAM_TEXT = (
    "litellm.APIError: Status 500 from https://internal-gw.corp/v1/chat "
    f'body: {{"error":"quota exceeded","credential":"{SECRET}"}}'
)


class _ExplodingRouter:
    """Router whose every attempt fails with a secret-bearing message."""

    async def route_request(self, request_data, request_id):
        raise RuntimeError(UPSTREAM_TEXT)


def _integration() -> RouterIntegration:
    """RouterIntegration without its (config/network-heavy) __init__."""
    integ = RouterIntegration.__new__(RouterIntegration)
    integ.router = _ExplodingRouter()
    return integ


@pytest.mark.asyncio
async def test_handle_with_router_detail_is_sanitized():
    integ = _integration()

    with pytest.raises(HTTPException) as exc_info:
        await integ._handle_with_router(
            {"model": "router/coding-smart", "messages": []}, "req_sanitized_1"
        )

    http_exc = exc_info.value
    assert http_exc.status_code == 500
    detail = str(http_exc.detail)
    assert SECRET not in detail
    assert "internal-gw.corp" not in detail
    assert "quota exceeded" not in detail
    # Client still gets something actionable to correlate with server logs.
    assert "req_sanitized_1" in detail
    assert "RuntimeError" in detail


@pytest.mark.asyncio
async def test_handle_with_router_still_logs_full_error(caplog):
    integ = _integration()

    with caplog.at_level(logging.ERROR, logger="proxy_app.router_integration"):
        with pytest.raises(HTTPException):
            await integ._handle_with_router(
                {"model": "router/coding-smart", "messages": []}, "req_logged_1"
            )

    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "req_logged_1" in logged
    assert SECRET in logged, "full upstream error must stay available server-side"


@pytest.mark.asyncio
async def test_chat_completions_wrapper_detail_is_sanitized(monkeypatch):
    integ = _integration()

    async def _boom(request_data, request_id):
        raise RuntimeError(UPSTREAM_TEXT)

    monkeypatch.setattr(integ, "_handle_with_router", _boom)

    with pytest.raises(HTTPException) as exc_info:
        await integ.chat_completions(
            {"model": "router/coding-smart", "messages": []}, raw_request=None
        )

    detail = str(exc_info.value.detail)
    assert exc_info.value.status_code == 500
    assert SECRET not in detail
    assert "internal-gw.corp" not in detail


@pytest.mark.asyncio
async def test_chat_handler_last_resort_is_sanitized(monkeypatch):
    """main.py's own fallback must not echo str(e) either."""

    def _broken_router():
        raise RuntimeError(UPSTREAM_TEXT)

    monkeypatch.setattr(main_module, "get_router", _broken_router)

    request = _FakeRequest({"model": "router/coding-smart", "messages": []})

    with pytest.raises(HTTPException) as exc_info:
        await main_module.chat_completions(request)

    detail = str(exc_info.value.detail)
    assert exc_info.value.status_code == 500
    assert SECRET not in detail
    assert "internal-gw.corp" not in detail


@pytest.mark.asyncio
async def test_stream_error_chunk_is_sanitized(monkeypatch):
    """The SSE error chunk is client-visible too."""

    class _StreamFailingRouter:
        async def handle_chat_completions(self, request_data, raw_request):
            async def _gen():
                raise RuntimeError(UPSTREAM_TEXT)
                yield  # pragma: no cover - makes this an async generator

            return _gen()

    monkeypatch.setattr(main_module, "get_router", lambda: _StreamFailingRouter())

    request = _FakeRequest(
        {"model": "router/coding-smart", "messages": [], "stream": True}
    )
    response = await main_module.chat_completions(request)

    body = ""
    async for chunk in response.body_iterator:
        body += chunk.decode() if isinstance(chunk, bytes) else chunk

    assert "data: [DONE]" in body
    assert SECRET not in body
    assert "internal-gw.corp" not in body
    # The user still sees a meaningful in-band error message.
    assert "Gateway stream error" in body


def test_safe_error_detail_is_metadata_only():
    class Weird(Exception):
        pass

    detail = safe_error_detail(Weird(UPSTREAM_TEXT), request_id="req_meta")
    assert SECRET not in detail
    assert detail == "Router error (Weird); request_id=req_meta"


def test_safe_error_detail_handles_odd_type_names():
    # The exception class name is attacker-visible metadata: anything that is
    # not an identifier character must be dropped, and the result truncated.
    Weird = type("Weird" + "\nINTERNAL", (Exception,), {})
    detail = safe_error_detail(Weird("boom"), request_id="r" * 500)
    assert "\n" not in detail
    assert len(detail) <= 200


class _FakeRequest:
    """Minimal stand-in for the Starlette Request used by the chat handler."""

    def __init__(self, payload):
        self._payload = payload
        self.headers = {}

    async def json(self):
        return self._payload
