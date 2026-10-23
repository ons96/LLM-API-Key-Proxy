"""Same leak as test_error_detail_leak.py, remaining sites.

PR #327 sanitized the live chat path (router_integration + main.py). The same
`str(e)`-into-client-text pattern also existed in `enhanced_proxy.py` (its own
chat handler, SSE wrappers, /v1/search, router diagnostics) and in
`provider_adapter.py` (G4F completions, three places). These tests pin those
boundaries too: client-visible text is metadata only (exception class +
request id), the full message stays in the log.
"""

import sys
import types
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proxy_app import enhanced_proxy

SECRET = "UPSTREAM_SECRET_SENTINEL"
UPSTREAM_TEXT = (
    "litellm.APIError: Status 500 from https://internal-gw.corp/v1/chat "
    f'body: {{"error":"quota exceeded","credential":"{SECRET}"}}'
)


class _Boom(Exception):
    pass


async def _boom(*args, **kwargs):
    raise _Boom(UPSTREAM_TEXT)


class _FakeFailingIntegration:
    """Router integration whose every chat call dies with a secret-bearing error."""

    def __init__(self):
        self.calls = 0

    async def chat_completions(self, *args, **kwargs):
        self.calls += 1
        raise _Boom(UPSTREAM_TEXT)


class _FakeStreamIntegration:
    """Router integration that hands back a stream which dies on first read."""

    def __init__(self):
        self.calls = 0

    async def chat_completions(self, *args, **kwargs):
        self.calls += 1

        async def _gen():
            raise _Boom(UPSTREAM_TEXT)
            yield  # pragma: no cover - makes this an async generator

        return _gen()


@pytest.fixture(autouse=True)
def open_auth(monkeypatch):
    """Disable the optional proxy API key so requests reach the handlers.

    The tests exercise handler error paths, not authentication; with
    PROXY_API_KEY set in the environment the chat endpoint answers 401 before
    the code under test runs.
    """
    from proxy_app import main as main_module

    monkeypatch.setattr(main_module, "PROXY_API_KEY", None)


@pytest.fixture(scope="module")
def client():
    """HTTP client without app lifespan.

    `with TestClient(app)` runs the startup/shutdown hooks, and shutdown here
    waits on the provider status tracker, which makes live calls to upstream
    providers. Those calls fail in CI and in an offline checkout, turning a
    passing test into a teardown error. None of these tests need startup state:
    the handlers under test are patched directly.
    """
    from fastapi.testclient import TestClient

    return TestClient(enhanced_proxy.app)


def _assert_clean(text: str) -> None:
    assert SECRET not in text
    assert "internal-gw.corp" not in text
    assert "quota exceeded" not in text


class _FakeRequest:
    """Minimal stand-in for the Starlette request the handlers read."""

    def __init__(self, payload):
        self._payload = payload
        self.headers = {}

    async def json(self):
        return self._payload


async def _drain(async_iter) -> str:
    out = ""
    async for chunk in async_iter:
        out += chunk.decode() if isinstance(chunk, bytes) else chunk
    return out


def test_chat_handler_detail_is_sanitized(monkeypatch):
    # Note: this handler is called directly, because the shared app registers
    # main.py's /v1/chat/completions first and this route is shadowed at the
    # HTTP layer. The sanitization still matters if the route is ever exposed.
    fake = _FakeFailingIntegration()
    monkeypatch.setattr(enhanced_proxy, "_router_integration", fake)

    import asyncio

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            enhanced_proxy.enhanced_chat_completions(
                _FakeRequest(
                    {
                        "model": "router/coding-smart",
                        "messages": [{"role": "user", "content": "hi"}],
                    }
                ),
                auth=None,
            )
        )

    detail = str(exc_info.value.detail)
    assert exc_info.value.status_code == 500
    _assert_clean(detail)
    # Client still gets a correlate-able request id.
    assert "request_id=" in detail
    # The failing router was actually used (no silent fallback to a real one).
    assert fake.calls == 1


def test_stream_error_chunk_is_sanitized(monkeypatch):
    fake = _FakeStreamIntegration()
    monkeypatch.setattr(enhanced_proxy, "_router_integration", fake)

    import asyncio

    async def _call():
        response = await enhanced_proxy.enhanced_chat_completions(
            _FakeRequest(
                {
                    "model": "router/coding-smart",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                }
            ),
            auth=None,
        )
        return await _drain(response.body_iterator)

    body = asyncio.run(_call())

    _assert_clean(body)
    assert fake.calls == 1, "the fake stream router must have been used"


def test_search_error_detail_is_sanitized():
    # Body parsing fails with a secret-bearing error, which is the except path
    # that used to echo str(e) straight into the 500 detail.
    class _ExplodingRequest:
        headers = {}

        async def json(self):
            raise _Boom(UPSTREAM_TEXT)

    import asyncio

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(enhanced_proxy.perform_search(_ExplodingRequest()))

    assert exc_info.value.status_code == 500
    _assert_clean(str(exc_info.value.detail))


def test_router_diagnostics_do_not_leak(client, monkeypatch):
    monkeypatch.setattr(enhanced_proxy, "_router_integration", None)
    monkeypatch.setattr(enhanced_proxy, "get_rotating_client", _boom)

    response = client.get("/v1/router/status")

    assert response.status_code == 200
    _assert_clean(response.text)


def _fake_g4f_module(monkeypatch):
    fake = types.ModuleType("g4f")

    class _ChatCompletion:
        @staticmethod
        def create_async(*args, **kwargs):
            raise _Boom(UPSTREAM_TEXT)

    fake.ChatCompletion = _ChatCompletion
    monkeypatch.setitem(sys.modules, "g4f", fake)


def test_g4f_non_stream_error_is_sanitized(monkeypatch):
    from proxy_app.provider_adapter import G4FAdapter

    _fake_g4f_module(monkeypatch)
    adapter = G4FAdapter.__new__(G4FAdapter)

    with pytest.raises(HTTPException) as exc_info:
        import asyncio

        asyncio.run(adapter._non_stream_g4f_completion("hello", "model-x"))

    _assert_clean(str(exc_info.value.detail))


def test_g4f_stream_error_chunk_is_sanitized(monkeypatch):
    from proxy_app.provider_adapter import G4FAdapter

    _fake_g4f_module(monkeypatch)
    adapter = G4FAdapter.__new__(G4FAdapter)

    import asyncio

    async def _collect():
        chunks = []
        async for chunk in adapter._stream_g4f_completion("hello", "model-x"):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(_collect())

    _assert_clean(str(chunks))
    assert any("error" in chunk for chunk in chunks), "should report the failure in-band"


def test_g4f_convert_error_is_sanitized():
    from proxy_app.provider_adapter import G4FAdapter

    adapter = G4FAdapter.__new__(G4FAdapter)
    exc = adapter._convert_error(_Boom(UPSTREAM_TEXT))

    assert isinstance(exc, HTTPException)
    _assert_clean(str(exc.detail))
