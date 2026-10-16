"""Regression tests for #122 / BUGS.md item 2: router wiring cannot be bypassed.

BUGS.md claimed the main entry point bypassed RouterCore/RouterIntegration, so
virtual models and fallback logic were inactive for /v1/chat/completions. By
the time this was investigated the wiring was already in place: both model
entry points call `get_router().handle_chat_completions(...)`, which is
RouterWrapper -> RouterIntegration -> RouterCore.route_request, and no direct
provider-client calls remain in those handlers.

These tests pin that delegation so the wiring cannot silently regress into a
bypass again.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proxy_app import main as main_module


class _FakeRequest:
    """Minimal stand-in for the Starlette request the handlers read."""

    def __init__(self, payload):
        self._payload = payload
        self.headers = {}

    async def json(self):
        return self._payload


class _RecorderRouter:
    """Stands in for RouterWrapper and records what it is asked to do."""

    def __init__(self, reply):
        self.calls = []
        self._reply = reply

    async def handle_chat_completions(self, request_data, raw_request):
        self.calls.append(request_data)
        return self._reply


REPLY = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "model": "test_model",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
}


def _chat_body(stream: bool = False):
    body = {
        "model": "router/coding-smart",
        "messages": [{"role": "user", "content": "hi"}],
    }
    if stream:
        body["stream"] = True
    return body


def test_chat_completions_delegates_to_router(monkeypatch):
    router = _RecorderRouter(REPLY)
    monkeypatch.setattr(main_module, "get_router", lambda: router)

    result = asyncio.run(main_module.chat_completions(_FakeRequest(_chat_body())))

    assert len(router.calls) == 1, "handler must call the router exactly once"
    assert router.calls[0]["model"] == "router/coding-smart"
    assert router.calls[0]["messages"] == [{"role": "user", "content": "hi"}]
    # The router reply is what the client gets; nothing re-routes it.
    assert json.loads(result.body) == REPLY


def test_chat_completions_stream_delegates_to_router(monkeypatch):
    async def _gen():
        yield {"id": "chatcmpl-test", "choices": [{"delta": {"content": "ok"}}]}

    router = _RecorderRouter(_gen())
    monkeypatch.setattr(main_module, "get_router", lambda: router)

    result = asyncio.run(
        main_module.chat_completions(_FakeRequest(_chat_body(stream=True)))
    )

    assert len(router.calls) == 1
    assert router.calls[0]["stream"] is True
    assert hasattr(result, "body_iterator"), "stream path must return a stream"


def test_responses_endpoint_delegates_to_router(monkeypatch):
    router = _RecorderRouter(REPLY)
    monkeypatch.setattr(main_module, "get_router", lambda: router)

    result = asyncio.run(
        main_module.responses_endpoint(
            _FakeRequest(
                {
                    "model": "router/coding-smart",
                    "input": [
                        {"type": "message", "role": "user", "content": "hi"}
                    ],
                }
            )
        )
    )

    assert len(router.calls) == 1, "responses API must go through the router too"
    assert router.calls[0]["model"] == "router/coding-smart"
    assert router.calls[0]["messages"] == [{"role": "user", "content": "hi"}]
    translated = json.loads(result.body)
    assert translated["model"] == REPLY["model"]
    assert translated["output"][0]["content"][0]["text"] == "ok"


def test_handlers_do_not_call_provider_clients_directly():
    """Static guard: the model paths must not grow a direct provider call."""
    source = Path(main_module.__file__).read_text(encoding="utf-8")

    for banned in ("acompletion(", "acreate(", "ChatCompletion.create(", "client.chat("):
        assert banned not in source, f"direct provider call found in main.py: {banned}"

    assert source.count("handle_chat_completions(") >= 2, (
        "both /v1/chat/completions and /v1/responses must delegate to the router"
    )
