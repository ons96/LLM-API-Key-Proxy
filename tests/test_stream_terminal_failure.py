"""Terminal stream failure must not look like a successful completion (#371).

A provider that dies mid-stream used to be reported on the wire as
``finish_reason: "stop"`` plus ``data: [DONE]``: byte-identical to a normal
answer, so every client counted a failed call as a success.
"""

import re

import pytest

import proxy_app.main as main_module
from tests.test_error_detail_leak import _FakeRequest


def _finish_reasons(body: str) -> list[str]:
    """Every non-null finish_reason in the SSE body, in order.

    Substring scanning rather than JSON parsing: the sanitised error text is
    multi-line, so a chunk is not guaranteed to sit on one line.
    """
    return re.findall(r'"finish_reason"\s*:\s*"([a-z_]+)"', body)


@pytest.mark.asyncio
async def test_terminal_stream_failure_is_not_a_success(monkeypatch):
    class _StreamFailingRouter:
        async def handle_chat_completions(self, request_data, raw_request):
            async def _gen():
                yield 'data: {"id": "x", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "partial"}, "index": 0}]}\n\n'
                raise RuntimeError("upstream died after the first chunk")

            return _gen()

    monkeypatch.setattr(main_module, "get_router", lambda: _StreamFailingRouter())

    request = _FakeRequest({"model": "router/coding-smart", "messages": [], "stream": True})
    response = await main_module.chat_completions(request)

    body = ""
    async for chunk in response.body_iterator:
        body += chunk.decode() if isinstance(chunk, bytes) else chunk

    finish_reasons = _finish_reasons(body)

    assert "data: [DONE]" not in body, "a failed stream must not send the success terminator"
    assert "stop" not in finish_reasons, f"failure reported as success: {finish_reasons}"
    assert "error" in finish_reasons, f"no terminal error marker: {finish_reasons}"
    assert "Gateway stream error" in body
