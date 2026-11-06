#!/usr/bin/env python3
"""The Responses API must speak Responses-API SSE, not chat-completions SSE.

Codex consumes ``/v1/responses`` with ``stream: true`` and waits for the
event sequence ``response.created`` -> ``*.delta`` -> ``response.completed``.
A chat-shaped stream never produces those events, so the client retries a
working request and then gives up. These tests pin the translation.
"""

import json

import pytest


def _load_main():
    from proxy_app import main as main_module

    return main_module


async def _chunks(items):
    for item in items:
        yield item


def _parse_sse(raw: bytes) -> list[tuple[str, dict]]:
    """Parse an SSE payload into (event_name, data) pairs."""
    events = []
    for block in raw.decode("utf-8").split("\n\n"):
        if not block.strip():
            continue
        name = None
        data = None
        for line in block.split("\n"):
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        if name is not None:
            events.append((name, data))
    return events


async def _collect(gen) -> bytes:
    return b"".join([chunk async for chunk in gen])


@pytest.mark.asyncio
async def test_sse_emits_the_responses_event_sequence():
    main = _load_main()

    chunks = _chunks(
        [
            {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}]},
        ]
    )
    raw = await _collect(main._responses_sse(chunks, "test/model"))
    events = _parse_sse(raw)
    names = [name for name, _ in events]

    assert names[0] == "response.created"
    assert "response.output_text.delta" in names
    assert names[-1] == "response.completed"


@pytest.mark.asyncio
async def test_sse_deltas_carry_the_text():
    main = _load_main()

    chunks = _chunks(
        [
            {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}]},
        ]
    )
    raw = await _collect(main._responses_sse(chunks, "test/model"))
    events = _parse_sse(raw)

    deltas = [d["delta"] for name, d in events if name == "response.output_text.delta"]
    assert deltas == ["Hello", " world"]

    done = [d["text"] for name, d in events if name == "response.output_text.done"]
    assert done == ["Hello world"]


@pytest.mark.asyncio
async def test_completed_event_carries_the_message_and_usage():
    main = _load_main()

    chunks = _chunks(
        [
            {"choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}],
             "usage": {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9}},
        ]
    )
    raw = await _collect(main._responses_sse(chunks, "test/model"))
    events = _parse_sse(raw)

    completed = [d for name, d in events if name == "response.completed"]
    assert len(completed) == 1
    response = completed[0]["response"]
    assert response["status"] == "completed"
    assert response["usage"]["total_tokens"] == 9
    text = response["output"][0]["content"][0]["text"]
    assert text == "Hi"


@pytest.mark.asyncio
async def test_empty_stream_still_completes():
    main = _load_main()

    raw = await _collect(main._responses_sse(_chunks([]), "test/model"))
    events = _parse_sse(raw)

    assert events[-1][0] == "response.completed"
    response = events[-1][1]["response"]
    assert response["status"] == "completed"


@pytest.mark.asyncio
async def test_non_dict_chunks_are_ignored():
    main = _load_main()

    chunks = _chunks(["a string", {"choices": [{"delta": {"content": "ok"}}]}])
    raw = await _collect(main._responses_sse(chunks, "test/model"))
    events = _parse_sse(raw)

    deltas = [d["delta"] for name, d in events if name == "response.output_text.delta"]
    assert deltas == ["ok"]


@pytest.mark.asyncio
async def test_mid_stream_failure_emits_an_error_event():
    main = _load_main()

    async def _failing():
        yield {"choices": [{"delta": {"content": "partial"}, "finish_reason": None}]}
        raise RuntimeError("upstream died")

    raw = await _collect(main._responses_sse(_failing(), "test/model"))
    events = _parse_sse(raw)

    errors = [d for name, d in events if name == "error"]
    assert len(errors) == 1
    assert "upstream died" in errors[0]["message"]


def test_normalize_accepts_input_text():
    """Codex sends input_text; only accepting "text" empties the message."""

    main = _load_main()

    content = [{"type": "input_text", "text": "hello codex"}]
    assert main._normalize_responses_content(content) == "hello codex"


def test_normalize_accepts_output_text():
    main = _load_main()

    content = [{"type": "output_text", "text": "assistant text"}]
    assert main._normalize_responses_content(content) == "assistant text"


def test_normalize_accepts_plain_text():
    main = _load_main()

    content = [{"type": "text", "text": "plain"}]
    assert main._normalize_responses_content(content) == "plain"


def test_normalize_joins_multiple_parts():
    main = _load_main()

    content = [
        {"type": "input_text", "text": "one"},
        {"type": "input_text", "text": "two"},
    ]
    assert main._normalize_responses_content(content) == "one\ntwo"


def test_normalize_handles_a_bare_string():
    main = _load_main()

    assert main._normalize_responses_content("plain string") == "plain string"


def test_normalize_handles_a_dict():
    main = _load_main()

    assert main._normalize_responses_content({"text": "from dict"}) == "from dict"
