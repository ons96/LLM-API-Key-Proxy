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


# --- tool calling -----------------------------------------------------------
#
# A harness is only useful if it can call tools. The Responses API nests a
# function tool differently from chat completions and reports tool calls as
# separate ``function_call`` output items, so both directions need translating.


def test_tools_are_converted_to_chat_shape():
    main = _load_main()
    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ]
    chat = main._responses_tools_to_chat(tools)
    assert chat == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]


def test_builtin_tools_are_dropped_and_chat_shape_passes_through():
    main = _load_main()
    already_chat = {"type": "function", "function": {"name": "keep", "parameters": {}}}
    chat = main._responses_tools_to_chat([already_chat, {"type": "web_search"}, "junk"])
    assert chat == [already_chat]


def test_tool_choice_converts():
    main = _load_main()
    assert main._responses_tool_choice_to_chat("auto") == "auto"
    assert main._responses_tool_choice_to_chat({"type": "function", "name": "f"}) == {
        "type": "function",
        "function": {"name": "f"},
    }
    assert main._responses_tool_choice_to_chat(None) is None


def test_function_call_items_become_assistant_tool_calls():
    main = _load_main()
    messages = main._responses_input_to_messages(
        {
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
                {"type": "function_call", "call_id": "call_1", "name": "f", "arguments": '{"a":1}'},
                {"type": "function_call_output", "call_id": "call_1", "output": "result"},
            ]
        }
    )
    assert messages[0] == {"role": "user", "content": "hi"}
    assert messages[1]["role"] == "assistant"
    assert messages[1]["tool_calls"][0]["id"] == "call_1"
    assert messages[1]["tool_calls"][0]["function"] == {"name": "f", "arguments": '{"a":1}'}
    assert messages[2] == {"role": "tool", "tool_call_id": "call_1", "content": "result"}


def test_reasoning_items_are_skipped():
    main = _load_main()
    messages = main._responses_input_to_messages(
        {"input": [{"type": "reasoning", "encrypted_content": "x"}, {"type": "message", "role": "user", "content": "hi"}]}
    )
    assert messages == [{"role": "user", "content": "hi"}]


def test_chat_usage_is_normalized_to_responses_fields():
    main = _load_main()
    assert main._to_responses_usage({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}) == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    }
    assert main._to_responses_usage(None) is None


@pytest.mark.asyncio
async def test_sse_emits_function_call_items():
    main = _load_main()
    chunks = [
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "f", "arguments": '{"a"'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ":1}"}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]
    events = _parse_sse(await _collect(main._responses_sse(_chunks(chunks), "test/model")))
    names = [name for name, _ in events]
    assert "response.output_item.added" in names
    assert "response.function_call_arguments.delta" in names
    assert "response.function_call_arguments.done" in names
    assert names[-1] == "response.completed"

    added = [d for n, d in events if n == "response.output_item.added"][1]
    assert added["item"]["type"] == "function_call"
    assert added["item"]["name"] == "f"
    assert added["output_index"] == 1

    completed = [d for n, d in events if n == "response.completed"][0]
    call_items = [i for i in completed["response"]["output"] if i["type"] == "function_call"]
    assert len(call_items) == 1
    assert call_items[0]["arguments"] == '{"a":1}'
    assert call_items[0]["call_id"] == "call_1"
