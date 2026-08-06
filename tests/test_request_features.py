"""Tests for proxy_app.routing.request_features (#478)."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proxy_app.routing.request_features import (
    Capabilities,
    PromptBucket,
    TaskClass,
    extract_request_features,
)


def _req(messages, **extra):
    body = {"messages": messages}
    body.update(extra)
    return body


def _text_msg(content, role="user"):
    return {"role": role, "content": content}


# ---------------------------------------------------------------------------
# Capability fixtures (AC: fixture correctness)
# ---------------------------------------------------------------------------


def test_plain_text_is_text_only():
    f = extract_request_features(_req([_text_msg("hello there")]))
    assert f.capabilities == Capabilities.TEXT
    assert not f.requires(Capabilities.VISION)
    assert not f.requires(Capabilities.TOOL_CALLING)
    assert not f.requires(Capabilities.FILE_PARSING)


def test_base64_image_sets_vision():
    body = _req([
        {"role": "user", "content": [
            {"type": "text", "text": "what is in this image?"},
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAE="}},
        ]}
    ])
    f = extract_request_features(body)
    assert f.requires(Capabilities.VISION)
    assert f.image_count == 1


def test_url_image_sets_vision():
    body = _req([
        {"role": "user", "content": [
            {"type": "text", "text": "describe the photo"},
            {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
        ]}
    ])
    f = extract_request_features(body)
    assert f.requires(Capabilities.VISION)
    assert f.image_count == 1


def test_tool_declarations_set_tool_calling():
    body = _req(
        [_text_msg("what is the weather in SF?")],
        tools=[{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }],
    )
    f = extract_request_features(body)
    assert f.requires(Capabilities.TOOL_CALLING)
    assert f.tool_count == 1


def test_empty_tools_array_is_not_tool_calling():
    f = extract_request_features(_req([_text_msg("hi")], tools=[]))
    assert not f.requires(Capabilities.TOOL_CALLING)


def test_pdf_attachment_sets_file_parsing():
    body = _req([
        {"role": "user", "content": [
            {"type": "text", "text": "summarize this pdf"},
            {"type": "file", "file_url": {"url": "https://example.com/report.pdf"}},
        ]}
    ])
    f = extract_request_features(body)
    assert f.requires(Capabilities.FILE_PARSING)
    assert f.file_count == 1


def test_docx_attachment_sets_file_parsing():
    body = _req([
        {"role": "user", "content": [
            {"type": "text", "text": "parse this document"},
            {"type": "file", "file_url": {"url": "data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,AAAA"}},
        ]}
    ])
    f = extract_request_features(body)
    assert f.requires(Capabilities.FILE_PARSING)


def test_no_attachment_is_text_only():
    f = extract_request_features(_req([_text_msg("explain quantum computing")]))
    assert f.capabilities == Capabilities.TEXT
    assert f.image_count == 0 and f.file_count == 0


def test_assistant_tool_calls_in_history_flag_agentic_context():
    body = _req([
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "f", "arguments": "{}"}}]},
        _text_msg("continue"),
    ])
    f = extract_request_features(body)
    assert f.requires(Capabilities.TOOL_CALLING)


def test_tool_choice_auto_sets_tool_calling():
    f = extract_request_features(_req([_text_msg("go")], tool_choice="auto"))
    assert f.requires(Capabilities.TOOL_CALLING)


def test_tool_choice_none_is_not_tool_calling():
    f = extract_request_features(_req([_text_msg("go")], tool_choice="none"))
    assert not f.requires(Capabilities.TOOL_CALLING)


def test_modalities_image_sets_vision():
    f = extract_request_features(_req([_text_msg("hi")], modalities=["text", "image"]))
    assert f.requires(Capabilities.VISION)


# ---------------------------------------------------------------------------
# Task classes
# ---------------------------------------------------------------------------


def test_greeting_class():
    f = extract_request_features(_req([_text_msg("hi")]))
    assert f.task_class == TaskClass.GREETING


def test_code_edit_class():
    f = extract_request_features(
        _req([_text_msg("fix the bug in this function:\n```python\ndef f(x):\n    return x +\n```")])
    )
    assert f.task_class == TaskClass.CODE_EDIT


def test_code_gen_class():
    f = extract_request_features(
        _req([_text_msg("write a python function that reverses a string")])
    )
    assert f.task_class == TaskClass.CODE_GEN


def test_reasoning_class():
    f = extract_request_features(
        _req([_text_msg("prove that sqrt(2) is irrational using a step by step argument")])
    )
    assert f.task_class == TaskClass.REASONING


def test_summarization_class():
    f = extract_request_features(
        _req([_text_msg("summarize the key points of this long article into bullet points")])
    )
    assert f.task_class == TaskClass.SUMMARIZATION


def test_agentic_class_with_tools_and_length():
    f = extract_request_features(
        _req(
            [_text_msg("use the available tools to research this topic, gather data from multiple sources, "
                       "cross-check the numbers, then produce a final report with citations and footnotes")],
            tools=[{"type": "function", "function": {"name": "search"}}],
        )
    )
    assert f.task_class == TaskClass.AGENTIC


def test_vision_caption_class():
    body = _req([
        {"role": "user", "content": [
            {"type": "text", "text": "what is in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]}
    ])
    f = extract_request_features(body)
    assert f.task_class == TaskClass.VISION_CAPTION


def test_file_analysis_class():
    body = _req([
        {"role": "user", "content": [
            {"type": "text", "text": "analyze this document"},
            {"type": "file", "file_url": {"url": "https://example.com/data.pdf"}},
        ]}
    ])
    f = extract_request_features(body)
    assert f.task_class == TaskClass.FILE_ANALYSIS


def test_default_short_qa():
    f = extract_request_features(_req([_text_msg("what time is it?")]))
    assert f.task_class == TaskClass.SHORT_QA


# ---------------------------------------------------------------------------
# Estimates, buckets, flags
# ---------------------------------------------------------------------------


def test_token_estimate_chars_over_4():
    text = "a" * 100
    f = extract_request_features(_req([_text_msg(text)]))
    assert f.estimated_input_tokens == 100 // 4


def test_image_surcharge_added():
    body = _req([
        {"role": "user", "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]}
    ])
    f = extract_request_features(body)
    # 4 chars "look" -> 1 token + 1 image * 1024
    assert f.estimated_input_tokens == 1024 + 1


def test_prompt_buckets():
    assert extract_request_features(_req([_text_msg("x" * 50)])).prompt_length_bucket == PromptBucket.SHORT
    assert extract_request_features(_req([_text_msg("x" * 700)])).prompt_length_bucket == PromptBucket.MEDIUM
    assert extract_request_features(_req([_text_msg("x" * 3000)])).prompt_length_bucket == PromptBucket.LONG
    assert extract_request_features(_req([_text_msg("x" * 9000)])).prompt_length_bucket == PromptBucket.VERY_LONG


def test_stream_and_max_tokens_flags():
    f = extract_request_features(_req([_text_msg("hi")], stream=True, max_tokens=100))
    assert f.stream and f.has_max_tokens


def test_reasoning_effort_captured():
    f = extract_request_features(_req([_text_msg("hi")], reasoning_effort="high"))
    assert f.reasoning_effort == "high"


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def test_none_content_ignored():
    f = extract_request_features(_req([{"role": "assistant", "content": None}]))
    assert f.capabilities == Capabilities.TEXT


def test_empty_messages():
    f = extract_request_features({"messages": []})
    assert f.capabilities == Capabilities.TEXT


def test_non_dict_blocks_ignored():
    body = _req([
        {"role": "user", "content": ["plain string block", 42, None]}
    ])
    f = extract_request_features(body)
    assert f.capabilities == Capabilities.TEXT


def test_missing_messages_key():
    f = extract_request_features({"model": "coding-smart"})
    assert f.capabilities == Capabilities.TEXT
    assert f.task_class == TaskClass.SHORT_QA


def test_mixed_multimodal_content():
    body = _req([
        {"role": "user", "content": [
            {"type": "text", "text": "caption + pdf both here"},
            {"type": "image_url", "image_url": {"url": "https://e.com/a.png"}},
            {"type": "file", "file_url": {"url": "https://e.com/b.docx"}},
        ]}
    ])
    f = extract_request_features(body)
    assert f.requires(Capabilities.VISION)
    assert f.requires(Capabilities.FILE_PARSING)
    assert f.image_count == 1 and f.file_count == 1


# ---------------------------------------------------------------------------
# Performance sanity (soft; AC: <5ms p50, <10ms p95 on VPS-40)
# ---------------------------------------------------------------------------


def test_parse_1000_requests_under_2s():
    body = _req(
        [_text_msg("write a function that does something useful with this data")],
        tools=[{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
        stream=True,
        max_tokens=500,
    )
    start = time.perf_counter()
    for _ in range(1000):
        extract_request_features(body)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"1000 parses took {elapsed:.3f}s (avg {elapsed * 1000:.1f}us)"
