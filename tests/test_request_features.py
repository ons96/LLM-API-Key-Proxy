"""Unit tests for deterministic request feature extraction."""

from proxy_app.routing.request_features import (
    Capability,
    PromptLength,
    TaskClass,
    extract_request_features,
)


def test_base64_image_fixture_requires_vision():
    body = {
        "model": "vision-model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Caption this image."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,"
                            + "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"
                        },
                    },
                ],
            }
        ],
    }

    features = extract_request_features(body)

    assert features.capabilities == Capability.TEXT | Capability.VISION
    assert features.task_class is TaskClass.VISION_CAPTION


def test_url_image_fixture_requires_vision():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is shown here?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.test/photo.jpg"},
                    },
                ],
            }
        ]
    }

    features = extract_request_features(body)

    assert features.has_capability(Capability.VISION)
    assert features.task_class is TaskClass.VISION_CAPTION


def test_tool_declarations_require_tool_calling():
    body = {
        "messages": [{"role": "user", "content": "Get the weather, then report it."}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {"type": "object"},
                },
            }
        ],
        "tool_choice": "auto",
    }

    features = extract_request_features(body)

    assert features.has_capability(Capability.TEXT)
    assert features.has_capability(Capability.TOOL_CALLING)
    assert features.task_class is TaskClass.AGENTIC_MULTI_STEP


def test_pdf_docx_attachment_requires_file_parsing():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {
                            "filename": "report.pdf",
                            "mime_type": "application/pdf",
                        },
                    },
                    {
                        "type": "file",
                        "file": {
                            "filename": "appendix.docx",
                            "mime_type": "application/vnd.openxmlformats-officedocument"
                            ".wordprocessingml.document",
                        },
                    },
                    {"type": "text", "text": "Analyze these files."},
                ],
            }
        ]
    }

    features = extract_request_features(body)

    assert features.has_capability(Capability.TEXT)
    assert features.has_capability(Capability.FILE_PARSING)
    assert features.task_class is TaskClass.FILE_ANALYSIS


def test_plain_text_is_text_only_and_uses_chars_over_four():
    prompt = "What is 2 + 2?"
    body = {"messages": [{"role": "user", "content": prompt}]}

    features = extract_request_features(body)

    assert features.capabilities == Capability.TEXT
    assert features.task_class is TaskClass.GREETING_TRIVIA
    assert features.input_tokens == len(prompt) // 4
    assert features.prompt_length is PromptLength.SHORT


def test_request_metadata_fields_are_preserved():
    body = {
        "messages": [{"role": "user", "content": "Explain this briefly."}],
        "stream": True,
        "max_tokens": 120,
        "reasoning_effort": "low",
    }

    features = extract_request_features(body)

    assert features.stream is True
    assert features.has_max_tokens is True
    assert features.reasoning_effort == "low"
    assert features.modalities == ()


def test_empty_tools_array_does_not_require_tool_calling():
    body = {
        "messages": [{"role": "user", "content": "Hello"}],
        "tools": [],
        "tool_choice": "none",
    }

    features = extract_request_features(body)

    assert features.capabilities == Capability.TEXT
    assert not features.has_capability(Capability.TOOL_CALLING)
    assert features.task_class is TaskClass.GREETING_TRIVIA


def test_missing_messages_key_is_handled_deterministically():
    features = extract_request_features({})

    assert features.capabilities == Capability(0)
    assert features.input_tokens == 0
    assert features.prompt_length is PromptLength.SHORT
    assert features.task_class is TaskClass.SHORT_QA
    assert features.stream is False
    assert features.has_max_tokens is False
    assert features.modalities == ()
    assert features.reasoning_effort is None


def test_malformed_message_entries_are_skipped_without_error():
    body = {
        "messages": [
            None,
            42,
            "bare string entry",
            {"role": "user", "content": 12345},
            {"role": "user", "content": "Hello there"},
        ]
    }

    features = extract_request_features(body)

    assert features.capabilities == Capability.TEXT
    assert features.input_tokens == len("Hello there") // 4
    assert features.task_class is TaskClass.GREETING_TRIVIA


def test_non_string_modalities_and_reasoning_effort_are_ignored():
    body = {
        "messages": [{"role": "user", "content": "Describe this."}],
        "modalities": [42, "image"],
        "reasoning_effort": 7,
    }

    features = extract_request_features(body)

    assert features.modalities == ("image",)
    assert features.has_capability(Capability.VISION)
    assert features.task_class is TaskClass.VISION_CAPTION
    assert features.reasoning_effort is None
