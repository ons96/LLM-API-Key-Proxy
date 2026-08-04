"""Deterministic, dependency-free request feature extraction.

The extractor intentionally does not perform model calls, network access, or
provider lookups. It accepts an already-decoded chat-completions request body
and returns routing metadata that downstream selectors can consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntFlag
from typing import Any, Iterator, Mapping
from urllib.parse import urlparse


class Capability(IntFlag):
    """Bit flags describing request capabilities."""

    TEXT = 1
    VISION = 2
    TOOL_CALLING = 4
    FILE_PARSING = 8


class TaskClass(str, Enum):
    """Coarse deterministic task categories for routing."""

    GREETING_TRIVIA = "greeting/trivia"
    SHORT_QA = "short-qa"
    CODE_EDIT = "code-edit"
    CODE_GEN = "code-gen"
    REASONING = "reasoning"
    AGENTIC_MULTI_STEP = "agentic-multi-step"
    SUMMARIZATION = "summarization"
    VISION_CAPTION = "vision-caption"
    FILE_ANALYSIS = "file-analysis"


class PromptLength(str, Enum):
    """Prompt size buckets based on estimated input tokens."""

    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"
    VERY_LONG = "very-long"


@dataclass(frozen=True)
class RequestFeatures:
    """Routing metadata extracted from one request body."""

    capabilities: int
    task_class: TaskClass
    input_tokens: int
    prompt_length: PromptLength
    stream: bool = False
    has_max_tokens: bool = False
    modalities: tuple[str, ...] = ()
    reasoning_effort: str | None = None

    @property
    def estimated_input_tokens(self) -> int:
        """Compatibility name for the chars/4 input-token estimate."""
        return self.input_tokens

    @property
    def prompt_length_bucket(self) -> PromptLength:
        """Compatibility name for the prompt-length bucket."""
        return self.prompt_length

    def has_capability(self, capability: Capability) -> bool:
        """Return whether the request requires a capability flag."""
        return bool(self.capabilities & capability)


_FILE_EXTENSIONS = frozenset(
    {
        ".csv",
        ".doc",
        ".docx",
        ".html",
        ".json",
        ".md",
        ".pdf",
        ".ppt",
        ".pptx",
        ".rtf",
        ".tsv",
        ".txt",
        ".xls",
        ".xlsx",
        ".xml",
    }
)
_FILE_MIME_PREFIXES = ("text/", "application/pdf")
_FILE_MIME_TYPES = frozenset(
    {
        "application/msword",
        "application/rtf",
        "application/vnd.ms-excel",
        "application/vnd.ms-powerpoint",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
)


def extract_request_features(body: Mapping[str, Any]) -> RequestFeatures:
    """Extract request capabilities and task metadata in one deterministic pass."""
    capabilities = 0
    text_parts: list[str] = []
    has_vision = False
    has_file = False
    has_tools = False
    modalities: tuple[str, ...] = ()
    marks = {"vision": False, "file": False}

    messages = body.get("messages")
    if isinstance(messages, list):
        message_iter: Iterator[Any] = iter(messages)
    else:
        message_iter = iter(())

    for message in message_iter:
        if not isinstance(message, Mapping):
            continue
        _collect_content(
            message.get("content"),
            text_parts,
            lambda: marks.__setitem__("vision", True),
            lambda: marks.__setitem__("file", True),
        )
        # Tool results and function arguments are text-like content even when
        # they are not represented in a content block.
        if isinstance(message.get("tool_calls"), list):
            has_tools = True
        if isinstance(message.get("function_call"), Mapping):
            has_tools = True

    tools = body.get("tools")
    functions = body.get("functions")
    # An empty tools array is intentionally not a tool requirement. Presence of
    # a usable declaration (or an explicit non-none tool choice) is.
    if isinstance(tools, list) and tools:
        has_tools = True
    if isinstance(functions, list) and functions:
        has_tools = True
    if body.get("tool_choice") not in (None, "none"):
        has_tools = True

    has_vision = marks["vision"]
    has_file = marks["file"]

    if isinstance(body.get("modalities"), list):
        modalities = tuple(
            item for item in body["modalities"] if isinstance(item, str)
        )
        if any(item != "text" for item in modalities):
            has_vision = True

    capabilities |= Capability.TEXT
    if has_vision:
        capabilities |= Capability.VISION
    if has_tools:
        capabilities |= Capability.TOOL_CALLING
    if has_file:
        capabilities |= Capability.FILE_PARSING

    text = "\n".join(text_parts)
    input_tokens = max(0, (len(text) + 3) // 4)
    prompt_length = _prompt_length(input_tokens)
    task_class = _classify_task(
        text,
        has_vision=has_vision,
        has_file=has_file,
        has_tools=has_tools,
        input_tokens=input_tokens,
    )

    return RequestFeatures(
        capabilities=capabilities,
        task_class=task_class,
        input_tokens=input_tokens,
        prompt_length=prompt_length,
        stream=bool(body.get("stream", False)),
        has_max_tokens="max_tokens" in body,
        modalities=modalities,
        reasoning_effort=(
            body.get("reasoning_effort")
            if isinstance(body.get("reasoning_effort"), str)
            else None
        ),
    )


def _collect_content(
    content: Any,
    text_parts: list[str],
    mark_vision: Any,
    mark_file: Any,
) -> None:
    """Collect content blocks without recursively walking arbitrary objects."""
    if isinstance(content, str):
        text_parts.append(content)
        return
    if not isinstance(content, list):
        if isinstance(content, Mapping):
            _collect_content_block(content, text_parts, mark_vision, mark_file)
        return

    for block in content:
        if isinstance(block, str):
            text_parts.append(block)
        elif isinstance(block, Mapping):
            _collect_content_block(block, text_parts, mark_vision, mark_file)


def _collect_content_block(
    block: Mapping[str, Any],
    text_parts: list[str],
    mark_vision: Any,
    mark_file: Any,
) -> None:
    block_type = block.get("type")
    if block_type in {"image_url", "input_image", "image"}:
        mark_vision()
        return
    if block_type in {"file", "input_file", "file_reference", "document"}:
        mark_file()
        return
    if isinstance(block.get("text"), str):
        text_parts.append(block["text"])
    if isinstance(block.get("content"), str):
        text_parts.append(block["content"])

    source = block.get("file")
    if not isinstance(source, Mapping):
        source = block
    if _looks_like_file(source):
        mark_file()

    image_url = block.get("image_url")
    if isinstance(image_url, Mapping) and image_url.get("url"):
        mark_vision()


def _looks_like_file(value: Mapping[str, Any]) -> bool:
    for key in ("filename", "file_name", "name", "file_url", "url", "uri"):
        candidate = value.get(key)
        if not isinstance(candidate, str):
            continue
        path = urlparse(candidate).path.lower()
        if any(path.endswith(extension) for extension in _FILE_EXTENSIONS):
            return True
    for key in ("mime_type", "mime", "content_type"):
        mime = value.get(key)
        if isinstance(mime, str):
            normalized = mime.lower().split(";", 1)[0].strip()
            if normalized.startswith(_FILE_MIME_PREFIXES) or normalized in _FILE_MIME_TYPES:
                return True
    return False


def _prompt_length(input_tokens: int) -> PromptLength:
    if input_tokens <= 256:
        return PromptLength.SHORT
    if input_tokens <= 2048:
        return PromptLength.MEDIUM
    if input_tokens <= 8192:
        return PromptLength.LONG
    return PromptLength.VERY_LONG


def _classify_task(
    text: str,
    *,
    has_vision: bool,
    has_file: bool,
    has_tools: bool,
    input_tokens: int,
) -> TaskClass:
    lowered = text.lower()
    if has_vision:
        return TaskClass.VISION_CAPTION
    if has_file:
        return TaskClass.FILE_ANALYSIS
    if has_tools and any(
        marker in lowered
        for marker in ("step by step", "workflow", "then", "agent", "execute")
    ):
        return TaskClass.AGENTIC_MULTI_STEP
    if any(marker in lowered for marker in ("summarize", "summary", "tldr", "tl;dr")):
        return TaskClass.SUMMARIZATION
    if any(
        marker in lowered
        for marker in ("prove", "derive", "reason", "why", "tradeoff", "analyze")
    ):
        return TaskClass.REASONING
    if any(
        marker in lowered
        for marker in ("fix this", "bug", "refactor", "edit", "patch", "change this code")
    ):
        return TaskClass.CODE_EDIT
    if any(
        marker in lowered
        for marker in ("write code", "implement", "function", "class", "script", "program")
    ):
        return TaskClass.CODE_GEN
    if input_tokens <= 32 and any(
        marker in lowered
        for marker in ("hello", "hi", "hey", "what is", "who is", "when is", "where is")
    ):
        return TaskClass.GREETING_TRIVIA
    return TaskClass.SHORT_QA if input_tokens <= 256 else TaskClass.REASONING
