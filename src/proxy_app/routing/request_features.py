"""
Request Feature Extractor for Auto Smart Routing (#478)

Single-pass deterministic parser for /v1/chat/completions request bodies.
Extracts capability flags, task class, and an input-token estimate with zero
network/model calls. Pure functions only; safe to import anywhere.

Consumed by tier_classifier (#476), latency_predictor (#477) and
chain_selector (#480) at the top of the USE_DYNAMIC_CHAIN middleware.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, IntFlag
from typing import Any, Dict, List, Optional, Union

# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


class Capabilities(IntFlag):
    """Capability bits a request requires from any candidate model."""

    TEXT = 1
    VISION = 2
    TOOL_CALLING = 4
    FILE_PARSING = 8


# ---------------------------------------------------------------------------
# Task classes
# ---------------------------------------------------------------------------


class TaskClass(str, Enum):
    """Coarse task class; drives tier floors and output-token estimation."""

    GREETING = "greeting"
    SHORT_QA = "short-qa"
    CODE_EDIT = "code-edit"
    CODE_GEN = "code-gen"
    REASONING = "reasoning"
    AGENTIC = "agentic-multi-step"
    SUMMARIZATION = "summarization"
    VISION_CAPTION = "vision-caption"
    FILE_ANALYSIS = "file-analysis"


class PromptBucket(str, Enum):
    """Input-length bucket (chars, not tokens)."""

    SHORT = "short"      # < 500 chars
    MEDIUM = "medium"    # 500..1999
    LONG = "long"        # 2000..7999
    VERY_LONG = "very-long"  # >= 8000


# ---------------------------------------------------------------------------
# Static maps
# ---------------------------------------------------------------------------

# File extensions that imply content parsing (FILE_PARSING), not vision.
FILE_EXTENSIONS = {
    "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "csv", "tsv",
    "txt", "md", "json", "xml", "yaml", "yml", "toml", "ini", "log",
    "py", "js", "ts", "tsx", "jsx", "go", "rs", "java", "c", "cpp",
    "h", "hpp", "cs", "rb", "php", "sh", "sql", "html", "css", "scss",
    "zip", "tar", "gz", "7z", "rar", "epub",
}
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp", "svg", "bmp", "avif"}

# MIME prefixes -> capability. Longer/より specific keys win (checked first).
_MIME_RULES: List[tuple] = [
    (re.compile(r"^image/"), Capabilities.VISION),
    (re.compile(r"^application/pdf$"), Capabilities.FILE_PARSING),
    (re.compile(r"^text/"), Capabilities.FILE_PARSING),
    (re.compile(r"^application/(msword|vnd\.openxmlformats|vnd\.ms-|zip|x-tar|gzip|json|csv|octet-stream)"),
     Capabilities.FILE_PARSING),
]

# Per-image / per-file token surcharge applied on top of the chars/4 text
# estimate (deterministic; documented constants, not measured).
IMAGE_TOKENS = 1024
FILE_TOKENS = 512

# ---------------------------------------------------------------------------
# Classification patterns (module-level compiled once)
# ---------------------------------------------------------------------------

_GREETING_RE = re.compile(
    r"^(hi|hello|hey|yo|sup|good (morning|afternoon|evening)|howdy)[\s,.!?]*$",
    re.IGNORECASE,
)
_CODE_FENCE_RE = re.compile(r"```|~~~")
_CODE_KW_RE = re.compile(
    r"\b(function|def|class|import|const|let|var|"
    r"public|private|static|void|lambda|async|await|if\s*\()"
)
_EDIT_VERB_RE = re.compile(
    r"\b(fix|debug|refactor|change|update|modify|correct|improve|migrate|"
    r"rewrite|review|explain this code|why (does|is|are))\b",
    re.IGNORECASE,
)
_GEN_VERB_RE = re.compile(
    r"\b(write|create|generate|implement|build|code|script|program|"
    r"function|class|module)\b",
    re.IGNORECASE,
)
_MATH_RE = re.compile(r"[+\-*/^=<>]|\b(solve|calculate|derive|compute|prove)\b", re.IGNORECASE)
_REASON_RE = re.compile(
    r"\b(why|how|explain|reason|logic|compare|analyze|evaluate|"
    r"step by step|let's think)\b",
    re.IGNORECASE,
)
_SUMMARIZE_RE = re.compile(
    r"\b(summarize|summarise|condense|tldr|tl;dr|key points|bullet points|"
    r"extract the main|executive summary)\b",
    re.IGNORECASE,
)
_AGENTIC_RE = re.compile(
    r"\b(then|first|next|finally|gather|cross-check|execute|multiple)\b",
    re.IGNORECASE,
)
_VISION_ASK_RE = re.compile(
    r"\b(describe|caption|what('s| is) in|what do you see|ocr|read (this|the) (image|picture|photo|screenshot))\b",
    re.IGNORECASE,
)
_FILE_ASK_RE = re.compile(
    r"\b(analyze|parse|read|process|summarize|extract|convert)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Feature dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RequestFeatures:
    """Extracted, deterministic features of one chat completion request."""

    capabilities: Capabilities = Capabilities.TEXT
    task_class: TaskClass = TaskClass.SHORT_QA
    estimated_input_tokens: int = 0
    prompt_length_bucket: PromptBucket = PromptBucket.SHORT
    stream: bool = False
    has_max_tokens: bool = False
    reasoning_effort: Optional[str] = None
    text_char_count: int = 0
    image_count: int = 0
    file_count: int = 0
    tool_count: int = 0
    # diagnostic only (frozen dataclass -> replace() or plain assignment below)
    detail: str = field(default="", repr=False)

    def requires(self, capability: Capabilities) -> bool:
        """True if the request needs `capability` from a candidate model."""
        return bool(self.capabilities & capability)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _capability_for_mime(mime: str) -> Optional[Capabilities]:
    for pattern, cap in _MIME_RULES:
        if pattern.match(mime):
            return cap
    return None


def _capability_for_url(url: str) -> Optional[Capabilities]:
    """Classify a url/data-uri by extension or mime prefix."""
    if url.startswith("data:"):
        # data:image/png;base64,... | data:application/pdf;base64,...
        mime = url[5:].split(";", 1)[0].split(",", 1)[0].lower()
        return _capability_for_mime(mime)
    path = url.split("?", 1)[0].split("#", 1)[0]
    dot = path.rfind(".")
    if dot == -1:
        return None
    ext = path[dot + 1 :].lower()
    if ext in IMAGE_EXTENSIONS:
        return Capabilities.VISION
    if ext in FILE_EXTENSIONS:
        return Capabilities.FILE_PARSING
    return None


def _block_capability(block: Dict[str, Any]) -> Optional[Capabilities]:
    """Capability implied by one content block dict (OpenAI/Anthropic-ish)."""
    btype = str(block.get("type", "")).lower()
    if btype in ("image_url", "input_image", "image", "image_urls"):
        return Capabilities.VISION
    if btype in ("file", "file_url", "input_file", "document", "attachment",
                 "document_url", "pdf"):
        return Capabilities.FILE_PARSING
    if btype in ("text", "input_text", "output_text", "text_delta"):
        return None  # text handled separately
    return None


def _block_text(block: Dict[str, Any]) -> str:
    """Best-effort text extraction from a content block dict."""
    for key in ("text", "content", "input_text"):
        val = block.get(key)
        if isinstance(val, str):
            return val
    return ""


def _file_name_from_url(url: str) -> str:
    path = url.split("?", 1)[0].split("#", 1)[0]
    return path.rsplit("/", 1)[-1] or ""


def _classify(features: "RequestFeatures", text: str) -> TaskClass:
    """Deterministic task-class assignment (documented precedence order)."""
    low = text.lower()
    n_chars = len(text)

    # 1. Vision caption / file analysis (capability-driven, highest precedence)
    if features.requires(Capabilities.VISION) and _VISION_ASK_RE.search(low):
        return TaskClass.VISION_CAPTION
    if features.requires(Capabilities.FILE_PARSING) and _FILE_ASK_RE.search(low):
        return TaskClass.FILE_ANALYSIS

    # 2. Agentic: tool-calling request with multi-step language
    if features.requires(Capabilities.TOOL_CALLING) and (
        n_chars > 300 or _AGENTIC_RE.search(low)
    ):
        return TaskClass.AGENTIC

    # 3. Greeting fast path (short, no task verbs)
    if n_chars <= 120 and _GREETING_RE.match(text.strip()):
        return TaskClass.GREETING

    # 4. Code edit / code gen (code presence decides)
    is_code = bool(_CODE_FENCE_RE.search(text) or _CODE_KW_RE.search(text))
    if is_code or _EDIT_VERB_RE.search(low):
        if _EDIT_VERB_RE.search(low):
            return TaskClass.CODE_EDIT
        if _GEN_VERB_RE.search(low) or is_code:
            return TaskClass.CODE_GEN

    # 5. Summarization
    if _SUMMARIZE_RE.search(low):
        return TaskClass.SUMMARIZATION

    # 6. Reasoning: math symbols or reasoning vocabulary, longer prompts
    if _MATH_RE.search(text) and n_chars > 40 or (
        _REASON_RE.search(low) and n_chars > 80
    ):
        return TaskClass.REASONING

    # 7. Defaults
    if n_chars <= 200:
        return TaskClass.SHORT_QA
    return TaskClass.REASONING


def _prompt_bucket(n_chars: int) -> PromptBucket:
    if n_chars < 500:
        return PromptBucket.SHORT
    if n_chars < 2000:
        return PromptBucket.MEDIUM
    if n_chars < 8000:
        return PromptBucket.LONG
    return PromptBucket.VERY_LONG


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_request_features(body: Dict[str, Any]) -> RequestFeatures:
    """Extract features from a /v1/chat/completions request body.

    Single pass, deterministic, no I/O. Missing/odd fields degrade to
    TEXT-only / SHORT_QA rather than raising.
    """
    capabilities = Capabilities.TEXT
    text_parts: List[str] = []
    image_count = 0
    file_count = 0
    tool_count = 0

    messages = body.get("messages") or []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        # Assistant tool_calls in history imply tool-using context.
        if msg.get("tool_calls"):
            tool_count += 1
            capabilities |= Capabilities.TOOL_CALLING
        content = msg.get("content")
        if content is None:
            continue
        if isinstance(content, str):
            if content:
                text_parts.append(content)
            continue
        if isinstance(content, list):
            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                    continue
                if not isinstance(block, dict):
                    continue
                cap = _block_capability(block)
                if cap == Capabilities.VISION:
                    image_count += 1
                    capabilities |= Capabilities.VISION
                elif cap == Capabilities.FILE_PARSING:
                    file_count += 1
                    capabilities |= Capabilities.FILE_PARSING
                # URL/data-uri inside known keys (only if block type gave no cap,
                # otherwise the same attachment would be double-counted)
                if cap is None:
                    for key in ("url", "file_url", "source_url", "image_url"):
                        url = block.get(key)
                        if isinstance(url, str):
                            url_cap = _capability_for_url(url)
                            if url_cap == Capabilities.VISION:
                                image_count += 1
                                capabilities |= Capabilities.VISION
                            elif url_cap == Capabilities.FILE_PARSING:
                                file_count += 1
                                capabilities |= Capabilities.FILE_PARSING
                        elif isinstance(url, dict):
                            inner = url.get("url")
                            if isinstance(inner, str):
                                url_cap = _capability_for_url(inner)
                                if url_cap == Capabilities.VISION:
                                    image_count += 1
                                    capabilities |= Capabilities.VISION
                                elif url_cap == Capabilities.FILE_PARSING:
                                    file_count += 1
                                    capabilities |= Capabilities.FILE_PARSING
                text_parts.append(_block_text(block))

    # Tools / tool_choice
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        for tool in tools:
            if isinstance(tool, dict):
                tool_count += 1
        capabilities |= Capabilities.TOOL_CALLING
    tool_choice = body.get("tool_choice")
    if tool_choice not in (None, "none"):
        capabilities |= Capabilities.TOOL_CALLING
        if isinstance(tool_choice, dict):
            tool_count += 1

    # Modalities field (responses API style)
    modalities = body.get("modalities")
    if isinstance(modalities, list) and "image" in modalities:
        capabilities |= Capabilities.VISION
        image_count += 1

    # Text length -> token estimate (chars/4 heuristic, per prototype).
    # Tools/function JSON counts toward prompt size but must NOT drive the
    # task classification (it is full of code-looking keywords).
    classify_text = "\n".join(part for part in text_parts if part)
    token_text = classify_text
    if isinstance(tools, list) and tools:
        token_text += "\n" + str(tools)
    n_chars = len(classify_text)
    estimated_tokens = (len(token_text) // 4) + image_count * IMAGE_TOKENS + file_count * FILE_TOKENS

    features = RequestFeatures(
        capabilities=capabilities,
        estimated_input_tokens=estimated_tokens,
        prompt_length_bucket=_prompt_bucket(n_chars),
        stream=bool(body.get("stream")),
        has_max_tokens="max_tokens" in body or "max_completion_tokens" in body,
        reasoning_effort=body.get("reasoning_effort")
        if isinstance(body.get("reasoning_effort"), str)
        else None,
        text_char_count=n_chars,
        image_count=image_count,
        file_count=file_count,
        tool_count=tool_count,
    )
    # dataclass is frozen; build task_class via object.__setattr__ once
    object.__setattr__(features, "task_class", _classify(features, classify_text))
    return features


# -- demo / self-check ------------------------------------------------------


def _demo() -> None:
    """Runnable self-check (python -m proxy_app.routing.request_features)."""
    import json

    cases = {
        "plain-text": {"messages": [{"role": "user", "content": "hi"}]},
        "tools": {
            "messages": [{"role": "user", "content": "call the tool"}],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
        },
        "vision": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what's in this image?"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    ],
                }
            ]
        },
        "file": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "summarize this pdf"},
                        {"type": "file", "file_url": {"url": "https://x.example/report.pdf"}},
                    ],
                }
            ]
        },
    }
    for name, body in cases.items():
        f = extract_request_features(body)
        print(f"{name:>10}: caps={f.capabilities!r} task={f.task_class.value} "
              f"tok={f.estimated_input_tokens} imgs={f.image_count} files={f.file_count}")
    assert extract_request_features(cases["plain-text"]).capabilities == Capabilities.TEXT
    assert extract_request_features(cases["tools"]).requires(Capabilities.TOOL_CALLING)
    assert extract_request_features(cases["vision"]).requires(Capabilities.VISION)
    assert extract_request_features(cases["file"]).requires(Capabilities.FILE_PARSING)
    print("OK")


if __name__ == "__main__":
    _demo()
