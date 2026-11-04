"""Bounded model-session state and deterministic intent compatibility helpers.

The legacy gateway kept session model choices in an unbounded process-global
dict and used regex-based intent detection. This module preserves the useful
behavior while bounding session memory and keeping routing vocabulary within
the sidecar's accepted aliases.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import threading
import time
from typing import Iterable

DEFAULT_MODEL = "auto-chat"
MAX_INTENT_TEXT = 32768
MAX_INTENT_MESSAGES = 128


def valid_session_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= 512
        and not any(ord(char) < 32 for char in value)
    )


class ModelSessionStore:
    """Process-local, bounded, expiring mapping of session ID to model alias."""

    def __init__(
        self,
        aliases: Iterable[str],
        *,
        max_sessions: int = 1024,
        ttl_seconds: float = 86400.0,
        now: float | None = None,
    ) -> None:
        available = frozenset(aliases)
        if DEFAULT_MODEL not in available:
            raise ValueError("default model is not an available alias")
        if isinstance(max_sessions, bool) or not isinstance(max_sessions, int):
            raise ValueError("max_sessions must be an integer")
        if not 1 <= max_sessions <= 100000:
            raise ValueError("max_sessions must be between 1 and 100000")
        if (
            isinstance(ttl_seconds, bool)
            or not isinstance(ttl_seconds, (int, float))
            or not math.isfinite(ttl_seconds)
            or ttl_seconds <= 0
        ):
            raise ValueError("ttl_seconds must be positive and finite")
        self.available = available
        self.max_sessions = max_sessions
        self.ttl_seconds = float(ttl_seconds)
        self._items: dict[str, tuple[str, float]] = {}
        self._lock = threading.Lock()
        self._clock = time.time if now is None else lambda: now

    def _current_time(self, now: float | None) -> float:
        value = self._clock() if now is None else now
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("now must be finite")
        return float(value)

    def _prune(self, current: float) -> None:
        cutoff = current - self.ttl_seconds
        expired = [key for key, (_, updated) in self._items.items() if updated < cutoff]
        for key in expired:
            self._items.pop(key, None)

    def get(self, session_id: str, *, now: float | None = None) -> str:
        if not valid_session_id(session_id):
            raise ValueError("invalid session_id")
        current = self._current_time(now)
        with self._lock:
            self._prune(current)
            item = self._items.get(session_id)
            if item is None:
                return DEFAULT_MODEL
            model, _ = item
            return model

    def set(
        self, session_id: str, model: str, *, now: float | None = None
    ) -> str | None:
        if not valid_session_id(session_id):
            raise ValueError("invalid session_id")
        if model not in self.available:
            raise ValueError("invalid model")
        current = self._current_time(now)
        with self._lock:
            self._prune(current)
            previous = self._items.get(session_id)
            self._items[session_id] = (model, current)
            if len(self._items) > self.max_sessions:
                oldest = min(self._items, key=lambda key: self._items[key][1])
                self._items.pop(oldest, None)
            return None if previous is None else previous[0]

    def count(self) -> int:
        with self._lock:
            return len(self._items)


UNSENSORED_PATTERNS = (
    r"\*[^*]+\*",
    r"\([^)]+\)",
    r"--[^-]+--",
    r"\{\{[^}]+\}\}",
    r"\b(roleplay|rp|in character|IC|OOC|out of character)\b",
    r"\b(continue the story|continue roleplay|as [^,]+,|playing as)\b",
    r"\b(uncensored|no filter|no restrictions|explicit|nsfw)\b",
    r"\b(narrate|narration|scene|setting)\b",
    r"\b(character|persona|protagonist|antagonist)\b",
    r"\b(once upon a time|in a land|the story begins)\b",
    r"\b(write a story|tell me a story|creative writing)\b",
    r"\b(passionate|intimate|sensual|romantic|affectionate)\b",
    r"\b(whisper(ed|ing)?|gaze|touch|embrace)\b",
)

CODING_PATTERNS = (
    r"\b(function|class|method|variable|const|let|var)\b",
    r"\b(def |async def|import |from |return |if __name__)\b",
    r"\b(python|javascript|typescript|java|c\+\+|rust|go)\b",
    r"\b(debug|error|bug|fix|implement|refactor)\b",
    r"\b(code|script|program|algorithm|syntax)\b",
    r"\b(API|endpoint|request|response|JSON|HTTP)\b",
    r"\b(git|commit|push|pull|merge|branch)\b",
    r"\b(test|unit test|integration|pytest|jest)\b",
    r"\b(database|SQL|query|table|schema)\b",
    r"\b(react|vue|angular|node|express|fastapi|django)\b",
    r"```[a-z]*\n",
    r"`[^`]+`",
    r"\b(indentation|indent|bracket|brace)\b",
    r"\b(architecture|design pattern|system design)\b",
    r"\b(optimize|performance|scalability)\b",
    r"\b(security|vulnerability|authentication)\b",
)

COMPLEX_REASONING_PATTERNS = (
    r"\b(analyze|analysis|evaluate|assessment)\b",
    r"\b(compare|contrast|difference|similarity)\b",
    r"\b(pros and cons|advantages|disadvantages)\b",
    r"\b(implications|consequences|impact)\b",
    r"\b(why|how come|explain in detail|elaborate)\b",
    r"\b(philosophy|ethical|moral|implication)\b",
    r"\b(theory|concept|hypothesis|premise)\b",
    r"\b(consider|take into account|factor in)\b",
    r"\b(research|investigate|explore|examine)\b",
    r"\b(comprehensive|thorough|in-depth|detailed)\b",
)

FAST_CHAT_PATTERNS = (
    r"^(hi|hello|hey|yo|sup)\s*[!.]?$",
    r"^(thanks|thank you|thx)\s*[!.]?$",
    r"^(ok|okay|k|sure|yes|no)\s*[!.]?$",
    r"^(bye|goodbye|see ya)\s*[!.]?$",
    r"^(what is|who is|where is|when is|how (do|to))\s+\w+\s*[?]?$",
    r"\b(quick(ly)?|fast|simple|short|brief)\b.*\b(question|answer|response)\b",
)

INTENT_MODELS = {
    "chat-rp": "chat-rp",
    "coding-elite": "coding-elite",
    "coding-fast": "coding-fast",
    "chat-smart": "chat-smart",
    "chat-fast": "chat-fast",
}


@dataclass(frozen=True)
class IntentResult:
    intent: str | None
    confidence: float
    detected_keywords: list[str]
    suggested_model: str
    reasoning: str

    def as_dict(self) -> dict:
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "suggested_model": self.suggested_model,
            "detected_keywords": self.detected_keywords,
            "reasoning": self.reasoning,
        }


def _message_text(messages: object) -> tuple[str, str]:
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    if len(messages) > MAX_INTENT_MESSAGES:
        raise ValueError(f"messages must contain at most {MAX_INTENT_MESSAGES} entries")
    chunks: list[str] = []
    system = ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            chunks.append(content)
            if message.get("role") == "system" and not system:
                system = content
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
    return " ".join(chunks)[:MAX_INTENT_TEXT], system[:MAX_INTENT_TEXT]


def _matches(patterns: Iterable[str], text: str) -> list[str]:
    found: list[str] = []
    for pattern in patterns:
        found.extend(str(value) for value in re.findall(pattern, text, re.IGNORECASE))
    return found


def _keywords(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = value.strip()
        if text and text not in result:
            result.append(text[:80])
    return result[:10]


def detect_intent(messages: object, current_model: str | None = None) -> IntentResult:
    """Classify the latest text with the legacy keyword heuristic.

    Suggestions are sidecar aliases. The legacy `uncensored-chat` target is
    represented by `chat-rp`, which is the available roleplay alias here.
    """
    if current_model is not None and not isinstance(current_model, str):
        raise ValueError("current_model must be a string")
    text, system = _message_text(messages)
    keywords: list[str] = []
    scores: dict[str, float] = {}

    unsensored = _matches(UNSENSORED_PATTERNS, text)
    if unsensored:
        keywords.extend(_keywords(unsensored)[:5])
        scores["chat-rp"] = min(0.9, 0.5 + len(unsensored) * 0.1)

    coding = _matches(CODING_PATTERNS, text)
    if coding:
        keywords.extend(_keywords(coding)[:5])
        complex_markers = bool(
            re.search(
                r"\b(architecture|design|system|optimize|security)\b",
                text,
                re.IGNORECASE,
            )
            or "```" in text
        )
        target = "coding-elite" if complex_markers else "coding-fast"
        scores[target] = (
            min(0.9, 0.6 + len(coding) * 0.05)
            if complex_markers
            else min(0.85, 0.5 + len(coding) * 0.05)
        )

    reasoning = _matches(COMPLEX_REASONING_PATTERNS, text)
    if reasoning:
        keywords.extend(_keywords(reasoning)[:5])
        scores["chat-smart"] = min(0.85, 0.5 + len(reasoning) * 0.1)

    fast = _matches(FAST_CHAT_PATTERNS, text.strip())
    if fast and not any(score > 0.7 for score in scores.values()):
        scores["chat-fast"] = min(0.7, 0.4 + len(fast) * 0.15)

    if not scores:
        if current_model:
            return IntentResult(
                intent=None,
                confidence=0.0,
                detected_keywords=[],
                suggested_model=current_model,
                reasoning="No clear intent detected, using current model",
            )
        if any(
            hint in system.lower()
            for hint in ("roleplay", "rp", "uncensored", "character")
        ):
            return IntentResult(
                intent="chat-rp",
                confidence=0.6,
                detected_keywords=["system_prompt_hint"],
                suggested_model="chat-rp",
                reasoning="System prompt suggests roleplay/uncensored content",
            )
        return IntentResult(
            intent="chat-fast",
            confidence=0.5,
            detected_keywords=[],
            suggested_model="chat-fast",
            reasoning="No clear intent, defaulting to fast chat",
        )

    best = max(scores, key=scores.get)
    confidence = scores[best]
    return IntentResult(
        intent=best,
        confidence=confidence,
        detected_keywords=_keywords(keywords),
        suggested_model=INTENT_MODELS[best],
        reasoning=(
            f"Detected {best} with {confidence:.0%} confidence based on "
            f"{len(keywords)} keywords"
        ),
    )
