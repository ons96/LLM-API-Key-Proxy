"""Bounded prompt classifier for gateway routing (#949).

Stdlib-only heuristic classifier. Acts as a chain-selector ABOVE the existing
dynamic_chain / router_core health-quota-policy gates: it only picks the
initial virtual model/chain, never bypasses provider health, quota, policy or
fallback controls. Safe fallback when uncertain/unavailable/stale.

Env config:
    PROMPT_CLASSIFIER_ENABLED=1|0   (default 0; opt-in)
    PROMPT_CLASSIFIER_THRESHOLD      (default 0.6; below => fallback)
    PROMPT_CLASSIFIER_DEFAULT_CHAIN  (default chat-fast)
"""

from __future__ import annotations

import os
import re
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger_name = "prompt_classifier"

try:
    import logging

    _log = logging.getLogger(logger_name)
except Exception:  # pragma: no cover
    _log = None


class PromptCategory(str, Enum):
    CODE = "code"
    REASONING = "reasoning"
    CHAT = "chat"
    RP = "rp"
    AGENTIC = "agentic"
    RETRIEVAL = "retrieval"
    UNKNOWN = "unknown"


# Bounded routing taxonomy -> virtual model chain.
CATEGORY_CHAIN: Dict[PromptCategory, str] = {
    PromptCategory.CODE: "coding-smart",
    PromptCategory.REASONING: "chat-smart",
    PromptCategory.CHAT: "chat-fast",
    PromptCategory.RP: "chat-rp",
    PromptCategory.AGENTIC: "coding-elite",
    PromptCategory.RETRIEVAL: "chat-fast",
    PromptCategory.UNKNOWN: "chat-fast",
}

# Tool-capable chains (mirror semantic_router.TOOL_CAPABLE_CHAINS subset).
TOOL_CAPABLE = {
    "coding-elite",
    "coding-smart",
    "coding-fast",
    "chat-smart",
    "chat-elite",
    "agent-oracle",
    "agent-build",
    "agent-explore",
}
DEFAULT_TOOL_CHAIN = "coding-fast"

# ---------------------------------------------------------------------------
# Pattern tables (compiled once)
# ---------------------------------------------------------------------------

_CODE_RE = re.compile(
    r"\b(function|class|method|variable|const|let|var|def |async def|import |from |"
    r"return |if __name__|python|javascript|typescript|java|rust|go|debug|error|bug|"
    r"fix|implement|refactor|code|script|program|algorithm|syntax|API|endpoint|"
    r"request|response|JSON|HTTP|git|commit|test|pytest|jest|database|SQL|query|"
    r"react|vue|angular|node|express|fastapi|django|architecture|design pattern|"
    r"traceback|exception|stack trace|undefined|null pointer|segfault|\w+Error)\b",
    re.IGNORECASE,
)
_REASON_RE = re.compile(
    r"\b(analyze|analysis|evaluate|assessment|compare|contrast|pros and cons|"
    r"advantages|disadvantages|tradeoff|tradeoffs|implications|consequences|impact|explain|explain in detail|"
    r"elaborate|philosophy|ethical|moral|theory|concept|hypothesis|premise|"
    r"research|investigate|explore|examine|comprehensive|thorough|in-depth|detailed|"
    r"why|how come|step by step|let's think)\b",
    re.IGNORECASE,
)
_RP_RE = re.compile(
    r"\*[^*]+\*|--[^-]+--|\{\{[^}]+\}\}|"
    r"\b(roleplay|rp|in character|IC|OOC|continue the story|continue roleplay|"
    r"playing as|uncensored|no filter|no restrictions|narrate|narration|"
    r"scene|setting|persona|protagonist|once upon a time|in a land|"
    r"(write|tell)( me)? (a|an) (short )?(story|poem|tale)|creative writing|passionate|intimate|sensual|"
    r"romantic|whisper|embrace)\b",
    re.IGNORECASE,
)
_AGENTIC_RE = re.compile(
    r"\b(plan|step by step|multi-step|agent|tool|tools|function call|execute|run command|"
    r"workflow|pipeline|orchestrate|automate|batch|open a PR|open a pull request|"
    r"clone the repo|run the (test suite|tests))\b",
    re.IGNORECASE,
)
_RETRIEVAL_RE = re.compile(
    r"\b(what is|what time|who is|where is|when is|find|search|lookup|look up|define|definition|"
    r"meaning of|fact|facts about)\b",
    re.IGNORECASE,
)
_GREETING_RE = re.compile(
    r"^(hi|hello|hey|yo|sup|thanks|thank you|thx|ok|okay|sure|yes|no|bye|"
    r"ok sure|sure thing|sounds good|got it|good (morning|night|evening|afternoon)|"
    r"lol.*|haha.*|how are you( doing)?( today)?|what'?s up)\s*[!.?]?$",
    re.IGNORECASE,
)


@dataclass
class ClassificationResult:
    category: PromptCategory
    confidence: float
    suggested_chain: str
    reasoning: str
    fallback: bool = False
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Metrics (in-process, thread-safe enough for counters)
# ---------------------------------------------------------------------------

_METRICS_LOCK = threading.Lock()
_METRICS: Dict[str, Any] = {
    "total": 0,
    "fallback": 0,
    "per_category": {c.value: 0 for c in PromptCategory},
    "latency_ms_sum": 0.0,
    "latency_ms_count": 0,
}


def get_metrics() -> Dict[str, Any]:
    with _METRICS_LOCK:
        total = _METRICS["total"]
        fb = _METRICS["fallback"]
        cnt = _METRICS["latency_ms_count"]
        s = _METRICS["latency_ms_sum"]
        return {
            "total": total,
            "fallback": fb,
            "fallback_rate": (fb / total) if total else 0.0,
            "per_category": dict(_METRICS["per_category"]),
            "latency_ms_avg": (s / cnt) if cnt else 0.0,
            "latency_ms_count": cnt,
        }


def reset_metrics() -> None:
    with _METRICS_LOCK:
        _METRICS["total"] = 0
        _METRICS["fallback"] = 0
        for k in _METRICS["per_category"]:
            _METRICS["per_category"][k] = 0
        _METRICS["latency_ms_sum"] = 0.0
        _METRICS["latency_ms_count"] = 0


def _record(category: PromptCategory, fallback: bool, latency_ms: float) -> None:
    with _METRICS_LOCK:
        _METRICS["total"] += 1
        if fallback:
            _METRICS["fallback"] += 1
        _METRICS["per_category"][category.value] += 1
        _METRICS["latency_ms_sum"] += latency_ms
        _METRICS["latency_ms_count"] += 1


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _enabled() -> bool:
    return os.getenv("PROMPT_CLASSIFIER_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}


def _threshold() -> float:
    try:
        return float(os.getenv("PROMPT_CLASSIFIER_THRESHOLD", "0.6"))
    except ValueError:
        return 0.6


def _default_chain() -> str:
    return os.getenv("PROMPT_CLASSIFIER_DEFAULT_CHAIN", "chat-fast").strip() or "chat-fast"


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _extract_text(request: Any) -> str:
    if isinstance(request, dict):
        msgs = request.get("messages", []) or []
    elif isinstance(request, list):
        msgs = request
    else:
        msgs = []
    parts: List[str] = []
    for msg in msgs:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for p in content:
                if isinstance(p, dict) and p.get("type") == "text":
                    parts.append(p.get("text", "") or "")
    return " ".join(parts)


def classify(request: Any) -> ClassificationResult:
    """Classify a request/prompt into a bounded PromptCategory.

    Never raises. On any error returns unknown + fallback=True.
    """
    t0 = time.perf_counter()
    try:
        text = _extract_text(request)
        needs_tools = False
        if isinstance(request, dict):
            needs_tools = bool(request.get("tools") or request.get("tool_choice"))

        scores: Dict[PromptCategory, float] = {c: 0.0 for c in PromptCategory}
        matches: Dict[PromptCategory, int] = {c: 0 for c in PromptCategory}

        stripped = text.strip()
        if _GREETING_RE.match(stripped) and len(stripped) < 40:
            scores[PromptCategory.CHAT] = max(scores[PromptCategory.CHAT], 0.7)
            matches[PromptCategory.CHAT] += 1

        n_code = len(_CODE_RE.findall(text))
        n_reason = len(_REASON_RE.findall(text))
        n_rp = len(_RP_RE.findall(text))
        n_agent = len(_AGENTIC_RE.findall(text))
        n_retrieval = len(_RETRIEVAL_RE.findall(text))

        if n_code:
            matches[PromptCategory.CODE] += n_code
            scores[PromptCategory.CODE] = min(0.95, 0.55 + n_code * 0.1)
        if n_reason:
            matches[PromptCategory.REASONING] += n_reason
            scores[PromptCategory.REASONING] = min(0.95, 0.55 + n_reason * 0.1)
        if n_rp:
            matches[PromptCategory.RP] += n_rp
            scores[PromptCategory.RP] = min(0.95, 0.55 + n_rp * 0.1)
        if n_agent:
            matches[PromptCategory.AGENTIC] += n_agent
            scores[PromptCategory.AGENTIC] = min(0.95, 0.55 + n_agent * 0.1)
        if n_retrieval and not n_code and not n_reason:
            matches[PromptCategory.RETRIEVAL] += n_retrieval
            scores[PromptCategory.RETRIEVAL] = min(0.8, 0.5 + n_retrieval * 0.1)

        # Code blocks strongly imply code.
        if "```" in text or "~~~" in text:
            scores[PromptCategory.CODE] = max(scores[PromptCategory.CODE], 0.85)
            matches[PromptCategory.CODE] += 1

        # Pick best.
        best_cat = max(scores, key=lambda c: scores[c])
        best_score = scores[best_cat]

        if best_score < 0.4 or best_cat == PromptCategory.UNKNOWN and best_score == 0.0:
            # No strong signal -> unknown / fallback
            if best_score == 0.0:
                best_cat = PromptCategory.UNKNOWN
                best_score = 0.3
            # else keep best_cat but mark fallback if below threshold

        threshold = _threshold()
        fallback = best_score < threshold

        # Tool-aware chain selection.
        chain = CATEGORY_CHAIN.get(best_cat, _default_chain())
        if needs_tools and chain not in TOOL_CAPABLE:
            chain = DEFAULT_TOOL_CHAIN

        if fallback:
            # Safe default chain when uncertain.
            if needs_tools:
                chain = DEFAULT_TOOL_CHAIN
            else:
                chain = _default_chain()

        reasoning = (
            f"category={best_cat.value} score={best_score:.2f} "
            f"matches(code={n_code},reason={n_reason},rp={n_rp},agent={n_agent},retr={n_retrieval}) "
            f"threshold={threshold:.2f} fallback={fallback}"
        )

        latency_ms = (time.perf_counter() - t0) * 1000.0
        result = ClassificationResult(
            category=best_cat,
            confidence=round(best_score, 3),
            suggested_chain=chain,
            reasoning=reasoning,
            fallback=fallback,
            latency_ms=round(latency_ms, 3),
        )
        _record(best_cat, fallback, latency_ms)
        return result
    except Exception as exc:  # pragma: no cover - defensive
        latency_ms = (time.perf_counter() - t0) * 1000.0
        result = ClassificationResult(
            category=PromptCategory.UNKNOWN,
            confidence=0.0,
            suggested_chain=_default_chain(),
            reasoning=f"classifier error: {exc!r}",
            fallback=True,
            latency_ms=round(latency_ms, 3),
        )
        _record(PromptCategory.UNKNOWN, True, latency_ms)
        return result


def is_enabled() -> bool:
    return _enabled()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _self_test() -> None:
    cases = [
        ("def reverse(s): return s[::-1]", PromptCategory.CODE),
        ("hi", PromptCategory.CHAT),
        ("*walks into the tavern*", PromptCategory.RP),
        ("Compare REST and gRPC tradeoffs", PromptCategory.REASONING),
    ]
    for text, expected in cases:
        r = classify({"messages": [{"role": "user", "content": text}]})
        print(f"  {expected.value:10s} -> {r.category.value:10s} conf={r.confidence:.2f} chain={r.suggested_chain} fb={r.fallback}")
    print("self-test OK")


if __name__ == "__main__":
    _self_test()
