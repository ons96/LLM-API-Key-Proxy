"""
Semantic Auto-Router for the 'auto' virtual model.

Resolves `model="auto"` to a concrete virtual-model chain via embedding
similarity (semantic-router + FastEmbed local encoder). Falls back to the
existing keyword intent_detector when the local stack is unavailable or
confidence is below threshold.

Tool-capability guard: if the request carries `tools`/`tool_choice`, the
resolved chain is restricted to a tool-capable allowlist. Non-tool intents
(FAST_CHAT, ROLEPLAY) are upgraded to a safe tool-capable chain.

Config (env):
    AUTO_ROUTE_MODE=all|ambiguous  (default: all)
        all       = always route via semantic_router
        ambiguous = only route when keyword detector confidence < 0.6
    AUTO_ROUTE_THRESHOLD=0.30      (default: 0.30, cosine similarity cutoff)

Refs: task-board #197, #194. Depends on #163 for full embedding corpus.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .intent_detector import IntentResult, MessageIntent, detect_intent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent -> virtual-model chain mapping
# ---------------------------------------------------------------------------

# Tool-capable chains: providers in these chains have supports_tools=True
# for at least the first few fallback entries (verified via virtual_models.yaml).
TOOL_CAPABLE_CHAINS = {
    "coding-elite",
    "coding-smart",
    "coding-fast",
    "chat-elite",
    "chat-smart",
    "agent-oracle",
    "agent-build",
    "agent-explore",
    "agent-librarian",
    "agent-metis",
    "agent-momus",
    "glm5-elite",
}

# Non-tool chains (chat-fast, chat-rp, title-fast) — excluded when tools needed.

# Map MessageIntent -> chain name. Mirrors intent_detector.py's suggested_model
# values but centralized so we can reroute for tool capability.
INTENT_CHAIN_MAP: Dict[MessageIntent, str] = {
    MessageIntent.CODING_COMPLEX: "coding-elite",
    MessageIntent.CODING_FAST: "coding-fast",
    MessageIntent.COMPLEX_REASONING: "chat-smart",
    MessageIntent.FAST_CHAT: "chat-fast",
    MessageIntent.UNSENSORED: "chat-rp",
    MessageIntent.ROLEPLAY: "chat-rp",
    MessageIntent.UNKNOWN: "chat-fast",
}

# Default fallback when semantic_router is unavailable or confidence is low.
DEFAULT_CHAIN = "chat-fast"
# Safe tool-capable chain when request needs tools but intent routed to non-tool.
DEFAULT_TOOL_CHAIN = "coding-fast"


# ---------------------------------------------------------------------------
# semantic-router integration (lazy-loaded)
# ---------------------------------------------------------------------------

_SR_LAYER = None  # type: Optional[Any]
_SR_LOAD_FAILED = False
_SR_LOAD_ATTEMPTED = False


def _load_semantic_router():
    """Lazy-load semantic-router with FastEmbed local encoder.

    Returns the RouteLayer (or SemanticRouter) instance, or None if the
    library is not installed or loading failed. Failures are sticky so we
    don't retry on every request.
    """
    global _SR_LAYER, _SR_LOAD_FAILED, _SR_LOAD_ATTEMPTED
    if _SR_LOAD_ATTEMPTED:
        return _SR_LAYER
    _SR_LOAD_ATTEMPTED = True

    try:
        # ponytail: local extras only — no API keys, no network at runtime.
        from semantic_router import Route  # noqa: F401
        from semantic_router.encoders import FastEmbedEncoder
        from semantic_router.layer import RouteLayer  # type: ignore
    except ImportError:
        logger.info(
            "semantic-router not installed; 'auto' will use keyword intent_detector fallback"
        )
        _SR_LOAD_FAILED = True
        return None
    except Exception as exc:  # pragma: no cover - FastEmbed model download errors
        logger.warning(f"semantic-router load failed: {exc!r}; falling back to keywords")
        _SR_LOAD_FAILED = True
        return None

    try:
        encoder = FastEmbedEncoder()
        routes = _build_routes()
        _SR_LAYER = RouteLayer(encoder=encoder, routes=routes)
        logger.info(
            "semantic-router loaded with FastEmbed local encoder (%d routes)",
            len(routes),
        )
    except Exception as exc:  # pragma: no cover - encoder model download
        logger.warning(f"semantic-router init failed: {exc!r}; falling back to keywords")
        _SR_LAYER = None
        _SR_LOAD_FAILED = True
    return _SR_LAYER


def _build_routes() -> List[Any]:
    """Build Route objects with example utterances per intent.

    Utterances are short, diverse prompts that characterize each intent.
    Tunable; this is the seed corpus. #163 will expand this via telemetry.
    """
    from semantic_router import Route

    route_specs = {
        "coding_complex": [
            "Refactor this module to use async generators",
            "Design a rate-limited retry policy for an LLM gateway",
            "Why is my PostgreSQL query doing a sequential scan?",
            "Implement a binary search tree with deletion in Rust",
            "Review this PR for security issues and edge cases",
            "Write a unit test for the cooldown manager using asyncio.Lock",
            "Debug a memory leak in our asyncio event loop",
        ],
        "coding_fast": [
            "What does the @lru_cache decorator do?",
            "Convert this list comprehension to a for loop",
            "How do I split a string on whitespace in Python?",
            "What's the syntax for an f-string with a format spec?",
            "Show me a quick regex for matching email addresses",
        ],
        "complex_reasoning": [
            "Compare REST and gRPC for internal microservices",
            "What are the tradeoffs of eventual consistency?",
            "Analyze the implications of switching to a monorepo",
            "Explain the CAP theorem with examples",
            "Evaluate whether we should adopt SQLite or PostgreSQL",
            "What are the pros and cons of horizontal scaling?",
        ],
        "fast_chat": [
            "hi",
            "thanks!",
            "ok sure",
            "what time is it?",
            "who won the world series in 2016?",
            "what's the capital of France?",
        ],
        "roleplay": [
            "The dragon approaches, what do you do?",
            "Continue the story where we left off",
            "As the barista, greet the customer warmly",
            "*walks into the tavern and orders a mead*",
            "Let's roleplay a cyberpunk detective scene",
        ],
    }
    routes = []
    for name, utterances in route_specs.items():
        routes.append(Route(name=name, utterances=utterances))
    return routes


# Map semantic-router route names back to MessageIntent for chain lookup.
_ROUTE_NAME_TO_INTENT: Dict[str, MessageIntent] = {
    "coding_complex": MessageIntent.CODING_COMPLEX,
    "coding_fast": MessageIntent.CODING_FAST,
    "complex_reasoning": MessageIntent.COMPLEX_REASONING,
    "fast_chat": MessageIntent.FAST_CHAT,
    "roleplay": MessageIntent.ROLEPLAY,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class AutoRouteResult:
    """Result of resolving `model="auto"`."""

    chain: str
    intent: MessageIntent
    confidence: float
    source: str  # "semantic" | "keyword" | "default"
    reasoning: str


def resolve_auto(
    request: Dict[str, Any],
    current_model: Optional[str] = None,
) -> AutoRouteResult:
    """Resolve `model="auto"` to a concrete virtual-model chain.

    Steps:
    1. Try semantic-router (FastEmbed local). If confidence >= threshold, use it.
    2. Else fall back to keyword intent_detector.
    3. Apply tool-capability guard: if request has tools, force tool-capable chain.
    """
    threshold = float(os.getenv("AUTO_ROUTE_THRESHOLD", "0.30"))
    mode = os.getenv("AUTO_ROUTE_MODE", "all").lower()
    needs_tools = bool(request.get("tools") or request.get("tool_choice"))

    # Step 1: semantic-router
    if mode == "all":
        sr = _load_semantic_router()
        if sr is not None:
            query = _extract_query(request)
            if query:
                try:
                    decision = sr(query)
                    # getattr guard: semantic-router RouteChoice.score attr name
                    # varies across versions (.score / .similarity_score / None).
                    score = getattr(decision, "score", None) if decision else None
                    if decision is not None and score is not None:
                        confidence = float(score)
                        route_name = decision.name
                        intent = _ROUTE_NAME_TO_INTENT.get(route_name, MessageIntent.UNKNOWN)
                        if confidence >= threshold and intent != MessageIntent.UNKNOWN:
                            chain = INTENT_CHAIN_MAP[intent]
                            chain = _apply_tool_guard(chain, needs_tools)
                            return AutoRouteResult(
                                chain=chain,
                                intent=intent,
                                confidence=confidence,
                                source="semantic",
                                reasoning=f"semantic_router route={route_name} score={confidence:.3f}",
                            )
                        logger.debug(
                            "semantic_router low confidence (%.3f < %.3f) or unknown route %r; falling back",
                            confidence,
                            threshold,
                            route_name,
                        )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(f"semantic_router query failed: {exc!r}; falling back to keywords")

    # Step 2: keyword fallback
    messages = request.get("messages", [])
    kw_result: IntentResult = detect_intent(messages, current_model=current_model)
    chain = INTENT_CHAIN_MAP.get(kw_result.intent, DEFAULT_CHAIN)
    chain = _apply_tool_guard(chain, needs_tools)
    return AutoRouteResult(
        chain=chain,
        intent=kw_result.intent,
        confidence=kw_result.confidence,
        source="keyword",
        reasoning=f"keyword fallback: {kw_result.reasoning}",
    )


def _apply_tool_guard(chain: str, needs_tools: bool) -> str:
    """If the request needs tools and the chosen chain isn't tool-capable,
    reroute to a safe tool-capable default. Never silently serve a non-tool
    chain when the client asked for tools — that breaks OpenCode and any
    agent runtime."""
    if not needs_tools:
        return chain
    if chain in TOOL_CAPABLE_CHAINS:
        return chain
    logger.info(
        "auto tool-guard: chain %r is not tool-capable; rerouting to %r",
        chain,
        DEFAULT_TOOL_CHAIN,
    )
    return DEFAULT_TOOL_CHAIN


def _extract_query(request: Dict[str, Any]) -> str:
    """Pull the most recent user message text for embedding lookup.

    Truncated to 512 chars: FastEmbed all-MiniLM-L6-v2 has a 256-token context
    window; longer input is silently truncated by the tokenizer but the encoder
    still allocates/encodes the full string first. 512 chars ≈ 128 tokens, safe
    headroom under the limit and enough signal for intent classification.
    """
    messages = request.get("messages", []) or []
    # Walk backwards to find the last user text content.
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content[:512]
        if isinstance(content, list):
            parts = [
                (p.get("text", "") or "").strip()
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            text = " ".join(part for part in parts if part).strip()
            if text:
                return text[:512]
    return ""


# ---------------------------------------------------------------------------
# Self-test (run with: python -m proxy_app.semantic_router)
# ---------------------------------------------------------------------------


def _self_test() -> None:
    """Smoke test: route a few sample prompts and assert sane chains."""
    cases = [
        ({"messages": [{"role": "user", "content": "hi"}]}, False),
        ({"messages": [{"role": "user", "content": "def f(x): return x*2"}]}, False),
        (
            {
                "messages": [{"role": "user", "content": "list files in this dir"}],
                "tools": [{"type": "function", "function": {"name": "ls"}}],
            },
            True,
        ),
        (
            {
                "messages": [
                    {"role": "user", "content": "continue the story"}
                ],
                "tools": [{"type": "function", "function": {"name": "dice"}}],
            },
            True,
        ),
    ]
    for req, needs_tools_expected in cases:
        result = resolve_auto(req)
        assert result.chain, f"empty chain for {req}"
        if needs_tools_expected:
            assert result.chain in TOOL_CAPABLE_CHAINS, (
                f"tool guard failed: chain={result.chain} for {req}"
            )
        print(
            f"  source={result.source:8s} intent={result.intent.name:20s} "
            f"chain={result.chain:14s} conf={result.confidence:.2f}  "
            f"q={req['messages'][-1]['content'][:40]!r}"
        )
    print("self-test OK")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _self_test()
