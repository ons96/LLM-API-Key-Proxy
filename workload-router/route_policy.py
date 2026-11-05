"""Hard feasibility gates, three-tier costs and top-N Thompson exploration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
import json
import math
import random

from quota_ledger import QuotaLedger, Route, nonnegative, token_count


STEPS = {"chat", "explore", "plan", "implement", "verify", "debug",
         "deep-debug", "final-review", "post-failure-debug"}


@dataclass(frozen=True)
class Work:
    profile: str
    step: str
    input_tokens: int
    output_tokens: int
    tools: bool = False
    structured_output: bool = False
    web_search: bool = False
    streaming: bool = False
    session_id: str = ""

    def __post_init__(self) -> None:
        if self.profile not in {"auto-coding", "auto-chat"} or self.step not in STEPS:
            raise ValueError("unknown profile or step")
        token_count(self.input_tokens, "input_tokens")
        if token_count(self.output_tokens, "output_tokens") < 1:
            raise ValueError("output_tokens must be positive")


@dataclass(frozen=True)
class Weights:
    coding_time_value: float = 0.003
    chat_time_value: float = 0.001
    quota_lambda: float = 0.01
    escalation_cost: float = 0.03
    top_n: int = 3

    def __post_init__(self) -> None:
        for name in ("coding_time_value", "chat_time_value", "quota_lambda", "escalation_cost"):
            nonnegative(getattr(self, name), name)
        if token_count(self.top_n, "top_n") < 1:
            raise ValueError("top_n must be positive")


@dataclass(frozen=True)
class Scored:
    route: Route
    score: float
    effective_money: float
    seconds: float
    failure_probability: float
    alpha: float
    beta: float
    pressure: float
    reason: str = "available"


def score_route(route: Route, work: Work, ledger: QuotaLedger, *,
                free_only: bool = True, weights: Weights = Weights(),
                preferred_id: str | None = None, cached_tokens: int = 0,
                lost_cached_tokens: int = 0) -> Scored:
    """Money + elapsed time + escalation, divided by probability of success."""
    reason = "available"
    if work.profile not in route.profiles:
        reason = "profile"
    elif route.context_window is None or route.output_limit is None:
        reason = "unknown_context_or_output_limit"
    elif work.input_tokens + work.output_tokens > route.context_window or work.output_tokens > route.output_limit:
        reason = "context_or_output_limit"
    elif work.tools and route.tool_call is not True:
        reason = "tool_support"
    elif work.streaming and route.stream is not True:
        reason = "stream_support"
    elif work.structured_output and route.structured_output is not True:
        reason = "schema_support"
    elif work.web_search and route.web_search is not True:
        reason = "web_search_support"
    elif free_only and route.free_class == "paid":
        reason = "paid_excluded"
    elif route.quota_type in {"unlimited", "rate_limited"} and (route.input_price != 0 or route.output_price != 0):
        reason = "free_price_not_verified"
    capacity = ledger.capacity(route, work.input_tokens, work.output_tokens)
    if reason == "available" and not capacity.feasible:
        reason = capacity.reason
    alpha, beta, latency_ms = ledger.evidence(route, work.step)
    failure = beta / (alpha + beta)
    cached = min(work.input_tokens, max(0, cached_tokens)) if work.profile == "auto-coding" else 0
    money = route.money(work.input_tokens, work.output_tokens, cached)
    if route.quota_type == "credit_based":
        # Free credits have a zero customer price. Their notional credit value
        # is what makes one-time allowances deplete and rank below unlimited
        # routes; the customer-facing cost estimate still uses `money`.
        effective = (route.credit_money(work.input_tokens, work.output_tokens)
                     * route.shadow_price + weights.quota_lambda * capacity.pressure)
    elif route.quota_type == "unlimited" and not (route.limits or route.provider_limits):
        effective = 0.0
    else:
        effective = money + weights.quota_lambda * capacity.pressure
    seconds = latency_ms / 1000
    if work.profile == "auto-coding" and preferred_id is not None and preferred_id != route.route_id:
        seconds += (route.handoff_ms + lost_cached_tokens * route.prefill_ms_per_token) / 1000
    time_value = weights.coding_time_value if work.profile == "auto-coding" else weights.chat_time_value
    score = (effective + seconds * time_value + failure * weights.escalation_cost) / (1 - failure)
    return Scored(route, score if reason == "available" else math.inf, effective,
                  seconds, failure, alpha, beta, capacity.pressure, reason)


def rank_routes(routes: list[Route], work: Work, ledger: QuotaLedger, *,
                free_only: bool = True, weights: Weights = Weights(),
                preferred_id: str | None = None, cache_tokens: dict[str, int] | None = None,
                explore: bool = True, rng: random.Random | None = None) -> list[Scored]:
    """Choose a sampled winner only from the top-N feasible cost candidates."""
    cache = cache_tokens or {}
    scored = [score_route(route, work, ledger, free_only=free_only, weights=weights,
                          preferred_id=preferred_id, cached_tokens=cache.get(route.route_id, 0),
                          lost_cached_tokens=cache.get(preferred_id or "", 0)) for route in routes]
    ordered = sorted((item for item in scored if math.isfinite(item.score)),
                     key=lambda item: (item.score, item.route.route_id))
    if explore and len(ordered) > 1:
        generator = rng or random
        value = weights.coding_time_value if work.profile == "auto-coding" else weights.chat_time_value

        def sampled(item: Scored) -> float:
            success = max(1e-9, generator.betavariate(item.alpha, item.beta))
            return (item.effective_money + item.seconds * value + (1 - success) * weights.escalation_cost) / success

        winner = min(ordered[:weights.top_n], key=sampled)
        ordered.remove(winner)
        ordered.insert(0, winner)
    return ordered


def input_token_bound(payload: dict) -> int:
    """A UTF-8 byte bound, not a fabricated tokenizer count; includes tools/schema.

    Text-only requests use at most one token per encoded byte plus framing.
    Multimodal token accounting needs a provider tokenizer and is rejected here.
    """
    messages = payload["messages"]
    for message in messages:
        if message.get("content") is not None and not isinstance(message["content"], str):
            raise ValueError("multimodal content requires an explicit tokenizer")
    fields = {key: value for key, value in payload.items()
              if key not in {"model", "stream", "max_tokens", "max_completion_tokens",
                             "context_tokens", "cached_tokens", "continuation"}}
    size = len(json.dumps(fields, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    return size + len(messages) * 16 + 64


def compact_json_tools(payload: dict, enabled: bool = False) -> tuple[dict, int, int]:
    """Opt-in whitespace-only JSON tool compaction, preserving numeric spellings.

    Duplicate keys, non-standard constants and non-JSON text stay byte-identical.
    This preserves JSON values, not byte-exact tool-output formatting.
    """
    baseline = input_token_bound(payload)
    if not enabled:
        return payload, baseline, baseline
    result = deepcopy(payload)

    def unique(pairs):
        keys = [key for key, _ in pairs]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate JSON keys")
        return dict(pairs)

    def invalid_constant(_):
        raise ValueError("non-standard JSON constant")

    for message in result["messages"]:
        source = message.get("content")
        if message.get("role") != "tool" or not isinstance(source, str):
            continue
        try:
            value = json.loads(source, object_pairs_hook=unique, parse_float=Decimal,
                               parse_constant=invalid_constant)
            if not isinstance(value, (dict, list)):
                continue
        except (ValueError, RecursionError):
            continue
        quoted = escaped = False
        compact = []
        for char in source:
            if quoted or char not in " \t\r\n":
                compact.append(char)
            if escaped:
                escaped = False
            elif quoted and char == "\\":
                escaped = True
            elif char == '"':
                quoted = not quoted
        message["content"] = "".join(compact)
    return result, baseline, input_token_bound(result)
