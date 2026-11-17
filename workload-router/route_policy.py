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
    """Policy knobs; `time_value` is lambda_T, the money/time exchange rate.

    Design v2 §2.1 defines one lambda_T in currency per second. It is a policy
    value the owner sets and never something the router learns, so code must not
    pick it. Left unset, the profile defaults apply (0.003 coding, 0.001 chat,
    which is what revision 2 scored); set, one value overrides both profiles.
    """

    coding_time_value: float = 0.003
    chat_time_value: float = 0.001
    quota_lambda: float = 0.01
    escalation_cost: float = 0.03
    top_n: int = 3
    time_value: float | None = None

    def __post_init__(self) -> None:
        for name in ("coding_time_value", "chat_time_value", "quota_lambda", "escalation_cost"):
            nonnegative(getattr(self, name), name)
        if self.time_value is not None:
            nonnegative(self.time_value, "time_value")
        if token_count(self.top_n, "top_n") < 1:
            raise ValueError("top_n must be positive")


@dataclass(frozen=True)
class Scored:
    """One route's cost vector, kept apart until the moment it is scored.

    `cost_money` is out-of-pocket `$`, `cost_opportunity` is the shadow value of
    quota and credits `O`, and `seconds` is wall-clock `T`. They are summed only
    inside `effective_money`, so a report can still read dollars, opportunity
    and time as three separate numbers (design v2 §2.1).
    """

    route: Route
    score: float
    cost_money: float
    cost_opportunity: float
    seconds: float
    failure_probability: float
    alpha: float
    beta: float
    pressure: float
    reason: str = "available"

    @property
    def effective_money(self) -> float:
        """The blended money term the score uses; never stored in this form."""
        return self.cost_money + self.cost_opportunity


def time_value_for(weights: Weights, profile: str) -> float:
    """lambda_T for a profile: the explicit override, else the default."""
    if weights.time_value is not None:
        return weights.time_value
    return weights.coding_time_value if profile == "auto-coding" else weights.chat_time_value


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
    # Split the cost vector (§2.1) without changing the ranking: the two terms
    # still add up to the single blended number revision 2 scored. Summing them
    # here, at decision time, is the only place they are ever combined.
    if route.quota_type == "credit_based":
        # Free credits have a zero customer price, so all of their cost is
        # opportunity: the notional credit tariff that depletes the allowance,
        # scaled by the route's shadow price. Customer-facing estimates still
        # use `money`, and the credit itself is stored as `credit_used`.
        cost_money = 0.0
        cost_opportunity = (route.credit_money(work.input_tokens, work.output_tokens)
                            * route.shadow_price + weights.quota_lambda * capacity.pressure)
    elif route.quota_type == "unlimited" and not (route.limits or route.provider_limits):
        cost_money = cost_opportunity = 0.0
    else:
        cost_money = money
        cost_opportunity = weights.quota_lambda * capacity.pressure
    effective = cost_money + cost_opportunity
    seconds = latency_ms / 1000
    if work.profile == "auto-coding" and preferred_id is not None and preferred_id != route.route_id:
        seconds += (route.handoff_ms + lost_cached_tokens * route.prefill_ms_per_token) / 1000
    time_value = time_value_for(weights, work.profile)
    score = (effective + seconds * time_value + failure * weights.escalation_cost) / (1 - failure)
    return Scored(route, score if reason == "available" else math.inf, cost_money,
                  cost_opportunity, seconds, failure, alpha, beta, capacity.pressure, reason)


def rank_routes(routes: list[Route], work: Work, ledger: QuotaLedger, *,
                free_only: bool = True, weights: Weights = Weights(),
                preferred_id: str | None = None, cache_tokens: dict[str, int] | None = None,
                explore: bool = True, rng: random.Random | None = None,
                include_infeasible: bool = False) -> list[Scored]:
    """Choose a sampled winner only from the top-N feasible cost candidates."""
    cache = cache_tokens or {}
    scored = [score_route(route, work, ledger, free_only=free_only, weights=weights,
                          preferred_id=preferred_id, cached_tokens=cache.get(route.route_id, 0),
                          lost_cached_tokens=cache.get(preferred_id or "", 0)) for route in routes]
    if include_infeasible:
        return sorted(scored, key=lambda item: (item.score, item.route.route_id))
    ordered = sorted((item for item in scored if math.isfinite(item.score)),
                     key=lambda item: (item.score, item.route.route_id))
    if explore and len(ordered) > 1:
        generator = rng or random
        value = time_value_for(weights, work.profile)

        def sampled(item: Scored) -> float:
            success = max(1e-9, generator.betavariate(item.alpha, item.beta))
            return (item.effective_money + item.seconds * value + (1 - success) * weights.escalation_cost) / success

        winner = min(ordered[:weights.top_n], key=sampled)
        ordered.remove(winner)
        ordered.insert(0, winner)
    return ordered


def is_priced(route: Route) -> bool:
    """True when a customer-facing price is non-zero, i.e. `$ > 0` (§6.5)."""
    return any((price or 0) > 0
               for price in (route.input_price, route.output_price, route.cached_input_price))


def free_only_audit(routes: list[Route], free_only: bool = True) -> dict:
    """Report the priced routes a free-only process excludes, without hiding them.

    Revision 3 asked for a startup error here. Priced routes are also the only
    thing `/v1/cost-estimate` can quote, and the scorer already makes them
    unreachable under `free_only`, so this is an audit for startup logging and the
    diagnostics endpoint instead of a boot failure that would take the sidecar
    down over one catalog row.
    """
    priced = [route.route_id for route in routes if is_priced(route)]
    return {"free_only": free_only,
            "priced_routes": priced if free_only else [],
            "excluded": len(priced) if free_only else 0}


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
