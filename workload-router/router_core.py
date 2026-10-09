"""Small deterministic P0 routing core with no third-party dependencies."""

ROUTER_VERSION = "0.1.0"

from dataclasses import dataclass, replace
from enum import Enum
import math
import time


class Capability(str, Enum):
    FAST_GENERAL = "fast_general"
    FAST_TOOL = "fast_tool"
    CODE_IMPLEMENT = "code_implement"
    CODE_VERIFY = "code_verify"
    CODE_DEBUG = "code_debug"
    DEEP_REASONING = "deep_reasoning"


PHASES = ("explore", "plan", "implement", "verify", "debug", "deep_debug")


def normalize_phase(value: str | None) -> str:
    """Normalize client phase hints to a bounded state vocabulary."""
    phase = (value or "").strip().lower().replace("-", "_")
    aliases = {"implementation": "implement", "verification": "verify", "deepdebug": "deep_debug"}
    phase = aliases.get(phase, phase)
    return phase if phase in PHASES else ""


def estimate_context_tokens(messages: list[dict[str, object]]) -> int:
    """Use a conservative, dependency-free character estimate for filtering."""
    characters = sum(len(str(message.get("content", ""))) for message in messages)
    return max(1, (characters + 3) // 4)


@dataclass(frozen=True)
class RequestFeatures:
    prompt: str
    context_tokens: int = 0
    tools: bool = False
    structured_output: bool = False
    phase: str = ""
    model_group: str = ""
    continuation: bool = False
    cached_tokens: int = 0


@dataclass(frozen=True)
class Deployment:
    deployment_id: str
    capability: Capability
    context_limit: int
    tools: bool = False
    structured_output: bool = False
    healthy: bool = True
    latency_ms: float = 1000.0
    success_rate: float = 0.5
    provider: str = "unconfigured"
    base_url: str = ""
    config_fingerprint: str = "default"
    quota_remaining: int | None = None
    quota_class: str = "unknown"
    model: str = ""
    group: str = ""
    chain_priority: int = 1000
    quota_reset_at: float | None = None
    cache_ttl_seconds: float = 600.0
    env_prefix: str = ""

    @property
    def chain_group(self) -> str:
        """Return the configured model group or the capability fallback group."""
        return self.group or self.capability.value


@dataclass(frozen=True)
class CacheHint:
    """Recent provider cache evidence used for a bounded routing decision."""

    deployment_id: str
    cached_tokens: int
    observed_at: float
    ttl_seconds: float = 600.0

    def is_fresh(self, now: float | None = None) -> bool:
        current = time.time() if now is None else now
        return self.cached_tokens > 0 and current - self.observed_at <= self.ttl_seconds


@dataclass(frozen=True)
class RoutePlan:
    """Ordered candidates and the reason for the first candidate."""

    candidates: tuple[Deployment, ...]
    reason: str
    cache_saved_ms: float = 0.0


DEFAULT_CACHE_MS_PER_TOKEN = 0.015
DEFAULT_CHAIN_STEP_PENALTY_MS = 25.0
DEFAULT_CACHE_MIN_TOKENS = 128
DEFAULT_MAX_WARM_CACHE_WAIT_SECONDS = 2.0


def finite_nonnegative(value: object, default: float | None = 0.0) -> float | None:
    """Return a finite non-negative float, or the safe caller-provided default."""
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(parsed):
        return default
    return max(0.0, parsed)


class HealthRegistry:
    """In-memory operational cooldowns; capability policy remains separate."""

    DEFAULT_COOLDOWNS = {
        "rate_limit": 30.0,
        "timeout": 5.0,
        "upstream_failure": 15.0,
        "network_error": 10.0,
        "invalid_response": 60.0,
        "empty_stream": 30.0,
        "provider_rejection": 60.0,
        "provider_error": 30.0,
    }

    def __init__(self, cooldown_seconds: float = 30.0) -> None:
        self.cooldown_seconds = cooldown_seconds
        self._until: dict[str, float] = {}
        self._error_class: dict[str, str] = {}

    def mark_failure(
        self,
        deployment_id: str,
        now: float | None = None,
        cooldown_seconds: float | None = None,
        error_class: str | None = None,
        retry_after_seconds: float | None = None,
        quota_reset_at: float | None = None,
    ) -> None:
        current = time.monotonic() if now is None else now
        base = finite_nonnegative(self.cooldown_seconds, 0.0)
        if cooldown_seconds is not None:
            base = finite_nonnegative(cooldown_seconds, base)
        if error_class is not None:
            base = self.DEFAULT_COOLDOWNS.get(error_class, base)
            self._error_class[deployment_id] = error_class
        retry_after = finite_nonnegative(retry_after_seconds, None)
        if retry_after is not None:
            base = max(base, retry_after)
        quota_reset = finite_nonnegative(quota_reset_at, None)
        if quota_reset is not None:
            reset_delay = quota_reset - time.time()
            if math.isfinite(reset_delay):
                base = max(base, reset_delay)
        self._until[deployment_id] = current + max(0.0, base)

    def mark_success(self, deployment_id: str) -> None:
        """Clear an operational cooldown after a deployment recovers."""
        self._until.pop(deployment_id, None)
        self._error_class.pop(deployment_id, None)

    def is_healthy(self, deployment_id: str, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        return current >= self._until.get(deployment_id, 0.0)

    def retry_after(self, deployment_id: str, now: float | None = None) -> float:
        """Return remaining cooldown seconds for response headers and telemetry."""
        current = time.monotonic() if now is None else now
        return max(0.0, self._until.get(deployment_id, 0.0) - current)

    def error_class(self, deployment_id: str) -> str | None:
        return self._error_class.get(deployment_id)


def classify(features: RequestFeatures) -> Capability:
    """Classify obvious requests using bounded metadata and lexical signals."""
    text = features.prompt.lower()
    if features.phase == "implement":
        return Capability.CODE_IMPLEMENT
    if features.phase in {"verify", "debug"} or any(
        word in text for word in ("test", "pytest", "compile", "lint", "failing")
    ):
        return Capability.CODE_VERIFY if features.phase == "verify" else Capability.CODE_DEBUG
    if features.tools:
        return Capability.FAST_TOOL
    if any(word in text for word in ("implement", "refactor", "fix the code", "write code")):
        return Capability.CODE_IMPLEMENT
    if any(word in text for word in ("prove", "derive", "analyze deeply", "step by step")):
        return Capability.DEEP_REASONING
    return Capability.FAST_GENERAL


def group_for(features: RequestFeatures) -> str:
    """Resolve an explicit model group before applying deterministic classification."""
    return features.model_group or classify(features).value


def eligible(features: RequestFeatures, deployments: list[Deployment], health: HealthRegistry | None = None) -> list[Deployment]:
    """Apply hard requirements before any preference scoring."""
    return [
        deployment
        for deployment in deployments
        if deployment.healthy
        and (health is None or health.is_healthy(deployment.deployment_id))
        and features.context_tokens <= deployment.context_limit
        and (not features.tools or deployment.tools)
        and (not features.structured_output or deployment.structured_output)
    ]


def select(features: RequestFeatures, deployments: list[Deployment], health: HealthRegistry | None = None) -> Deployment:
    """Return the fastest reliable eligible deployment in the capability group."""
    group = group_for(features)
    candidates = [deployment for deployment in eligible(features, deployments, health) if deployment.chain_group == group or (not deployment.group and deployment.capability.value == group)]
    if not candidates and not features.model_group:
        candidates = [deployment for deployment in eligible(features, deployments, health) if deployment.capability.value == classify(features).value]
    if not candidates:
        raise LookupError(f"no eligible deployment for {group}")
    return min(candidates, key=_score)


def _score(item: Deployment) -> float:
    """Prefer expected success time, with a soft penalty for exhausted quota."""
    quota_penalty = 10_000.0 if item.quota_remaining == 0 else 0.0
    if item.quota_reset_at is not None and item.quota_reset_at > time.time():
        quota_penalty += 10_000.0
    return item.latency_ms / max(item.success_rate, 0.01) + quota_penalty


def _matching_candidates(
    features: RequestFeatures,
    deployments: list[Deployment],
    health: HealthRegistry | None,
) -> list[Deployment]:
    group = group_for(features)
    candidates = [
        deployment
        for deployment in eligible(features, deployments, health)
        if deployment.chain_group == group or (not deployment.group and deployment.capability.value == group)
    ]
    if not candidates and not features.model_group:
        candidates = [
            deployment
            for deployment in eligible(features, deployments, health)
            if deployment.capability.value == classify(features).value
        ]
    return candidates


def route_plan(
    features: RequestFeatures,
    deployments: list[Deployment],
    preferred_id: str | None = None,
    health: HealthRegistry | None = None,
    cache_hints: dict[str, CacheHint] | None = None,
    now: float | None = None,
    cache_ms_per_token: float = DEFAULT_CACHE_MS_PER_TOKEN,
    chain_step_penalty_ms: float = DEFAULT_CHAIN_STEP_PENALTY_MS,
    cache_min_tokens: int = DEFAULT_CACHE_MIN_TOKENS,
) -> RoutePlan:
    """Build a deterministic chain order with bounded cache-aware stickiness.

    New requests start at the lowest chain priority. Continuations retain the
    compatible deployment. A fresh request may reuse a lower-ranked warm cache
    only when the estimated cache savings exceed the chain-position penalty.
    Provider failures are handled by the execution layer, which consumes this
    ordered list without waiting synchronously for a cooldown to expire.
    """
    candidates = _matching_candidates(features, deployments, health)
    if not candidates:
        raise LookupError(f"no eligible deployment for {group_for(features)}")
    ordered = sorted(candidates, key=lambda item: (item.chain_priority, _score(item), item.deployment_id))
    if preferred_id and features.continuation:
        preferred = next((item for item in ordered if item.deployment_id == preferred_id), None)
        if preferred is not None:
            ordered = [preferred] + [item for item in ordered if item.deployment_id != preferred_id]
            return RoutePlan(tuple(ordered), "continuation_sticky")
    if not cache_hints:
        return RoutePlan(tuple(ordered), "chain_top")
    current = time.time() if now is None else now
    top = ordered[0]
    warm = [
        item for item in ordered
        if item.deployment_id in cache_hints
        and cache_hints[item.deployment_id].is_fresh(current)
        and cache_hints[item.deployment_id].cached_tokens >= cache_min_tokens
    ]
    if not warm:
        return RoutePlan(tuple(ordered), "chain_top")
    warm.sort(key=lambda item: (item.chain_priority, _score(item), item.deployment_id))
    cached = warm[0]
    if cached.deployment_id == top.deployment_id:
        return RoutePlan(tuple(ordered), "chain_top_warm_cache")
    cached_tokens = cache_hints[cached.deployment_id].cached_tokens
    saved_ms = cached_tokens * cache_ms_per_token
    rank_penalty = max(0, cached.chain_priority - top.chain_priority) * chain_step_penalty_ms
    if saved_ms >= rank_penalty:
        ordered = [cached] + [item for item in ordered if item.deployment_id != cached.deployment_id]
        return RoutePlan(tuple(ordered), "warm_cache_override", saved_ms)
    return RoutePlan(tuple(ordered), "chain_top_cache_below_break_even", saved_ms)


def warm_cache_wait_seconds(
    error_class: str,
    retry_after_seconds: float | None,
    hint: CacheHint | None,
    *,
    now: float | None = None,
    cache_ms_per_token: float = DEFAULT_CACHE_MS_PER_TOKEN,
    chain_step_penalty_ms: float = DEFAULT_CHAIN_STEP_PENALTY_MS,
    cache_min_tokens: int = DEFAULT_CACHE_MIN_TOKENS,
    max_wait_seconds: float = DEFAULT_MAX_WARM_CACHE_WAIT_SECONDS,
) -> float:
    """Return a bounded wait for a warm rate-limited deployment, else zero.

    Waiting is deliberately narrower than failover: only a short rate-limit
    hint and a fresh cache with enough estimated savings can justify keeping
    the provider. Timeout and upstream failures immediately use the chain.
    """
    if error_class != "rate_limit" or retry_after_seconds is None:
        return 0.0
    wait = max(0.0, retry_after_seconds)
    if wait <= 0 or wait > max_wait_seconds or hint is None:
        return 0.0
    current = time.time() if now is None else now
    if not hint.is_fresh(current) or hint.cached_tokens < cache_min_tokens:
        return 0.0
    cache_savings = hint.cached_tokens * cache_ms_per_token
    if cache_savings < wait * 1000.0 + chain_step_penalty_ms:
        return 0.0
    return wait


def select_with_stickiness(
    features: RequestFeatures,
    deployments: list[Deployment],
    preferred_id: str | None = None,
    health: HealthRegistry | None = None,
    cache_hints: dict[str, CacheHint] | None = None,
) -> Deployment:
    """Keep a compatible deployment when session state says to stay put."""
    # Preserve the legacy helper's explicit sticky contract.  The HTTP route
    # path calls route_plan directly and only sets continuation when the client
    # confirms this is a continuation request.
    sticky_features = replace(features, continuation=True)
    return route_plan(sticky_features, deployments, preferred_id, health, cache_hints).candidates[0]
