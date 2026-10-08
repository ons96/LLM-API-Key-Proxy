"""Small deterministic P0 routing core with no third-party dependencies."""

ROUTER_VERSION = "0.1.0"

from dataclasses import dataclass
from enum import Enum
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
    config_fingerprint: str = "default"
    quota_remaining: int | None = None
    quota_class: str = "unknown"
    model: str = ""


class HealthRegistry:
    """In-memory operational cooldowns; capability policy remains separate."""

    def __init__(self, cooldown_seconds: float = 30.0) -> None:
        self.cooldown_seconds = cooldown_seconds
        self._until: dict[str, float] = {}

    def mark_failure(self, deployment_id: str, now: float | None = None) -> None:
        current = time.monotonic() if now is None else now
        self._until[deployment_id] = current + self.cooldown_seconds

    def is_healthy(self, deployment_id: str, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        return current >= self._until.get(deployment_id, 0.0)


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
    group = classify(features)
    candidates = [deployment for deployment in eligible(features, deployments, health) if deployment.capability == group]
    if not candidates:
        raise LookupError(f"no eligible deployment for {group.value}")
    return min(candidates, key=_score)


def _score(item: Deployment) -> float:
    """Prefer expected success time, with a soft penalty for exhausted quota."""
    quota_penalty = 10_000.0 if item.quota_remaining == 0 else 0.0
    return item.latency_ms / max(item.success_rate, 0.01) + quota_penalty


def select_with_stickiness(
    features: RequestFeatures,
    deployments: list[Deployment],
    preferred_id: str | None = None,
    health: HealthRegistry | None = None,
) -> Deployment:
    """Keep a compatible deployment when session state says to stay put."""
    candidates = eligible(features, deployments, health)
    group = classify(features)
    matching = [item for item in candidates if item.capability == group]
    if preferred_id:
        preferred = next((item for item in matching if item.deployment_id == preferred_id), None)
        if preferred is not None:
            return preferred
    if not matching:
        raise LookupError(f"no eligible deployment for {group.value}")
    return min(matching, key=_score)
