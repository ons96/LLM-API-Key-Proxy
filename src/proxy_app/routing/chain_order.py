"""Lexicographic chain ordering.

Ordering is not a weighted sum. Hard tiers come first and the score only breaks
ties *inside* a tier, because a cheap tier-0 model must never outrank a tier-1
model merely by scoring well -- that is how a one-time credit gets spent.

Order, in this exact sequence:
  1. capability filter (hard: tools / vision / json_schema / context window)
  2. quota tier (0 unlimited, 1 recurring, 2 rpd cap, 3 one-time, 4 paid=excluded)
  3. quality        benchmark prior blended with observed success by sample count
  4. speed          observed TPS, with a prior for unsampled models
  5. reliability    Wilson lower bound on success, shrunk to 0.5 when n < 5
  6. freshness      recently verified beats stale
  7. (provider, model) as the final tiebreak, so output is byte-identical

Determinism is a feature: the same catalog and the same telemetry produce the
same bytes, so a routing decision can be replayed and diffed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

PAID_TIER = 4
MIN_SAMPLES_FOR_TELEMETRY = 5


class CapabilityError(Exception):
    """No candidate satisfied a hard capability requirement."""

    error_type = "capability_mismatch"
    status_code = 400

    def __init__(self, requirement: str, detail: str):
        self.requirement = requirement
        self.detail = detail
        super().__init__(f"no candidate satisfies {requirement}: {detail}")


@dataclass(frozen=True)
class Requirements:
    needs_tools: bool = False
    needs_vision: bool = False
    needs_json_schema: bool = False
    min_context_tokens: int = 0

    def describe(self) -> str:
        parts = []
        if self.needs_tools:
            parts.append("tools")
        if self.needs_vision:
            parts.append("vision")
        if self.needs_json_schema:
            parts.append("json_schema")
        if self.min_context_tokens:
            parts.append(f"context>={self.min_context_tokens}")
        return ", ".join(parts) or "none"


@dataclass
class Candidate:
    provider: str
    model: str
    quota_tier: int = 0
    quality_score: Optional[float] = None  # 0..100 benchmark composite
    speed_tps: Optional[float] = None
    observed_ttft_ms: Optional[float] = None
    samples: int = 0
    successes: int = 0
    context_window: Optional[int] = None
    capabilities: set[str] = field(default_factory=set)
    last_ok_at: Optional[float] = None
    pinned: bool = False

    def supports(self, capability: str) -> bool:
        if not self.capabilities:
            # Undeclared capabilities are unknown, not "yes": a tools-required
            # request must not land on a model nobody declared tool-capable.
            return False
        aliases = {
            "tools": {"tools", "function_calling", "tool_use"},
            "vision": {"vision", "image", "multimodal"},
            "json_schema": {"json_schema", "structured_output", "json"},
        }.get(capability, {capability})
        return bool(self.capabilities & aliases)


def wilson_lower_bound(successes: int, n: int, z: float = 1.96) -> float:
    """Wilson score lower bound. Handles small n without exploding to 0 or 1."""
    if n <= 0:
        return 0.0
    # Telemetry can report more successes than samples (retries, partial rows).
    # Clamp rather than raise: a broken counter must not break routing.
    successes = max(0, min(successes, n))
    phat = successes / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return max(0.0, (centre - margin) / denom)


def reliability(candidate: Candidate) -> float:
    """Observed reliability, shrunk toward 0.5 when there is little evidence."""
    if candidate.samples <= 0:
        return 0.5
    bound = wilson_lower_bound(candidate.successes, candidate.samples)
    if candidate.samples >= MIN_SAMPLES_FOR_TELEMETRY:
        return bound
    # Shrink small samples toward the neutral prior.
    weight = candidate.samples / MIN_SAMPLES_FOR_TELEMETRY
    return weight * bound + (1 - weight) * 0.5


def effective_quality(candidate: Candidate, prior: float = 50.0) -> float:
    """Benchmark prior, blended with observed reliability as evidence accrues.

    This is the sparse-telemetry rule: an unsampled top-quartile benchmark model
    must not sink below a well-sampled weak one just because it has no history.
    """
    benchmark = candidate.quality_score if candidate.quality_score is not None else prior
    if candidate.samples < MIN_SAMPLES_FOR_TELEMETRY:
        return benchmark
    observed = candidate.quality_score if candidate.quality_score is not None else prior
    # Telemetry nudges, it does not replace the benchmark.
    return observed * reliability(candidate) ** 0.15


def effective_speed(candidate: Candidate, prior_tps: float = 20.0) -> float:
    if candidate.speed_tps and candidate.speed_tps > 0:
        return float(candidate.speed_tps)
    if candidate.observed_ttft_ms and candidate.observed_ttft_ms > 0:
        return 1000.0 / candidate.observed_ttft_ms
    return prior_tps


def _capability_failures(
    candidate: Candidate, requirements: Requirements
) -> list[str]:
    failures: list[str] = []
    if requirements.needs_tools and not candidate.supports("tools"):
        failures.append("tools")
    if requirements.needs_vision and not candidate.supports("vision"):
        failures.append("vision")
    if requirements.needs_json_schema and not candidate.supports("json_schema"):
        failures.append("json_schema")
    if requirements.min_context_tokens:
        window = candidate.context_window or 0
        if window < requirements.min_context_tokens:
            failures.append(
                f"context {window} < required {requirements.min_context_tokens}"
            )
    return failures


def rank(
    candidates: Iterable[Candidate],
    requirements: Requirements | None = None,
    *,
    exclude_paid: bool = True,
    allow_one_time: bool = False,
) -> list[Candidate]:
    """Order candidates. Raises CapabilityError when nothing satisfies the request.

    ``allow_one_time`` is False for automatic routing: tier 3 is reserved and only
    reached when the caller explicitly pins it, or when nothing else is eligible.
    """
    requirements = requirements or Requirements()
    eligible: list[Candidate] = []
    rejected_for_capability: list[str] = []

    for candidate in candidates:
        if exclude_paid and candidate.quota_tier >= PAID_TIER:
            continue
        failures = _capability_failures(candidate, requirements)
        if failures:
            rejected_for_capability.append(
                f"{candidate.provider}/{candidate.model}: {', '.join(failures)}"
            )
            continue
        eligible.append(candidate)

    if not eligible and rejected_for_capability and (
        requirements.needs_tools
        or requirements.needs_vision
        or requirements.needs_json_schema
        or requirements.min_context_tokens
    ):
        raise CapabilityError(requirements.describe(), "; ".join(rejected_for_capability[:5]))

    def sort_key(candidate: Candidate) -> tuple:
        tier = candidate.quota_tier
        # Pinned candidates float to the head of their tier, never above tier 0/1
        # ordering: a pin is a preference, not an exemption from the tier ladder.
        pin_bucket = 0 if candidate.pinned else 1
        freshness = -(candidate.last_ok_at or 0.0)
        return (
            tier,
            pin_bucket,
            -effective_quality(candidate),
            -effective_speed(candidate),
            -reliability(candidate),
            freshness,
            candidate.provider,
            candidate.model,
        )

    ordered = sorted(eligible, key=sort_key)

    if not allow_one_time:
        renewing = [c for c in ordered if c.quota_tier < 3]
        one_time = [c for c in ordered if c.quota_tier >= 3]
        # Tier 3 stays at the back, in its own tier order.
        ordered = renewing + one_time

    return ordered


def to_dicts(candidates: Iterable[Candidate]) -> list[dict[str, Any]]:
    """Stable serialisation used by the determinism test and by trace replay."""
    return [
        {
            "provider": c.provider,
            "model": c.model,
            "quota_tier": c.quota_tier,
            "pinned": c.pinned,
        }
        for c in candidates
    ]
