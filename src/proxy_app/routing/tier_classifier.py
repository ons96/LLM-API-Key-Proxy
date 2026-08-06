"""Tier classifier + tier floors for smart routing (#476).

Five request tiers T0..T4 (trivial .. frontier), derived deterministically
from a RequestFeatures dataclass (no model calls, no I/O, microseconds).

Composition note: the keyword/math/question-density signals listed in #465
are already folded into task_class by request_features.extract_request_features
(code blocks, edit/gen verbs, math symbols, reasoning markers, ...). This
module maps task_class -> base tier, then escalates on length / tool count /
attachment count. Keeping the raw-text heuristics in the extractor avoids
re-parsing the prompt twice.

Tier floors (T1..T4) bound the minimum model capability score a candidate
must have to serve that tier. #465 wanted 0.5*tau3-Banking + 0.3*Terminal-
Bench 2.1 + 0.2*Intelligence Index; model_rankings.yaml has no AA composite
column, so composite_score/100 (aggregator composite, which feeds agentic
coding) is used as the documented proxy. Models with no score never satisfy
T1+ floors (they fail the floor check), per spec "no score -> T1 floor,
never T0".
"""

from __future__ import annotations

import os
from enum import IntEnum
from typing import Any, Dict, Optional

from .request_features import PromptBucket, RequestFeatures, TaskClass

# ---------------------------------------------------------------------------
# Tiers
# ---------------------------------------------------------------------------


class Tier(IntEnum):
    """Request complexity tier; higher = needs stronger/agentic models."""

    T0 = 0  # trivial: greeting, trivia
    T1 = 1  # simple: short qa, one-shot basics
    T2 = 2  # standard: code edit, summarization, vision caption
    T3 = 3  # hard: code gen, reasoning, file analysis
    T4 = 4  # frontier: agentic multi-step


TASK_TIER: Dict[TaskClass, Tier] = {
    TaskClass.GREETING: Tier.T0,
    TaskClass.SHORT_QA: Tier.T1,
    TaskClass.VISION_CAPTION: Tier.T2,
    TaskClass.SUMMARIZATION: Tier.T2,
    TaskClass.CODE_EDIT: Tier.T2,
    TaskClass.FILE_ANALYSIS: Tier.T3,
    TaskClass.CODE_GEN: Tier.T3,
    TaskClass.REASONING: Tier.T3,
    TaskClass.AGENTIC: Tier.T4,
}

# Escalation thresholds (chars, counts). Tuned constants; env-overridable.
_VERY_LONG_ESCALATE_CHARS = int(os.environ.get("TIER_VERY_LONG_CHARS", "8000"))
_LONG_ESCALATE_CHARS = int(os.environ.get("TIER_LONG_CHARS", "2000"))
_MANY_TOOLS = int(os.environ.get("TIER_MANY_TOOLS", "3"))
_MANY_ATTACHMENTS = int(os.environ.get("TIER_MANY_ATTACHMENTS", "2"))

# ---------------------------------------------------------------------------
# Tier floors (model capability score, composite/100 scale [0, 1])
# ---------------------------------------------------------------------------

TIER_FLOORS: Dict[Tier, float] = {
    Tier.T0: 0.0,
    Tier.T1: 0.25,
    Tier.T2: 0.40,
    Tier.T3: 0.55,
    Tier.T4: 0.70,
}

_DEFAULT_RANKINGS_PATH = os.environ.get(
    "MODEL_RANKINGS_PATH",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))),
        "config", "model_rankings.yaml",
    ),
)


def parse_tier_header(value: Optional[str]) -> Optional[Tier]:
    """Parse X-Route-Tier header ('T0'..'T4', '0'..'4'; case-insensitive).

    Invalid/unknown values return None so the caller falls back to heuristics.
    """
    if not value:
        return None
    v = value.strip().upper()
    if v.startswith("T") and v[1:].isdigit():
        v = v[1:]
    if not v.isdigit():
        return None
    n = int(v)
    if n in (t.value for t in Tier):
        return Tier(n)
    return None


def classify_request(
    features: RequestFeatures,
    header_override: Optional[str] = None,
) -> Tier:
    """Deterministic tier for a parsed request.

    Precedence: valid X-Route-Tier header wins, then task_class base tier,
    then length/tool/attachment escalations (capped at T4).
    """
    override = parse_tier_header(header_override)
    if override is not None:
        return override

    tier = TASK_TIER.get(features.task_class, Tier.T1)

    chars = features.text_char_count
    if chars >= _VERY_LONG_ESCALATE_CHARS and tier < Tier.T3:
        tier = Tier.T3
    elif chars >= _LONG_ESCALATE_CHARS and tier == Tier.T1:
        tier = Tier.T2

    if features.tool_count >= _MANY_TOOLS and tier < Tier.T3:
        tier = Tier.T3
    if features.image_count + features.file_count >= _MANY_ATTACHMENTS \
            and tier < Tier.T3:
        tier = Tier.T3
    if (features.image_count or features.file_count) and tier < Tier.T2:
        tier = Tier.T2

    return tier


# ---------------------------------------------------------------------------
# Model scores + floor checks (for chain filtering, consumed by #480)
# ---------------------------------------------------------------------------


def load_model_scores(rankings_path: Optional[str] = None) -> Dict[str, float]:
    """model_id ('provider/model') -> composite score on [0, 1].

    Reads config/model_rankings.yaml scores.composite_score (100-scale),
    normalized. Missing/unreadable file -> {} (callers treat as no floors).
    """
    path = rankings_path or _DEFAULT_RANKINGS_PATH
    if not os.path.exists(path):
        return {}
    try:
        import yaml  # runtime-only dep; kept out of the hot classify path
    except ImportError:
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    scores: Dict[str, float] = {}
    for entry in data.get("models", []) or []:
        mid = entry.get("id")
        if not isinstance(mid, str):
            continue
        comp = (entry.get("scores") or {}).get("composite_score")
        if isinstance(comp, (int, float)):
            scores[mid] = max(0.0, min(1.0, comp / 100.0))
    return scores


def model_meets_floor(score: float, tier: Tier) -> bool:
    """True if a model with capability `score` may serve `tier`.

    T0 has no floor; a model with no score (score <= 0) only ever meets T0.
    """
    if tier == Tier.T0:
        return True
    return score >= TIER_FLOORS[tier]


def _demo() -> None:
    """Self-check: run with PYTHONPATH=src python3 -m proxy_app.routing.tier_classifier"""
    from .request_features import Capabilities, extract_request_features

    cases = [
        ("greeting", {"messages": [{"role": "user", "content": "hi"}]}, Tier.T0),
        ("long code-gen", {"messages": [{"role": "user", "content": "write a full parser with error handling and streaming, " * 200}]}, Tier.T3),
        ("agentic", {"messages": [{"role": "user", "content": "first gather logs then cross-check then execute"}],
                     "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}] * 3}, Tier.T4),
        ("header override", {"messages": [{"role": "user", "content": "hi"}]}, Tier.T4, "T4"),
    ]
    for name, body, want, *rest in cases:
        feats = extract_request_features(body)
        got = classify_request(feats, rest[0] if rest else None)
        assert got == want, f"{name}: {got} != {want}"
    floors = load_model_scores()
    assert floors, "model_rankings.yaml not loaded"
    assert all(0.0 <= v <= 1.0 for v in floors.values()), "score range"
    print(f"tier_classifier OK ({len(floors)} scores, {len(cases)} cases)")


if __name__ == "__main__":
    _demo()
