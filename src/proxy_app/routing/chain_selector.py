"""Per-request chain selection for smart routing (#480).

Coordinates sibling routing modules (request_features, tier_classifier,
latency_predictor, output_estimator) to build the per-request fallback chain:
  [fastest capable candidate, 2nd, ...] + escalation pool (tier-above models).

Pipeline (pure, deterministic, no I/O inside select()):
  1. pins first  (config/chain_pins.yaml style, #472)
  2. capability mask filter   (request must be satisfiable)
  3. tier floor filter        (model score >= floor for request tier)
  4. E[time] sort via LatencyPredictor; confidence < 0.7 for ALL candidates
     -> static fallback (input order preserved, no-regression guarantee)
  5. escalation pool: candidates whose tier floor is ABOVE the request tier
  6. sticky sessions: escalate-only stickiness per session key (#465)

Inert module: wired into USE_DYNAMIC_CHAIN middleware by #481.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .latency_predictor import LatencyPredictor, Prediction
from .request_features import Capabilities, RequestFeatures
from .tier_classifier import Tier, model_meets_floor

# ---------------------------------------------------------------------------
# Static tables
# ---------------------------------------------------------------------------

# Confidence threshold: below this a prediction is advisory only, and the
# selector falls back to the static chain (AC3 no-regression).
CONFIDENT_CONFIDENCE = 0.7
# Session stickiness TTL + cap (escalate-only, #465).
STICKY_TTL_S = 600.0
STICKY_MAX_ENTRIES = 512

# Capability keywords found in providers_database.yaml / override files.
_TOOL_KEYWORDS = ("tools", "tool", "function_calling", "functions")
_VISION_KEYWORDS = ("vision", "multimodal", "image", "images")
_FILE_KEYWORDS = ("document", "documents", "file", "files", "file_parsing")

# Name-heuristic fallbacks when no capability metadata is known.
_VISION_NAME_RE = re.compile(
    r"vl|vision|vlm|omni|multimodal|llava|pixtral|internvl|glm-4v|glm-4.1v|"
    r"minicpm-v|phi-3\.5-vision|phi-4-vision|idefics|moondream|fuyu|paligemma|"
    r"qwen2\.5-vl|qwen2-vl|qwen3-vl|cogvlm|qwen-vl",
    re.IGNORECASE,
)
_MEDIA_ONLY_RE = re.compile(
    r"embed|rerank|tts|stt|asr|whisper|dall-e|dalle|flux|sdxl|stable-diffusion|"
    r"image-gen|text-to-image|ttv|transcribe|audio",
    re.IGNORECASE,
)

# Transient statuses -> try next candidate. Capability/format statuses ->
# escalate. Matches error policy in the issue spec.
TRANSIENT_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
CAPABILITY_STATUSES = frozenset({400, 404, 422})
_CAPABILITY_HINTS = ("context", "capabilit", "unsupported", "modality", "format")


@dataclass(frozen=True)
class ChainCandidate:
    """One entry of the selected chain."""

    provider: str
    model: str
    priority: int
    capabilities: Capabilities = Capabilities.TEXT
    score: float = 0.0
    e_time_ms: Optional[int] = None
    confidence: float = 0.0
    source: str = "static"
    reason: str = ""


@dataclass
class SelectResult:
    chain: List[ChainCandidate] = field(default_factory=list)
    escalation: List[ChainCandidate] = field(default_factory=list)
    static_fallback: bool = False
    header: str = ""


# ---------------------------------------------------------------------------
# Loaders (file I/O lives here, not in select())
# ---------------------------------------------------------------------------


def load_virtual_chain(virtual_models_path: str, virtual_model_id: str) -> List[Dict]:
    """Return fallback_chain entries for one virtual model (mirror reorder_chains)."""
    try:
        import yaml

        with open(virtual_models_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        chain = (
            data.get("virtual_models", {}).get(virtual_model_id, {}).get("fallback_chain", [])
        )
        return [e for e in chain if isinstance(e, dict) and e.get("provider") and e.get("model")]
    except (OSError, ValueError, AttributeError):
        return []


def load_capabilities(
    providers_db_path: Optional[str] = None,
    overrides_path: Optional[str] = None,
) -> Dict[str, Capabilities]:
    """Build provider/model -> Capabilities map.

    Cascade: overrides file wins, then providers_database.yaml per-model +
    provider-level (merged into every model of that provider). Missing files
    return {} and callers fall back to name heuristics.
    """
    result: Dict[str, Capabilities] = {}

    def _caps_from_keywords(keywords: Sequence[str]) -> Capabilities:
        caps = Capabilities.TEXT
        lowered = " ".join(k.lower() for k in keywords)
        if any(k in lowered for k in _TOOL_KEYWORDS):
            caps |= Capabilities.TOOL_CALLING
        if any(k in lowered for k in _VISION_KEYWORDS):
            caps |= Capabilities.VISION
        if any(k in lowered for k in _FILE_KEYWORDS):
            caps |= Capabilities.FILE_PARSING
        return caps

    try:
        import yaml
    except ImportError:
        return result

    if overrides_path:
        try:
            with open(overrides_path, "r", encoding="utf-8") as fh:
                overrides = yaml.safe_load(fh) or {}
            for key, value in overrides.items():
                if isinstance(value, list):
                    result[str(key)] = _caps_from_keywords(value)
        except (OSError, ValueError, AttributeError):
            pass

    if providers_db_path:
        try:
            with open(providers_db_path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            for prov in data.get("providers", []):
                if not isinstance(prov, dict):
                    continue
                pid = prov.get("id")
                if not pid:
                    continue
                prov_caps = _caps_from_keywords(prov.get("capabilities", []))
                for fm in prov.get("free_models", []) or []:
                    if not isinstance(fm, dict) or not fm.get("id"):
                        continue
                    mid = f"{pid}/{fm['id']}"
                    caps = _caps_from_keywords(fm.get("capabilities", [])) | prov_caps
                    result[mid] = caps
        except (OSError, ValueError, AttributeError):
            pass

    return result


def _heuristic_capabilities(provider: str, model: str) -> Capabilities:
    """Name-based capability guess, used only when metadata is unknown."""
    name = f"{provider} {model}"
    if _MEDIA_ONLY_RE.search(name):
        return Capabilities(0)  # embeddings/tts/etc: not chat-capable
    caps = Capabilities.TEXT
    if _VISION_NAME_RE.search(name):
        caps |= Capabilities.VISION
    return caps


# ---------------------------------------------------------------------------
# Pure selection logic
# ---------------------------------------------------------------------------


def on_error(
    chain: Sequence[ChainCandidate],
    escalation: Sequence[ChainCandidate],
    status: Optional[int],
    err_text: str = "",
) -> Optional[ChainCandidate]:
    """Next candidate after a failed attempt, per #480 error policy.

    Capability error (400/404/422 or capability hints in body) -> escalate.
    Transient (429/5xx/timeout) -> next in chain, else escalate. Else None.
    """
    err_text = (err_text or "").lower()
    if status in CAPABILITY_STATUSES or any(h in err_text for h in _CAPABILITY_HINTS):
        return escalation[0] if escalation else None
    if status in TRANSIENT_STATUSES or "timeout" in err_text or "timed out" in err_text:
        if len(chain) > 1:
            return chain[1]
        return escalation[0] if escalation else None
    return None


class ChainSelector:
    """Per-request chain builder. select() is pure (no I/O)."""

    def __init__(
        self,
        model_scores: Optional[Dict[str, float]] = None,
        capabilities: Optional[Dict[str, Capabilities]] = None,
        predictor: Optional[LatencyPredictor] = None,
        pins: Optional[Sequence[Dict]] = None,
    ) -> None:
        self._model_scores = model_scores or {}
        self._capabilities = capabilities or {}
        self._predictor = predictor
        self._pins = list(pins or [])
        # session_key -> (tier, provider, model, timestamp)  # escalate-only
        self._sticky: Dict[str, Tuple[Tier, str, str, float]] = {}

    # -- helpers -----------------------------------------------------------

    def _candidate_caps(self, provider: str, model: str) -> Capabilities:
        return self._capabilities.get(f"{provider}/{model}") or _heuristic_capabilities(
            provider, model
        )

    def _score(self, provider: str, model: str) -> float:
        return self._model_scores.get(f"{provider}/{model}", 0.0)

    # -- main entry --------------------------------------------------------

    def select(
        self,
        features: RequestFeatures,
        tier: Tier,
        candidates: Sequence[Dict],
        session_key: Optional[str] = None,
        output_tokens: int = 200,
        now: Optional[float] = None,
    ) -> SelectResult:
        now = time.time() if now is None else now
        required = features.capabilities if features.capabilities else Capabilities.TEXT
        result = SelectResult()

        # 1. Expand input entries into candidates.
        entries: List[Dict] = []
        for idx, entry in enumerate(candidates):
            entries.append(
                {
                    "provider": entry["provider"],
                    "model": entry["model"],
                    "priority": entry.get("priority", idx + 1),
                }
            )

        # 2. Pins first (respect capability + tier filter; dropped pins logged).
        pin_keys = {(p["provider"], p["model"]) for p in self._pins}
        pinned: List[ChainCandidate] = []
        remaining: List[Dict] = []
        for entry in entries:
            key = (entry["provider"], entry["model"])
            if key in pin_keys and self._eligible(entry, required, tier):
                pinned.append(
                    ChainCandidate(
                        provider=entry["provider"],
                        model=entry["model"],
                        priority=len(pinned) + 1,
                        capabilities=self._candidate_caps(*key),
                        score=self._score(*key),
                        reason="PINNED",
                    )
                )
            else:
                remaining.append(entry)

        # 3+4. Capability + tier filter on the remainder.
        pool = [e for e in remaining if self._eligible(e, required, tier)]
        pool_candidates = [
            ChainCandidate(
                provider=e["provider"],
                model=e["model"],
                priority=e["priority"],
                capabilities=self._candidate_caps(e["provider"], e["model"]),
                score=self._score(e["provider"], e["model"]),
            )
            for e in pool
        ]

        # 5. Predict E[time]; fall back to static order when nothing confident.
        if self._predictor is not None:
            preds = self._predictor.predict_many(
                ((c.provider, c.model) for c in pool_candidates),
                output_tokens=output_tokens,
            )
            pred_map = {(p.provider, p.model): p for p in preds}
            rebuilt = []
            for cand in pool_candidates:
                p = pred_map.get((cand.provider, cand.model))
                if p is not None:
                    cand = ChainCandidate(
                        provider=cand.provider,
                        model=cand.model,
                        priority=cand.priority,
                        capabilities=cand.capabilities,
                        score=cand.score,
                        e_time_ms=p.e_time_ms,
                        confidence=p.confidence,
                        source=p.source,
                        reason="telemetry"
                        if p.confidence >= CONFIDENT_CONFIDENCE
                        else p.source,
                    )
                rebuilt.append(cand)
            pool_candidates = rebuilt
            confident = any(
                c.confidence >= CONFIDENT_CONFIDENCE for c in pool_candidates
            )
        else:
            confident = False

        if confident:
            pool_candidates.sort(
                key=lambda c: (
                    c.e_time_ms if c.e_time_ms is not None else 1 << 60,
                    -c.confidence,
                    c.priority,
                )
            )
        else:
            result.static_fallback = True
            pool_candidates.sort(key=lambda c: c.priority)

        # 6. Escalation pool: tier-floor-above candidates, E[time]-sorted.
        escalation = [
            ChainCandidate(
                provider=c.provider,
                model=c.model,
                priority=c.priority,
                capabilities=c.capabilities,
                score=c.score,
                reason="escalation",
            )
            for c in pool_candidates
            if self._floor_above(c, tier)
        ]
        escalation.sort(key=lambda c: c.e_time_ms if c.e_time_ms is not None else 1 << 60)

        # 7. Sticky sessions: escalate-only stickiness (#465).
        if session_key:
            stored = self._sticky.get(session_key)
            if stored and now - stored[3] < STICKY_TTL_S:
                stored_tier, sprov, smodel, _ = stored
                # Never downgrade below the stored tier.
                pool_candidates = [c for c in pool_candidates if self._floor_ok(c, stored_tier)]
                for i, c in enumerate(pool_candidates):
                    if c.provider == sprov and c.model == smodel:
                        pool_candidates.insert(0, pool_candidates.pop(i))
                        break
            # Record current choice for next request (escalate-only memory).
            if pool_candidates:
                best = pool_candidates[0]
                self._sticky[session_key] = (tier, best.provider, best.model, now)
                if len(self._sticky) > STICKY_MAX_ENTRIES:
                    oldest = min(self._sticky, key=lambda k: self._sticky[k][3])
                    del self._sticky[oldest]

        result.chain = pinned + pool_candidates
        result.escalation = escalation
        return result

    # -- filtering helpers -------------------------------------------------

    def _meets_floor(self, provider: str, model: str, tier: Tier) -> bool:
        key = f"{provider}/{model}"
        if key not in self._model_scores:
            # Unranked model: assume T1 floor (#480 spec: never T0-only, but
            # eligible for trivial/light work).
            return tier <= Tier.T1
        return model_meets_floor(self._model_scores[key], tier)

    def _eligible(self, entry: Dict, required: Capabilities, tier: Tier) -> bool:
        provider, model = entry["provider"], entry["model"]
        caps = self._candidate_caps(provider, model)
        if not caps & Capabilities.TEXT:  # non-chat media-only models
            return False
        if (caps & required) != required:
            return False
        return self._meets_floor(provider, model, tier)

    def _floor_ok(self, cand: ChainCandidate, tier: Tier) -> bool:
        return self._meets_floor(cand.provider, cand.model, tier)

    def _floor_above(self, cand: ChainCandidate, tier: Tier) -> bool:
        """Candidate meets the next tier's floor but not the current one."""
        if tier >= Tier.T4:
            return False
        return self._meets_floor(cand.provider, cand.model, Tier(tier + 1)) and not self._floor_ok(
            cand, tier
        )


# ---------------------------------------------------------------------------
# Debug header
# ---------------------------------------------------------------------------


def to_debug_header(result: SelectResult, virtual_model_id: str) -> str:
    """X-Route-Chain style header, matching reorder_chains dry-run format.

    '[<id>] #1 provider/model: reason; #2 ...' + fallback marker.
    """
    parts = []
    for cand in result.chain:
        reason = cand.reason or ("static-fallback" if result.static_fallback else "chain")
        parts.append(f"#{cand.priority} {cand.provider}/{cand.model}: {reason}")
    if result.escalation:
        esc = ", ".join(
            f"{c.provider}/{c.model}" for c in result.escalation[:3]
        )
        parts.append(f"esc: {esc}")
    if result.static_fallback:
        parts.append("static-fallback")
    return f"[{virtual_model_id}] " + "; ".join(parts)


def _demo() -> None:
    """Self-check: build a chain for a vision request with mixed candidates."""
    from .request_features import extract_request_features

    selector = ChainSelector(
        model_scores={"groq/llama-3.1-8b-instant": 0.55, "openai/gpt-4o-mini": 0.62},
        capabilities={
            "groq/llama-3.1-8b-instant": Capabilities.TEXT | Capabilities.TOOL_CALLING,
            "openai/gpt-4o-mini": Capabilities.TEXT | Capabilities.VISION,
        },
        pins=[{"provider": "openai", "model": "gpt-4o-mini"}],
    )
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is in this image?"},
                    {"type": "image_url", "image_url": {"url": "https://x/i.png"}},
                ],
            }
        ]
    }
    features = extract_request_features(body)
    from .tier_classifier import classify_request

    tier = classify_request(features)
    candidates = [
        {"provider": "groq", "model": "llama-3.1-8b-instant", "priority": 1},
        {"provider": "openai", "model": "gpt-4o-mini", "priority": 2},
    ]
    result = selector.select(features, tier, candidates)
    assert result.chain and result.chain[0].model == "gpt-4o-mini", "pin must lead for vision"
    assert result.static_fallback, "no telemetry -> static fallback"
    print(to_debug_header(result, "demo"))
    print("chain_selector OK")


if __name__ == "__main__":
    _demo()
