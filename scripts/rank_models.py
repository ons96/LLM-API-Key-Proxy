#!/usr/bin/env python3
"""
U-formula model ranking: U = A * (I^w / (C_opp * T)).

Pure scoring module. Reads from llm-leaderboard-aggregate db (benchmark
scores) + llm-provider-manager db (provider categories) + reorder_chains
telemetry dict + tier_config.yaml. Returns ranked chains.

Refs: task-board #195, user-approved formula (m1028).
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("rank_models")

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TIER_CONFIG = _REPO_ROOT / "config" / "tier_config.yaml"
DEFAULT_BENCHMARK_DB = Path(
    os.environ.get(
        "LLM_BENCHMARK_DB",
        str(Path.home() / "CodingProjects/llm-leaderboard-aggregate/db/models.db"),
    )
)
DEFAULT_PROVIDER_DB = Path(
    os.environ.get(
        "LLM_PROVIDERS_DB",
        str(Path.home() / "CodingProjects/llm-provider-manager/llm_providers.db"),
    )
)

# Intelligence floor if model has no benchmark score (small models, untested).
NULL_I_FLOOR = 0.20
# TPS for latency estimate when telemetry missing (rough mid-tier).
DEFAULT_TPS_FALLBACK = 20.0
# TTFT in ms when telemetry missing.
DEFAULT_TTFT_MS_FALLBACK = 2000.0
# Cap C_opp to avoid div-by-zero / runaway scaling.
C_OPP_CAP = 100.0
# One-time-credit provider K constant (scarce, preserve for complex tasks).
ONE_TIME_K = 10.0


@dataclass
class BenchmarkScore:
    """Per-model benchmark row (normalized to [0,1] at load time)."""

    model_id: str
    coding: float = 0.0  # agentic coding score [0,1]
    chat: float = 0.0  # reasoning chat score [0,1]


@dataclass
class ProviderCategory:
    """Free-tier classification for C_opp computation."""

    name: str
    kind: str = "paid"  # free_unlimited|free_daily|free_one_time|no_key|paid
    rate_limit_rpm: int = 0
    rate_limit_daily_tokens: int = 0


@dataclass
class TierConfig:
    """Per-virtual-model U-formula params."""

    w: int = 2
    I_floor: float = 0.30
    n_out_tokens: int = 4000
    purpose: str = "coding"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_benchmark_scores(db_path: Path = DEFAULT_BENCHMARK_DB) -> Dict[str, BenchmarkScore]:
    """Read models_unique from llm-leaderboard-aggregate db. Key = model_id.

    Normalizes raw scores to [0,1]:
      - agentic_coding / 100 (top scores ~75)
      - reasoning_chat / 100
    """
    if not db_path.exists():
        logger.warning("benchmark DB not found at %s; all I = NULL_I_FLOOR", db_path)
        return {}
    out: Dict[str, BenchmarkScore] = {}
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT model_id,
                   avg_agentic_coding_score,
                   avg_reasoning_chat_score
            FROM models_unique
            WHERE avg_agentic_coding_score IS NOT NULL
               OR avg_reasoning_chat_score IS NOT NULL
            """
        )
        for row in cur.fetchall():
            mid = row["model_id"]
            coding_raw = row["avg_agentic_coding_score"]
            chat_raw = row["avg_reasoning_chat_score"]
            coding = max(0.0, min(1.0, (coding_raw or 0.0) / 100.0))
            chat = max(0.0, min(1.0, (chat_raw or 0.0) / 100.0))
            out[mid] = BenchmarkScore(model_id=mid, coding=coding, chat=chat)
        conn.close()
        logger.info("loaded %d benchmark scores from %s", len(out), db_path)
    except sqlite3.Error as exc:
        logger.warning("benchmark DB read failed: %r", exc)
    return out


def load_provider_categories(
    db_path: Path = DEFAULT_PROVIDER_DB,
) -> Dict[str, ProviderCategory]:
    """Read providers from llm-provider-manager db. Key = key_name."""
    if not db_path.exists():
        logger.warning("provider DB not found at %s; all C_opp = paid", db_path)
        return {}
    out: Dict[str, ProviderCategory] = {}
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT key_name,
                   no_api_key_required,
                   free_unlimited,
                   free_daily,
                   free_one_time,
                   rate_limit_rpm,
                   rate_limit_daily_tokens
            FROM providers
            WHERE key_name IS NOT NULL
            """
        )
        for row in cur.fetchall():
            name = row["key_name"]
            if row["no_api_key_required"]:
                kind = "no_key"
            elif row["free_unlimited"]:
                kind = "free_unlimited"
            elif row["free_daily"]:
                kind = "free_daily"
            elif row["free_one_time"]:
                kind = "free_one_time"
            else:
                kind = "paid"
            out[name] = ProviderCategory(
                name=name,
                kind=kind,
                rate_limit_rpm=int(row["rate_limit_rpm"] or 0),
                rate_limit_daily_tokens=int(row["rate_limit_daily_tokens"] or 0),
            )
        conn.close()
        logger.info("loaded %d provider categories from %s", len(out), db_path)
    except sqlite3.Error as exc:
        logger.warning("provider DB read failed: %r", exc)
    return out


def load_tier_config(config_path: Path = DEFAULT_TIER_CONFIG) -> Dict[str, TierConfig]:
    """Load tier_config.yaml. Returns dict keyed by virtual model name."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(f"PyYAML required: {exc}") from exc
    if not config_path.exists():
        logger.warning("tier_config not found at %s; using defaults", config_path)
        return {"default": TierConfig()}
    with config_path.open("r") as f:
        raw = yaml.safe_load(f) or {}
    default_raw = raw.get("default", {})
    default = TierConfig(
        w=int(default_raw.get("w", 2)),
        I_floor=float(default_raw.get("I_floor", 0.30)),
        n_out_tokens=int(default_raw.get("n_out_tokens", 4000)),
        purpose=str(default_raw.get("purpose", "coding")),
    )
    out: Dict[str, TierConfig] = {"default": default}
    for name, cfg in (raw.get("virtual_models") or {}).items():
        out[name] = TierConfig(
            w=int(cfg.get("w", default.w)),
            I_floor=float(cfg.get("I_floor", default.I_floor)),
            n_out_tokens=int(cfg.get("n_out_tokens", default.n_out_tokens)),
            purpose=str(cfg.get("purpose", default.purpose)),
        )
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


_ALIASES = [
    # (regex, canonical model_id). Order matters — first match wins.
    # Canonical names verified against llm-leaderboard-aggregate db/models.db.
    (r"^meta-llama/llama-3\.3-70b", "llama-3-3-70b-instruct"),
    (r"^llama-3\.3-70b", "llama-3-3-70b-instruct"),
    (r"^llama-3\.1-8b", "llama-3-1-8b-instruct"),
    (r"^llama-3\.1-70b", "llama-3-1-70b-instruct"),
    (r"^llama-3-70b", "llama-3-70b-instruct"),
    (r"^llama-3-8b", "llama-3-8b-instruct"),
    (r"^gpt-4o-mini", "gpt-4o-mini"),
    (r"^gpt-5", "gpt-5-2"),
    (r"^gemini-2\.5-flash", "gemini-2.5-flash"),
    (r"^gemini-3-flash", "gemini-3-flash"),
    (r"^gemini-3-pro", "gemini-3-pro"),
    (r"^claude-opus-4", "claude-opus-4-6"),
    (r"^claude-sonnet-4", "claude-sonnet-4-6"),
    (r"^deepseek-v3", "deepseek-v3-2"),
    (r"^qwen3", "qwen3-max"),
    (r"^glm-5", "glm-5"),
    (r"^nemotron", "nvidia-llama-3-1-nemotron-70b-instruct"),
    (r"^mistral-large", "mistralai/mistral-large-3-675b-instruct-2512"),
    (r"^mistral-embed|codestral-embed", "mistral-embed"),
]


def _normalize_model_id(model: str) -> str:
    """Strip provider prefix + apply alias map. 'groq/llama-3.3-70b-versatile' → 'meta-llama/llama-3.3-70b-instruct'."""
    if not model:
        return ""
    # Drop provider prefix
    if "/" in model:
        model = model.split("/", 1)[1]
    for pattern, canonical in _ALIASES:
        if re.match(pattern, model, re.IGNORECASE):
            return canonical
    return model


def compute_intelligence(
    model: str,
    purpose: str,
    benchmark: Dict[str, BenchmarkScore],
) -> float:
    """I in [0,1]. coding → agentic_coding_score; chat → reasoning_chat_score.

    If no benchmark row, returns NULL_I_FLOOR (so untested models can still
    appear in low-floor chains like coding-fast, but won't win elite tiers).
    """
    norm = _normalize_model_id(model)
    score = benchmark.get(norm)
    if score is None:
        # Fallback: try direct match on normalized name without provider prefix.
        score = benchmark.get(model)
    if score is None:
        return NULL_I_FLOOR
    return score.coding if purpose == "coding" else score.chat


def compute_opportunity_cost(
    provider: str,
    categories: Dict[str, ProviderCategory],
    telemetry_used_today: int = 0,
) -> float:
    """C_opp >= 1.0.

    free_unlimited / no_key: 1.0 (renewable, low cost)
    free_daily: 1.0 + used_today/daily_limit (rises as quota consumed)
    free_one_time: ONE_TIME_K / (remaining+1) (scarce, preserve for complex)
    paid: C_OPP_CAP (avoid; only as last resort)
    """
    cat = categories.get(provider)
    if cat is None:
        # Unknown provider — assume paid (safe default).
        return C_OPP_CAP
    if cat.kind in ("free_unlimited", "no_key"):
        return 1.0
    if cat.kind == "free_daily":
        daily_limit = cat.rate_limit_daily_tokens or 100000
        if daily_limit <= 0:
            return 1.0
        used_ratio = min(1.0, telemetry_used_today / daily_limit)
        return min(C_OPP_CAP, 1.0 + used_ratio)
    if cat.kind == "free_one_time":
        # ponytail: don't track remaining credits yet — assume half-spent.
        # K/(remaining+1) with remaining=1 → K/2 = 5.0. Preserves scarce.
        return min(C_OPP_CAP, ONE_TIME_K / 2.0)
    return C_OPP_CAP


def compute_latency(
    ttft_ms: Optional[float],
    tps: Optional[float],
    n_out_tokens: int,
) -> float:
    """T in seconds. T = ttft + n_out_tokens / tps.

    Missing telemetry → fallback mid-tier estimates (no zero division).
    """
    ttft_s = (ttft_ms or DEFAULT_TTFT_MS_FALLBACK) / 1000.0
    rate = tps if (tps and tps > 0) else DEFAULT_TPS_FALLBACK
    return ttft_s + (n_out_tokens / rate)


def compute_U(
    intelligence: float,
    w: int,
    opportunity_cost: float,
    latency_s: float,
    I_floor: float,
) -> Tuple[float, str]:
    """U = A * (I^w / (C_opp * T)). Returns (U, reason).

    A = 0 if I < I_floor (kill switch), else 1.
    """
    A = 1.0 if intelligence >= I_floor else 0.0
    if A == 0.0:
        return 0.0, f"A=0 (I={intelligence:.2f} < floor={I_floor:.2f})"
    I_w = intelligence ** w
    c_opp = max(1.0, min(C_OPP_CAP, opportunity_cost))
    t = max(0.1, latency_s)  # avoid div-by-zero
    u = A * (I_w / (c_opp * t))
    reason = (
        f"A=1 I={intelligence:.2f}^{w}={I_w:.4f} "
        f"/ (C_opp={c_opp:.2f} * T={t:.2f}s) = {u:.4f}"
    )
    return u, reason


# ---------------------------------------------------------------------------
# Chain ranking
# ---------------------------------------------------------------------------


@dataclass
class RankReason:
    provider: str
    model: str
    score: float
    reason: str
    I: float = 0.0
    c_opp: float = 0.0
    T: float = 0.0


def rank_chain(
    chain: List[Dict],
    telemetry_stats: Dict[Tuple[str, str], "object"],
    penalties: Dict[Tuple[str, str], float],
    benchmark: Dict[str, BenchmarkScore],
    categories: Dict[str, ProviderCategory],
    tier: TierConfig,
    penalty_floor: float = 0.5,
) -> Tuple[List[Dict], List[RankReason]]:
    """Rank one fallback chain via U formula. Returns (ranked_chain, reasons).

    Stable sort: scored entries first (DESC by U), unscored entries after
    (preserve original order). Penalty multiplier reduces U (sinks failing
    providers without removing them).
    """
    scored: List[Tuple[int, float, RankReason]] = []
    unscored: List[Tuple[int, Dict]] = []
    for idx, entry in enumerate(chain):
        provider = entry.get("provider", "")
        model = entry.get("model", "")
        stat = telemetry_stats.get((provider, model))
        pen = penalties.get((provider, model), 0.0)

        I = compute_intelligence(model, tier.purpose, benchmark)
        c_opp = compute_opportunity_cost(provider, categories)
        ttft_ms = getattr(stat, "avg_ttft_ms", None) if stat else None
        tps = getattr(stat, "avg_tps", None) if stat else None
        T = compute_latency(ttft_ms, tps, tier.n_out_tokens)
        u, reason_text = compute_U(I, tier.w, c_opp, T, tier.I_floor)

        # ponytail: penalty multiplier. pen=0 → 1.0 (no penalty). pen=5 → 0.5.
        # penalty_floor=0.5 means pen=5 cuts U in half. Doesn't remove provider.
        pen_mult = max(penalty_floor, 1.0 / (1.0 + pen * 0.2))
        u_penalized = u * pen_mult

        reason = RankReason(
            provider=provider,
            model=model,
            score=u_penalized,
            reason=f"{reason_text} pen_mult={pen_mult:.2f}",
            I=I,
            c_opp=c_opp,
            T=T,
        )
        if stat is None:
            unscored.append((idx, entry))
        else:
            scored.append((idx, u_penalized, reason))

    # Sort scored DESC, then unscored by original idx.
    scored.sort(key=lambda t: t[1], reverse=True)
    new_chain: List[Dict] = []
    reasons: List[RankReason] = []
    for new_idx, (idx, _u, reason) in enumerate(scored):
        new_entry = dict(chain[idx])
        new_entry["priority"] = new_idx + 1
        new_chain.append(new_entry)
        reasons.append(reason)
    offset = len(scored)
    for new_idx, (idx, entry) in enumerate(unscored):
        new_entry = dict(chain[idx])
        new_entry["priority"] = offset + new_idx + 1
        new_chain.append(new_entry)
        provider = entry.get("provider", "")
        model = entry.get("model", "")
        reasons.append(
            RankReason(
                provider=provider,
                model=model,
                score=0.0,
                reason="no telemetry — preserved original order",
            )
        )
    return new_chain, reasons


def get_tier(tier_config: Dict[str, TierConfig], virtual_model: str) -> TierConfig:
    """Look up tier config for a virtual model, falling back to default."""
    return tier_config.get(virtual_model) or tier_config["default"]


def _self_test() -> None:
    """Smoke test. python -m rank_models"""
    bench = {
        "gemini-3-flash": BenchmarkScore("gemini-3-flash", coding=0.747, chat=0.591),
        "gpt-5-2": BenchmarkScore("gpt-5-2", coding=0.708, chat=0.583),
        "llama-small": BenchmarkScore("llama-small", coding=0.30, chat=0.25),
    }
    cats = {
        "nvidia": ProviderCategory("nvidia", "free_unlimited", 40, 0),
        "groq": ProviderCategory("groq", "free_unlimited", 30, 14400),
        "agentrouter": ProviderCategory("agentrouter", "free_one_time", 0, 0),
        "openai": ProviderCategory("openai", "paid", 0, 0),
    }
    tier = TierConfig(w=3, I_floor=0.55, n_out_tokens=4000, purpose="coding")

    class _Stat:
        def __init__(self, tps: float, ttft: float) -> None:
            self.avg_tps = tps
            self.avg_ttft_ms = ttft

    telemetry = {
        ("nvidia", "gemini-3-flash"): _Stat(50.0, 400.0),
        ("groq", "llama-small"): _Stat(250.0, 200.0),
    }
    chain = [
        {"provider": "groq", "model": "llama-small", "priority": 1},
        {"provider": "nvidia", "model": "gemini-3-flash", "priority": 2},
        {"provider": "agentrouter", "model": "gpt-5-2", "priority": 3},
        {"provider": "openai", "model": "gpt-5-2", "priority": 4},
    ]
    new_chain, reasons = rank_chain(chain, telemetry, {}, bench, cats, tier)
    print("=== Self-test ===")
    for r in reasons:
        print(f"  {r.provider}/{r.model}: U={r.score:.4f} I={r.I:.2f} C={r.c_opp:.2f} T={r.T:.2f}s — {r.reason}")
    # Expected: nvidia/gemini-3-flash #1 (high I, free, fast), groq/llama-small
    # killed (I=0.30 < floor 0.55, A=0), agentrouter/gpt-5-2 mid (scarce, no
    # telemetry), openai/gpt-5-2 last (paid, C_opp=100).
    assert new_chain[0]["provider"] == "nvidia", f"expected nvidia #1, got {new_chain[0]['provider']}"
    assert new_chain[-1]["provider"] == "openai", f"expected openai last, got {new_chain[-1]['provider']}"
    print("self-test OK")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _self_test()
