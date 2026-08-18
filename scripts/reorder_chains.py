#!/usr/bin/env python3
"""
Dynamic fallback chain auto-reorder from live telemetry.

Reads telemetry SQLite (`llm_events` table) + PenaltyStore + existing
`config/virtual_models.yaml`, computes a composite per-(provider, model)
score for each virtual model's fallback_chain, rewrites priority order
(healthiest first). Backs up prior config to `virtual_models.yaml.bak-<ts>`.

Composite score (env-overridable weights below; Phase 2.A cache affinity;
#472 quality + pins):
    success_rate * 0.288 (24h window, min_samples threshold, else skip reorder)
    + tps_norm * 0.216   (avg tps / max_observed_tps, clipped [0,1])
    + ttft_norm * 0.144  (1 - avg_ttft_ms / 30000, clipped [0,1])
    + penalty_inv * 0.072 (1 / (1 + penalty_score))
    + cache_score * 0.080 (min(avg_cached_tokens / 1000, 1); prompt-cache affinity)
    + quality_score * 0.20 (model_rankings.yaml composite_score, normalized;
                            REORDER_W_QUALITY=0 restores pure telemetry)

Chain pins (#472): optional static chain-heads per virtual model in
`config/chain_pins.yaml` ({virtual_model: [{provider, model}, ...]}).
Pinned entries are forced to the chain head in pin order; the remainder is
reordered by composite. No pins file = current behavior.

Smoothing:
    - 24h telemetry window (REORDER_WINDOW_H env, default 24)
    - min 5 samples per (provider, model) (REORDER_MIN_SAMPLES env, default 5)
    - if below threshold, keep original priority (no thrash on outliers)

Usage:
    python scripts/reorder_chains.py [--config config/virtual_models.yaml]
                                      [--telemetry-db /dev/shm/telemetry.db]
                                      [--dry-run] [--verbose]

Exit codes:
    0 = success (or dry-run)
    1 = config/telemetry error
    2 = no reorder needed (all chains already optimal)

Refs: task-board #195, #194. Depends on #196 (PenaltyStore), #163 (telemetry).
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Allow running as script (no package import) or as module.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

try:
    import yaml
except ImportError as exc:
    print(f"ERROR: PyYAML required: {exc}", file=sys.stderr)
    sys.exit(1)

logger = logging.getLogger("reorder_chains")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "virtual_models.yaml"
DEFAULT_TIER_CONFIG = _REPO_ROOT / "config" / "tier_config.yaml"
DEFAULT_TELEMETRY_DB = os.environ.get("TELEMETRY_DB_PATH", "/dev/shm/telemetry.db")
DEFAULT_WINDOW_H = int(os.environ.get("REORDER_WINDOW_H", "24"))
DEFAULT_MIN_SAMPLES = int(os.environ.get("REORDER_MIN_SAMPLES", "5"))
DEFAULT_MAX_TPS = float(os.environ.get("REORDER_MAX_TPS", "3000"))
DEFAULT_MAX_TTFT_MS = float(os.environ.get("REORDER_MAX_TTFT_MS", "30000"))
STATIC_VIRTUAL_MODELS_PATH = _REPO_ROOT / "config" / "static_virtual_models.yaml"

# ---------------------------------------------------------------------------
# Health-probe classification (task-board #484).
#
# Post-launch health probes emit tiny responses (a handful of tokens) measured
# almost entirely over connection overhead, and therefore land at artificially
# low throughput (tps ~0.3–10). Mixed into per-(provider, model) AVG(tps) they
# dragged real throughput down 10–100x, skewing reorder_chains scoring. These
# rows are NOT real usage, so we exclude them from telemetry aggregations.
# Real small completions (<= 46 tokens) still stream at much higher tps, so the
# combined predicate isolates probes without dropping legitimate calls.
PROBE_COMPLETION_TOKENS_MAX = 46
PROBE_TPS_MIN = 0.3
PROBE_TPS_MAX = 10.0

# SQL fragment appended to every llm_events aggregation WHERE clause.
PROBE_EXCLUDE_SQL = (
    "AND NOT (COALESCE(completion_tokens, 0) <= {max_tok}"
    " AND tps >= {tps_min} AND tps <= {tps_max})"
).format(
    max_tok=PROBE_COMPLETION_TOKENS_MAX,
    tps_min=PROBE_TPS_MIN,
    tps_max=PROBE_TPS_MAX,
)


def load_static_virtual_models(path: Path) -> Dict[str, object]:
    """#405: load policy-fixed virtual models from a separate YAML.

    These chains (e.g. safe-coding via PrivAiTe) must survive the telemetry
    rebuild that reorder_config performs. Returns the inner `virtual_models`
    mapping; empty dict if the file is absent or invalid.
    """
    if not path.exists():
        return {}
    try:
        with path.open("r") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning("could not parse %s: %r", path, exc)
        return {}
    return cfg.get("virtual_models", {}) or {}


def is_probe_row(completion_tokens, tps) -> bool:
    """True when a telemetry row looks like a health probe (not real usage)."""
    if tps is None:
        return False
    return (
        (completion_tokens or 0) <= PROBE_COMPLETION_TOKENS_MAX
        and PROBE_TPS_MIN <= tps <= PROBE_TPS_MAX
    )


# Composite weights (sum to 1.0). Override via env if needed.
# ponytail: 6-factor post-#472. Telemetry terms rescaled proportionally
# from Phase-2.A 0.36/0.27/0.18/0.09/0.10 to make room for W_QUALITY=0.20.
# Set REORDER_W_QUALITY=0 to restore pure-telemetry behavior (Phase-2.A).
W_SUCCESS = float(os.environ.get("REORDER_W_SUCCESS", "0.288"))
W_TPS = float(os.environ.get("REORDER_W_TPS", "0.216"))
W_TTFT = float(os.environ.get("REORDER_W_TTFT", "0.144"))
W_PENALTY = float(os.environ.get("REORDER_W_PENALTY", "0.072"))
W_CACHE = float(os.environ.get("REORDER_W_CACHE", "0.080"))
W_QUALITY = float(os.environ.get("REORDER_W_QUALITY", "0.20"))

# U-formula paths (lazy import from rank_models sibling module).
_RANK_MODELS = None


def _get_rank_models():
    """Lazy import of rank_models sibling module."""
    global _RANK_MODELS
    if _RANK_MODELS is None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "rank_models", _REPO_ROOT / "scripts" / "rank_models.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["rank_models"] = mod
        spec.loader.exec_module(mod)
        _RANK_MODELS = mod
    return _RANK_MODELS


@dataclass
class TelemetryStat:
    """Aggregated telemetry for one (provider, model) over the window."""

    provider: str
    model: str
    samples: int
    success_rate: float  # [0, 1]
    avg_tps: float
    avg_ttft_ms: float
    avg_cached_tokens: float = 0.0  # Phase 2.A prompt-cache affinity signal


@dataclass
class ChainEntry:
    """One row of a virtual model's fallback_chain."""

    provider: str
    model: str
    priority: int
    original_index: int


# ---------------------------------------------------------------------------
# Telemetry read
# ---------------------------------------------------------------------------


def load_telemetry(
    db_path: str,
    window_h: int = DEFAULT_WINDOW_H,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> Dict[Tuple[str, str], TelemetryStat]:
    """Read `llm_events` table, aggregate per (provider, model) over window.

    Returns dict keyed by (provider, model). Entries below min_samples are
    still returned (with samples count) so callers can decide to skip.
    """
    if not os.path.exists(db_path):
        logger.warning("telemetry DB not found at %s; no reorder possible", db_path)
        return {}

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute("PRAGMA table_info(llm_events)")
        cols = {row["name"] for row in cur.fetchall()}
        if not cols:
            logger.warning("llm_events table missing in %s", db_path)
            return {}
        required = {"provider", "model", "status", "tps", "ttft_ms", "ts_start"}
        if not required.issubset(cols):
            logger.warning(
                "llm_events schema mismatch; need %s, got %s",
                required,
                cols,
            )
            return {}

        cutoff = time.time() - (window_h * 3600)
        # ponytail: prefer (concrete_provider, concrete_model) when present
        # (recorded post #195 reorder follow-up); fall back to (provider, model)
        # alias-level for older rows. sqlite NULL-aware COALESCE via CASE.
        has_concrete = {"concrete_provider", "concrete_model"}.issubset(cols)
        # Phase 2.A: cache cols may be absent on pre-upgrade DBs — degrade to 0.
        has_cache = "cached_tokens" in cols
        cache_avg_expr = (
            "AVG(COALESCE(cached_tokens, 0)) AS avg_cached_tokens"
            if has_cache
            else "0.0 AS avg_cached_tokens"
        )
        if has_concrete:
            cur = conn.execute(
                f"""
                SELECT
                    COALESCE(NULLIF(concrete_provider, ''), provider) AS provider,
                    COALESCE(NULLIF(concrete_model, ''), model) AS model,
                    COUNT(*) AS samples,
                    SUM(CASE WHEN status='success' THEN 1.0 ELSE 0.0 END) AS successes,
                    AVG(tps) AS avg_tps,
                    AVG(ttft_ms) AS avg_ttft_ms,
                    {cache_avg_expr}
                FROM llm_events
                WHERE ts_start >= ?
                  AND provider IS NOT NULL
                  AND model IS NOT NULL
                  {PROBE_EXCLUDE_SQL}
                GROUP BY COALESCE(NULLIF(concrete_provider, ''), provider),
                         COALESCE(NULLIF(concrete_model, ''), model)
                """,
                (cutoff,),
            )
        else:
            logger.info(
                "llm_events lacks concrete_provider/concrete_model; using alias columns"
            )
            cur = conn.execute(
                f"""
                SELECT provider, model,
                       COUNT(*) AS samples,
                       SUM(CASE WHEN status='success' THEN 1.0 ELSE 0.0 END) AS successes,
                       AVG(tps) AS avg_tps,
                       AVG(ttft_ms) AS avg_ttft_ms,
                       {cache_avg_expr}
                FROM llm_events
                WHERE ts_start >= ?
                  AND provider IS NOT NULL
                  AND model IS NOT NULL
                  {PROBE_EXCLUDE_SQL}
                GROUP BY provider, model
                """,
                (cutoff,),
            )
        stats: Dict[Tuple[str, str], TelemetryStat] = {}
        for row in cur.fetchall():
            samples = int(row["samples"] or 0)
            successes = float(row["successes"] or 0.0)
            stat = TelemetryStat(
                provider=row["provider"],
                model=row["model"],
                samples=samples,
                success_rate=(successes / samples) if samples > 0 else 0.0,
                avg_tps=float(row["avg_tps"] or 0.0),
                avg_ttft_ms=float(row["avg_ttft_ms"] or 0.0),
                avg_cached_tokens=float(row["avg_cached_tokens"] or 0.0),
            )
            stats[(stat.provider, stat.model)] = stat
        logger.info(
            "loaded %d (provider, model) stats from %s (window=%dh, min_samples=%d)",
            len(stats),
            db_path,
            window_h,
            min_samples,
        )
        return stats
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# PenaltyStore read (optional — graceful if #196 not merged)
# ---------------------------------------------------------------------------


def load_penalty_scores(now: Optional[float] = None) -> Dict[Tuple[str, str], float]:
    """Best-effort read of PenaltyStore scores. Returns {} if unavailable."""
    try:
        from proxy_app.penalty_store import PenaltyStore  # type: ignore

        store = PenaltyStore.get()
        entries = store.get_entries(now=now)
        scores: Dict[Tuple[str, str], float] = {}
        for e in entries:
            key = (e.provider, e.model)
            # Sum decayed score across failure types (get_entries already decays).
            scores[key] = scores.get(key, 0.0) + e.score
        return scores
    except Exception as exc:
        logger.debug("penalty_store unavailable, skipping penalty factor: %r", exc)
        return {}


# ---------------------------------------------------------------------------
# Quality scores + chain pins (#472)
# ---------------------------------------------------------------------------

DEFAULT_RANKINGS_PATH = _REPO_ROOT / "config" / "model_rankings.yaml"
DEFAULT_PINS_PATH = _REPO_ROOT / "config" / "chain_pins.yaml"


def load_quality_scores(rankings_path: Optional[Path] = None) -> Dict[str, float]:
    """Read model_rankings.yaml -> {model_id: composite_score_norm}.

    model_id is the rankings `id` field ("provider/model"), value is
    composite_score normalized to [0, 1] (rankings use a 0-100 scale).
    Missing/unreadable file -> {} (quality term contributes 0).
    """
    path = rankings_path or DEFAULT_RANKINGS_PATH
    if not path.exists():
        logger.warning("model_rankings not found at %s; quality term = 0", path)
        return {}
    try:
        with path.open("r") as f:
            data = yaml.safe_load(f)
    except Exception as exc:
        logger.warning("failed to load %s: %r", path, exc)
        return {}
    scores: Dict[str, float] = {}
    for m in (data or {}).get("models", []) or []:
        mid = m.get("id")
        comp = ((m.get("scores") or {}).get("composite_score")) if m else None
        if mid and isinstance(comp, (int, float)) and comp > 0:
            scores[mid] = max(0.0, min(1.0, float(comp) / 100.0))
    logger.info("loaded %d quality scores from %s", len(scores), path)
    return scores


def load_chain_pins(pins_path: Optional[Path] = None) -> Dict[str, List[Dict]]:
    """Read chain_pins.yaml -> {virtual_model_id: [entry, ...]}.

    Entry shape mirrors a fallback_chain row: {provider, model, priority}.
    Missing file -> {} (no pins = current behavior). Entries referencing a
    (provider, model) absent from a chain are ignored with a warning.
    """
    path = pins_path or DEFAULT_PINS_PATH
    if not path.exists():
        return {}
    try:
        with path.open("r") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning("failed to load %s: %r", path, exc)
        return {}
    pins: Dict[str, List[Dict]] = {}
    for model_id, entries in data.items():
        if not isinstance(entries, list):
            continue
        clean = [
            {"provider": e.get("provider", ""), "model": e.get("model", "")}
            for e in entries
            if isinstance(e, dict) and e.get("provider") and e.get("model")
        ]
        if clean:
            pins[model_id] = clean
    logger.info("loaded chain pins for %d virtual models from %s", len(pins), path)
    return pins


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------


def compute_composite(
    stat: Optional[TelemetryStat],
    penalty: float,
    min_samples: int,
    max_tps: float,
    max_ttft_ms: float,
    quality: float = 0.0,
) -> Tuple[float, str]:
    """Return (score, reason). Score in [0, 1.1] (1.1 = perfect + no penalty).

    If stat is None or below min_samples, returns (-1.0, "insufficient_samples")
    so the caller knows to keep original priority.
    """
    if stat is None:
        return -1.0, "no_telemetry"
    if stat.samples < min_samples:
        return -1.0, f"insufficient_samples({stat.samples}<{min_samples})"

    success = max(0.0, min(1.0, stat.success_rate))
    tps_norm = max(0.0, min(1.0, stat.avg_tps / max_tps)) if max_tps > 0 else 0.0
    ttft_norm = (
        max(0.0, 1.0 - (stat.avg_ttft_ms / max_ttft_ms)) if max_ttft_ms > 0 else 0.0
    )
    penalty_inv = 1.0 / (1.0 + max(0.0, penalty))
    # ponytail: fixed 1000-token normalization; per-provider cap table if
    # cache-heavy chains need sharper differentiation.
    cache_score = min(stat.avg_cached_tokens / 1000.0, 1.0)
    quality_score = max(0.0, min(1.0, quality))

    score = (
        W_SUCCESS * success
        + W_TPS * tps_norm
        + W_TTFT * ttft_norm
        + W_PENALTY * penalty_inv
        + W_CACHE * cache_score
        + W_QUALITY * quality_score
    )
    reason = (
        f"s={success:.2f}*{W_SUCCESS:.2f} + "
        f"tps={tps_norm:.2f}*{W_TPS:.2f} + "
        f"ttft={ttft_norm:.2f}*{W_TTFT:.2f} + "
        f"pen={penalty_inv:.2f}*{W_PENALTY:.2f} + "
        f"cache={cache_score:.2f}*{W_CACHE:.2f} + "
        f"qual={quality_score:.2f}*{W_QUALITY:.2f} "
        f"(samples={stat.samples}, tps={stat.avg_tps:.1f}, ttft={stat.avg_ttft_ms:.0f}ms, "
        f"pen={penalty:.2f}, cache_tok={stat.avg_cached_tokens:.0f})"
    )
    return score, reason


# ---------------------------------------------------------------------------
# Config rewrite
# ---------------------------------------------------------------------------


def reorder_chain(
    chain: List[Dict],
    stats: Dict[Tuple[str, str], TelemetryStat],
    penalties: Dict[Tuple[str, str], float],
    min_samples: int,
    max_tps: float,
    max_ttft_ms: float,
    quality_scores: Optional[Dict[str, float]] = None,
    pins: Optional[List[Dict]] = None,
) -> Tuple[List[Dict], List[str]]:
    """Reorder one fallback_chain. Returns (new_chain, reasons).

    Entries with insufficient samples keep their original relative order
    (stable sort), but are placed AFTER all entries with valid telemetry.

    Pinned entries (#472) — when `pins` is non-empty, entries whose
    (provider, model) matches a pin are forced to the chain head in pin
    order; the remaining entries are reordered by composite as usual.
    Pins referencing entries absent from the chain are ignored (warned).
    """
    entries: List[Tuple[ChainEntry, float, str]] = []
    pinned: List[Dict] = []
    pin_keys = {(p.get("provider"), p.get("model")) for p in (pins or [])}
    for idx, c in enumerate(chain):
        provider = c.get("provider", "")
        model = c.get("model", "")
        priority = int(c.get("priority", idx))
        if (provider, model) in pin_keys:
            pinned.append(c)
            continue
        entry = ChainEntry(provider, model, priority, idx)
        stat = stats.get((provider, model))
        pen = penalties.get((provider, model), 0.0)
        quality = (quality_scores or {}).get(f"{provider}/{model}", 0.0)
        score, reason = compute_composite(
            stat, pen, min_samples, max_tps, max_ttft_ms, quality
        )
        entries.append((entry, score, reason))

    # ponytail: stable sort — entries with score=-1 (insufficient samples)
    # keep their original relative order, but come after scored entries.
    # Sort key: (has_score, score). has_score=True (1) sorts before False (0).
    entries.sort(
        key=lambda t: (1 if t[1] >= 0 else 0, t[1] if t[1] >= 0 else 0.0),
        reverse=True,
    )

    new_chain: List[Dict] = []
    reasons: List[str] = []
    for new_idx, entry in enumerate(pinned):
        new_entry = dict(entry)  # preserve extra keys
        new_entry["priority"] = new_idx + 1
        new_chain.append(new_entry)
        reasons.append(
            f"  #{new_idx + 1} {entry.get('provider')}/{entry.get('model')}: "
            "PINNED (chain head)"
        )
    for new_idx, (entry, score, reason) in enumerate(entries):
        new_entry = dict(chain[entry.original_index])  # preserve extra keys
        new_entry["priority"] = len(pinned) + new_idx + 1
        new_chain.append(new_entry)
        reasons.append(
            f"  #{len(pinned) + new_idx + 1} {entry.provider}/{entry.model}: "
            f"score={score:.3f} — {reason}"
        )
    return new_chain, reasons


# ---------------------------------------------------------------------------
# Free-model baseline enforcement (#343)
# ---------------------------------------------------------------------------


def is_free_provider(provider: str, categories: Dict[str, object]) -> bool:
    """True if the provider is classified as free-tier.

    Duck-types ProviderCategory.kind from rank_models — avoids importing
    rank_models here. Free kinds: free_unlimited, free_daily, free_one_time,
    no_key. Anything else (paid, unknown, missing) is not free.
    """
    cat = categories.get(provider)
    if cat is None:
        return False
    kind = getattr(cat, "kind", "paid")
    return kind in ("free_unlimited", "free_daily", "free_one_time", "no_key")


def apply_free_baseline(
    chain: List[Dict],
    score_lookup: Dict[Tuple[str, str], float],
    categories: Dict[str, object],
    baseline_floor_min: float = 0.0,
) -> Tuple[List[Dict], List[str]]:
    """Demote non-free entries scoring below the free-tier quality floor.

    Baseline = min score among free entries with a valid (>=0) score.
    Floor = max(baseline, baseline_floor_min). Any non-free entry with a
    valid (>=0) score strictly below the floor is moved to a last-resort
    priority slot (99, 100, ...). Entries with invalid scores (e.g. -1
    insufficient samples) are never demoted — they keep their ranked
    position so untested-but-potentially-good models aren't penalised.

    Returns (new_chain, log_lines). When no free entries have valid scores,
    returns the chain unchanged (current-behaviour fallback per #343).
    """
    free_scores = [
        score_lookup.get((c.get("provider", ""), c.get("model", "")))
        for c in chain
        if is_free_provider(c.get("provider", ""), categories)
    ]
    free_scores = [s for s in free_scores if s is not None and s >= 0]
    if not free_scores:
        return chain, []  # no free data -> fall back to current behaviour

    baseline = min(free_scores)
    floor = max(baseline, baseline_floor_min)

    kept: List[Dict] = []
    demoted: List[Dict] = []
    log_lines: List[str] = []
    for c in chain:
        provider = c.get("provider", "")
        model = c.get("model", "")
        score = score_lookup.get((provider, model))
        if (
            not is_free_provider(provider, categories)
            and score is not None
            and score >= 0
            and score < floor
        ):
            demoted.append(c)
            log_lines.append(
                f"[BASELINE] {provider}/{model} score={score:.4f} "
                f"demoted below free baseline {floor:.4f}"
            )
        else:
            kept.append(c)

    if not demoted:
        return chain, log_lines  # order unchanged, but log_lines may be empty

    new_chain: List[Dict] = []
    for idx, c in enumerate(kept):
        new_entry = dict(c)
        new_entry["priority"] = idx + 1
        new_chain.append(new_entry)
    for idx, c in enumerate(demoted):
        new_entry = dict(c)
        new_entry["priority"] = 99 + idx
        new_chain.append(new_entry)
    return new_chain, log_lines


def reorder_config(
    config_path: Path,
    telemetry_db: str,
    window_h: int,
    min_samples: int,
    max_tps: float,
    max_ttft_ms: float,
    dry_run: bool = False,
    use_u_formula: bool = False,
    tier_config_path: Optional[Path] = None,
) -> Tuple[int, int, List[str]]:
    """Reorder all virtual_models in config. Returns (total_models, reordered_count, log).

    total_models = number of virtual models in config.
    reordered_count = number of models whose chain order changed.

    When use_u_formula=True, uses U = A * (I^w / (C_opp * T)) from
    rank_models.py (joins benchmark scores + provider categories + telemetry).
    Otherwise uses the original telemetry-only composite.
    """
    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    virtual_models = config.get("virtual_models", {})
    if not virtual_models:
        logger.warning("no virtual_models: block in %s", config_path)
        return 0, 0, ["no virtual_models found"]

    stats = load_telemetry(telemetry_db, window_h, min_samples)
    penalties = load_penalty_scores()
    quality_scores = load_quality_scores()
    chain_pins = load_chain_pins()

    # U-formula loads (only when enabled — ponytail: pay only what you use).
    tier_cfg: Dict[str, "object"] = {}
    bench: Dict[str, "object"] = {}
    cats: Dict[str, "object"] = {}
    if use_u_formula:
        rm = _get_rank_models()
        tier_path = tier_config_path or DEFAULT_TIER_CONFIG
        tier_cfg = rm.load_tier_config(tier_path)
        bench = rm.load_benchmark_scores()
        cats = rm.load_provider_categories()
        log_lines_intro = (
            f"U-formula mode: tier_cfg={tier_path.name} "
            f"bench={len(bench)} cats={len(cats)} "
            f"stats={len(stats)} pen={len(penalties)}"
        )
    else:
        # #343: provider categories needed for the free-baseline filter even
        # in composite mode. Loaded once, reused across all chains.
        try:
            rm = _get_rank_models()
            cats = rm.load_provider_categories()
        except Exception as exc:
            logger.debug(
                "provider categories unavailable, free-baseline skipped: %r",
                exc,
            )
        log_lines_intro = (
            f"composite mode: stats={len(stats)} pen={len(penalties)} cats={len(cats)}"
        )

    log_lines: List[str] = [log_lines_intro]
    reordered_count = 0

    for model_id, model_cfg in virtual_models.items():
        chain = model_cfg.get("fallback_chain", [])
        if not chain:
            continue
        original_order = [(c.get("provider"), c.get("model")) for c in chain]
        if use_u_formula:
            rm = _get_rank_models()
            tier = rm.get_tier(tier_cfg, model_id)
            # #756: pass chain pins through in u-formula mode too (was
            # silently ignored — only the composite branch honored them).
            new_chain, reasons_raw = rm.rank_chain(
                chain, stats, penalties, bench, cats, tier,
                pins=chain_pins.get(model_id),
            )
            # Build score lookup from RankReason objects for #343 baseline.
            score_lookup: Dict[Tuple[str, str], float] = {
                (r.provider, r.model): r.score for r in reasons_raw
            }
            reasons = [
                f"  #{i + 1} {r.provider}/{r.model}: U={r.score:.4f} — {r.reason}"
                for i, r in enumerate(reasons_raw)
            ]
        else:
            new_chain, reasons = reorder_chain(
                chain,
                stats,
                penalties,
                min_samples,
                max_tps,
                max_ttft_ms,
                quality_scores,
                chain_pins.get(model_id),
            )
            # Build score lookup by recomputing composite per entry.
            score_lookup = {}
            for c in chain:
                prov = c.get("provider", "")
                mdl = c.get("model", "")
                stat = stats.get((prov, mdl))
                pen = penalties.get((prov, mdl), 0.0)
                quality = quality_scores.get(f"{prov}/{mdl}", 0.0)
                s, _ = compute_composite(
                    stat, pen, min_samples, max_tps, max_ttft_ms, quality
                )
                score_lookup[(prov, mdl)] = s

        # #343: free-model baseline — demote non-free entries scoring below
        # the worst free entry to last-resort (priority 99). Only applies
        # when provider categories are available.
        if cats:
            new_chain, baseline_logs = apply_free_baseline(
                new_chain, score_lookup, cats
            )
            if baseline_logs:
                log_lines.append(f"[{model_id}] free-baseline demotions:")
                log_lines.extend(f"  {line}" for line in baseline_logs)

        new_order = [(c.get("provider"), c.get("model")) for c in new_chain]
        if new_order != original_order:
            reordered_count += 1
            log_lines.append(f"[{model_id}] REORDERED ({len(chain)} entries):")
        else:
            log_lines.append(f"[{model_id}] unchanged ({len(chain)} entries):")
        log_lines.extend(reasons)
        model_cfg["fallback_chain"] = new_chain

    # #405: inject policy-fixed virtual models (e.g. safe-coding via PrivAiTe)
    # from a separate static file so the telemetry rebuild never clobbers them.
    # ponytail: separate file = explicit, visible injection, not string-concat.
    static_vms = load_static_virtual_models(STATIC_VIRTUAL_MODELS_PATH)
    if static_vms:
        injected = 0
        for sid, scfg in static_vms.items():
            if sid in virtual_models:
                logger.warning(
                    "static virtual model %s already present in config, skipping", sid
                )
                continue
            virtual_models[sid] = scfg
            injected += 1
            log_lines.append(
                f"[{sid}] STATIC injected from {STATIC_VIRTUAL_MODELS_PATH.name}"
            )
        if injected:
            log_lines.append(
                f"injected {injected} static virtual model(s) from "
                f"{STATIC_VIRTUAL_MODELS_PATH.name}"
            )

    # Update metadata.generated_at (if present) for audit trail.
    metadata = config.get("metadata", {})
    if metadata:
        metadata["last_reorder_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        metadata["last_reorder_stats"] = {
            "total_models": len(virtual_models),
            "reordered": reordered_count,
            "window_h": window_h,
            "min_samples": min_samples,
            "mode": "u_formula" if use_u_formula else "composite",
        }
        config["metadata"] = metadata

    if dry_run:
        log_lines.insert(1, f"DRY RUN — would rewrite {config_path}")
    else:
        backup_path = config_path.with_suffix(f".yaml.bak-{int(time.time())}")
        shutil.copy2(config_path, backup_path)
        logger.info("backed up %s -> %s", config_path, backup_path)
        with config_path.open("w") as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
        log_lines.insert(1, f"REWROTE {config_path} (backup: {backup_path.name})")

    return len(virtual_models), reordered_count, log_lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reorder virtual_models.yaml fallback chains from live telemetry"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to virtual_models.yaml (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--telemetry-db",
        default=DEFAULT_TELEMETRY_DB,
        help=f"Path to telemetry SQLite (default: {DEFAULT_TELEMETRY_DB})",
    )
    parser.add_argument(
        "--window-h",
        type=int,
        default=DEFAULT_WINDOW_H,
        help=f"Telemetry window in hours (default: {DEFAULT_WINDOW_H})",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=DEFAULT_MIN_SAMPLES,
        help=f"Min samples per (provider, model) to reorder (default: {DEFAULT_MIN_SAMPLES})",
    )
    parser.add_argument(
        "--max-tps",
        type=float,
        default=DEFAULT_MAX_TPS,
        help=f"TPS normalization ceiling (default: {DEFAULT_MAX_TPS})",
    )
    parser.add_argument(
        "--max-ttft-ms",
        type=float,
        default=DEFAULT_MAX_TTFT_MS,
        help=f"TTFT normalization ceiling in ms (default: {DEFAULT_MAX_TTFT_MS})",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print plan, don't write"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    parser.add_argument(
        "--use-u-formula",
        action="store_true",
        help="Use U = A * (I^w / (C_opp * T)) scoring (joins benchmark + provider categories)",
    )
    parser.add_argument(
        "--tier-config",
        type=Path,
        default=DEFAULT_TIER_CONFIG,
        help=f"Path to tier_config.yaml (default: {DEFAULT_TIER_CONFIG})",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.config.exists():
        logger.error("config not found: %s", args.config)
        return 1

    total, reordered, log_lines = reorder_config(
        config_path=args.config,
        telemetry_db=args.telemetry_db,
        window_h=args.window_h,
        min_samples=args.min_samples,
        max_tps=args.max_tps,
        max_ttft_ms=args.max_ttft_ms,
        dry_run=args.dry_run,
        use_u_formula=args.use_u_formula,
        tier_config_path=args.tier_config,
    )

    for line in log_lines:
        print(line)

    logger.info(
        "done: %d/%d virtual models reordered (window=%dh, min_samples=%d)",
        reordered,
        total,
        args.window_h,
        args.min_samples,
    )
    if reordered == 0:
        return 2  # no reorder needed
    return 0


if __name__ == "__main__":
    sys.exit(main())
