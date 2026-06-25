"""Telemetry-driven fallback chain re-ranking (task-board #251).

Replaces the static config-ordered fallback chain with a runtime ranking
derived from recent real request outcomes.

GOTCHA (Opus 4.8, 2026-06-24): the #251 spec says read `telemetry/events.jsonl`.
That file DOES NOT EXIST. The real telemetry is a SQLite table `llm_events`
at /dev/shm/telemetry.db, written by src/proxy_app/telemetry/logger.py.
This module reads that table. Do not build against the spec's fictional path.

Score (per candidate provider, all components normalized 0..1 across the
candidate set):
    score = 0.40 * uptime_ema          # recency-weighted success rate, 30min half-life
          + 0.25 * quality             # static model/provider quality, 0..1
          + 0.15 * (1 - fail_penalty)  # recent-failure recency penalty
          + 0.10 * cost_eff            # cheaper => higher
          + 0.10 * load_spread_bonus   # less-used-this-window => higher

Constraints honored: stdlib only (sqlite3 + math), numpy-free, <20MB RAM
(single aggregate query, no full-table load), recompute throttled to >=30s.

Backwards-compat: if telemetry is missing/empty or USE_DYNAMIC_CHAIN is off,
rank() returns the input order unchanged (caller keeps the static chain).
"""

from __future__ import annotations

import math
import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# ---- tunables (spec-fixed; change only with a spec change) -----------------
WEIGHT_UPTIME = 0.40
WEIGHT_QUALITY = 0.25
WEIGHT_FAIL = 0.15
WEIGHT_COST = 0.10
WEIGHT_SPREAD = 0.10

UPTIME_HALFLIFE_S = 30 * 60          # 30 min
FAIL_PENALTY_FLOOR_S = 60            # < this since last failure => full penalty
FAIL_PENALTY_ZERO_S = 5 * 60        # >= this => no penalty (linear between)
RECOMPUTE_MIN_INTERVAL_S = 30        # throttle recompute
COLD_START_S = 10 * 60               # first N seconds: trust static order
LOOKBACK_S = 60 * 60                 # only consider events from the last hour

DEFAULT_DB_PATH = "/dev/shm/telemetry.db"


@dataclass
class ProviderStats:
    provider: str
    uptime_ema: float = 0.0          # 0..1, recency-weighted success fraction
    last_failure_age_s: Optional[float] = None
    request_count: int = 0           # in window, for load spread
    seen: bool = False               # had any telemetry at all


@dataclass
class DynamicChainRanker:
    db_path: str = DEFAULT_DB_PATH
    quality: Dict[str, float] = field(default_factory=dict)   # provider -> 0..1
    cost: Dict[str, float] = field(default_factory=dict)      # provider -> $/unit (lower better)
    enabled: bool = True
    _started_at: float = field(default_factory=time.time)
    _last_compute: float = 0.0
    _cached_order: Tuple[str, ...] = ()
    _cache_key: Tuple[str, ...] = ()

    # -- public ------------------------------------------------------------
    def rank(
        self,
        candidates: Sequence[str],
        *,
        now: Optional[float] = None,
        force: bool = False,
    ) -> List[str]:
        """Return candidates re-ordered best-first.

        Falls back to the input order (stable) whenever dynamic ranking
        cannot apply: disabled, cold-start window, no telemetry, or a recent
        cached result for the same candidate set within the throttle window.
        """
        now = time.time() if now is None else now
        candidates = [c.lower() for c in candidates]
        if not self.enabled or len(candidates) <= 1:
            return list(candidates)

        # Cold start: not enough runtime history to trust telemetry yet.
        if now - self._started_at < COLD_START_S:
            return list(candidates)

        key = tuple(candidates)
        # Throttle: reuse cache unless forced (e.g. on a 429/5xx event) or stale.
        if (
            not force
            and key == self._cache_key
            and (now - self._last_compute) < RECOMPUTE_MIN_INTERVAL_S
            and self._cached_order
        ):
            return list(self._cached_order)

        stats = self._load_stats(candidates, now)
        if not any(s.seen for s in stats.values()):
            # No telemetry at all -> keep static order, but don't thrash the DB.
            self._last_compute = now
            self._cache_key = key
            self._cached_order = key
            return list(candidates)

        scored = self._score(candidates, stats, now)
        ordered = [p for _, p in scored]

        self._last_compute = now
        self._cache_key = key
        self._cached_order = tuple(ordered)
        return ordered

    # -- internals ---------------------------------------------------------
    def _load_stats(self, candidates: Sequence[str], now: float) -> Dict[str, ProviderStats]:
        stats = {c: ProviderStats(provider=c) for c in candidates}
        if not os.path.exists(self.db_path):
            return stats
        since = now - LOOKBACK_S
        try:
            # read-only, immutable-ish: don't create the file, short timeout.
            conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro", uri=True, timeout=0.5
            )
        except sqlite3.Error:
            return stats
        try:
            conn.row_factory = sqlite3.Row
            # Pull only window rows for the candidate providers. Aggregation
            # (EMA, last-failure) is done in Python because SQLite has no EMA.
            qmarks = ",".join("?" * len(candidates))
            rows = conn.execute(
                f"""
                SELECT provider, ts_start, status
                FROM llm_events
                WHERE ts_start >= ?
                  AND lower(provider) IN ({qmarks})
                ORDER BY ts_start ASC
                """,
                (since, *candidates),
            ).fetchall()
        except sqlite3.Error:
            return stats
        finally:
            conn.close()

        # EMA accumulators per provider.
        num: Dict[str, float] = {c: 0.0 for c in candidates}   # sum w*success
        den: Dict[str, float] = {c: 0.0 for c in candidates}   # sum w
        for r in rows:
            prov = (r["provider"] or "").lower()
            if prov not in stats:
                continue
            ts = r["ts_start"]
            if ts is None:
                continue
            age = max(0.0, now - float(ts))
            w = 0.5 ** (age / UPTIME_HALFLIFE_S)
            success = 1.0 if _is_ok(r["status"]) else 0.0
            num[prov] += w * success
            den[prov] += w
            s = stats[prov]
            s.seen = True
            s.request_count += 1
            if success == 0.0:
                # rows are ASC, so this keeps the most-recent failure age
                s.last_failure_age_s = age

        for c in candidates:
            if den[c] > 0:
                stats[c].uptime_ema = num[c] / den[c]
        return stats

    def _score(
        self, candidates: Sequence[str], stats: Dict[str, ProviderStats], now: float
    ) -> List[Tuple[float, str]]:
        # Raw component values per candidate.
        uptime = {c: stats[c].uptime_ema for c in candidates}
        fail_pen = {c: _fail_penalty(stats[c].last_failure_age_s) for c in candidates}
        quality = {c: self.quality.get(c, 0.5) for c in candidates}
        # cost_eff: lower cost -> higher eff. Unknown cost -> mid (0.5 after norm).
        raw_cost = {c: self.cost.get(c) for c in candidates}
        load = {c: float(stats[c].request_count) for c in candidates}

        # Normalize each component to 0..1 across the candidate set.
        n_uptime = _minmax(uptime)
        n_quality = _minmax(quality)
        # fail penalty is already 0..1 and directional; keep as-is.
        cost_eff = _cost_eff(raw_cost)
        # spread bonus: fewer requests this window => higher bonus
        spread = _spread_bonus(load)

        scored: List[Tuple[float, str]] = []
        for c in candidates:
            score = (
                WEIGHT_UPTIME * n_uptime[c]
                + WEIGHT_QUALITY * n_quality[c]
                + WEIGHT_FAIL * (1.0 - fail_pen[c])
                + WEIGHT_COST * cost_eff[c]
                + WEIGHT_SPREAD * spread[c]
            )
            scored.append((score, c))

        # Sort best-first. Deterministic tiebreak (#251 spec order):
        #   higher quality > higher uptime(proxy for TPM throughput)
        #   > less-used-this-window (unused-this-hour) > alphabetical.
        idx = {c: i for i, c in enumerate(candidates)}  # stable input fallback
        scored.sort(
            key=lambda t: (
                -round(t[0], 9),
                -quality[t[1]],
                -uptime[t[1]],
                load[t[1]],
                t[1],
                idx[t[1]],
            )
        )
        return scored


# ---- pure helpers (each independently testable) ----------------------------
def _is_ok(status: Optional[str]) -> bool:
    if status is None:
        return False
    s = str(status).strip().lower()
    return s in ("ok", "success", "200", "completed")


def _fail_penalty(age_s: Optional[float]) -> float:
    """1.0 if last failure < 60s ago, linear to 0 at 5min, 0 if older/none."""
    if age_s is None:
        return 0.0
    if age_s < FAIL_PENALTY_FLOOR_S:
        return 1.0
    if age_s >= FAIL_PENALTY_ZERO_S:
        return 0.0
    span = FAIL_PENALTY_ZERO_S - FAIL_PENALTY_FLOOR_S
    return 1.0 - (age_s - FAIL_PENALTY_FLOOR_S) / span


def _minmax(values: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalize to 0..1. All-equal -> all 0.5 (neutral)."""
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if hi - lo < 1e-12:
        return {k: 0.5 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def _cost_eff(raw_cost: Dict[str, Optional[float]]) -> Dict[str, float]:
    """Lower cost -> higher efficiency (0..1). Unknown cost -> neutral 0.5."""
    known = {k: v for k, v in raw_cost.items() if v is not None}
    if not known:
        return {k: 0.5 for k in raw_cost}
    inv = {k: -v for k, v in known.items()}  # invert so cheaper is larger
    norm = _minmax(inv)
    return {k: norm.get(k, 0.5) for k in raw_cost}


def _spread_bonus(load: Dict[str, float]) -> Dict[str, float]:
    """Fewer requests this window -> higher bonus (0..1)."""
    if not load:
        return {}
    inv = {k: -v for k, v in load.items()}
    return _minmax(inv)


# ---- runnable self-test (no framework; `python dynamic_chain.py`) ----------
def _demo() -> None:
    eps = 1e-9

    # _fail_penalty boundaries
    assert _fail_penalty(None) == 0.0
    assert _fail_penalty(0) == 1.0
    assert _fail_penalty(59) == 1.0
    assert _fail_penalty(60) == 1.0
    assert abs(_fail_penalty(180) - 0.5) < eps     # midpoint of [60,300]
    assert _fail_penalty(300) == 0.0
    assert _fail_penalty(9999) == 0.0

    # _minmax neutral + spread
    assert _minmax({"a": 5, "b": 5}) == {"a": 0.5, "b": 0.5}
    mm = _minmax({"a": 0.0, "b": 1.0, "c": 0.5})
    assert abs(mm["a"]) < eps and abs(mm["b"] - 1.0) < eps and abs(mm["c"] - 0.5) < eps

    # cost_eff: cheaper wins, unknown stays neutral
    ce = _cost_eff({"cheap": 1.0, "pricey": 10.0, "unknown": None})
    assert ce["cheap"] > ce["pricey"]
    assert ce["unknown"] == 0.5

    # spread: less-used wins
    sb = _spread_bonus({"busy": 100.0, "idle": 1.0})
    assert sb["idle"] > sb["busy"]

    # End-to-end against a temp SQLite mirroring the REAL llm_events schema.
    import tempfile

    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    conn = sqlite3.connect(tmp.name)
    conn.execute(
        """CREATE TABLE llm_events (
            id INTEGER PRIMARY KEY, request_id TEXT,
            ts_start REAL, ts_end REAL, ts_first_token REAL,
            model TEXT, provider TEXT, stream INT,
            prompt_tokens INT, completion_tokens INT,
            ttft_ms REAL, total_ms REAL, tps REAL, cost_usd REAL,
            caller TEXT, agent_session TEXT, status TEXT, error TEXT,
            concrete_provider TEXT, concrete_model TEXT,
            waited_for_429 INT, wait_duration_s REAL)"""
    )
    now = time.time()
    # groq: all recent successes. flaky: a failure 30s ago. stale: old successes.
    rows = []
    for i in range(10):
        rows.append((now - i * 5, "groq", "ok"))
    for i in range(8):
        rows.append((now - i * 5, "flaky", "ok"))
    rows.append((now - 30, "flaky", "error"))         # recent failure -> penalty
    for i in range(5):
        rows.append((now - 3000 - i * 5, "stale", "ok"))  # old but ok
    conn.executemany(
        "INSERT INTO llm_events(ts_start, provider, status) VALUES (?,?,?)", rows
    )
    conn.commit()
    conn.close()

    r = DynamicChainRanker(
        db_path=tmp.name,
        quality={"groq": 0.9, "flaky": 0.9, "stale": 0.9},
    )
    r._started_at = now - COLD_START_S - 1  # bypass cold start for the test
    order = r.rank(["stale", "flaky", "groq"], now=now, force=True)
    # groq (recent + no failures) must beat flaky (recent failure penalty).
    # NOTE: uptime_ema is a recency-weighted success *ratio*, so decay cancels
    # in num/den -- an old-but-100%-success provider scores as well on uptime
    # as a fresh one. "Recency of evidence" is intentionally NOT a penalty here
    # (spec uses load_spread_bonus, which actually rewards the less-used one).
    # So `stale` outranking `flaky` is correct: flaky carries a failure penalty.
    assert order.index("groq") < order.index("flaky"), order

    # Cold-start returns input order unchanged.
    r2 = DynamicChainRanker(db_path=tmp.name)
    assert r2.rank(["a", "b", "c"], now=now) == ["a", "b", "c"]

    # Missing DB -> input order unchanged, no crash.
    r3 = DynamicChainRanker(db_path="/nonexistent/telemetry.db")
    r3._started_at = now - COLD_START_S - 1
    assert r3.rank(["a", "b"], now=now) == ["a", "b"]

    os.unlink(tmp.name)
    print("dynamic_chain self-test: OK")


if __name__ == "__main__":
    _demo()
