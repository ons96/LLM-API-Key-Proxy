"""Per-provider billing-archetype classification + within-provider model ranking
(task-board #253).

Classifies each provider into one of three free-tier billing archetypes so the
gateway can pick models that stretch free quotas as far as possible:

    A  fixed credit pool (one-time signup bonus). Each call burns from a finite
       pool that does NOT reset (or resets rarely). -> prefer the cheapest
       good-enough model for low-priority traffic; save quality for high-value.
    B  flat quota, rate-limited but effectively unlimited resets (RPM/RPD caps,
       no per-call price differential). -> always pick the best model; model-
       side price labels are irrelevant.
    C  daily-resetting / check-in credits (temporal). -> spend the free quota on
       the highest-value work first, since it refills each day.

GOTCHA (Opus 4.8, 2026-06-24): the #253 spec references columns that DO NOT
EXIST in llm-provider-manager/llm_providers.db. Spec asked for `pricing_tier`,
`free_credit_daily`, `model_per_token_cost`, `cost_archetype`,
`quota_reset_strategy`. The REAL providers schema has:
    free_tier, free_one_time, free_daily, free_unlimited,
    checkin_required, checkin_unlimited, no_api_key_required,
    rate_limit_rpm, rate_limit_daily_tokens
and models has: tier, tps, context_window (NO per-token cost anywhere).

Consequence for Archetype A: there is NO per-token cost data, so the spec's
"rank by quality / credit_cost" cannot use a real $ cost. We degrade honestly:
within an A provider we rank by quality but, for low-priority traffic, bias
toward *smaller/faster* models (lower tps-adjusted footprint) as a cost proxy,
and we expose `cost_per_call_estimate=None` in telemetry rather than fabricating
a number. If a real per-token-cost column is added later, swap the proxy in
_cost_proxy() for the real value -- that is the only change needed.

Stdlib only (sqlite3). No new deps. Read-only DB access.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence

Archetype = Literal["A", "B", "C"]

DEFAULT_DB = os.path.expanduser(
    "~/CodingProjects/llm-provider-manager/llm_providers.db"
)

# Backwards-compat default: unknown provider -> B (flat quota) is the safest
# assumption per spec (treat as "just pick the best model").
DEFAULT_ARCHETYPE: Archetype = "B"


@dataclass
class ProviderProfile:
    key_name: str
    archetype: Archetype
    rate_limit_rpm: Optional[int]
    rate_limit_daily_tokens: Optional[int]
    free_unlimited: bool
    free_one_time: bool
    free_daily: bool
    checkin_required: bool
    reason: str


def classify_from_flags(
    *,
    free_one_time: int = 0,
    free_daily: int = 0,
    free_unlimited: int = 0,
    checkin_required: int = 0,
    checkin_unlimited: int = 0,
) -> tuple[Archetype, str]:
    """Pure classifier from the REAL provider flag columns.

    Precedence is deliberate and ordered by how much the choice of MODEL
    affects how far the free tier stretches:

      1. one-time credit pool (A): a finite pool that doesn't refill -> model
         choice matters most (burn cheap on low-value work).
      2. daily/check-in reset (C): refills, so temporal "spend best work first".
         checkin_unlimited collapses to B (refills with no finite cap).
      3. unlimited-with-rate-limit (B): flat quota, model choice doesn't change
         how many calls you get -> just pick the best.

    Returns (archetype, human-readable reason).
    """
    if free_one_time:
        return "A", "free_one_time: finite credit pool, no reset"
    if checkin_unlimited:
        # check-in but unlimited once checked in -> behaves like flat quota
        return "B", "checkin_unlimited: flat quota after check-in"
    if free_daily or checkin_required:
        return "C", "daily/check-in reset: temporal quota, refills"
    if free_unlimited:
        return "B", "free_unlimited: flat rate-limited quota"
    return DEFAULT_ARCHETYPE, "no free-tier flags set -> default B (safest)"


class CostEfficiencyClassifier:
    """Loads provider profiles from the DB and classifies archetypes.

    One DB read at construction (or on refresh()); per-request lookups are
    dict hits (Oracle-micro friendly, no per-request query).
    """

    def __init__(self, db_path: str = DEFAULT_DB):
        self.db_path = db_path
        self._profiles: Dict[str, ProviderProfile] = {}
        self.refresh()

    def refresh(self) -> None:
        self._profiles.clear()
        if not os.path.exists(self.db_path):
            return
        try:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=1.0)
        except sqlite3.Error:
            return
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT key_name, free_one_time, free_daily, free_unlimited,
                          checkin_required, checkin_unlimited,
                          rate_limit_rpm, rate_limit_daily_tokens
                   FROM providers"""
            ).fetchall()
        except sqlite3.Error:
            return
        finally:
            conn.close()
        for r in rows:
            arche, reason = classify_from_flags(
                free_one_time=r["free_one_time"] or 0,
                free_daily=r["free_daily"] or 0,
                free_unlimited=r["free_unlimited"] or 0,
                checkin_required=r["checkin_required"] or 0,
                checkin_unlimited=r["checkin_unlimited"] or 0,
            )
            key = (r["key_name"] or "").lower()
            self._profiles[key] = ProviderProfile(
                key_name=key,
                archetype=arche,
                rate_limit_rpm=r["rate_limit_rpm"],
                rate_limit_daily_tokens=r["rate_limit_daily_tokens"],
                free_unlimited=bool(r["free_unlimited"]),
                free_one_time=bool(r["free_one_time"]),
                free_daily=bool(r["free_daily"]),
                checkin_required=bool(r["checkin_required"]),
                reason=reason,
            )

    def classify_provider(self, provider_id: str) -> Archetype:
        prof = self._profiles.get((provider_id or "").lower())
        return prof.archetype if prof else DEFAULT_ARCHETYPE

    def profile(self, provider_id: str) -> Optional[ProviderProfile]:
        return self._profiles.get((provider_id or "").lower())

    def rank_within_provider(
        self,
        provider_id: str,
        models: Sequence[dict],
        *,
        priority: str = "normal",
        quality_floor: float = 0.0,
    ) -> List[dict]:
        """Order a provider's candidate models per its archetype.

        `models`: dicts with at least {"model_id", "quality" (0..1)} and
        optionally {"tps", "context_window"}. Returns a new ordered list,
        best-first for the archetype.

        `priority`: "low" | "normal" | "high". Only archetype A uses it (low
        priority -> bias toward cheap proxy; high -> quality).
        """
        cands = [m for m in models if m.get("quality", 0.0) >= quality_floor]
        if not cands:
            cands = list(models)  # quality_floor wiped everything -> ignore it
        arche = self.classify_provider(provider_id)

        if arche == "B":
            # flat quota: model choice doesn't change call budget -> best quality.
            key = lambda m: (-m.get("quality", 0.0), m.get("model_id", ""))
        elif arche == "C":
            # temporal refill: spend best work first -> highest quality first.
            # (No hours_until_reset in DB; daily reset is the common case, so
            # quality-first is the correct default. Documented limitation.)
            key = lambda m: (-m.get("quality", 0.0), m.get("model_id", ""))
        else:  # "A" finite pool: cost-aware
            if priority == "high":
                key = lambda m: (-m.get("quality", 0.0), m.get("model_id", ""))
            else:
                # cost proxy: prefer cheaper (smaller/faster) model that still
                # clears the floor. quality/cost where cost proxy is _cost_proxy.
                key = lambda m: (
                    -(m.get("quality", 0.0) / _cost_proxy(m)),
                    m.get("model_id", ""),
                )
        return sorted(cands, key=key)

    def cost_per_call_estimate(self, provider_id: str, model: dict) -> Optional[float]:
        """Telemetry field. None when no real cost data exists (the honest value).

        For archetype B/C the per-call cost against the free quota is "1 call"
        (flat), so we return None ($-cost unknown/irrelevant). For archetype A
        we'd return a real $ estimate IF the DB had per-token cost -- it does
        not, so we return None and rely on cost_archetype for auditing.
        """
        return None  # ponytail: no per-token cost column exists; don't fabricate


def _cost_proxy(model: dict) -> float:
    """Stand-in for per-call cost when no $ data exists.

    Bigger context window + slower tps ~ heavier/more-expensive model. Normalize
    to a positive multiplier >= 1.0 so quality/cost stays well-defined. This is
    a PROXY; replace with real per-token cost if/when the column is added.
    # ponytail: crude heuristic, swap for real cost column when it exists.
    """
    ctx = model.get("context_window") or 8000
    # heavier context -> higher proxy cost, gently (log-ish via sqrt-free ratio)
    cost = 1.0 + (ctx / 200_000.0)
    return max(1.0, cost)


# ---- runnable self-test (no framework; `python cost_efficiency.py`) --------
def _demo() -> None:
    # classify_from_flags: the decision table
    assert classify_from_flags(free_one_time=1)[0] == "A"
    assert classify_from_flags(free_daily=1)[0] == "C"
    assert classify_from_flags(checkin_required=1)[0] == "C"
    assert classify_from_flags(free_unlimited=1)[0] == "B"
    assert classify_from_flags(checkin_unlimited=1)[0] == "B"
    assert classify_from_flags()[0] == "B"  # nothing set -> safe default
    # precedence: one-time beats everything (finite pool dominates)
    assert classify_from_flags(free_one_time=1, free_unlimited=1, free_daily=1)[0] == "A"
    # daily beats unlimited (temporal refill signal wins over flat)
    assert classify_from_flags(free_daily=1, free_unlimited=1)[0] == "C"

    # rank_within_provider on a synthetic DB-less classifier
    clf = CostEfficiencyClassifier(db_path="/nonexistent.db")  # empty profiles
    clf._profiles["acme_a"] = ProviderProfile(
        "acme_a", "A", None, None, False, True, False, False, "test"
    )
    clf._profiles["acme_b"] = ProviderProfile(
        "acme_b", "B", 30, None, True, False, False, False, "test"
    )
    models = [
        {"model_id": "big-great", "quality": 0.95, "context_window": 200_000},
        {"model_id": "small-good", "quality": 0.80, "context_window": 8_000},
    ]
    # Archetype B: always best quality first, regardless of cost.
    b_order = [m["model_id"] for m in clf.rank_within_provider("acme_b", models)]
    assert b_order[0] == "big-great", b_order

    # Archetype A, high priority: quality first.
    a_high = [m["model_id"] for m in clf.rank_within_provider("acme_a", models, priority="high")]
    assert a_high[0] == "big-great", a_high

    # Archetype A, low priority: cost proxy favors the cheaper small model
    # (0.80 / ~1.04) vs (0.95 / 2.0) -> small-good wins.
    a_low = [m["model_id"] for m in clf.rank_within_provider("acme_a", models, priority="low")]
    assert a_low[0] == "small-good", a_low

    # quality_floor filters
    floored = clf.rank_within_provider("acme_b", models, quality_floor=0.9)
    assert [m["model_id"] for m in floored] == ["big-great"], floored

    # cost estimate is honest None (no real cost column)
    assert clf.cost_per_call_estimate("acme_a", models[0]) is None

    # unknown provider -> default B
    assert clf.classify_provider("never-heard-of-it") == "B"

    # Against the REAL DB if present: spot-check known providers.
    real = CostEfficiencyClassifier()
    if real._profiles:
        # groq/gemini/nvidia are free_unlimited rate-limited -> B
        for p in ("groq", "gemini", "nvidia", "opencode_zen"):
            if real.profile(p):
                assert real.classify_provider(p) == "B", (p, real.classify_provider(p))
        # freetheai has checkin_required=1 AND checkin_unlimited=1: "check in
        # daily, then ALL models unlimited for 24h." Once checked in, model
        # choice doesn't stretch the quota -> B is correct (checkin_unlimited
        # takes precedence over the free_daily flag). This is the intended
        # behavior; B for an unlimited-after-checkin provider is right.
        if real.profile("freetheai"):
            assert real.classify_provider("freetheai") == "B"
        # A genuine daily-reset-with-cap provider (free_daily, no checkin_unlimited)
        # would classify C; verify the rule directly via the pure classifier.
        assert classify_from_flags(free_daily=1, checkin_required=1)[0] == "C"
        print(f"real DB: classified {len(real._profiles)} providers")

    print("cost_efficiency self-test: OK")


if __name__ == "__main__":
    _demo()
