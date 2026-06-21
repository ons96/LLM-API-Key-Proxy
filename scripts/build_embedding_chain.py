#!/usr/bin/env python3
"""Build free-embedding fallback chain from llm-provider-manager DB.

Joins:
  (1) MTEB retrieval benchmark CSV (#163 output, optional)
  (2) Free-tier provider metadata from llm_providers.db
  (3) Live telemetry SQLite (chat proxy; embedding calls not yet wired)

Outputs:
  - config/embedding_fallback_chain.yaml (LiteLLM Router model_list format)
  - config/embedding_fallback_chain.csv (human-readable scoring breakdown)

Composite: quality*W_Q + rpm_headroom*W_R + latency*W_L + reliability*W_RL
Defaults: 0.50, 0.25, 0.20, 0.05 (env: EMB_W_*).

Single-user workload assumption: 1-5 emb/min sustained, 20/min burst.
Reject providers whose RPM < EMB_MIN_RPM (default 5).

Refs: task-board #207, #194. Depends on #163 (MTEB pipeline) for quality.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    print("PyYAML required: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

logger = logging.getLogger("build_embedding_chain")

DEFAULT_DB = os.environ.get(
    "LLM_PROVIDERS_DB",
    str(Path.home() / "CodingProjects/llm-provider-manager/llm_providers.db"),
)
DEFAULT_TELEMETRY_DB = os.environ.get("TELEMETRY_DB_PATH", "/dev/shm/telemetry.db")
DEFAULT_MTEB_CSV = os.environ.get("MTEB_CSV_PATH", "")
DEFAULT_OUTPUT_YAML = "config/embedding_fallback_chain.yaml"
DEFAULT_OUTPUT_CSV = "config/embedding_fallback_chain.csv"

W_QUALITY = float(os.environ.get("EMB_W_QUALITY", "0.50"))
W_RPM = float(os.environ.get("EMB_W_RPM", "0.25"))
W_LATENCY = float(os.environ.get("EMB_W_LATENCY", "0.20"))
W_RELIABILITY = float(os.environ.get("EMB_W_RELIABILITY", "0.05"))
MIN_RPM_DEFAULT = int(os.environ.get("EMB_MIN_RPM", "5"))
BURST_RPM = int(os.environ.get("EMB_BURST_RPM", "20"))
MAX_TTFT_MS = float(os.environ.get("EMB_MAX_TTFT_MS", "5000"))


@dataclass
class EmbeddingCandidate:
    provider: str
    model: str  # full model_id as stored in DB (may include provider/ prefix)
    base_url: str
    env_var: str
    no_api_key_required: bool
    rpm: int  # provider-level RPM; 0 if unknown
    quality: float = 0.5  # MTEB score; default when CSV absent
    avg_ttft_ms: float = 0.0  # from telemetry; 0 if no data
    success_rate: float = 0.5  # from telemetry; 0.5 default
    samples: int = 0
    composite: float = 0.0
    reject_reason: str = ""
    extras: Dict[str, str] = field(default_factory=dict)

    @property
    def litellm_model(self) -> str:
        """LiteLLM-format model identifier: '<provider>/<model>'."""
        m = self.model
        if "/" in m and m.split("/", 1)[0] == self.provider:
            return m  # already prefixed correctly
        return f"{self.provider}/{m}"


def load_provider_db(db_path: str) -> List[EmbeddingCandidate]:
    """Read llm_providers.db, return all free embedding candidates."""
    if not Path(db_path).exists():
        raise FileNotFoundError(f"provider DB not found: {db_path}")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT p.key_name AS provider, p.base_url, p.env_var,
                   p.no_api_key_required, p.free_tier, p.rate_limit_rpm,
                   m.model_id, m.display_name
            FROM models m
            JOIN providers p ON m.provider_id = p.id
            WHERE (m.model_id LIKE '%embed%' OR m.display_name LIKE '%embed%')
              AND (p.free_tier = 1 OR p.no_api_key_required = 1)
            """
        ).fetchall()
    finally:
        conn.close()

    candidates: List[EmbeddingCandidate] = []
    for r in rows:
        rpm = r["rate_limit_rpm"] or 0
        candidates.append(
            EmbeddingCandidate(
                provider=r["provider"],
                model=r["model_id"],
                base_url=r["base_url"] or "",
                env_var=r["env_var"] or "",
                no_api_key_required=bool(r["no_api_key_required"]),
                rpm=rpm,
            )
        )
    return candidates


def load_mteb_csv(csv_path: str) -> Dict[str, float]:
    """Read optional MTEB CSV. Returns {model_key: score}.

    Expected columns: model (or model_id), score (or mteb_score).
    Rows with non-numeric scores are skipped.
    """
    if not csv_path or not Path(csv_path).exists():
        return {}
    out: Dict[str, float] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row.get("model") or row.get("model_id") or ""
            score_str = row.get("score") or row.get("mteb_score") or ""
            if not model or not score_str:
                continue
            try:
                score = float(score_str)
            except ValueError:
                continue
            # Normalize to 0..1 if it looks like a percentage.
            if score > 1.0:
                score = score / 100.0
            out[model.strip()] = max(0.0, min(1.0, score))
    logger.info("MTEB CSV loaded: %d entries from %s", len(out), csv_path)
    return out


def load_telemetry(db_path: str) -> Dict[Tuple[str, str], Tuple[float, float, int]]:
    """Read llm_events table. Returns {(provider, model): (avg_ttft_ms, success_rate, samples)}.

    NOTE: Telemetry records CHAT events, not embedding calls. We use chat
    calls as a proxy for provider latency/reliability. Limitation documented
    in the chain YAML metadata.
    """
    if not db_path or not Path(db_path).exists():
        return {}
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            """
            SELECT provider, model,
                   AVG(ttft_ms) AS avg_ttft,
                   SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS success_rate,
                   COUNT(*) AS samples
            FROM llm_events
            WHERE ts_start > strftime('%s','now') - 86400
            GROUP BY provider, model
            """
        ).fetchall()
    except sqlite3.OperationalError:
        # Table doesn't exist or empty — return empty.
        return {}
    finally:
        conn.close()
    return {
        (r[0], r[1]): (r[2] or 0.0, r[3] or 0.0, r[4] or 0)
        for r in rows
    }


def compute_composite(
    c: EmbeddingCandidate, min_rpm: int = MIN_RPM_DEFAULT
) -> Tuple[float, str]:
    """Return (composite_score, reject_reason). reject_reason non-empty => drop."""
    if c.rpm < min_rpm and c.rpm != 0:
        return 0.0, f"rpm={c.rpm} < MIN_RPM={min_rpm}"
    if c.rpm == 0:
        # Unknown RPM: accept but flag. We prefer known RPM but don't reject
        # providers we haven't rate-limited yet.
        c.extras["rpm_unknown"] = "true"

    rpm_headroom = min(c.rpm / float(BURST_RPM), 1.0) if c.rpm else 0.5
    latency = 1.0 - min(c.avg_ttft_ms / MAX_TTFT_MS, 1.0) if c.avg_ttft_ms else 0.5
    reliability = c.success_rate

    composite = (
        c.quality * W_QUALITY
        + rpm_headroom * W_RPM
        + latency * W_LATENCY
        + reliability * W_RELIABILITY
    )
    c.composite = composite
    return composite, ""


def build_chain(
    candidates: List[EmbeddingCandidate],
    mteb: Dict[str, float],
    telemetry: Dict[Tuple[str, str], Tuple[float, float, int]],
    min_rpm: int = MIN_RPM_DEFAULT,
) -> Tuple[List[EmbeddingCandidate], List[EmbeddingCandidate]]:
    """Apply MTEB + telemetry, compute composite, split accepted/rejected."""
    for c in candidates:
        # Apply MTEB score if we have one for this model.
        for key in (c.model, c.litellm_model, c.model.split("/")[-1]):
            if key in mteb:
                c.quality = mteb[key]
                break
        # Apply telemetry if present.
        tkey = (c.provider, c.model.split("/")[-1])
        if tkey in telemetry:
            c.avg_ttft_ms, c.success_rate, c.samples = telemetry[tkey]
        score, reason = compute_composite(c, min_rpm=min_rpm)
        if reason:
            c.reject_reason = reason
        else:
            c.composite = score

    accepted = [c for c in candidates if not c.reject_reason]
    rejected = [c for c in candidates if c.reject_reason]
    # Stable sort: highest composite first.
    accepted.sort(key=lambda x: x.composite, reverse=True)
    return accepted, rejected


def write_yaml(candidates: List[EmbeddingCandidate], out_path: str) -> None:
    """Write LiteLLM Router model_list format YAML."""
    model_list = []
    for c in candidates:
        params: Dict[str, str] = {
            "model": c.litellm_model,
        }
        if c.base_url:
            params["api_base"] = c.base_url
        if c.env_var:
            params["api_key"] = f"os.environ/{c.env_var}"
        model_list.append({"model_name": "embeddings", "litellm_params": params})

    doc = {
        "model_list": model_list,
        "metadata": {
            "description": "Free embedding fallback chain (auto-generated by scripts/build_embedding_chain.py)",
            "generated_at": _now_iso(),
            "weights": {
                "quality": W_QUALITY,
                "rpm_headroom": W_RPM,
                "latency": W_LATENCY,
                "reliability": W_RELIABILITY,
            },
            "notes": [
                "quality defaults to 0.5 when MTEB CSV absent (see #163)",
                "latency/reliability proxied from chat telemetry (embedding-specific hook is future work)",
                "top of chain = best composite; bottom = degraded fallback",
            ],
        },
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False)
    logger.info("wrote YAML chain: %s (%d entries)", out_path, len(model_list))


def write_csv(candidates: List[EmbeddingCandidate], out_path: str) -> None:
    """Write human-readable CSV with all scoring components."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank", "provider", "model", "rpm", "quality", "avg_ttft_ms",
        "success_rate", "samples", "composite", "reject_reason",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, c in enumerate(candidates, start=1):
            w.writerow({
                "rank": i,
                "provider": c.provider,
                "model": c.model,
                "rpm": c.rpm,
                "quality": f"{c.quality:.3f}",
                "avg_ttft_ms": f"{c.avg_ttft_ms:.1f}",
                "success_rate": f"{c.success_rate:.3f}",
                "samples": c.samples,
                "composite": f"{c.composite:.3f}",
                "reject_reason": c.reject_reason,
            })
    logger.info("wrote CSV breakdown: %s (%d entries)", out_path, len(candidates))


def _now_iso() -> str:
    import datetime
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=DEFAULT_DB, help="llm_providers.db path")
    p.add_argument("--mteb-csv", default=DEFAULT_MTEB_CSV, help="MTEB scores CSV (optional)")
    p.add_argument("--telemetry-db", default=DEFAULT_TELEMETRY_DB, help="telemetry SQLite path")
    p.add_argument("--output-yaml", default=DEFAULT_OUTPUT_YAML)
    p.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    p.add_argument("--min-rpm", type=int, default=MIN_RPM_DEFAULT)
    p.add_argument("--dry-run", action="store_true", help="print plan, write nothing")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    # ponytail: parameter-pass, not module-level global. Cleaner for tests.
    try:
        candidates = load_provider_db(args.db)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if not candidates:
        print("ERROR: 0 free embedding models found in DB", file=sys.stderr)
        return 1

    mteb = load_mteb_csv(args.mteb_csv)
    telemetry = load_telemetry(args.telemetry_db)
    accepted, rejected = build_chain(candidates, mteb, telemetry, min_rpm=args.min_rpm)

    print(f"candidates: {len(candidates)} (accepted: {len(accepted)}, rejected: {len(rejected)})")
    for r in rejected:
        print(f"  REJECT {r.provider}/{r.model}: {r.reject_reason}")
    if not accepted:
        print("ERROR: 0 candidates passed filter", file=sys.stderr)
        return 1
    for i, c in enumerate(accepted, 1):
        print(f"  [{i:2d}] {c.provider:12s} {c.model:50s} comp={c.composite:.3f}")

    if args.dry_run:
        print("dry-run: no files written")
        return 2

    write_yaml(accepted, args.output_yaml)
    write_csv(accepted + rejected, args.output_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
