#!/usr/bin/env python3
"""Stage 2 decision-comparison report (task-board #1090).

Aggregates the workload-router dry-run JSONL against the live chains in
config/virtual_models.yaml into a prompt-free report: agreement rate,
disagreement buckets by phase, and no_eligible_deployment share.

Read-only: touches nothing but the JSONL and the chain config.

Usage: python3 tools/workload_router_stage2_report.py [jsonl_path]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_LOG = REPO / "logs" / "workload-router-dry-run.jsonl"
CHAINS = REPO / "config" / "virtual_models.yaml"

# Alias -> tier the live chain is designed to serve. Agreement = shadow tier
# matches the live alias's intent.
ALIAS_TIER = {
    "coding-fast": "fast_general",
    "coding-elite": "code_implement",
    "chat-elite": "deep_reasoning",
    "chat-smart": "deep_reasoning",
    "chat-fast": "fast_general",
    "auto": None,  # delegated to classifier; always "agree" if tier is valid
}


def live_tier_for(alias: str) -> str | None:
    return ALIAS_TIER.get(alias)


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LOG
    if not path.exists():
        print(f"no shadow log at {path} -- nothing to report")
        return 1

    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    total = len(records)
    if not total:
        print("shadow log empty -- nothing to report")
        return 1

    why = Counter(r.get("why", "") for r in records)
    phase = Counter(r.get("phase", "") or "(none)" for r in records)
    tier = Counter(r.get("tier", "") for r in records)
    deployment = Counter(r.get("deployment") or "(none)" for r in records)

    no_elig = why.get("no_eligible_deployment", 0)
    agree = disagree = unclassified = 0
    disagree_buckets: Counter[str] = Counter()
    for r in records:
        alias = r.get("live_model", "")
        want = live_tier_for(alias)
        got = r.get("tier", "")
        if want is None and alias == "auto":
            agree += 1  # classifier-delegated: any valid tier counts
        elif want is None:
            unclassified += 1
        elif want == got:
            agree += 1
        else:
            disagree += 1
            disagree_buckets[f"{r.get('phase') or '(none)'}: want {want} got {got}"] += 1

    comparable = agree + disagree
    agreement_rate = (agree / comparable * 100) if comparable else 0.0

    print("# Stage 2 shadow report -- workload-router vs live chain")
    print(f"records: {total}  (log: {path})")
    print()
    print("## why distribution (Stage 1 expected set gate)")
    for k, v in why.most_common():
        print(f"  {k or '(empty)'}: {v}")
    print()
    print("## tier / deployment / phase")
    print(f"  tiers: {dict(tier.most_common())}")
    print(f"  deployments: {dict(deployment.most_common())}")
    print(f"  phases: {dict(phase.most_common())}")
    print()
    print("## agreement vs live alias intent")
    print(f"  agreement: {agree}/{comparable} ({agreement_rate:.1f}%)")
    print(f"  unclassified aliases: {unclassified}")
    print(f"  no_eligible_deployment share: {no_elig}/{total} ({no_elig / total * 100:.1f}%)")
    if disagree_buckets:
        print("  disagreement buckets:")
        for k, v in disagree_buckets.most_common():
            print(f"    {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
