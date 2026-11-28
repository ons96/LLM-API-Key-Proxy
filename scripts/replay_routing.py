#!/usr/bin/env python3
"""Golden routing replay (#1422).

Replays the pinned traces in ``scripts/golden/routing_replay.json`` through the
workload-router routing core and diffs the outcome against the pinned golden
expectations. This script never rewrites the golden file: regenerating
expectations is a human-reviewed change because the golden file is protected by
``.github/CODEOWNERS``.

Exit code 0 means every trace replayed to its golden decision, 1 means at least
one trace drifted or failed to route.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "workload-router"))

from router_core import Capability, Deployment, RequestFeatures, select  # noqa: E402

GOLDEN_PATH = REPO_ROOT / "scripts" / "golden" / "routing_replay.json"


def load_golden(path: Path) -> Dict[str, Any]:
    """Load the pinned golden fixture."""
    return json.loads(path.read_text(encoding="utf-8"))


def build_deployments(entries: List[Dict[str, Any]]) -> List[Deployment]:
    """Materialize golden deployment records."""
    deployments = []
    for entry in entries:
        capability = Capability(entry["capability"])
        deployments.append(
            Deployment(
                deployment_id=entry["deployment_id"],
                capability=capability,
                context_limit=int(entry.get("context_limit", 0)),
                tools=bool(entry.get("tools", False)),
                structured_output=bool(entry.get("structured_output", False)),
                healthy=bool(entry.get("healthy", True)),
                latency_ms=float(entry.get("latency_ms", 1000.0)),
                success_rate=float(entry.get("success_rate", 0.5)),
                provider=entry.get("provider", "unconfigured"),
                model=entry.get("model", ""),
                quota_remaining=entry.get("quota_remaining"),
            )
        )
    return deployments


def replay_trace(trace: Dict[str, Any], deployments: List[Deployment]) -> str:
    """Replay one trace and return the selected deployment id."""
    features = RequestFeatures(
        prompt=trace.get("prompt", ""),
        context_tokens=int(trace.get("context_tokens", 0)),
        tools=bool(trace.get("tools", False)),
        structured_output=bool(trace.get("structured_output", False)),
        phase=trace.get("phase", ""),
    )
    return select(features, deployments).deployment_id


def run(path: Path) -> int:
    """Replay every golden trace and report drift."""
    golden = load_golden(path)
    deployments = build_deployments(golden.get("deployments", []))
    traces = golden.get("traces", [])
    if not deployments or not traces:
        print(f"golden replay: malformed golden fixture at {path}", file=sys.stderr)
        return 1

    mismatches = []
    for trace in traces:
        name = trace.get("name", "<unnamed>")
        expected = trace.get("expected")
        try:
            actual = replay_trace(trace, deployments)
        except LookupError as exc:
            mismatches.append(f"{name}: no eligible deployment ({exc})")
            continue
        if actual != expected:
            mismatches.append(f"{name}: expected {expected}, got {actual}")

    if mismatches:
        print("golden replay: FAILED", file=sys.stderr)
        for line in mismatches:
            print(f"  - {line}", file=sys.stderr)
        return 1

    print(f"golden replay: OK ({len(traces)} traces matched the golden file)")
    return 0


def main(argv: List[str]) -> int:
    """CLI entry point."""
    path = Path(argv[1]) if len(argv) > 1 else GOLDEN_PATH
    if not path.exists():
        print(f"golden replay: missing golden file {path}", file=sys.stderr)
        return 1
    return run(path)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
