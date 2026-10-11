"""Aggregate redacted routing outcomes into calibration metrics."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def calibrate(path: str | Path) -> dict[str, object]:
    totals: dict[str, int] = defaultdict(int)
    successes: dict[str, int] = defaultdict(int)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        deployment = record.get("deployment")
        outcome = record.get("outcome")
        if not isinstance(deployment, str) or not deployment:
            raise ValueError("each record requires a deployment")
        if outcome not in {"success", "operational_failure", "semantic_failure", "escalated"}:
            raise ValueError("unsupported outcome")
        totals[deployment] += 1
        if outcome == "success":
            successes[deployment] += 1
    deployments = {
        name: {"samples": totals[name], "successes": successes[name],
               "success_rate": successes[name] / totals[name]}
        for name in sorted(totals)
    }
    return {"total_samples": sum(totals.values()), "deployments": deployments}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(calibrate(args.input), indent=2) + "\n"
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
