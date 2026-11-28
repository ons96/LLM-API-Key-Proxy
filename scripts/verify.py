#!/usr/bin/env python3
"""Immutable verifier (#1422).

The verifier is the authority for "does this change pass?". It runs three
pinned stages:

1. the existing pytest suite (the same modules CI already gates on),
2. the golden routing replay (``scripts/replay_routing.py``),
3. the quality-floor check (``scripts/quality_floor.py``).

The verifier's own exit code is the sole pass criterion: 0 means every stage
passed, any non-zero exit means the change does not pass. The verifier and the
files it depends on are protected by ``.github/CODEOWNERS``, so changing them
requires human review.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTEST_TARGETS = [
    "tests/test_fault_injection.py",
    "tests/test_fallback_logic.py",
    "tests/test_chain_selector.py",
    "tests/test_chain_policy.py",
    "tests/test_free_only_mode.py",
    "tests/test_configuration.py",
    "tests/test_cooldown_policy.py",
    "tests/test_verifier_contract.py",
]


def run_stage(name: str, command: Sequence[str]) -> int:
    """Run one verifier stage and report its exit code."""
    print(f"--- verifier stage: {name}", flush=True)
    result = subprocess.run(list(command), cwd=REPO_ROOT, check=False)
    if result.returncode == 0:
        print(f"--- verifier stage: {name} OK", flush=True)
    else:
        print(
            f"--- verifier stage: {name} FAILED (exit {result.returncode})",
            file=sys.stderr,
            flush=True,
        )
    return result.returncode


def main(argv: List[str]) -> int:
    """CLI entry point."""
    args = argv[1:]
    targets = [arg for arg in args if arg.endswith(".py") and arg.startswith("tests/")]
    if targets:
        pytest_targets = targets
    else:
        pytest_targets = PYTEST_TARGETS

    failures = []
    pytest_code = run_stage(
        "pytest",
        [sys.executable, "-m", "pytest", "-q", *pytest_targets],
    )
    if pytest_code != 0:
        failures.append(f"pytest exited {pytest_code}")

    replay_code = run_stage(
        "golden replay",
        [sys.executable, str(REPO_ROOT / "scripts" / "replay_routing.py")],
    )
    if replay_code != 0:
        failures.append(f"golden replay exited {replay_code}")

    floor_code = run_stage(
        "quality floor",
        [sys.executable, str(REPO_ROOT / "scripts" / "quality_floor.py")],
    )
    if floor_code != 0:
        failures.append(f"quality floor exited {floor_code}")

    if failures:
        print("verifier: FAILED", file=sys.stderr)
        for line in failures:
            print(f"  - {line}", file=sys.stderr)
        return 1

    print("verifier: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
