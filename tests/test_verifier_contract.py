"""Contract tests for the immutable verifier suite (#1422).

These tests pin the mechanical guarantees: the verifier authority exists, the
protected paths are owned by a human, and the protected-path gate actually
rejects edits to them.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CODEOWNERS = REPO_ROOT / ".github" / "CODEOWNERS"
CHECKER = REPO_ROOT / "scripts" / "check_protected_paths.py"

PROTECTED_PATHS = (
    "tests/test_router_core.py",
    "scripts/replay_routing.py",
    "scripts/quality_floor.py",
    "scripts/verify.py",
    "src/proxy_app/telemetry/logger.py",
    ".github/workflows/verifier.yml",
    ".github/CODEOWNERS",
)

UNPROTECTED_PATHS = (
    "src/proxy_app/main.py",
    "src/rotator_library/client.py",
    "config/router_config.yaml",
    "README.md",
)


def _run_checker(paths):
    result = subprocess.run(
        [sys.executable, str(CHECKER), "--paths-from-stdin"],
        input="\n".join(paths),
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    return result


def _checker_protected_paths():
    """Load the checker's protected path logic."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from check_protected_paths import protected_patterns, protected_paths

    return protected_paths(PROTECTED_PATHS, protected_patterns())


def test_verifier_authority_files_exist():
    for relative in (
        "scripts/verify.py",
        "scripts/replay_routing.py",
        "scripts/quality_floor.py",
        "scripts/check_protected_paths.py",
        "scripts/golden/routing_replay.json",
        ".github/CODEOWNERS",
        ".github/workflows/verifier.yml",
    ):
        assert (REPO_ROOT / relative).is_file(), f"missing {relative}"


def test_codeowners_owns_every_protected_path():
    protected = _checker_protected_paths()
    for path in PROTECTED_PATHS:
        assert path in protected, f"{path} is not protected by CODEOWNERS"


def test_protected_path_gate_rejects_protected_edits():
    result = _run_checker(PROTECTED_PATHS)
    assert result.returncode == 1, result.stdout


def test_protected_path_gate_allows_ordinary_edits():
    result = _run_checker(UNPROTECTED_PATHS)
    assert result.returncode == 0, result.stderr


def test_golden_replay_passes():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "replay_routing.py")],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_quality_floor_check_passes():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "quality_floor.py")],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stderr
