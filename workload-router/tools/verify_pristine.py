"""Pristine-worktree oracle for verified routing.

Implements the client side of ``docs/verification-protocol.md`` section 8:
V4 (diff invariants), V0 (compile) and V2 (the base commit's own test command)
run against a detached worktree where only the non-test source diff was
applied. There is no network access, no LLM call and no third-party dependency.

The runner command is read from the *base* commit and the runner manifest is
restored from the base commit after applying the patch, so a patch that
rewrites ``scripts.test`` or a Makefile target cannot grade itself green.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any
from urllib import request

VERDICTS = frozenset({
    "pass", "fail_build", "fail_lint", "fail_test",
    "fail_scope", "fail_tamper", "timeout", "unknown",
})
EXIT_CODES = {
    "pass": 0, "unknown": 2, "fail_scope": 3, "fail_tamper": 4,
    "fail_test": 5, "fail_build": 6, "timeout": 7,
}
TEST_DIRS = {"tests", "test", "__tests__", "__snapshots__", "snapshots"}
DENY_NAMES = {
    "pytest.ini", "tox.ini", "conftest.py", "noxfile.py",
    "poetry.lock", "uv.lock", "package-lock.json", "yarn.lock",
    "pnpm-lock.yaml", "go.sum", "Cargo.lock",
}
MANIFEST_NAMES = (
    "go.mod", "Cargo.toml", "pyproject.toml", "pytest.ini",
    "package.json", "Makefile",
)
DIFF_EXCLUDES = (
    "tests/**", "test/**", "**/tests/**", "**/test/**",
    "**/*_test.go", "**/*.test.*", "**/*.spec.*",
    "**/__snapshots__/**", "**/snapshots/**", "**/conftest.py",
    ".github/workflows/**", "**/pytest.ini", "**/tox.ini",
    "**/jest.config.*", "**/vitest.config.*",
)
TAMPER = re.compile(
    r"(pytest\.mark\.skip|pytest\.mark\.xfail|@Disabled|\.only\(|"
    r"it\.skip\(|t\.Skip\(|t\.skip\()"
)


def _git(repo: Path, *args: str, check: bool = True, input_bytes: bytes | None = None):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check,
        input=input_bytes,
        capture_output=True,
    )


def _is_test_path(path: str) -> bool:
    parts = path.split("/")
    if any(part in TEST_DIRS for part in parts[:-1]):
        return True
    name = parts[-1]
    return (
        name == "conftest.py"
        or (name.startswith("test_") and name.endswith(".py"))
        or name.endswith("_test.py")
        or ".test." in name
        or ".spec." in name
    )


def _is_denied(path: str) -> bool:
    parts = path.split("/")
    if ".github" in parts and "workflows" in parts:
        return True
    name = parts[-1]
    if name in DENY_NAMES:
        return True
    if name.startswith("requirements") and name.endswith((".txt", ".in")):
        return True
    return name.startswith(("jest.config.", "vitest.config."))


def _changed_paths(repo: Path) -> list[str]:
    output = _git(repo, "diff", "--name-only", "HEAD").stdout
    return [line for line in output.decode("utf-8", "replace").splitlines() if line]


def _collect_diff(repo: Path) -> bytes:
    # New files must be visible to `git diff`; intent-to-add does not stage content.
    _git(repo, "add", "-N", ".")
    return _git(repo, "diff", "--binary", "HEAD", "--", ".").stdout


def _tamper(diff: bytes) -> bool:
    for line in diff.splitlines():
        if line.startswith(b"+") and not line.startswith(b"+++"):
            if TAMPER.search(line.decode("utf-8", "replace")):
                return True
    return False


def _base_manifests(repo: Path, base: str) -> dict[str, bytes]:
    manifests: dict[str, bytes] = {}
    for name in MANIFEST_NAMES:
        result = _git(repo, "show", f"{base}:{name}", check=False)
        if result.returncode == 0:
            manifests[name] = result.stdout
    return manifests


def _runner(manifests: dict[str, bytes]) -> list[str] | None:
    """Mirror the protocol order; the manifest set comes from the base commit."""
    if "go.mod" in manifests:
        return ["go", "test", "./..."]
    if "Cargo.toml" in manifests:
        return ["cargo", "test", "--quiet"]
    if "pyproject.toml" in manifests or "pytest.ini" in manifests:
        return ["python3", "-m", "pytest", "-q"]
    if "package.json" in manifests:
        return ["npm", "test", "--silent"]
    if "Makefile" in manifests:
        return ["make", "check"]
    return None


def _selected_tests(worktree: Path, changed: list[str], runner: list[str]) -> list[str]:
    """Cheap changed-subset selection; unknown shape falls back to the full suite."""
    if runner[:3] != ["python3", "-m", "pytest"]:
        return []
    stems = {
        Path(path).stem for path in changed
        if path.endswith(".py") and Path(path).stem != "__init__"
    }
    if not stems or len(stems) != len([p for p in changed if p.endswith(".py")]):
        return []
    selected: list[str] = []
    for path in worktree.rglob("*.py"):
        name = path.name
        if not (name.startswith("test_") or name.endswith("_test.py")):
            continue
        stem = name[5:] if name.startswith("test_") else name[:-9]
        if stem in stems:
            selected.append(str(path.relative_to(worktree)))
    return sorted(selected)


def _restore_manifests(worktree: Path, manifests: dict[str, bytes]) -> None:
    for name in MANIFEST_NAMES:
        target = worktree / name
        if name in manifests:
            target.write_bytes(manifests[name])
        elif target.exists():
            target.unlink()


def _apply(worktree: Path, diff: bytes) -> bool:
    command = ["git", "-C", str(worktree), "apply", "--binary"]
    for pattern in DIFF_EXCLUDES:
        command.extend(["--exclude", pattern])
    return subprocess.run(command, input=diff, capture_output=True).returncode == 0


def _v0(worktree: Path, changed: list[str]) -> str | None:
    files = [
        path for path in changed
        if path.endswith(".py") and not _is_test_path(path)
        and (worktree / path).is_file()
    ]
    if not files:
        return None
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", *files],
        cwd=worktree, capture_output=True, timeout=30,
    )
    return None if result.returncode == 0 else "fail_build"


def _v2(worktree: Path, runner: list[str], timeout: float) -> bool | None:
    env = {**os.environ, "CI": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("PYTEST_ADDOPTS", None)
    try:
        result = subprocess.run(
            runner, cwd=worktree, env=env, capture_output=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None
    return result.returncode == 0


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.json"


def _cache_key(diff: bytes, base_sha: str, runner: list[str], flake_runs: int) -> str:
    digest = hashlib.sha256()
    digest.update(diff)
    digest.update(base_sha.encode("ascii"))
    digest.update(json.dumps([runner, flake_runs]).encode("utf-8"))
    return digest.hexdigest()


def _result(verdict: str, oracle: str, started: float, *, cached: bool = False,
            flaky: bool = False, reason: str = "") -> dict[str, Any]:
    return {
        "verdict": verdict,
        "oracle": oracle,
        "duration_ms": int((time.monotonic() - started) * 1000),
        "cached": cached,
        "flaky": flaky,
        "reason": reason,
    }


def verify(repo: str | Path, base: str = "HEAD", *, flake_runs: int = 1,
           timeout: float = 180.0, cache_dir: str | Path | None = None,
           use_cache: bool = True) -> dict[str, Any]:
    """Grade the working-tree diff against the base commit.

    Returns the closed verdict enum plus tier metadata. The result never
    contains stderr, file paths or prompt text.
    """
    started = time.monotonic()
    repo = Path(repo).resolve()
    if not (repo / ".git").exists():
        return _result("unknown", "none", started, reason="not_a_repo")
    base_sha = _git(repo, "rev-parse", base).stdout.decode().strip()
    diff = _collect_diff(repo)
    changed = _changed_paths(repo)
    if not diff or not changed:
        return _result("unknown", "none", started, reason="empty_diff")

    # V4 before any worktree work: tamper wins over scope, both are cheap.
    if _tamper(diff):
        return _result("fail_tamper", "V4", started, reason="skip_inserted")
    if any(_is_test_path(path) for path in changed):
        return _result("fail_scope", "V4", started, reason="test_path")
    if any(_is_denied(path) for path in changed):
        return _result("fail_scope", "V4", started, reason="denied_path")

    manifests = _base_manifests(repo, base_sha)
    runner = _runner(manifests)
    if runner is None:
        return _result("unknown", "none", started, reason="no_runner")

    cache = Path(cache_dir or os.environ.get(
        "ORACLE_CACHE_DIR", Path.home() / ".cache" / "oracle"))
    key = _cache_key(diff, base_sha, runner, flake_runs)
    if use_cache:
        cached = _cache_path(cache, key)
        if cached.is_file():
            try:
                stored = json.loads(cached.read_text(encoding="utf-8"))
                verdict = stored["verdict"]
                if verdict in VERDICTS:
                    return _result(verdict, stored.get("oracle", "V2"), started,
                                   cached=True)
            except (OSError, ValueError, KeyError):
                pass

    worktree = Path(tempfile.mkdtemp(prefix="oracle-"))
    added = False
    try:
        try:
            _git(repo, "worktree", "add", "--detach", str(worktree), base_sha)
            added = True
        except subprocess.CalledProcessError:
            return _result("unknown", "none", started, reason="worktree_failed")
        if not _apply(worktree, diff):
            return _result("fail_scope", "V4", started, reason="patch_does_not_apply")
        # Verification inputs come from the base commit, never the patch.
        _restore_manifests(worktree, manifests)
        selected = _selected_tests(worktree, changed, runner)
        command = runner + selected
        build = _v0(worktree, changed)
        if build is not None:
            return _result(build, "V0", started, reason="compile_failed")
        outcomes: list[bool] = []
        for _ in range(max(1, flake_runs)):
            outcome = _v2(worktree, command, timeout)
            if outcome is None:
                return _result("timeout", "V2", started, reason="budget_exceeded")
            outcomes.append(outcome)
        if len(set(outcomes)) > 1:
            return _result("unknown", "V2", started, flaky=True, reason="flaky_suite")
        verdict = "pass" if outcomes[0] else "fail_test"
        if use_cache:
            try:
                cache.mkdir(parents=True, exist_ok=True)
                _cache_path(cache, key).write_text(
                    json.dumps({"verdict": verdict, "oracle": "V2"}),
                    encoding="utf-8",
                )
            except OSError:
                pass
        return _result(verdict, "V2", started)
    finally:
        if added:
            subprocess.run(
                ["git", "-C", str(repo), "worktree", "remove", "--force", str(worktree)],
                capture_output=True,
            )
        elif worktree.exists():
            subprocess.run(["rm", "-rf", str(worktree)], capture_output=True)


def post_verdict(url: str, attempt_id: str, verdict: str, oracle: str,
                 duration_ms: int, *, token: str | None = None,
                 timeout: float = 5.0) -> int:
    """Post one bounded grade. Never send stderr, paths or prompt text."""
    if verdict not in VERDICTS:
        raise ValueError("unsupported verdict")
    if not isinstance(attempt_id, str) or not attempt_id or len(attempt_id) > 256:
        raise ValueError("invalid attempt_id")
    body = {
        "attempt_id": attempt_id,
        "grade": verdict == "pass",
        "verdict": verdict,
        "oracle": oracle,
        "duration_ms": int(duration_ms),
    }
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    req = request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        return response.status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pristine-worktree verification oracle")
    parser.add_argument("repo")
    parser.add_argument("--base", default="HEAD")
    parser.add_argument("--flake-runs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--cache-dir")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--post")
    parser.add_argument("--attempt-id")
    parser.add_argument("--api-key-env")
    arguments = parser.parse_args(argv)
    result = verify(
        arguments.repo, arguments.base,
        flake_runs=max(1, arguments.flake_runs),
        timeout=arguments.timeout,
        cache_dir=arguments.cache_dir,
        use_cache=not arguments.no_cache,
    )
    print(json.dumps(result, sort_keys=True))
    if arguments.post:
        token = os.environ.get(arguments.api_key_env) if arguments.api_key_env else None
        post_verdict(arguments.post, arguments.attempt_id or "", result["verdict"],
                     result["oracle"], result["duration_ms"], token=token)
    return EXIT_CODES[result["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
