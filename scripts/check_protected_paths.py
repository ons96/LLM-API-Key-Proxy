#!/usr/bin/env python3
"""Protected-path gate (#1422).

The verifier authority is the path list in ``.github/CODEOWNERS``. This script
answers one question mechanically: does a change touch a protected path, and if
so, did a human approve it?

Modes:

* default (CI): compare the PR or push against its base and fail when a
  protected path changed without a human approving review;
* ``--paths-from-stdin``: fail when any listed path is protected (used by the
  auto-merge workflows to skip such pull requests);
* ``--list-protected``: print the protected path patterns.

Exit code 0 means the change may proceed, 1 means human review is required.
"""

from __future__ import annotations

import fnmatch
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
CODEOWNERS_PATH = REPO_ROOT / ".github" / "CODEOWNERS"
BOT_SUFFIX = "[bot]"


def protected_patterns(path: Path = CODEOWNERS_PATH) -> List[str]:
    """Read the protected path patterns from CODEOWNERS."""
    if not path.exists():
        return []
    patterns = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        patterns.append(parts[0])
    return patterns


def is_protected(file_path: str, patterns: Sequence[str]) -> bool:
    """Return True when a repository-relative path is protected."""
    candidate = file_path.strip()
    for pattern in patterns:
        normalized = pattern.strip()
        if normalized.startswith("/"):
            normalized = normalized[1:]
        if normalized.endswith("/"):
            prefix = normalized
            if candidate == prefix.rstrip("/") or candidate.startswith(prefix):
                return True
            continue
        if fnmatch.fnmatch(candidate, normalized):
            return True
        if candidate == normalized or candidate == normalized.lstrip("/"):
            return True
    return False


def protected_paths(paths: Sequence[str], patterns: Sequence[str]) -> List[str]:
    """Filter the paths that are protected."""
    return [path for path in paths if is_protected(path, patterns)]


def run_git(args: Sequence[str]) -> str:
    """Run a git command and return stdout."""
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def changed_files(base: str, head: str) -> List[str]:
    """List files changed between two revisions."""
    output = run_git(["diff", "--name-only", f"{base}...{head}"])
    if not output:
        output = run_git(["diff", "--name-only", base, head])
    return [line for line in output.splitlines() if line.strip()]


def gh_json(args: Sequence[str]) -> object | None:
    """Run a gh command that returns JSON, or None when it fails."""
    result = subprocess.run(
        ["gh", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def human_approved(pr_number: str) -> bool:
    """Return True when a human (not the PR author, not a bot) approved."""
    payload = gh_json(
        ["pr", "view", pr_number, "--json", "reviews,author,authorAssociation"]
    )
    if not isinstance(payload, dict):
        return False
    author = str((payload.get("author") or {}).get("login", ""))
    reviews = payload.get("reviews") or []
    for review in reviews:
        if not isinstance(review, dict):
            continue
        if str(review.get("state", "")).upper() != "APPROVED":
            continue
        login = str((review.get("author") or {}).get("login", ""))
        if not login or login == author or login.endswith(BOT_SUFFIX):
            continue
        return True
    return False


def pull_request_for_commit(sha: str) -> str | None:
    """Find the pull request that delivered a commit, if any."""
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    owner = os.environ.get("GITHUB_REPOSITORY_OWNER", "")
    if not repository and "/" in os.environ.get("GITHUB_REPOSITORY", ""):
        repository = os.environ["GITHUB_REPOSITORY"]
    if not owner:
        owner = repository.split("/")[0] if repository else ""
    if not repository:
        return None
    payload = gh_json(
        [
            "api",
            f"repos/{repository}/commits/{sha}/pulls",
            "--jq",
            ".[0].number",
        ]
    )
    if payload is None:
        return None
    number = str(payload).strip()
    return number or None


def event_context() -> tuple[str, str, str | None]:
    """Resolve (event name, base revision, head revision, pr number)."""
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    event_path = os.environ.get("GITHUB_EVENT_PATH", "")
    payload: dict = {}
    if event_path and Path(event_path).exists():
        try:
            payload = json.loads(Path(event_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}

    if event_name == "pull_request" or payload.get("pull_request"):
        pr = payload.get("pull_request") or {}
        base = str((pr.get("base") or {}).get("sha", ""))
        head = str((pr.get("head") or {}).get("sha", ""))
        number = str(pr.get("number", "")) or None
        return event_name or "pull_request", base, head, number

    before = str(payload.get("before", ""))
    after = str(payload.get("after", ""))
    if before and after:
        number = pull_request_for_commit(after)
        return event_name or "push", before, after, number

    return event_name, "", "", None


def gate() -> int:
    """Fail when a protected path changed without human approval."""
    patterns = protected_patterns()
    if not patterns:
        print("protected paths: no CODEOWNERS patterns found", file=sys.stderr)
        return 1

    _, base, head, number = event_context()
    if not base or not head:
        print(
            "protected paths: no base/head revision available; "
            "human review is required for protected paths",
            file=sys.stderr,
        )
        return 1

    touched = protected_paths(changed_files(base, head), patterns)
    if not touched:
        print("protected paths: OK (no protected path changed)")
        return 0

    print("protected paths: protected paths changed:", file=sys.stderr)
    for path in touched:
        print(f"  - {path}", file=sys.stderr)

    if number and human_approved(str(number)):
        print(f"protected paths: human review recorded on PR #{number}")
        return 0

    print(
        "protected paths: human review required; verifier changes must be "
        "approved by a human reviewer",
        file=sys.stderr,
    )
    return 1


def main(argv: List[str]) -> int:
    """CLI entry point."""
    args = argv[1:]
    patterns = protected_patterns()
    if "--list-protected" in args:
        for pattern in patterns:
            print(pattern)
        return 0
    if "--paths-from-stdin" in args:
        paths = [line.strip() for line in sys.stdin.read().splitlines() if line.strip()]
        touched = protected_paths(paths, patterns)
        if touched:
            for path in touched:
                print(f"protected path: {path}", file=sys.stderr)
            return 1
        return 0
    return gate()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
