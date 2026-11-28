#!/usr/bin/env python3
"""Machine-checked allow-list for meta-loop candidates.

A meta-loop candidate may retune routing configuration, but it must not be
able to change gateway code, tests or CI. Every path in a candidate diff
therefore has to live under ``config/``. The allowed set is a module constant
on purpose: the loop owns ``config/``, so a configurable allow-list would let
the loop widen its own permissions. There is deliberately no bypass flag, env
var or config key that changes the allowed set.

Usage::

    python3 scripts/check_meta_loop_config_surface.py --base origin/main
    python3 scripts/check_meta_loop_config_surface.py --base origin/main --head HEAD
    python3 scripts/check_meta_loop_config_surface.py --paths-file diff.txt

The process exits 0 when every changed path is inside the config surface and
1 with the named error ``META_LOOP_CONFIG_SURFACE_VIOLATION`` otherwise.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Iterable, List, Optional, Sequence, Tuple

ERROR_NAME = "META_LOOP_CONFIG_SURFACE_VIOLATION"
ALLOWED_ROOTS: Tuple[str, ...] = ("config",)


class ConfigSurfaceViolation(Exception):
    """Raised when a candidate changes paths outside the config surface."""

    def __init__(self, paths: Sequence[str]) -> None:
        self.paths: List[str] = list(paths)
        super().__init__(f"{ERROR_NAME}: " + ", ".join(self.paths))


def normalize_path(raw: str) -> str:
    """Return a repo-relative POSIX path for a raw diff path."""
    path = raw.strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path


def is_config_path(raw: str) -> bool:
    """Return True when the path is inside the allowed config surface."""
    path = normalize_path(raw)
    if not path:
        return False
    parts = PurePosixPath(path).parts
    if not parts or parts[0] in ("/", ".."):
        return False
    if any(part == ".." for part in parts):
        return False
    if len(parts) < 2:
        return False
    return parts[0] in ALLOWED_ROOTS


def paths_from_name_only(text: str) -> List[str]:
    """Parse the output of ``git diff --name-only``."""
    paths: List[str] = []
    for line in text.splitlines():
        path = normalize_path(line)
        if path:
            paths.append(path)
    return paths


def _strip_diff_prefix(path: str) -> str:
    """Remove the ``a/`` or ``b/`` prefix git puts on diff header paths."""
    if path.startswith("a/") or path.startswith("b/"):
        return path[2:]
    return path


def _split_diff_git_paths(payload: str) -> Tuple[str, str]:
    """Split a ``diff --git`` payload into its old and new path."""
    marker = " b/"
    index = payload.rfind(marker)
    if index == -1:
        return payload, payload
    old_path = _strip_diff_prefix(payload[:index])
    new_path = _strip_diff_prefix(payload[index + len(marker) :])
    return old_path, new_path


def paths_from_unified_diff(text: str) -> List[str]:
    """Parse the changed paths out of a unified diff."""
    paths: List[str] = []
    for line in text.splitlines():
        if not line.startswith("diff --git "):
            continue
        old_path, new_path = _split_diff_git_paths(line[len("diff --git ") :])
        for candidate in (old_path, new_path):
            path = normalize_path(candidate)
            if path and path not in paths:
                paths.append(path)
    return paths


def paths_from_diff(text: str) -> List[str]:
    """Parse changed paths from a name-only listing or a unified diff."""
    if "diff --git " in text:
        return paths_from_unified_diff(text)
    return paths_from_name_only(text)


def violations(paths: Iterable[str]) -> List[str]:
    """Return the unique changed paths that escape the config surface."""
    bad: List[str] = []
    for raw in paths:
        path = normalize_path(raw)
        if not path:
            continue
        if not is_config_path(path) and path not in bad:
            bad.append(path)
    return bad


def _run_git(args: Sequence[str]) -> str:
    """Run a git command and return its stdout."""
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "unknown git error"
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout


def changed_paths(base: str, head: Optional[str] = None) -> List[str]:
    """Return the paths a candidate changes relative to ``base``.

    With ``head`` the candidate is the commit or ref ``head``; without it the
    candidate is the current workspace, which is what the commit path checks
    before the change is committed.
    """
    if head:
        output = _run_git(["diff", "--name-only", "--no-renames", f"{base}...{head}"])
        return paths_from_name_only(output)
    output = _run_git(["diff", "--name-only", "--no-renames", base])
    tracked = paths_from_name_only(output)
    untracked = paths_from_name_only(
        _run_git(["ls-files", "--others", "--exclude-standard"])
    )
    return tracked + [path for path in untracked if path not in tracked]


def _build_parser() -> argparse.ArgumentParser:
    """Build the command line parser for the guard."""
    parser = argparse.ArgumentParser(
        description="Fail when a meta-loop candidate changes anything but config/."
    )
    parser.add_argument(
        "--base",
        default="origin/main",
        help="Base ref the candidate is diffed against.",
    )
    parser.add_argument(
        "--head",
        default=None,
        help="Candidate ref. Omit to check the current workspace.",
    )
    parser.add_argument(
        "--paths-file",
        default=None,
        help="Read the changed paths from a file instead of from git.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the guard and return the process exit code."""
    args = _build_parser().parse_args(argv)

    if args.paths_file:
        text = Path(args.paths_file).read_text(encoding="utf-8")
        paths = paths_from_diff(text)
    else:
        paths = changed_paths(args.base, args.head)

    bad = violations(paths)
    if bad:
        message = (
            f"{ERROR_NAME}: meta-loop candidate changes paths outside the config "
            f"surface ({'/'.join(ALLOWED_ROOTS)}/): " + ", ".join(bad)
        )
        print(f"::error title=Meta-loop config surface violation::{message}", file=sys.stderr)
        print(message, file=sys.stderr)
        return 1

    allowed = "/".join(ALLOWED_ROOTS) + "/"
    print(f"config surface OK: {len(paths)} changed path(s), all inside {allowed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
