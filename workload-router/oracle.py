"""Grade a change from the target itself: run its tests, else validate it.

Two graders, chosen by what the target can actually do:

- ``tests`` — a repository with a runnable test command. The command is
  detected from the repository (``package.json``, ``Makefile``, ``pyproject``,
  ``Cargo.toml``, ``go.mod``), so any repository an agent edits can be graded
  without keeping a per-repository list.
- ``validator`` — no test command, or no repository at all. JSON, YAML and TOML
  are parsed so a config edit that breaks syntax fails instead of shipping.
  This is how work like ``opencode.json`` provider edits gets graded.

A target with neither is ``none``: ungradable, reported as such rather than
counted as a pass. Nothing here calls a model or the network.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - 3.10 fallback
    tomllib = None  # type: ignore[assignment]

VALIDATOR_SUFFIXES = {".json", ".yaml", ".yml", ".toml"}
PLACEHOLDER_TESTS = {"no test specified", "echo \"Error: no test specified\" && exit 1"}


@dataclass(frozen=True)
class Grade:
    """The outcome of grading one target."""

    grader: str  # "tests" | "validator" | "none"
    passed: bool | None  # None means nothing could grade it
    detail: str
    command: tuple[str, ...] | None = None


def _json_script(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    scripts = payload.get("scripts")
    if not isinstance(scripts, dict):
        return None
    test = scripts.get("test")
    if not isinstance(test, str) or not test.strip():
        return None
    if test.strip() in PLACEHOLDER_TESTS:
        return None
    return "npm test --silent"


def _make_has_test(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return any(line.startswith("test:") for line in text.splitlines())


def _pyproject_has_pytest(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return "[tool.pytest" in text or "pytest" in text


def detect_test_command(root: str | Path) -> list[str] | None:
    """Return the repository's own test command, or None if it has none."""
    root = Path(root)
    if not root.is_dir():
        return None

    package_json = root / "package.json"
    if package_json.is_file():
        script = _json_script(package_json)
        if script:
            return script.split()

    makefile = root / "Makefile"
    if makefile.is_file() and _make_has_test(makefile):
        return ["make", "test"]

    if (root / "Cargo.toml").is_file():
        return ["cargo", "test", "--quiet"]

    if (root / "go.mod").is_file():
        return ["go", "test", "./..."]

    pyproject = root / "pyproject.toml"
    if pyproject.is_file() and _pyproject_has_pytest(pyproject):
        return ["python", "-m", "pytest", "-q"]

    if (root / "tests").is_dir() or (root / "pytest.ini").is_file():
        return ["python", "-m", "pytest", "-q"]

    return None


def validate_file(path: str | Path) -> tuple[bool, str]:
    """Parse one file; return (ok, detail). Unparseable formats are skipped."""
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return False, f"{path.name}: unreadable ({type(exc).__name__})"
    try:
        if suffix == ".json":
            json.loads(text)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ModuleNotFoundError:  # pragma: no cover
                return True, f"{path.name}: yaml parser unavailable, skipped"
            yaml.safe_load(text)
        elif suffix == ".toml":
            if tomllib is None:  # pragma: no cover
                return True, f"{path.name}: tomllib unavailable, skipped"
            tomllib.loads(text)
        else:
            return True, f"{path.name}: no validator for {suffix or 'this type'}"
    except Exception as exc:  # parse errors from any of the three parsers
        return False, f"{path.name}: {type(exc).__name__}"
    return True, f"{path.name}: parsed"


def grade(
    target: str | Path,
    changed: Sequence[str | Path] | None = None,
    runner: Callable[..., subprocess.CompletedProcess] | None = None,
    timeout: int = 300,
) -> Grade:
    """Grade `target`: its test command if it has one, else validate the edits."""
    root = Path(target)
    command = detect_test_command(root) if root.is_dir() else None

    if command:
        run = runner or subprocess.run
        try:
            completed = run(
                command,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError:
            return Grade("tests", False, f"{command[0]} not installed", tuple(command))
        except subprocess.TimeoutExpired:
            return Grade("tests", False, f"timed out after {timeout}s", tuple(command))
        output = ((completed.stdout or "") + (completed.stderr or "")).strip()
        tail = output.splitlines()[-1] if output else ""
        return Grade(
            "tests",
            completed.returncode == 0,
            f"exit {completed.returncode}" + (f": {tail}" if tail else ""),
            tuple(command),
        )

    candidates = [Path(item) for item in changed] if changed else ([root] if root.is_file() else [])
    validatable = [item for item in candidates if item.suffix.lower() in VALIDATOR_SUFFIXES]
    if validatable:
        details = [validate_file(item) for item in validatable]
        failures = [detail for ok, detail in details if not ok]
        if failures:
            return Grade("validator", False, "; ".join(failures))
        return Grade("validator", True, "; ".join(detail for _, detail in details))

    return Grade("none", None, "nothing to grade: no test command and no validatable files")
