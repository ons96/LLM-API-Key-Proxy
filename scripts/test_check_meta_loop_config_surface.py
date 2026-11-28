"""Unit tests for the meta-loop config surface allow-list."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "check_meta_loop_config_surface.py"


def _load_guard():
    """Load the guard script as a module."""
    spec = importlib.util.spec_from_file_location(
        "check_meta_loop_config_surface", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = _load_guard()

BAD_DIFF = """diff --git a/config/routing_policy.yaml b/config/routing_policy.yaml
--- a/config/routing_policy.yaml
+++ b/config/routing_policy.yaml
@@ -1 +1 @@
-old
+new
diff --git a/src/proxy_app/main.py b/src/proxy_app/main.py
--- a/src/proxy_app/main.py
+++ b/src/proxy_app/main.py
@@ -1 +1 @@
-old
+new
diff --git a/tests/test_router_core.py b/tests/test_router_core.py
--- a/tests/test_router_core.py
+++ b/tests/test_router_core.py
@@ -1 +1 @@
-old
+new
"""

GOOD_DIFF = """diff --git a/config/routing_policy.yaml b/config/routing_policy.yaml
--- a/config/routing_policy.yaml
+++ b/config/routing_policy.yaml
@@ -1 +1 @@
-old
+new
"""


def test_bad_diff_is_rejected_with_named_error():
    """A candidate touching tests/ or src/proxy_app/ fails by name."""
    paths = guard.paths_from_diff(BAD_DIFF)
    bad = guard.violations(paths)

    assert "config/routing_policy.yaml" not in bad
    assert "src/proxy_app/main.py" in bad
    assert "tests/test_router_core.py" in bad

    error = guard.ConfigSurfaceViolation(bad)
    assert guard.ERROR_NAME in str(error)
    assert "META_LOOP_CONFIG_SURFACE_VIOLATION" in str(error)


def test_config_only_diff_passes():
    """A candidate touching only config/routing_policy.yaml passes."""
    paths = guard.paths_from_diff(GOOD_DIFF)

    assert paths == ["config/routing_policy.yaml"]
    assert guard.violations(paths) == []


def test_cli_fails_on_bad_diff_and_passes_on_good_diff(tmp_path):
    """The command line entry point returns 1 for a bad diff and 0 for a good one."""
    bad_file = tmp_path / "bad.diff"
    bad_file.write_text(BAD_DIFF, encoding="utf-8")
    good_file = tmp_path / "good.diff"
    good_file.write_text(GOOD_DIFF, encoding="utf-8")

    bad_run = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--paths-file",
            str(bad_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    good_run = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--paths-file",
            str(good_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert bad_run.returncode == 1
    assert "META_LOOP_CONFIG_SURFACE_VIOLATION" in bad_run.stderr
    assert good_run.returncode == 0
    assert good_run.stderr == ""


@pytest.mark.parametrize(
    "path",
    [
        "src/proxy_app/main.py",
        "tests/test_router_core.py",
        "configfoo/routing_policy.yaml",
        "config/../src/proxy_app/main.py",
        "/etc/passwd",
        "config",
    ],
)
def test_paths_outside_the_config_surface_are_rejected(path):
    """Only files below config/ are inside the allow-list."""
    assert guard.is_config_path(path) is False
    assert guard.violations([path]) == [path]


@pytest.mark.parametrize(
    "path",
    [
        "config/routing_policy.yaml",
        "config/fallbacks/gemini.yaml",
        "config/chain_pins.yaml",
    ],
)
def test_config_paths_are_allowed(path):
    """Files below config/ are inside the allow-list."""
    assert guard.is_config_path(path) is True
    assert guard.violations([path]) == []


def test_allow_list_is_not_configurable(monkeypatch):
    """No env var or config key can widen the allow-list."""
    monkeypatch.setenv("META_LOOP_CONFIG_SURFACE", "src,tests")
    monkeypatch.setenv("META_LOOP_ALLOWED_PATHS", "src/proxy_app")
    monkeypatch.setenv("META_LOOP_BYPASS", "1")

    assert guard.ALLOWED_ROOTS == ("config",)
    assert guard.violations(["src/proxy_app/main.py"]) == ["src/proxy_app/main.py"]
    assert guard.violations(["tests/test_router_core.py"]) == ["tests/test_router_core.py"]
