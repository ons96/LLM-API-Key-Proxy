"""Tests for CooldownPolicy (#218): per-provider wait-vs-fallback on 429."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

# ponytail: load cooldown_manager.py directly to avoid importing rotator_library
# (which pulls in litellm at module level).
_SCRIPT = Path(__file__).resolve().parent.parent / "src" / "rotator_library" / "cooldown_manager.py"
_spec = importlib.util.spec_from_file_location("cdm", _SCRIPT)
cdm = importlib.util.module_from_spec(_spec)
sys.modules["cdm"] = cdm
_spec.loader.exec_module(cdm)

CooldownPolicy = cdm.CooldownPolicy
_DEFAULT_POLICY = cdm._DEFAULT_POLICY


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_policy(tmp_path: Path, providers: dict, default: dict | None = None) -> Path:
    """Write a cooldown_policy.yaml to tmp_path and return its path."""
    path = tmp_path / "cooldown_policy.yaml"
    data = {"providers": providers, "default": default or _DEFAULT_POLICY}
    path.write_text(yaml.safe_dump(data))
    return path


# ---------------------------------------------------------------------------
# TestLoad
# ---------------------------------------------------------------------------


class TestLoad:
    def test_load_real_config(self):
        # ponytail: use the actual shipped config to verify it parses cleanly
        p = CooldownPolicy()
        assert p.get_policy("anthropic")["wait_on_429"] is True
        assert p.get_policy("openai")["wait_on_429"] is True
        assert p.get_policy("gemini")["wait_on_429"] is True
        assert p.get_policy("groq")["wait_on_429"] is False  # default

    def test_load_missing_file_uses_default(self, tmp_path: Path):
        p = CooldownPolicy(str(tmp_path / "nope.yaml"))
        assert p.get_policy("anything")["wait_on_429"] is False
        assert p.get_policy("anything")["max_wait_s"] == 0

    def test_load_malformed_uses_default(self, tmp_path: Path):
        path = tmp_path / "bad.yaml"
        path.write_text(":::not yaml:::\n  - broken")
        p = CooldownPolicy(str(path))
        assert p.get_policy("anything")["wait_on_429"] is False

    def test_provider_keys_case_insensitive(self, tmp_path: Path):
        path = _write_policy(
            tmp_path,
            {"Anthropic": {"wait_on_429": True, "max_wait_s": 30, "reason": "x"}},
        )
        p = CooldownPolicy(str(path))
        assert p.get_policy("anthropic")["wait_on_429"] is True
        assert p.get_policy("ANTHROPIC")["wait_on_429"] is True
        assert p.get_policy("Anthropic")["wait_on_429"] is True


# ---------------------------------------------------------------------------
# TestShouldWaitOn429
# ---------------------------------------------------------------------------


class TestShouldWaitOn429:
    def test_wait_provider_with_valid_retry_after(self, tmp_path: Path):
        path = _write_policy(
            tmp_path,
            {"openai": {"wait_on_429": True, "max_wait_s": 120, "reason": "cache"}},
        )
        p = CooldownPolicy(str(path))
        wait, deadline = p.should_wait_on_429("openai", retry_after_s=30)
        assert wait is True
        assert deadline > time.time()

    def test_no_wait_for_default_provider(self, tmp_path: Path):
        path = _write_policy(tmp_path, {})
        p = CooldownPolicy(str(path))
        wait, _ = p.should_wait_on_429("groq", retry_after_s=30)
        assert wait is False

    def test_no_wait_when_retry_after_missing(self, tmp_path: Path):
        path = _write_policy(
            tmp_path,
            {"openai": {"wait_on_429": True, "max_wait_s": 120, "reason": "x"}},
        )
        p = CooldownPolicy(str(path))
        wait, _ = p.should_wait_on_429("openai", retry_after_s=None)
        assert wait is False
        wait, _ = p.should_wait_on_429("openai", retry_after_s=0)
        assert wait is False

    def test_no_wait_when_retry_after_exceeds_max(self, tmp_path: Path):
        path = _write_policy(
            tmp_path,
            {"openai": {"wait_on_429": True, "max_wait_s": 60, "reason": "x"}},
        )
        p = CooldownPolicy(str(path))
        # retry_after 300s > max 60s -> don't wait (fall back instead)
        wait, _ = p.should_wait_on_429("openai", retry_after_s=300)
        assert wait is False

    def test_wait_at_boundary_retry_after_equals_max(self, tmp_path: Path):
        path = _write_policy(
            tmp_path,
            {"openai": {"wait_on_429": True, "max_wait_s": 60, "reason": "x"}},
        )
        p = CooldownPolicy(str(path))
        wait, deadline = p.should_wait_on_429("openai", retry_after_s=60)
        assert wait is True
        assert deadline > time.time()


# ---------------------------------------------------------------------------
# TestWaitUntil
# ---------------------------------------------------------------------------


class TestWaitUntil:
    def test_wait_completes_after_deadline(self):
        p = CooldownPolicy()
        deadline = time.time() + 0.1
        ok = asyncio.run(p.wait_until("openai", deadline))
        assert ok is True

    def test_wait_returns_true_if_deadline_already_passed(self):
        p = CooldownPolicy()
        deadline = time.time() - 1  # past
        ok = asyncio.run(p.wait_until("openai", deadline))
        assert ok is True

    def test_wait_returns_false_on_cancel(self):
        p = CooldownPolicy()
        deadline = time.time() + 10

        async def _run():
            task = asyncio.create_task(p.wait_until("openai", deadline))
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                return await task
            except asyncio.CancelledError:
                return None

        ok = asyncio.run(_run())
        assert ok is None or ok is False  # cancelled path


# ---------------------------------------------------------------------------
# TestPolicyFields
# ---------------------------------------------------------------------------


class TestPolicyFields:
    def test_default_has_all_fields(self, tmp_path: Path):
        p = CooldownPolicy(str(tmp_path / "nope.yaml"))
        d = p.get_policy("anything")
        assert "wait_on_429" in d
        assert "max_wait_s" in d
        assert "reason" in d

    def test_provider_missing_fields_get_defaults(self, tmp_path: Path):
        path = _write_policy(tmp_path, {"foo": {}})  # no fields
        p = CooldownPolicy(str(path))
        pol = p.get_policy("foo")
        assert pol["wait_on_429"] is False
        assert pol["max_wait_s"] == 0
        assert pol["reason"] == "configured"
