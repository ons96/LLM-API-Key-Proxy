"""Tests for scripts/keys.py — multi-key management + dead-key verify CLI."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# ponytail: no pytest-asyncio dep — wrap async tests in asyncio.run().

# Load scripts/keys.py as a module (same pattern as test_reorder_chains.py).
SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "keys.py"
spec = importlib.util.spec_from_file_location("keys_mod", SCRIPT_PATH)
assert spec and spec.loader
keys_mod = importlib.util.module_from_spec(spec)
sys.modules["keys_mod"] = keys_mod
spec.loader.exec_module(keys_mod)

from keys_mod import (  # noqa: E402
    DEAD,
    KeyEntry,
    KeyManifest,
    RATE_LIMITED,
    REMOVED,
    UNREACHABLE,
    UNVERIFIED,
    WORKING,
    classify_response,
    verify_key,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manifest(tmp_path: Path, providers: dict | None = None) -> KeyManifest:
    """Create a KeyManifest rooted at tmp_path (no real ~/.secrets writes)."""
    man_path = tmp_path / "keys.json"
    man = KeyManifest(path=man_path)
    if providers:
        for prov, entries in providers.items():
            for e in entries:
                man.providers.setdefault(prov, []).append(
                    KeyEntry.from_dict(e) if isinstance(e, dict) else e
                )
        man.save()
    return man


# ---------------------------------------------------------------------------
# TestKeyManifest
# ---------------------------------------------------------------------------


class TestKeyManifest:
    def test_load_missing_creates_empty(self, tmp_path):
        man = KeyManifest(path=tmp_path / "missing.json")
        assert man.providers == {}
        assert man.list_providers() == []

    def test_add_key_persists(self, tmp_path, monkeypatch):
        monkeypatch.setattr(keys_mod, "SECRETS_DIR", tmp_path)
        man = _make_manifest(tmp_path)
        entry = man.add_key("groq", "gsk_test123", write_file=True)
        assert entry.id == "1"
        assert entry.status == UNVERIFIED
        assert Path(entry.file_path).exists()
        assert Path(entry.file_path).read_text() == "gsk_test123"

        # Reload manifest from disk.
        man2 = KeyManifest(path=man.path)
        assert "groq" in man2.providers
        assert len(man2.providers["groq"]) == 1
        assert man2.providers["groq"][0].id == "1"

    def test_remove_key_marks_status_not_delete(self, tmp_path, monkeypatch):
        monkeypatch.setattr(keys_mod, "SECRETS_DIR", tmp_path)
        man = _make_manifest(tmp_path)
        entry = man.add_key("groq", "gsk_test123", write_file=True)
        file_path = entry.file_path
        removed = man.remove_key("groq", "1")
        assert removed is not None
        assert removed.status == REMOVED
        # File MUST still exist (we never delete).
        assert Path(file_path).exists()

    def test_next_working_key_skips_dead(self, tmp_path):
        now = time.time()
        entries = [
            KeyEntry(id="1", status=WORKING, last_used=now - 100),
            KeyEntry(id="2", status=DEAD, last_used=now - 50),
            KeyEntry(id="3", status=WORKING, last_used=now - 10),
        ]
        man = _make_manifest(tmp_path, {"groq": [e.to_dict() for e in entries]})
        # Oldest working key is id=1 (last_used=now-100).
        picked = man.next_working_key("groq")
        assert picked is not None
        assert picked.id == "1"

    def test_next_working_key_returns_none_if_all_dead(self, tmp_path):
        entries = [
            KeyEntry(id="1", status=DEAD),
            KeyEntry(id="2", status=REMOVED),
        ]
        man = _make_manifest(tmp_path, {"groq": [e.to_dict() for e in entries]})
        assert man.next_working_key("groq") is None


# ---------------------------------------------------------------------------
# TestClassifyResponse
# ---------------------------------------------------------------------------


class TestClassifyResponse:
    def test_200_working(self):
        assert classify_response(200) == WORKING

    def test_401_dead(self):
        assert classify_response(401) == DEAD

    def test_403_dead(self):
        assert classify_response(403) == DEAD

    def test_429_rate_limited(self):
        assert classify_response(429) == RATE_LIMITED

    def test_500_unreachable(self):
        assert classify_response(500) == UNREACHABLE

    def test_503_unreachable(self):
        assert classify_response(503) == UNREACHABLE

    def test_unknown_status_unverified(self):
        assert classify_response(418) == UNVERIFIED


# ---------------------------------------------------------------------------
# TestVerifyKey (async, mocks httpx)
# ---------------------------------------------------------------------------


class TestVerifyKey:
    def test_working_key_returns_working(self):
        async def _run():
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.text = '{"data": []}'
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.get = AsyncMock(return_value=mock_resp)

            with patch("httpx.AsyncClient", return_value=mock_client):
                return await verify_key(
                    "groq", "https://api.groq.com/openai/v1", "gsk_valid"
                )

        status, detail = asyncio.run(_run())
        assert status == WORKING
        assert "200" in detail

    def test_dead_key_returns_dead(self):
        async def _run():
            mock_resp = MagicMock()
            mock_resp.status_code = 401
            mock_resp.text = '{"error": "invalid api key"}'
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.get = AsyncMock(return_value=mock_resp)

            with patch("httpx.AsyncClient", return_value=mock_client):
                return await verify_key(
                    "groq", "https://api.groq.com/openai/v1", "gsk_bogus"
                )

        status, detail = asyncio.run(_run())
        assert status == DEAD
        assert "401" in detail

    def test_empty_base_url_unreachable(self):
        async def _run():
            return await verify_key("foo", "", "any")

        status, detail = asyncio.run(_run())
        assert status == UNREACHABLE
        assert "base_url" in detail

    def test_empty_key_dead(self):
        async def _run():
            return await verify_key("foo", "https://example.com/v1", "")

        status, detail = asyncio.run(_run())
        assert status == DEAD
        assert "empty" in detail


# ---------------------------------------------------------------------------
# TestRotateProvider (CLI smoke)
# ---------------------------------------------------------------------------


class TestRotate:
    def test_rotate_picks_next_working_and_marks_used(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(keys_mod, "SECRETS_DIR", tmp_path)
        man = _make_manifest(tmp_path)
        man.add_key("groq", "k1", write_file=False)
        man.add_key("groq", "k2", write_file=False)
        man.update_status("groq", "1", WORKING, "mock")
        man.update_status("groq", "2", WORKING, "mock")

        args = MagicMock(provider="groq")
        rc = keys_mod._cmd_rotate(man, args)
        out = capsys.readouterr().out
        assert rc == 0
        assert "rotate groq" in out

    def test_rotate_no_working_key_errors(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(keys_mod, "SECRETS_DIR", tmp_path)
        man = _make_manifest(tmp_path)
        man.add_key("groq", "k1", write_file=False)
        # status defaults to UNVERIFIED, not WORKING → no rotation candidate.
        args = MagicMock(provider="groq")
        rc = keys_mod._cmd_rotate(man, args)
        err = capsys.readouterr().err
        assert rc == 1
        assert "no working key" in err


# ---------------------------------------------------------------------------
# TestVerifyManifest (end-to-end with mocked HTTP)
# ---------------------------------------------------------------------------


class TestVerifyManifest:
    def test_verify_manifest_classifies_each_key(self, tmp_path, monkeypatch):
        async def _run():
            monkeypatch.setattr(keys_mod, "SECRETS_DIR", tmp_path)
            man = _make_manifest(tmp_path)
            man.add_key("groq", "k_valid", write_file=True)
            man.add_key("groq", "k_bogus", write_file=True)

            async def fake_verify(provider, base_url, key_value, timeout=10):
                if "valid" in key_value:
                    return (WORKING, "HTTP 200")
                return (DEAD, "HTTP 401")

            monkeypatch.setattr(keys_mod, "verify_key", fake_verify)
            monkeypatch.setattr(
                keys_mod, "_provider_base_url", lambda p, db=None: ("KEY", "https://fake/v1")
            )

            return await keys_mod.verify_manifest(man, provider="groq")

        results = asyncio.run(_run())
        assert len(results) == 2
        statuses = {r[2] for r in results}
        assert WORKING in statuses
        assert DEAD in statuses

    def test_verify_manifest_skips_removed(self, tmp_path, monkeypatch):
        async def _run():
            monkeypatch.setattr(keys_mod, "SECRETS_DIR", tmp_path)
            man = _make_manifest(tmp_path)
            man.add_key("groq", "k1", write_file=False)
            man.remove_key("groq", "1")

            async def fake_verify(*a, **kw):
                return (WORKING, "should not be called")

            monkeypatch.setattr(keys_mod, "verify_key", fake_verify)
            monkeypatch.setattr(
                keys_mod, "_provider_base_url", lambda p, db=None: ("KEY", "https://fake/v1")
            )

            return await keys_mod.verify_manifest(man, provider="groq")

        results = asyncio.run(_run())
        assert len(results) == 1
        assert results[0][2] == REMOVED
        assert "skipped" in results[0][3]


# ---------------------------------------------------------------------------
# TestAddRemove (CLI smoke)
# ---------------------------------------------------------------------------


class TestAddRemove:
    def test_add_then_list(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(keys_mod, "SECRETS_DIR", tmp_path)
        man = _make_manifest(tmp_path)

        args = MagicMock(
            provider="gemini",
            key_value="AIzaTest",
            id=None,
            env_var="GEMINI_API_KEY",
            no_file=False,
        )
        rc = keys_mod._cmd_add(man, args)
        assert rc == 0
        assert "added gemini" in capsys.readouterr().out

        # List.
        args_list = MagicMock(provider=None)
        rc = keys_mod._cmd_list(man, args_list)
        out = capsys.readouterr().out
        assert rc == 0
        assert "gemini" in out
        assert "1" in out  # key id

    def test_remove_warns_when_no_active_keys_left(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(keys_mod, "SECRETS_DIR", tmp_path)
        man = _make_manifest(tmp_path)
        man.add_key("groq", "only_key", write_file=False)
        args = MagicMock(provider="groq", id="1")
        rc = keys_mod._cmd_remove(man, args)
        captured = capsys.readouterr()
        assert rc == 0
        assert "removed" in captured.out
        assert "no active keys" in captured.err

    def test_remove_unknown_key_errors(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(keys_mod, "SECRETS_DIR", tmp_path)
        man = _make_manifest(tmp_path)
        args = MagicMock(provider="groq", id="999")
        rc = keys_mod._cmd_remove(man, args)
        err = capsys.readouterr().err
        assert rc == 1
        assert "no key id=999" in err


# ---------------------------------------------------------------------------
# TestAutoId
# ---------------------------------------------------------------------------


class TestAutoId:
    def test_auto_id_increments(self, tmp_path, monkeypatch):
        monkeypatch.setattr(keys_mod, "SECRETS_DIR", tmp_path)
        man = _make_manifest(tmp_path)
        e1 = man.add_key("groq", "k1", write_file=False)
        e2 = man.add_key("groq", "k2", write_file=False)
        e3 = man.add_key("groq", "k3", write_file=False)
        assert e1.id == "1"
        assert e2.id == "2"
        assert e3.id == "3"

    def test_explicit_id_respected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(keys_mod, "SECRETS_DIR", tmp_path)
        man = _make_manifest(tmp_path)
        e = man.add_key("groq", "k_special", key_id="backup", write_file=False)
        assert e.id == "backup"
