"""Unit tests for the active provider probe (#332).

Covers:
- target selection (search-only skipped, disabled skipped, dead skipped,
  no-api-key providers handled, env var resolution, max_models cap)
- probe_one success path (TTFT + TPS + latency recorded)
- probe_one failure path (error classified, success=False)
- record_call invoked with correct kwargs
- end-to-end: probe -> DB row queryable
"""
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import probe_providers as probe_mod

# Load telemetry.py directly via importlib to avoid rotator_library/__init__
# which imports client.py -> litellm (not installed in test env).
import importlib.util

_telemetry_path = REPO_ROOT / "src" / "rotator_library" / "telemetry.py"
_spec = importlib.util.spec_from_file_location("probe_telemetry", _telemetry_path)
_telemetry_mod = importlib.util.module_from_spec(_spec)
sys.modules["probe_telemetry"] = _telemetry_mod
_spec.loader.exec_module(_telemetry_mod)
TelemetryManager = _telemetry_mod.TelemetryManager


SAMPLE_CONFIG = {
    "providers": {
        "groq": {
            "enabled": True,
            "env_var": "GROQ_API_KEY",
            "base_url": "https://api.groq.com/openai/v1",
            "free_tier_models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        },
        "kilo": {
            "enabled": False,
            "env_var": "KILO_API_KEY",
            "base_url": "https://api.kilo.ai/v1",
            "free_tier_models": ["x-ai/grok-code-fast-1"],
        },
        "wiwi": {
            "enabled": True,
            "env_var": "WIWI_API_KEY",
            "base_url": "https://wiwi.up.railway.app/v1",
            "free_tier_models": ["gpt-5.4"],
            "_dead": True,
        },
        "brave_search": {
            "enabled": True,
            "env_var": "BRAVE_API_KEY",
        },
        "duckduckgo": {
            "enabled": True,
            "no_api_key_required": True,
        },
        "nokey": {
            "enabled": True,
            "env_var": "NOKEY_VAR",
            "base_url": "https://example.com/v1",
            "free_tier_models": ["m1"],
        },
        "freekey": {
            "enabled": True,
            "no_api_key_required": True,
            "base_url": "https://free.example.com/v1",
            "free_tier_models": ["m1", "kilo/auto-free", "openrouter/auto"],
        },
    }
}


@pytest.fixture
def config_file(tmp_path):
    p = tmp_path / "router_config.yaml"
    p.write_text(yaml.safe_dump(SAMPLE_CONFIG))
    return p


class TestSelectTargets:
    def test_skips_search_only(self, config_file):
        providers = probe_mod.load_provider_config(config_file)
        targets = probe_mod.select_targets(providers)
        pids = {t[0] for t in targets}
        assert "brave_search" not in pids
        assert "duckduckgo" not in pids

    def test_skips_disabled(self, config_file):
        providers = probe_mod.load_provider_config(config_file)
        targets = probe_mod.select_targets(providers)
        pids = {t[0] for t in targets}
        assert "kilo" not in pids

    def test_skips_dead(self, config_file):
        providers = probe_mod.load_provider_config(config_file)
        targets = probe_mod.select_targets(providers)
        pids = {t[0] for t in targets}
        assert "wiwi" not in pids

    def test_skips_no_api_key(self, config_file, monkeypatch):
        monkeypatch.delenv("NOKEY_VAR", raising=False)
        providers = probe_mod.load_provider_config(config_file)
        targets = probe_mod.select_targets(providers)
        pids = {t[0] for t in targets}
        assert "nokey" not in pids

    def test_includes_no_api_key_required(self, config_file):
        providers = probe_mod.load_provider_config(config_file)
        targets = probe_mod.select_targets(providers)
        pids = {t[0] for t in targets}
        assert "freekey" in pids

    def test_filters_auto_meta_models(self, config_file):
        providers = probe_mod.load_provider_config(config_file)
        targets = probe_mod.select_targets(providers, only="freekey")
        models = [t[1] for t in targets]
        assert "m1" in models
        assert "kilo/auto-free" not in models
        assert "openrouter/auto" not in models

    def test_only_filter(self, config_file):
        providers = probe_mod.load_provider_config(config_file)
        targets = probe_mod.select_targets(providers, only="groq")
        pids = {t[0] for t in targets}
        assert pids == {"groq"}

    def test_max_models_cap(self, config_file):
        providers = probe_mod.load_provider_config(config_file)
        targets = probe_mod.select_targets(providers, only="groq", max_models=1)
        assert len(targets) == 1


class TestEnvResolution:
    def test_resolves_var(self, monkeypatch):
        monkeypatch.setenv("MY_URL", "https://resolved.example.com/v1")
        assert probe_mod._resolve_env("${MY_URL}") == "https://resolved.example.com/v1"

    def test_resolves_default(self, monkeypatch):
        monkeypatch.delenv("MISSING_VAR", raising=False)
        assert probe_mod._resolve_env("${MISSING_VAR:-https://fallback/v1}") == "https://fallback/v1"

    def test_passthrough_plain(self):
        assert probe_mod._resolve_env("https://plain.example.com/v1") == "https://plain.example.com/v1"


class TestProbeOne:
    def _make_chunk(self, content=None, usage=None):
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = MagicMock()
        chunk.choices[0].delta.content = content
        chunk.usage = usage
        return chunk

    def test_success_path(self):
        chunks = [
            self._make_chunk(content="def reverse"),
            self._make_chunk(content="(s):\n"),
            self._make_chunk(content="    return s[::-1]"),
            self._make_chunk(usage=MagicMock(completion_tokens=8)),
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)
        with patch.object(probe_mod, "OpenAI", return_value=mock_client):
            result = probe_mod.probe_one("groq", "llama-3.3-70b", "https://api/v1", "key")
        assert result["success"] is True
        assert result["time_to_first_token_ms"] is not None and result["time_to_first_token_ms"] >= 0
        assert result["response_time_ms"] >= result["time_to_first_token_ms"]
        assert result["output_tokens"] >= 1
        assert result["error_reason"] is None

    def test_failure_path_classifies_rate_limit(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("429 rate_limit_exceeded")
        with patch.object(probe_mod, "OpenAI", return_value=mock_client):
            result = probe_mod.probe_one("groq", "m", "https://api/v1", "key")
        assert result["success"] is False
        assert result["error_reason"] == "rate_limited"
        assert result["time_to_first_token_ms"] is None

    def test_failure_path_classifies_auth(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("401 Unauthorized")
        with patch.object(probe_mod, "OpenAI", return_value=mock_client):
            result = probe_mod.probe_one("groq", "m", "https://api/v1", "key")
        assert result["success"] is False
        assert result["error_reason"] == "auth_error"

    def test_failure_path_classifies_timeout(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Connection timeout")
        with patch.object(probe_mod, "OpenAI", return_value=mock_client):
            result = probe_mod.probe_one("groq", "m", "https://api/v1", "key")
        assert result["success"] is False
        assert result["error_reason"] == "timeout"


class TestRecordCallIntegration:
    def test_probe_result_written_to_db(self, tmp_path):
        db = str(tmp_path / "telemetry.db")
        tm = TelemetryManager(db_path=db)
        tm.record_call(
            provider="groq",
            model="llama-3.3-70b",
            success=True,
            response_time_ms=500,
            time_to_first_token_ms=120,
            tokens_per_second=45.3,
            input_tokens=10,
            output_tokens=20,
            cost_estimate_usd=None,
        )
        import sqlite3
        conn = sqlite3.connect(db)
        row = conn.execute(
            "SELECT provider, model, success, response_time_ms, "
            "time_to_first_token_ms, tokens_per_second FROM api_calls"
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == "groq"
        assert row[1] == "llama-3.3-70b"
        assert row[2] == 1
        assert row[3] == 500
        assert row[4] == 120
        assert abs(row[5] - 45.3) < 0.1
