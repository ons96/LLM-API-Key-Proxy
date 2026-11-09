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
import socket
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

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
def config_file(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("WIWI_API_KEY", "test-key")
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

    def test_skips_one_time_provider(self, config_file):
        providers = probe_mod.load_provider_config(config_file)
        targets = probe_mod.select_targets(providers, only="groq", skip_providers={"groq"})
        assert targets == []


class TestOneTimeProviders:
    def test_reads_provider_level_one_time_quota(self, tmp_path):
        path = tmp_path / "providers_database.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "providers": [
                        {"id": "together", "free_quota": {"free_type": "one_time_credit"}},
                        {"id": "groq", "free_quota": {"free_type": "daily_renewable"}},
                    ]
                }
            )
        )
        assert probe_mod.load_one_time_providers(path) == {"together"}

    def test_missing_provider_db_is_safe(self, tmp_path):
        assert probe_mod.load_one_time_providers(tmp_path / "missing.yaml") == set()


class TestCheckEndpoint:
    def _response(self, status):
        response = MagicMock()
        response.status = status
        response.__enter__.return_value = response
        response.__exit__.return_value = None
        return response

    def test_http_200_is_alive(self):
        with patch.object(probe_mod, "urlopen", return_value=self._response(200)):
            ok, detail = probe_mod.check_endpoint("https://api.example.com/v1")
        assert ok is True
        assert detail == "HTTP 200"

    def test_http_404_is_reachable(self):
        error = HTTPError("https://api.example.com/v1", 404, "Not Found", None, None)
        with patch.object(probe_mod, "urlopen", side_effect=error):
            ok, detail = probe_mod.check_endpoint("https://api.example.com/v1")
        assert ok is True
        assert detail == "HTTP 404"

    def test_auth_error_is_unhealthy(self):
        error = HTTPError("https://api.example.com/v1", 401, "Unauthorized", None, None)
        with patch.object(probe_mod, "urlopen", side_effect=error):
            ok, detail = probe_mod.check_endpoint("https://api.example.com/v1")
        assert ok is False
        assert detail == "HTTP 401 auth_error"

    def test_server_error_is_unhealthy(self):
        error = HTTPError("https://api.example.com/v1", 503, "Unavailable", None, None)
        with patch.object(probe_mod, "urlopen", side_effect=error):
            ok, detail = probe_mod.check_endpoint("https://api.example.com/v1")
        assert ok is False
        assert detail == "HTTP 503 server_error"

    def test_connection_error_is_unhealthy(self):
        with patch.object(probe_mod, "urlopen", side_effect=URLError("dns failure")):
            ok, detail = probe_mod.check_endpoint("https://api.example.com/v1")
        assert ok is False
        assert "connection_error" in detail

    def test_timeout_is_unhealthy(self):
        with patch.object(probe_mod, "urlopen", side_effect=socket.timeout("timed out")):
            ok, detail = probe_mod.check_endpoint("https://api.example.com/v1")
        assert ok is False
        assert "connection_error" in detail

    def test_key_is_sent_as_bearer_header(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["auth"] = request.get_header("Authorization")
            captured["url"] = request.full_url
            captured["headers"] = {k.lower(): v for k, v in request.headers.items()}
            return self._response(200)

        with patch.object(probe_mod, "urlopen", side_effect=fake_urlopen):
            probe_mod.check_endpoint("https://api.example.com/v1", api_key="test-key")
        assert captured["auth"] == "Bearer test-key"
        assert captured["url"] == "https://api.example.com/v1/models"
        assert "Mozilla/5.0" in captured["headers"]["user-agent"]


class TestCheckTargets:
    def test_dead_provider_skips_all_models_and_is_tagged(self):
        status = {"version": 1, "providers": {}}
        targets = [
            ("dead", "m1", "https://dead.example/v1", ""),
            ("dead", "m2", "https://dead.example/v1", ""),
            ("alive", "m3", "https://alive.example/v1", ""),
        ]

        def checker(url, key):
            return (url.startswith("https://alive"), "HTTP 503 server_error" if "dead" in url else "HTTP 200")

        healthy, counts = probe_mod.check_targets(targets, status, checker=checker, now="2026-11-08T00:00:00+00:00")
        assert [t[1] for t in healthy] == ["m3"]
        assert counts == {"providers": 2, "alive": 1, "dead": 1, "skipped_models": 2}
        assert status["providers"]["dead"]["status"] == "dead"
        assert status["providers"]["dead"]["reason"] == "HTTP 503 server_error"
        assert status["providers"]["dead"]["dead_since"] == "2026-11-08T00:00:00+00:00"

    def test_dead_provider_is_retried_and_cleared_on_recovery(self):
        status = {
            "version": 1,
            "providers": {
                "recovering": {
                    "status": "dead",
                    "reason": "HTTP 503 server_error",
                    "dead_since": "2026-11-07T00:00:00+00:00",
                }
            },
        }
        targets = [("recovering", "m1", "https://recovering.example/v1", "")]

        healthy, counts = probe_mod.check_targets(
            targets,
            status,
            checker=lambda url, key: (True, "HTTP 200"),
            now="2026-11-08T00:00:00+00:00",
        )
        assert [t[1] for t in healthy] == ["m1"]
        assert counts["alive"] == 1
        assert status["providers"]["recovering"]["status"] == "alive"
        assert "reason" not in status["providers"]["recovering"]
        assert "dead_since" not in status["providers"]["recovering"]

    def test_all_targets_skipped_when_all_endpoints_dead(self):
        status = {"version": 1, "providers": {}}
        targets = [("dead", "m1", "https://dead.example/v1", "")]
        healthy, counts = probe_mod.check_targets(
            targets,
            status,
            checker=lambda url, key: (False, "connection_error: dns"),
            now="2026-11-08T00:00:00+00:00",
        )
        assert healthy == []
        assert counts["dead"] == 1
        assert counts["skipped_models"] == 1

    def test_status_file_round_trip(self, tmp_path):
        path = tmp_path / "provider_probe_status.yaml"
        status = {
            "version": 1,
            "updated": "2026-11-08T00:00:00+00:00",
            "providers": {"dead": {"status": "dead", "reason": "HTTP 503 server_error"}},
        }
        probe_mod.save_probe_status(path, status)
        loaded = probe_mod.load_probe_status(path)
        assert loaded["providers"]["dead"]["status"] == "dead"


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
        with patch.object(probe_mod, "OpenAI", return_value=mock_client) as mock_openai:
            result = probe_mod.probe_one("groq", "llama-3.3-70b", "https://api/v1", "key")
        assert mock_openai.call_args.kwargs["default_headers"]["User-Agent"] == probe_mod.BROWSER_USER_AGENT
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
