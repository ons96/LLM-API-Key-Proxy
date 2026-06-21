"""Tests for scripts/build_embedding_chain.py.

Covers:
- compute_composite: perfect score, RPM rejection, MTEB-absent default
- load_provider_db: filters non-embed, paid providers; includes no-key providers
- build_chain: LiteLLM output format, sorted DESC, health check
- write_yaml / write_csv: format validity
- RPM headroom: burst RPM 20 = full headroom
"""

from __future__ import annotations

import csv
import importlib.util
import os
import sqlite3
import sys
from pathlib import Path
from typing import List, Tuple
from unittest import TestCase

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_embedding_chain.py"

# Load the module under test by file path (scripts/ is not a package).
_spec = importlib.util.spec_from_file_location("build_embedding_chain", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
bec = importlib.util.module_from_spec(_spec)
sys.modules["build_embedding_chain"] = bec
_spec.loader.exec_module(bec)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_db(db_path: Path, rows: List[dict]) -> None:
    """Create a minimal providers+models schema with the given rows."""
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE providers (
            id INTEGER PRIMARY KEY,
            key_name TEXT,
            display_name TEXT,
            base_url TEXT,
            env_var TEXT,
            enabled INTEGER DEFAULT 1,
            free_tier INTEGER DEFAULT 0,
            no_api_key_required INTEGER DEFAULT 0,
            signup_url TEXT,
            notes TEXT,
            source TEXT,
            capabilities TEXT,
            rate_limit_rpm INTEGER,
            rate_limit_daily_tokens INTEGER,
            last_verified TEXT,
            checkin_required INTEGER DEFAULT 0,
            checkin_unlimited INTEGER DEFAULT 0,
            free_one_time INTEGER DEFAULT 0,
            free_unlimited INTEGER DEFAULT 0,
            free_daily INTEGER DEFAULT 0
        );
        CREATE TABLE models (
            id INTEGER PRIMARY KEY,
            provider_id INTEGER,
            model_id TEXT,
            display_name TEXT,
            context_window INTEGER,
            tps REAL,
            capabilities TEXT,
            free_tier INTEGER DEFAULT 0,
            tier TEXT,
            fallback_only INTEGER DEFAULT 0,
            updated_at TEXT,
            last_verified TEXT,
            FOREIGN KEY(provider_id) REFERENCES providers(id)
        );
        """
    )
    for i, r in enumerate(rows, 1):
        pid = i
        conn.execute(
            """INSERT INTO providers
               (id, key_name, display_name, base_url, env_var,
                free_tier, no_api_key_required, rate_limit_rpm)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pid,
                r["provider"],
                r["provider"],
                r.get("base_url", "https://example.com/v1"),
                r.get("env_var", "EXAMPLE_KEY"),
                r.get("free_tier", 1),
                r.get("no_api_key", 0),
                r.get("rpm", 10),
            ),
        )
        conn.execute(
            """INSERT INTO models
               (id, provider_id, model_id, display_name, free_tier)
               VALUES (?, ?, ?, ?, ?)""",
            (i, pid, r["model"], r.get("display_name", r["model"]), 1),
        )
    conn.commit()
    conn.close()


def _make_telemetry_db(db_path: Path, rows: List[dict]) -> None:
    """Create llm_events schema and insert rows for telemetry tests."""
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE llm_events (
            id INTEGER PRIMARY KEY,
            request_id TEXT,
            ts_start REAL,
            ts_end REAL,
            ts_first_token REAL,
            model TEXT,
            provider TEXT,
            stream INTEGER,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            ttft_ms REAL,
            total_ms REAL,
            tps REAL,
            cost_usd REAL,
            caller TEXT,
            agent_session TEXT,
            status TEXT,
            error TEXT
        );
        """
    )
    import time

    now = time.time()
    for i, r in enumerate(rows, 1):
        conn.execute(
            """INSERT INTO llm_events
               (id, ts_start, ts_end, ts_first_token, model, provider,
                ttft_ms, total_ms, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                i,
                now - r.get("age_s", 0),
                now - r.get("age_s", 0) + 0.1,
                now - r.get("age_s", 0) + 0.05,
                r["model"],
                r["provider"],
                r.get("ttft_ms", 100.0),
                r.get("total_ms", 500.0),
                r.get("status", "success"),
            ),
        )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# TestComputeComposite
# ---------------------------------------------------------------------------


class TestComputeComposite(TestCase):
    def test_perfect_score(self) -> None:
        c = bec.EmbeddingCandidate(
            provider="nvidia",
            model="nv-embed-v1",
            base_url="",
            env_var="",
            no_api_key_required=False,
            rpm=40,
            quality=1.0,
            avg_ttft_ms=100.0,
            success_rate=1.0,
        )
        score, reason = bec.compute_composite(c)
        # quality(1.0)*0.5 + rpm_headroom(1.0)*0.25 + latency(0.98)*0.2 + reliability(1.0)*0.05
        # = 0.5 + 0.25 + 0.196 + 0.05 = 0.996
        self.assertAlmostEqual(score, 0.996, places=2)
        self.assertEqual(reason, "")

    def test_rpm_rejected_below_min(self) -> None:
        c = bec.EmbeddingCandidate(
            provider="mistral",
            model="mistral-embed",
            base_url="",
            env_var="",
            no_api_key_required=False,
            rpm=10,
        )
        score, reason = bec.compute_composite(c, min_rpm=15)
        self.assertEqual(score, 0.0)
        self.assertIn("rpm=10", reason)
        self.assertIn("MIN_RPM=15", reason)

    def test_mteb_absent_defaults_to_0_5(self) -> None:
        c = bec.EmbeddingCandidate(
            provider="gemini",
            model="gemini-embedding-001",
            base_url="",
            env_var="",
            no_api_key_required=False,
            rpm=15,
        )
        # No MTEB CSV loaded, quality should default to 0.5
        self.assertEqual(c.quality, 0.5)
        score, reason = bec.compute_composite(c)
        self.assertEqual(reason, "")
        # Should include quality*0.5 component = 0.25
        self.assertGreaterEqual(score, 0.25)


# ---------------------------------------------------------------------------
# TestLoadProviderDb
# ---------------------------------------------------------------------------


class TestLoadProviderDb(TestCase):
    def setUp(self) -> None:
        self.db = Path(__file__).resolve().parent / "test_embedding_chain.db"
        if self.db.exists():
            self.db.unlink()

    def tearDown(self) -> None:
        if self.db.exists():
            self.db.unlink()

    def test_filters_non_embed_models(self) -> None:
        _make_db(
            self.db,
            [
                {"provider": "nvidia", "model": "nv-embed-v1", "rpm": 40},
                {"provider": "openai", "model": "gpt-4o", "rpm": 60},
            ],
        )
        candidates = bec.load_provider_db(str(self.db))
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].model, "nv-embed-v1")

    def test_filters_paid_providers(self) -> None:
        _make_db(
            self.db,
            [
                {"provider": "nvidia", "model": "nv-embed-v1", "rpm": 40, "free_tier": 1},
                {"provider": "anthropic", "model": "claude-embed", "rpm": 50, "free_tier": 0},
            ],
        )
        candidates = bec.load_provider_db(str(self.db))
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].provider, "nvidia")

    def test_includes_no_api_key_providers(self) -> None:
        _make_db(
            self.db,
            [
                {"provider": "nvidia", "model": "nv-embed-v1", "rpm": 40, "free_tier": 1, "no_api_key": 0},
                {"provider": "g4f", "model": "g4f-embed", "rpm": 30, "free_tier": 0, "no_api_key": 1},
            ],
        )
        candidates = bec.load_provider_db(str(self.db))
        self.assertEqual(len(candidates), 2)
        providers = {c.provider for c in candidates}
        self.assertEqual(providers, {"nvidia", "g4f"})


# ---------------------------------------------------------------------------
# TestBuildChain
# ---------------------------------------------------------------------------


class TestBuildChain(TestCase):
    def test_sorted_desc_by_composite(self) -> None:
        candidates = [
            bec.EmbeddingCandidate(
                provider="gemini", model="gemini-embed", base_url="", env_var="",
                no_api_key_required=False, rpm=15, quality=0.6,
            ),
            bec.EmbeddingCandidate(
                provider="nvidia", model="nv-embed-v1", base_url="", env_var="",
                no_api_key_required=False, rpm=40, quality=0.9,
            ),
        ]
        accepted, rejected = bec.build_chain(candidates, {}, {})
        self.assertEqual(len(accepted), 2)
        self.assertEqual(len(rejected), 0)
        # nvidia has higher quality + higher RPM headroom
        self.assertEqual(accepted[0].provider, "nvidia")
        self.assertEqual(accepted[1].provider, "gemini")
        self.assertGreater(accepted[0].composite, accepted[1].composite)

    def test_rejected_entries_carried_through(self) -> None:
        candidates = [
            bec.EmbeddingCandidate(
                provider="mistral", model="mistral-embed", base_url="", env_var="",
                no_api_key_required=False, rpm=10,
            ),
            bec.EmbeddingCandidate(
                provider="nvidia", model="nv-embed-v1", base_url="", env_var="",
                no_api_key_required=False, rpm=40,
            ),
        ]
        accepted, rejected = bec.build_chain(candidates, {}, {}, min_rpm=15)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0].provider, "mistral")
        self.assertTrue(rejected[0].reject_reason)

    def test_empty_candidates_returns_empty(self) -> None:
        accepted, rejected = bec.build_chain([], {}, {})
        self.assertEqual(accepted, [])
        self.assertEqual(rejected, [])


# ---------------------------------------------------------------------------
# TestOutputFormats
# ---------------------------------------------------------------------------


class TestOutputFormats(TestCase):
    def setUp(self) -> None:
        import tempfile

        self.tmpdir = tempfile.mkdtemp(prefix="emb_chain_test_")
        self.yaml_path = os.path.join(self.tmpdir, "out.yaml")
        self.csv_path = os.path.join(self.tmpdir, "out.csv")
        self.candidates = [
            bec.EmbeddingCandidate(
                provider="nvidia", model="nvidia/nv-embed-v1",
                base_url="https://integrate.api.nvidia.com/v1",
                env_var="NVIDIA_API_KEY", no_api_key_required=False, rpm=40,
            ),
        ]

    def test_yaml_valid_litemllm_model_list(self) -> None:
        import yaml

        bec.write_yaml(self.candidates, self.yaml_path)
        with open(self.yaml_path) as f:
            doc = yaml.safe_load(f)
        self.assertIn("model_list", doc)
        self.assertEqual(len(doc["model_list"]), 1)
        entry = doc["model_list"][0]
        self.assertEqual(entry["model_name"], "embeddings")
        self.assertIn("litellm_params", entry)
        self.assertEqual(entry["litellm_params"]["model"], "nvidia/nv-embed-v1")
        self.assertEqual(entry["litellm_params"]["api_base"], "https://integrate.api.nvidia.com/v1")
        self.assertEqual(entry["litellm_params"]["api_key"], "os.environ/NVIDIA_API_KEY")
        self.assertIn("metadata", doc)

    def test_csv_has_all_columns(self) -> None:
        bec.write_csv(self.candidates + [], self.csv_path)
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            rows = list(reader)
        expected = ["rank", "provider", "model", "rpm", "quality", "avg_ttft_ms",
                    "success_rate", "samples", "composite", "reject_reason"]
        self.assertEqual(headers, expected)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["provider"], "nvidia")


# ---------------------------------------------------------------------------
# TestRpmHeadroom
# ---------------------------------------------------------------------------


class TestRpmHeadroom(TestCase):
    def test_burst_rpm_20_gets_full_headroom(self) -> None:
        # With rpm=20 and BURST_RPM=20, headroom = min(20/20, 1.0) = 1.0
        c = bec.EmbeddingCandidate(
            provider="test", model="test-embed", base_url="", env_var="",
            no_api_key_required=False, rpm=20,
        )
        score, reason = bec.compute_composite(c)
        self.assertEqual(reason, "")
        # quality(0.5)*0.5 + rpm_headroom(1.0)*0.25 + latency(0.5)*0.2 + reliability(0.5)*0.05
        # = 0.25 + 0.25 + 0.1 + 0.025 = 0.625
        self.assertAlmostEqual(score, 0.625, places=3)

    def test_rpm_unknown_treated_as_half_headroom(self) -> None:
        # rpm=0 = unknown; headroom=0.5 per ponytail default
        c = bec.EmbeddingCandidate(
            provider="unknown", model="mystery-embed", base_url="", env_var="",
            no_api_key_required=False, rpm=0,
        )
        score, reason = bec.compute_composite(c)
        self.assertEqual(reason, "")
        # quality(0.5)*0.5 + rpm_headroom(0.5)*0.25 + latency(0.5)*0.2 + reliability(0.5)*0.05
        # = 0.25 + 0.125 + 0.1 + 0.025 = 0.5
        self.assertAlmostEqual(score, 0.5, places=3)


# ---------------------------------------------------------------------------
# TestLoadTelemetry
# ---------------------------------------------------------------------------


class TestLoadTelemetry(TestCase):
    def setUp(self) -> None:
        import tempfile

        self.tmpdir = tempfile.mkdtemp(prefix="emb_tel_test_")
        self.db_path = os.path.join(self.tmpdir, "tel.db")

    def test_empty_db_returns_empty(self) -> None:
        # Missing file -> empty dict
        result = bec.load_telemetry("/nonexistent/path.db")
        self.assertEqual(result, {})

    def test_aggregates_per_provider_model(self) -> None:
        rows = [
            {"provider": "nvidia", "model": "nv-embed-v1", "ttft_ms": 100.0, "status": "success"},
            {"provider": "nvidia", "model": "nv-embed-v1", "ttft_ms": 200.0, "status": "success"},
            {"provider": "nvidia", "model": "nv-embed-v1", "ttft_ms": 0.0, "status": "error"},
        ]
        _make_telemetry_db(Path(self.db_path), rows)
        result = bec.load_telemetry(self.db_path)
        self.assertEqual(len(result), 1)
        key = ("nvidia", "nv-embed-v1")
        self.assertIn(key, result)
        ttft, success, samples = result[key]
        self.assertEqual(samples, 3)
        # avg_ttft = (100+200+0)/3 = 100.0
        self.assertAlmostEqual(ttft, 100.0, places=1)
        # success_rate = 2/3
        self.assertAlmostEqual(success, 2.0 / 3.0, places=2)


if __name__ == "__main__":
    import unittest

    unittest.main()
