"""Tests for scripts/reorder_chains.py — dynamic fallback chain reorder.

Verifies:
- Telemetry stats aggregate correctly from llm_events table
- Composite scoring weights success_rate > tps > ttft > penalty
- Insufficient-sample entries keep original order, moved to tail
- Healthy provider rises; failing provider drops
- Backup file created with timestamp
- Dry-run does not write
- No-thrash: below-min-samples chain stays unchanged

Run: python -m pytest tests/test_reorder_chains.py -v
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Dict, List, Tuple

# Make scripts/ importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
# Also allow `from reorder_chains import ...` (script has no .py extension issue)
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "reorder_chains",
    _REPO_ROOT / "scripts" / "reorder_chains.py",
)
assert _spec is not None and _spec.loader is not None
reorder_chains = importlib.util.module_from_spec(_spec)
# Register before exec_module so @dataclass can resolve the module by name.
sys.modules["reorder_chains"] = reorder_chains
_spec.loader.exec_module(reorder_chains)

from reorder_chains import (  # type: ignore
    ChainEntry,
    TelemetryStat,
    compute_composite,
    load_telemetry,
    reorder_chain,
    reorder_config,
)


def _make_telemetry_db(db_path: Path, rows: List[Dict]) -> None:
    """Create llm_events table and insert synthetic rows."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
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
            )
            """
        )
        conn.execute("CREATE INDEX idx_ts_start ON llm_events(ts_start)")
        conn.execute("CREATE INDEX idx_model ON llm_events(model)")
        conn.execute("CREATE INDEX idx_provider ON llm_events(provider)")
        for row in rows:
            conn.execute(
                """
                INSERT INTO llm_events (
                    request_id, ts_start, ts_end, ts_first_token,
                    model, provider, stream, prompt_tokens, completion_tokens,
                    ttft_ms, total_ms, tps, cost_usd, caller, agent_session, status, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row.get("request_id", "req-1"),
                    row.get("ts_start", time.time()),
                    row.get("ts_end", time.time() + 1),
                    row.get("ts_first_token", time.time() + 0.1),
                    row["model"],
                    row["provider"],
                    row.get("stream", 0),
                    row.get("prompt_tokens", 100),
                    row.get("completion_tokens", 50),
                    row.get("ttft_ms", 500.0),
                    row.get("total_ms", 1000.0),
                    row.get("tps", 50.0),
                    row.get("cost_usd", 0.0),
                    row.get("caller", "test"),
                    row.get("agent_session", ""),
                    row.get("status", "success"),
                    row.get("error", ""),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _make_config(yaml_path: Path, virtual_models: Dict) -> None:
    """Write a minimal virtual_models.yaml for testing."""
    import yaml

    config = {
        "metadata": {"generated_at": "2026-01-01T00:00:00Z"},
        "virtual_models": virtual_models,
    }
    with yaml_path.open("w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


class TestComputeComposite(unittest.TestCase):
    def test_perfect_score(self):
        stat = TelemetryStat(
            provider="groq",
            model="llama-3.3-70b",
            samples=100,
            success_rate=1.0,
            avg_tps=3000.0,  # = max_tps → tps_norm=1.0
            avg_ttft_ms=0.0,  # → ttft_norm=1.0
        )
        score, reason = compute_composite(stat, 0.0, 5, 3000.0, 30000.0)
        # success=0.4 + tps=0.3 + ttft=0.2 + penalty_inv=0.1 = 1.0
        self.assertAlmostEqual(score, 1.0, places=2)
        self.assertIn("samples=100", reason)

    def test_no_telemetry(self):
        score, reason = compute_composite(None, 0.0, 5, 3000.0, 30000.0)
        self.assertEqual(score, -1.0)
        self.assertEqual(reason, "no_telemetry")

    def test_insufficient_samples(self):
        stat = TelemetryStat("groq", "llama", 2, 1.0, 50.0, 500.0)
        score, reason = compute_composite(stat, 0.0, 5, 3000.0, 30000.0)
        self.assertEqual(score, -1.0)
        self.assertIn("insufficient", reason)

    def test_penalty_lowers_score(self):
        stat = TelemetryStat("groq", "llama", 100, 1.0, 0.0, 0.0)
        score_clean, _ = compute_composite(stat, 0.0, 5, 3000.0, 30000.0)
        score_penalized, _ = compute_composite(stat, 10.0, 5, 3000.0, 30000.0)
        self.assertLess(score_penalized, score_clean)

    def test_low_success_lowers_score(self):
        stat_good = TelemetryStat("groq", "llama", 100, 0.95, 50.0, 500.0)
        stat_bad = TelemetryStat("groq", "llama", 100, 0.10, 50.0, 500.0)
        score_good, _ = compute_composite(stat_good, 0.0, 5, 3000.0, 30000.0)
        score_bad, _ = compute_composite(stat_bad, 0.0, 5, 3000.0, 30000.0)
        self.assertGreater(score_good, score_bad)


class TestReorderChain(unittest.TestCase):
    def test_healthy_rises_failing_drops(self):
        chain = [
            {"provider": "bad", "model": "m-bad", "priority": 1},
            {"provider": "good", "model": "m-good", "priority": 2},
        ]
        stats = {
            ("bad", "m-bad"): TelemetryStat("bad", "m-bad", 100, 0.10, 5.0, 5000.0),
            ("good", "m-good"): TelemetryStat("good", "m-good", 100, 0.99, 200.0, 100.0),
        }
        new_chain, reasons = reorder_chain(chain, stats, {}, 5, 3000.0, 30000.0)
        self.assertEqual(new_chain[0]["provider"], "good")
        self.assertEqual(new_chain[1]["provider"], "bad")
        self.assertEqual(new_chain[0]["priority"], 1)
        self.assertEqual(new_chain[1]["priority"], 2)
        self.assertEqual(len(reasons), 2)

    def test_insufficient_samples_keeps_original_order_at_tail(self):
        chain = [
            {"provider": "novel", "model": "m-novel", "priority": 1},
            {"provider": "good", "model": "m-good", "priority": 2},
        ]
        stats = {
            ("good", "m-good"): TelemetryStat("good", "m-good", 100, 0.99, 200.0, 100.0),
            # novel has no stats
        }
        new_chain, _ = reorder_chain(chain, stats, {}, 5, 3000.0, 30000.0)
        # good (scored) first, novel (unscored) second
        self.assertEqual(new_chain[0]["provider"], "good")
        self.assertEqual(new_chain[1]["provider"], "novel")

    def test_empty_chain(self):
        new_chain, reasons = reorder_chain([], {}, {}, 5, 3000.0, 30000.0)
        self.assertEqual(new_chain, [])
        self.assertEqual(reasons, [])

    def test_preserves_extra_keys(self):
        chain = [
            {
                "provider": "good",
                "model": "m",
                "priority": 1,
                "capabilities": {"tools": True},
                "notes": "do not drop me",
            }
        ]
        stats = {("good", "m"): TelemetryStat("good", "m", 100, 1.0, 100.0, 100.0)}
        new_chain, _ = reorder_chain(chain, stats, {}, 5, 3000.0, 30000.0)
        self.assertEqual(new_chain[0]["capabilities"], {"tools": True})
        self.assertEqual(new_chain[0]["notes"], "do not drop me")

    def test_no_threash_all_below_min_samples(self):
        """If every entry is below min_samples, original order is preserved."""
        chain = [
            {"provider": "a", "model": "1", "priority": 1},
            {"provider": "b", "model": "2", "priority": 2},
            {"provider": "c", "model": "3", "priority": 3},
        ]
        # All have 2 samples (below default min_samples=5)
        stats = {
            ("a", "1"): TelemetryStat("a", "1", 2, 0.5, 10.0, 100.0),
            ("b", "2"): TelemetryStat("b", "2", 2, 0.1, 10.0, 100.0),
            ("c", "3"): TelemetryStat("c", "3", 2, 0.9, 10.0, 100.0),
        }
        new_chain, _ = reorder_chain(chain, stats, {}, 5, 3000.0, 30000.0)
        # All unscored → stable sort keeps original order
        self.assertEqual([c["provider"] for c in new_chain], ["a", "b", "c"])


class TestReorderConfig(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="reorder_test_")
        self.config_path = Path(self.tmpdir) / "virtual_models.yaml"
        self.db_path = Path(self.tmpdir) / "telemetry.db"

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_backup_created(self):
        _make_config(
            self.config_path,
            {
                "coding-fast": {
                    "description": "test",
                    "fallback_chain": [
                        {"provider": "bad", "model": "m-bad", "priority": 1},
                        {"provider": "good", "model": "m-good", "priority": 2},
                    ],
                }
            },
        )
        _make_telemetry_db(
            self.db_path,
            [
                {"provider": "bad", "model": "m-bad", "status": "error", "tps": 1.0, "ttft_ms": 5000.0}
                for _ in range(10)
            ]
            + [
                {"provider": "good", "model": "m-good", "status": "success", "tps": 200.0, "ttft_ms": 100.0}
                for _ in range(10)
            ],
        )
        total, reordered, log = reorder_config(
            self.config_path,
            str(self.db_path),
            window_h=24,
            min_samples=5,
            max_tps=3000.0,
            max_ttft_ms=30000.0,
            dry_run=False,
        )
        self.assertEqual(total, 1)
        self.assertEqual(reordered, 1)
        backups = list(self.config_path.parent.glob("virtual_models.yaml.bak-*"))
        self.assertEqual(len(backups), 1, f"expected 1 backup, got {backups}")

    def test_dry_run_does_not_write(self):
        _make_config(
            self.config_path,
            {
                "chat-fast": {
                    "description": "test",
                    "fallback_chain": [
                        {"provider": "a", "model": "1", "priority": 1},
                    ],
                }
            },
        )
        original_content = self.config_path.read_text()
        _make_telemetry_db(self.db_path, [])
        total, reordered, log = reorder_config(
            self.config_path,
            str(self.db_path),
            window_h=24,
            min_samples=5,
            max_tps=3000.0,
            max_ttft_ms=30000.0,
            dry_run=True,
        )
        self.assertEqual(self.config_path.read_text(), original_content)
        backups = list(self.config_path.parent.glob("virtual_models.yaml.bak-*"))
        self.assertEqual(len(backups), 0)

    def test_no_telemetry_db_returns_zero_reordered(self):
        _make_config(
            self.config_path,
            {
                "chat-fast": {
                    "description": "test",
                    "fallback_chain": [
                        {"provider": "a", "model": "1", "priority": 1},
                    ],
                }
            },
        )
        total, reordered, log = reorder_config(
            self.config_path,
            str(Path(self.tmpdir) / "nonexistent.db"),
            window_h=24,
            min_samples=5,
            max_tps=3000.0,
            max_ttft_ms=30000.0,
            dry_run=False,
        )
        self.assertEqual(reordered, 0)

    def test_metadata_last_reorder_at_set(self):
        _make_config(
            self.config_path,
            {
                "chat-fast": {
                    "description": "test",
                    "fallback_chain": [
                        {"provider": "a", "model": "1", "priority": 1},
                    ],
                }
            },
        )
        _make_telemetry_db(self.db_path, [])
        reorder_config(
            self.config_path,
            str(self.db_path),
            window_h=24,
            min_samples=5,
            max_tps=3000.0,
            max_ttft_ms=30000.0,
            dry_run=False,
        )
        import yaml
        with self.config_path.open() as f:
            config = yaml.safe_load(f)
        self.assertIn("last_reorder_at", config["metadata"])
        self.assertIn("last_reorder_stats", config["metadata"])
        self.assertEqual(config["metadata"]["last_reorder_stats"]["total_models"], 1)


class TestLoadTelemetry(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="reorder_lt_")
        self.db_path = Path(self.tmpdir) / "telemetry.db"

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_empty_db_returns_empty(self):
        stats = load_telemetry(str(self.db_path), window_h=24, min_samples=5)
        self.assertEqual(stats, {})

    def test_aggregates_per_provider_model(self):
        _make_telemetry_db(
            self.db_path,
            [
                {"provider": "groq", "model": "llama", "status": "success", "tps": 100.0, "ttft_ms": 100.0},
                {"provider": "groq", "model": "llama", "status": "success", "tps": 200.0, "ttft_ms": 200.0},
                {"provider": "groq", "model": "llama", "status": "error", "tps": 0.0, "ttft_ms": 5000.0},
            ],
        )
        stats = load_telemetry(str(self.db_path), window_h=24, min_samples=1)
        self.assertIn(("groq", "llama"), stats)
        stat = stats[("groq", "llama")]
        self.assertEqual(stat.samples, 3)
        self.assertAlmostEqual(stat.success_rate, 2.0 / 3.0, places=2)
        self.assertAlmostEqual(stat.avg_tps, 100.0, places=1)  # (100+200+0)/3

    def test_respects_window(self):
        """Events older than window_h are excluded."""
        old_ts = time.time() - (48 * 3600)  # 48h ago, outside 24h window
        _make_telemetry_db(
            self.db_path,
            [
                {"provider": "old", "model": "m", "status": "success", "tps": 1.0, "ttft_ms": 1.0, "ts_start": old_ts},
                {"provider": "new", "model": "m", "status": "success", "tps": 100.0, "ttft_ms": 100.0},
            ],
        )
        stats = load_telemetry(str(self.db_path), window_h=24, min_samples=1)
        self.assertIn(("new", "m"), stats)
        self.assertNotIn(("old", "m"), stats)


if __name__ == "__main__":
    unittest.main()
