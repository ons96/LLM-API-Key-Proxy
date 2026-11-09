"""Tests for the read-only provider coverage audit."""

from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

import yaml


def _load_audit_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_provider_coverage.py"
    spec = importlib.util.spec_from_file_location("audit_provider_coverage", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_db(path: Path) -> None:
    con = sqlite3.connect(path)
    try:
        con.executescript(
            """
            CREATE TABLE providers (
                id INTEGER,
                key_name TEXT,
                base_url TEXT,
                env_var TEXT,
                free_tier INTEGER,
                no_api_key_required INTEGER,
                rate_limit_rpm INTEGER,
                rate_limit_daily_tokens INTEGER
            );
            CREATE TABLE models (
                id INTEGER PRIMARY KEY,
                provider_id INTEGER,
                model_id TEXT,
                free_tier INTEGER
            );
            """
        )
        con.executemany(
            "INSERT INTO providers VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (1, "alpha", "https://alpha.example/v1", "ALPHA_API_KEY", 1, 0, 10, 0),
                (2, "gamma", "https://gamma.example/v1", "GAMMA_API_KEY", 1, 0, 0, 0),
                (3, "paid", "https://paid.example/v1", "PAID_API_KEY", 0, 0, 0, 0),
                (None, "orphan", "https://orphan.example/v1", "ORPHAN_KEY", 1, 0, 0, 0),
            ],
        )
        con.executemany(
            "INSERT INTO models (id, provider_id, model_id, free_tier) VALUES (?, ?, ?, ?)",
            [
                (1, 1, "alpha-free", 1),
                (2, 2, "gamma-free", 1),
                (3, 3, "paid-model", 0),
                (4, None, "orphan-model", 1),
            ],
        )
        con.commit()
    finally:
        con.close()


def test_audit_skips_orphans_and_finds_usable_gap(tmp_path):
    module = _load_audit_module()
    db_path = tmp_path / "providers.db"
    _make_db(db_path)

    config_path = tmp_path / "router_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "providers": {
                    "alpha": {
                        "base_url": "https://alpha.example/v1",
                        "env_var": "ALPHA_API_KEY",
                        "free_tier": True,
                    }
                },
                "router_models": {
                    "coding": {
                        "candidates": [
                            {"provider": "alpha", "model": "alpha-free"}
                        ]
                    }
                },
            }
        )
    )

    report = module.build_report(config_path, db_path, tmp_path / "missing-catalog.json.gz")

    assert report["configured_providers"] == 1
    assert report["fallback_chain_providers"] == ["alpha"]
    assert report["db_providers"] == 3
    assert report["db_free_providers"] == 2
    assert report["free_with_numeric_limit"] == 1
    assert report["free_without_numeric_limit"] == 1
    assert report["missing_free_providers"] == 1
    assert report["counts"]["usable"] == 1
    assert report["usable_with_numeric_limit"] == 0
    assert report["usable_without_numeric_limit"] == 1
    assert report["usable"][0]["name"] == "gamma"
    assert report["catalog"]["providers"] == 0
