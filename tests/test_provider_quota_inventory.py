"""Tests for the read-only provider/model quota inventory."""

from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path


def _load_inventory_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "export_provider_quota_inventory.py"
    )
    spec = importlib.util.spec_from_file_location("provider_quota_inventory", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quota_labels_and_probe_policy():
    module = _load_inventory_module()

    assert module._quota_class(True, True, False, False, False) == "unlimited"
    assert module._quota_class(True, False, True, False, False) == "daily"
    assert module._quota_class(True, False, False, True, False) == "one_time"
    assert module._quota_class(True, False, True, True, False) == "daily+one_time"
    assert module._probe_policy("free_one_time", True, True, "env") == "do_not_probe_one_time"
    assert module._probe_policy("fully_free", True, False, "env") == "probe_one_model"
    assert module._probe_policy("paid", False, False, "env") == "not_a_probe_target"
    assert module._label_state(1, 0, 0, 0, "free_tier_limited") == "labeled"
    assert module._label_state(1, 0, 0, 0, "internal") == "labeled"
    assert module._label_state(1, 0, 0, 0, "") == "unlabeled_free"
    assert module._label_state(0, 0, 0, 0, "paid") == "labeled"


def test_csv_output_is_ascii(tmp_path):
    module = _load_inventory_module()
    path = tmp_path / "report.csv"
    module._write_csv(path, [{"name": "\u4e2d\u6587", "n": 1}], ["name", "n"])
    path.read_bytes().decode("ascii")
    assert "\\u4e2d\\u6587" in path.read_text(encoding="ascii")


def test_orphans_include_null_and_dangling_provider_ids(tmp_path):
    module = _load_inventory_module()
    db_path = tmp_path / "providers.db"
    con = sqlite3.connect(db_path)
    try:
        con.executescript(
            """
            CREATE TABLE providers (
                id INTEGER,
                key_name TEXT,
                base_url TEXT,
                provider_type TEXT,
                free_tier INTEGER
            );
            CREATE TABLE models (
                provider_id INTEGER,
                model_id TEXT,
                free_tier INTEGER
            );
            INSERT INTO providers VALUES (1, 'valid', 'https://valid.example/v1', 'paid', 0);
            INSERT INTO providers VALUES (NULL, 'orphan-provider', NULL, 'internal', 1);
            INSERT INTO models VALUES (NULL, 'null-model', 1);
            INSERT INTO models VALUES (99, 'dangling-model', 0);
            INSERT INTO models VALUES (1, 'valid-model', 1);
            """
        )
        con.commit()
    finally:
        con.close()

    providers, models = module._load_orphans(db_path)

    assert [row["key_name"] for row in providers] == ["orphan-provider"]
    assert {row["model_id"] for row in models} == {"null-model", "dangling-model"}
    assert {row["orphan_reason"] for row in models} == {
        "null_provider_id",
        "dangling_provider_id",
    }
