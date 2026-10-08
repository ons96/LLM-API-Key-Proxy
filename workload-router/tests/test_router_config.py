import json

from router_config import default_deployments, load_deployments


def test_catalog_loads_valid_profiles(tmp_path):
    path = tmp_path / "deployments.json"
    path.write_text(
        json.dumps([{"deployment_id": "custom", "capability": "fast_general", "context_limit": 1234, "provider": "mock"}]),
        encoding="utf-8",
    )
    loaded = load_deployments(path)
    assert loaded[0].deployment_id == "custom"
    assert loaded[0].context_limit == 1234
    assert loaded[0].provider == "mock"


def test_catalog_invalid_file_uses_safe_defaults(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("not json", encoding="utf-8")
    assert [item.deployment_id for item in load_deployments(path)] == [
        item.deployment_id for item in default_deployments()
    ]


def test_catalog_invalid_ranges_use_safe_defaults(tmp_path):
    path = tmp_path / "invalid-range.json"
    path.write_text(json.dumps([{"deployment_id": "bad", "capability": "fast_general", "context_limit": 0, "success_rate": 2}]), encoding="utf-8")
    assert [item.deployment_id for item in load_deployments(path)] == [item.deployment_id for item in default_deployments()]
