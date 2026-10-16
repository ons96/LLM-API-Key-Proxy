import json

import pytest

from router_config import default_deployments, load_deployments
from router_core import RequestFeatures, route_plan


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("latency_ms", float("nan")),
        ("latency_ms", float("inf")),
        ("success_rate", float("nan")),
        ("success_rate", float("inf")),
        ("cache_ttl_seconds", float("nan")),
        ("cache_ttl_seconds", float("inf")),
        ("quota_reset_at", float("nan")),
        ("quota_reset_at", float("inf")),
    ],
)
def test_catalog_rejects_non_finite_float_values(tmp_path, field, value):
    path = tmp_path / f"non-finite-{field}.json"
    entry = {
        "deployment_id": "bad",
        "capability": "fast_general",
        "context_limit": 1234,
        field: value,
    }
    path.write_text(json.dumps([entry]), encoding="utf-8")
    assert [item.deployment_id for item in load_deployments(path)] == [
        item.deployment_id for item in default_deployments()
    ]


def test_default_catalog_covers_deep_reasoning():
    """Regression (#1095): deep_reasoning had no deployment, so every reasoning
    request resolved to no_eligible_deployment and could never be routed."""
    deployments = default_deployments()
    assert any(item.capability.value == "deep_reasoning" for item in deployments)

    plan = route_plan(
        RequestFeatures("prove step by step that this holds"), deployments
    )
    assert plan.candidates[0].capability.value == "deep_reasoning"


def test_catalog_unknown_capability_uses_safe_defaults(tmp_path):
    """Documented trap: one unknown capability string discards the whole catalog
    (load_deployments falls back to defaults), so capability typos hide as
    'missing deployment' instead of raising."""
    path = tmp_path / "unknown-capability.json"
    path.write_text(
        json.dumps(
            [
                {
                    "deployment_id": "weird",
                    "capability": "not_a_capability",
                    "context_limit": 1024,
                }
            ]
        ),
        encoding="utf-8",
    )
    assert [item.deployment_id for item in load_deployments(path)] == [
        item.deployment_id for item in default_deployments()
    ]
