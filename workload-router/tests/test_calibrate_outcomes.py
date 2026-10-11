import json

from tools.calibrate_outcomes import calibrate


def test_calibrate_outcomes_aggregates_without_prompt_data(tmp_path):
    path = tmp_path / "outcomes.jsonl"
    path.write_text("\n".join([
        json.dumps({"deployment": "general", "outcome": "success"}),
        json.dumps({"deployment": "general", "outcome": "semantic_failure"}),
        json.dumps({"deployment": "tool", "outcome": "success"}),
    ]), encoding="utf-8")
    report = calibrate(path)
    assert report["total_samples"] == 3
    assert report["deployments"]["general"]["success_rate"] == 0.5


def test_calibrate_outcomes_rejects_unknown_outcome(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"deployment": "general", "outcome": "unknown"}), encoding="utf-8")
    try:
        calibrate(path)
    except ValueError as error:
        assert "unsupported" in str(error)
    else:
        raise AssertionError("unknown outcome should be rejected")
