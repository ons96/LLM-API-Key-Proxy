import json

from router_config import default_deployments
from trace_replay import load_traces, replay, write_report


def test_trace_replay_is_offline_and_selects_capabilities(tmp_path):
    path = tmp_path / "traces.jsonl"
    path.write_text(
        "\n".join([
            json.dumps({"prompt": "answer a question"}),
            json.dumps({"prompt": "inspect files", "tools": True}),
            json.dumps({"prompt": "run tests", "phase": "verify"}),
        ]),
        encoding="utf-8",
    )
    result = replay(load_traces(path), default_deployments(), "general")
    assert result.total == result.routed == 3
    assert result.selected == ("general", "tool", "verify")
    assert result.baseline_matches == 1
    assert result.selected_counts == {"general": 1, "tool": 1, "verify": 1}
    report = tmp_path / "report.json"
    write_report(result, report)
    assert '"baseline_matches": 1' in report.read_text(encoding="utf-8")
    assert '"baseline_match_rate": 0.3333333333333333' in report.read_text(encoding="utf-8")


def test_trace_loader_rejects_missing_prompt(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"tools": True}), encoding="utf-8")
    try:
        load_traces(path)
    except ValueError as error:
        assert "prompt" in str(error)
    else:
        raise AssertionError("missing prompt should be rejected")
