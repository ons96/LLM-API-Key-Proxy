"""Tests for the gated workload-router dry run (task-board #1070).

The contract under test: the shadow path is off by default, records
tier/deployment/why without prompt content, never mutates the request, never
raises, and leaves the live routing decision byte-identical whether it is
enabled or not.
"""

import asyncio
import copy
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proxy_app import workload_router_shadow as shadow  # noqa: E402
from proxy_app.router_wrapper import RouterWrapper  # noqa: E402


TOOL_REQUEST = {
    "model": "coding-smart",
    "messages": [
        {"role": "user", "content": "inspect the files and list the symbols"}
    ],
    "tools": [{"type": "function", "function": {"name": "read_file"}}],
    "stream": False,
}


@pytest.fixture
def dry_run(tmp_path, monkeypatch):
    """Enable the dry run with an isolated log file for one test."""
    log = tmp_path / "shadow.jsonl"
    monkeypatch.setenv(shadow.ENV_ENABLE, "1")
    monkeypatch.setenv(shadow.ENV_LOG_PATH, str(log))
    monkeypatch.delenv(shadow.ENV_CATALOG, raising=False)
    monkeypatch.delenv(shadow.ENV_MAX_BYTES, raising=False)
    return log


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_disabled_by_default_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv(shadow.ENV_ENABLE, raising=False)
    log = tmp_path / "shadow.jsonl"
    monkeypatch.setenv(shadow.ENV_LOG_PATH, str(log))

    assert shadow.is_enabled() is False
    assert shadow.observe(dict(TOOL_REQUEST)) is None
    assert not log.exists()


def test_disabled_flag_variants(monkeypatch, tmp_path):
    monkeypatch.setenv(shadow.ENV_LOG_PATH, str(tmp_path / "s.jsonl"))
    for value in ("", "0", "false", "no", "off"):
        monkeypatch.setenv(shadow.ENV_ENABLE, value)
        assert shadow.is_enabled() is False
        assert shadow.observe(dict(TOOL_REQUEST)) is None
    monkeypatch.setenv(shadow.ENV_ENABLE, "1")
    assert shadow.is_enabled() is True


def test_records_tier_deployment_and_why(dry_run):
    record = shadow.observe(dict(TOOL_REQUEST), request_id="req_abc", live_model="coding-smart")

    assert record is not None
    assert record["tier"] == "fast_tool"
    assert record["deployment"] == "tool"
    assert record["why"] in {"chain_top", "chain_top_warm_cache"}
    assert record["capability"] == "fast_tool"
    assert record["live_model"] == "coding-smart"
    assert record["mode"] == "dry_run"
    assert record["request_id"] == "req_abc"

    rows = _read(dry_run)
    assert len(rows) == 1
    assert set(("tier", "deployment", "why")) <= set(rows[0])
    assert rows[0]["deployment"] == "tool"


def test_prompt_content_is_never_logged(dry_run):
    marker = "SECRET-CANARY-do-not-log-8fd2"
    request = copy.deepcopy(TOOL_REQUEST)
    request["messages"][0]["content"] = f"please {marker} inspect the files"

    shadow.observe(request)

    raw = dry_run.read_text()
    assert marker not in raw
    assert "inspect the files" not in raw
    # Counts and flags are logged instead of content.
    assert json.loads(raw)["messages"] == 1
    assert json.loads(raw)["tools"] is True


def test_request_is_not_mutated(dry_run):
    request = copy.deepcopy(TOOL_REQUEST)
    before = copy.deepcopy(request)

    shadow.observe(request)

    assert request == before
    assert shadow.observe(request) is not None
    assert request == before


def test_expected_miss_is_reported_not_raised(dry_run):
    """No eligible deployment must surface as a reason, not an exception."""
    request = dict(TOOL_REQUEST, response_format={"type": "json_object"})

    record = shadow.observe(request)

    assert record is not None
    assert record["deployment"] == ""
    assert record["why"] == "no_eligible_deployment"


def test_core_failure_is_contained(dry_run, monkeypatch):
    def boom():
        raise RuntimeError("core unavailable")

    monkeypatch.setattr(shadow, "_load_vendored", boom)

    assert shadow.observe(dict(TOOL_REQUEST)) is None
    assert not dry_run.exists()


def test_log_path_failure_is_contained(tmp_path, monkeypatch):
    monkeypatch.setenv(shadow.ENV_ENABLE, "1")
    # A path whose parent is an existing file -> mkdir raises.
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    monkeypatch.setenv(shadow.ENV_LOG_PATH, str(blocker / "shadow.jsonl"))

    assert shadow.observe(dict(TOOL_REQUEST)) is None


def test_phase_header_alias_maps_to_workload_phase(dry_run):
    headers = {"X-Phase-Router-Phase": "build_iteration"}

    record = shadow.observe(dict(TOOL_REQUEST), headers=headers)

    assert record["phase"] == "implement"
    assert record["tier"] == "code_implement"
    assert record["deployment"] == "implement"


def test_unknown_phase_stays_neutral(dry_run):
    record = shadow.observe(dict(TOOL_REQUEST), headers={"X-Phase-Router-Phase": "bogus"})

    assert record["phase"] == ""
    assert record["tier"] == "fast_tool"


def test_log_rotation_keeps_file_bounded(dry_run, monkeypatch):
    monkeypatch.setenv(shadow.ENV_MAX_BYTES, "300")
    for _ in range(4):
        shadow.observe(dict(TOOL_REQUEST))

    backup = dry_run.with_name(dry_run.name + ".1")
    assert backup.exists()
    assert dry_run.stat().st_size < 3000
    assert _read(backup)


class _StubIntegration:
    """Minimal router stand-in that records what routing actually received."""

    def __init__(self):
        self.seen = []

    async def chat_completions(self, request_data, raw_request):
        self.seen.append(copy.deepcopy(request_data))
        return {"ok": True, "model": request_data.get("model"), "choices": []}


class _StubRequest:
    def __init__(self, headers=None):
        self.headers = dict(headers or {})


def _wrapper(stub):
    wrapper = RouterWrapper.__new__(RouterWrapper)
    wrapper.router_integration = stub
    wrapper._initialized = True
    return wrapper


@pytest.mark.parametrize("enabled", [False, True])
def test_live_routing_is_identical_with_flag_on_or_off(tmp_path, monkeypatch, enabled):
    """The whole point: the shadow decision never reaches the router."""
    log = tmp_path / "shadow.jsonl"
    monkeypatch.setenv(shadow.ENV_LOG_PATH, str(log))
    if enabled:
        monkeypatch.setenv(shadow.ENV_ENABLE, "1")
    else:
        monkeypatch.delenv(shadow.ENV_ENABLE, raising=False)

    request = copy.deepcopy(TOOL_REQUEST)
    original = copy.deepcopy(request)
    stub = _StubIntegration()
    result = asyncio.run(_wrapper(stub).handle_chat_completions(request, _StubRequest()))

    assert result == {"ok": True, "model": "coding-smart", "choices": []}
    assert stub.seen == [original]
    assert request == original
    assert log.exists() is enabled


def test_flag_toggling_does_not_change_router_payload(tmp_path, monkeypatch):
    log = tmp_path / "shadow.jsonl"
    monkeypatch.setenv(shadow.ENV_LOG_PATH, str(log))

    payloads = []
    results = []
    for enabled in (False, True, False):
        if enabled:
            monkeypatch.setenv(shadow.ENV_ENABLE, "1")
        else:
            monkeypatch.delenv(shadow.ENV_ENABLE, raising=False)
        stub = _StubIntegration()
        request = copy.deepcopy(TOOL_REQUEST)
        results.append(
            asyncio.run(_wrapper(stub).handle_chat_completions(request, _StubRequest()))
        )
        payloads.append(stub.seen[0])

    assert results[0] == results[1] == results[2]
    assert payloads[0] == payloads[1] == payloads[2]
    assert sum(1 for line in log.read_text().splitlines() if line) == 1


def test_shadow_module_import_does_not_shadow_gateway_router_core():
    """Loading the vendored core must not leave a top-level router_core bound."""
    shadow.observe(dict(TOOL_REQUEST)) if shadow.is_enabled() else None
    shadow._load_vendored()
    assert "router_core" not in sys.modules

    from proxy_app import router_core as gateway_core

    assert hasattr(gateway_core, "RouterCore")
