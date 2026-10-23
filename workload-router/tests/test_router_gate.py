"""Tests for the cutover feature flag (default OFF) and dry-run logging."""

import io
import json

import router_gate


class Deployment:
    def __init__(self, deployment_id="dep-a", chain_priority=1):
        self.deployment_id = deployment_id
        self.chain_priority = chain_priority


def test_cutover_defaults_off():
    assert router_gate.cutover_enabled({}) is False
    assert router_gate.cutover_enabled({"ROUTER_CUTOVER_ENABLED": ""}) is False
    assert router_gate.cutover_enabled({"ROUTER_CUTOVER_ENABLED": "   "}) is False
    assert router_gate.cutover_enabled({"ROUTER_CUTOVER_ENABLED": "0"}) is False
    assert router_gate.cutover_enabled({"ROUTER_CUTOVER_ENABLED": "false"}) is False


def test_cutover_requires_explicit_truthy():
    assert router_gate.cutover_enabled({"ROUTER_CUTOVER_ENABLED": "1"}) is True
    assert router_gate.cutover_enabled({"ROUTER_CUTOVER_ENABLED": "true"}) is True
    assert router_gate.cutover_enabled({"ROUTER_CUTOVER_ENABLED": "YES"}) is True
    assert router_gate.cutover_enabled({"ROUTER_CUTOVER_ENABLED": "on"}) is True


def test_dry_run_logging_defaults_on_and_can_be_disabled():
    assert router_gate.dry_run_logging_enabled({}) is True
    assert router_gate.dry_run_logging_enabled({"ROUTER_DRY_RUN_LOG": "0"}) is False
    assert router_gate.dry_run_logging_enabled({"ROUTER_DRY_RUN_LOG": "off"}) is False


def test_dry_run_line_reports_tier_deployment_why():
    line = router_gate.dry_run_line(
        Deployment("dep-b", 2),
        "warm_cache_override",
        session_id="s1",
        phase="implement",
        env={},
    )
    record = json.loads(line)
    assert record["mode"] == "dry_run"
    assert record["tier"] == 2
    assert record["deployment"] == "dep-b"
    assert record["why"] == "warm_cache_override"
    assert record["session"] == "s1"
    assert record["phase"] == "implement"


def test_dry_run_line_mode_reflects_cutover_flag():
    record = json.loads(router_gate.dry_run_line(Deployment(), "chain_top", env={"ROUTER_CUTOVER_ENABLED": "1"}))
    assert record["mode"] == "cutover"


def test_log_dry_run_writes_one_json_line_to_stream():
    stream = io.StringIO()
    router_gate.log_dry_run(Deployment("dep-c", 0), "chain_top", session_id="s2", env={}, stream=stream)
    lines = stream.getvalue().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["deployment"] == "dep-c"


def test_log_dry_run_silent_when_disabled():
    stream = io.StringIO()
    router_gate.log_dry_run(Deployment(), "chain_top", env={"ROUTER_DRY_RUN_LOG": "0"}, stream=stream)
    assert stream.getvalue() == ""


def test_route_endpoint_reports_cutover_flag(monkeypatch):
    import threading
    from http.client import HTTPConnection
    from http.server import ThreadingHTTPServer

    from router_server import RouterHandler

    monkeypatch.delenv("ROUTER_CUTOVER_ENABLED", raising=False)
    server = ThreadingHTTPServer(("127.0.0.1", 0), RouterHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection(*server.server_address)
        connection.request(
            "POST",
            "/v1/router/route",
            json.dumps({"messages": [{"role": "user", "content": "inspect the files"}]}),
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
        connection.close()
        assert response.status == 200
        assert payload["cutover_enabled"] is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
