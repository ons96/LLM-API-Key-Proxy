"""Operator command behaviour, exercised without live providers or credentials."""

from __future__ import annotations

import json
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from newapi_operator import (OperatorError, apply_priorities, apply_routes,
                             boot_check, default_key_lookup, load_routes,
                             ordered_routes, preview_priorities, preview_routes,
                             refresh_catalogs, run_audit)
from quota_ledger import QuotaLedger, Route

TOOL = Path(__file__).resolve().parents[1] / "newapi_operator.py"


def route(route_id: str, channel_id: int) -> Route:
    return Route(route_id, "provider", "model", "opaque", "sc." + route_id, channel_id,
                 quota_type="unlimited", input_price=0, output_price=0,
                 context_window=32768, output_limit=4096)


class StubLedger:
    def __init__(self, routes):
        self._routes = list(routes)

    def routes(self):
        return list(self._routes)


class StubAdmin:
    def __init__(self, digest: str = "d" * 64) -> None:
        self.calls: list[tuple] = []
        self.digest = digest

    def audit(self, routes):
        self.calls.append(("audit", [item.route_id for item in routes]))
        return self.digest

    def audit_relay(self, key, token_id):
        self.calls.append(("relay", token_id))

    def set_priorities(self, ordered):
        self.calls.append(("priorities", [item.route_id for item in ordered]))


def test_audit_is_startup_equivalent_and_read_only():
    admin = StubAdmin()
    ledger = StubLedger([route("a", 7), route("b", 8)])

    result = run_audit(admin, ledger, "sk-relay", 9)

    assert result == {"routes": 2, "channels": [7, 8], "audit_digest": admin.digest}
    assert admin.calls == [("audit", ["a", "b"]), ("relay", 9)]


def test_preview_priorities_never_calls_admin():
    ledger = StubLedger([route("a", 7), route("b", 8)])

    result = preview_priorities(ledger, ["b", "a"])

    assert result["apply"] is False
    assert [row["route_id"] for row in result["order"]] == ["b", "a"]
    assert [row["priority"] for row in result["order"]] == [2, 1]
    assert [row["channel_id"] for row in result["order"]] == [8, 7]


def test_apply_priorities_passes_the_exact_order():
    admin = StubAdmin()
    ledger = StubLedger([route("a", 7), route("b", 8)])

    result = apply_priorities(admin, ledger, ["b", "a"])

    assert result["apply"] is True
    assert admin.calls == [("priorities", ["b", "a"])]


@pytest.mark.parametrize("order", [["a"], ["a", "b", "c"], ["a", "a"], []])
def test_incomplete_or_repeated_order_fails_closed(order):
    ledger = StubLedger([route("a", 7), route("b", 8)])

    with pytest.raises(OperatorError):
        ordered_routes(ledger, order)


class ModelsServer:
    """Minimal OpenAI-shaped /models endpoint for catalog refresh."""

    def __init__(self, models, status: int = 200) -> None:
        self.models = models
        self.status = status
        self.headers: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                outer.headers.append(dict(self.headers))
                body = json.dumps({"data": outer.models}).encode()
                self.send_response(outer.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)

    @property
    def base_url(self) -> str:
        return "http://127.0.0.1:%d/v1" % self.server.server_address[1]

    def __enter__(self):
        import threading
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def test_catalog_refresh_writes_models_and_sends_no_request_to_keyless_provider(tmp_path):
    ledger = QuotaLedger(tmp_path / "ledger.db")
    try:
        with ModelsServer([{"id": "m1"}, {"id": "m2"}]) as server:
            rows = [
                {"name": "keyless", "base_url": server.base_url, "env_var": None,
                 "no_api_key_required": 1},
                {"name": "keyed", "base_url": server.base_url, "env_var": "PROVIDER_KEY",
                 "no_api_key_required": 0},
            ]
            result = refresh_catalogs(ledger, rows, lambda row: "" if row["no_api_key_required"] else "sk-test")
    finally:
        ledger.close()

    assert result["providers"] == {"keyless": 2, "keyed": 2}
    assert len(server.headers) == 2
    assert "Authorization" not in server.headers[0]
    assert server.headers[1]["Authorization"] == "Bearer sk-test"


def test_empty_provider_catalog_is_rejected(tmp_path):
    ledger = QuotaLedger(tmp_path / "ledger.db")
    try:
        with ModelsServer([]) as server:
            rows = [{"name": "empty", "base_url": server.base_url, "env_var": None,
                     "no_api_key_required": 1}]
            with pytest.raises(Exception):
                refresh_catalogs(ledger, rows, lambda row: "")
    finally:
        ledger.close()


def test_missing_key_env_var_names_only_the_variable(monkeypatch):
    monkeypatch.delenv("PROVIDER_KEY", raising=False)

    with pytest.raises(OperatorError) as error:
        default_key_lookup({"name": "p", "env_var": "PROVIDER_KEY", "no_api_key_required": 0})

    assert "PROVIDER_KEY" in str(error.value)


def test_boot_check_counts_models_per_provider(tmp_path):
    config = tmp_path / "opencode.json"
    config.write_text(json.dumps({
        "provider": {
            "groq": {"models": {"llama": {}, "mixtral": {}}},
            "cerebras": {"models": {}},
        }
    }))

    result = boot_check(config, ["groq", "cerebras"])

    assert result["ok"] is False
    assert result["provider_count"] == 2
    assert result["models"] == {"groq": 2}
    assert result["missing"] == ["cerebras"]


@pytest.mark.parametrize("content", ["[]", "{not json", json.dumps({"provider": []})])
def test_boot_check_rejects_malformed_configurations(tmp_path, content):
    config = tmp_path / "opencode.json"
    config.write_text(content)

    with pytest.raises(OperatorError):
        boot_check(config, [])


def test_boot_check_requires_a_readable_file(tmp_path):
    with pytest.raises(OperatorError):
        boot_check(tmp_path / "absent.json", [])


def route_file(tmp_path, rows):
    path = tmp_path / "routes.json"
    path.write_text(json.dumps({"routes": rows}))
    return path


ROUTE_ROW = {"route_id": "a", "provider_id": "p", "model_id": "m", "key_id": "k",
             "alias": "sc.a", "channel_id": 7, "quota_type": "unlimited",
             "input_price": 0, "output_price": 0, "context_window": 32768,
             "output_limit": 4096}


def test_route_file_preview_does_not_write_the_ledger(tmp_path):
    path = route_file(tmp_path, [ROUTE_ROW])

    routes = load_routes(path)
    result = preview_routes(routes)

    assert result["apply"] is False
    assert result["routes"] == [{"route_id": "a", "alias": "sc.a",
                                 "channel_id": 7, "model_id": "m",
                                 "free_class": "free_unlimited", "quota_type": "unlimited"}]
    assert not (tmp_path / "ledger.db").exists()


def test_route_file_apply_configures_every_route(tmp_path):
    path = route_file(tmp_path, [ROUTE_ROW, {**ROUTE_ROW, "route_id": "b", "channel_id": 8}])
    ledger = QuotaLedger(tmp_path / "ledger.db")
    try:
        result = apply_routes(ledger, load_routes(path))
        stored = [item.route_id for item in ledger.routes()]
    finally:
        ledger.close()

    assert result["configured"] == 2
    assert stored == ["a", "b"]


@pytest.mark.parametrize("rows", [
    [],
    [{**ROUTE_ROW, "channel_id": 0}],
    [{**ROUTE_ROW, "unknown": 1}],
    [ROUTE_ROW, ROUTE_ROW],
])
def test_bad_route_files_fail_closed(tmp_path, rows):
    with pytest.raises(OperatorError):
        load_routes(route_file(tmp_path, rows))


def test_route_file_must_be_json(tmp_path):
    path = tmp_path / "routes.json"
    path.write_text("{not json")

    with pytest.raises(OperatorError):
        load_routes(path)


def test_cli_rejects_an_unknown_route_without_touching_newapi(tmp_path, monkeypatch):
    database = tmp_path / "ledger.db"
    QuotaLedger(database).close()
    monkeypatch.setenv("SIDECAR_DB", str(database))

    result = subprocess.run(
        [sys.executable, str(TOOL), "priorities", "--order", "absent"],
        capture_output=True, text=True, timeout=30,
    )

    assert result.returncode == 1
    assert "operator command failed" in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("models, expected", [({}, 1), ({"fixture": {}}, 0)])
def test_cli_boot_check_exit_status_matches_report(tmp_path, models, expected):
    config = tmp_path / "opencode.json"
    config.write_text(json.dumps({"provider": {"local": {"models": models}}}))
    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "boot-check",
            "--opencode-config",
            str(config),
            "--providers",
            "local",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == expected
    assert json.loads(result.stdout)["ok"] is (expected == 0)
    assert result.stderr == ""


def test_cli_help_succeeds():
    result = subprocess.run([sys.executable, str(TOOL), "--help"],
                            capture_output=True, text=True, timeout=30)

    assert result.returncode == 0
    assert "audit" in result.stdout and "classify" in result.stdout
    assert "boot-check" in result.stdout
