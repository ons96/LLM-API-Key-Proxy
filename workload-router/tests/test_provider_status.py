from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from http.client import HTTPConnection
import json
import sqlite3
import threading

from newapi_runtime import Runtime
from newapi_server import Server
from provider_status import ProviderStatusStore
from quota_ledger import QuotaLedger
from router_state import RouterState


def status_database(tmp_path, *, old_hours=2):
    path = tmp_path / "provider_status.db"
    now = datetime.now(timezone.utc)
    recent = (now - timedelta(seconds=30)).isoformat()
    old = (now - timedelta(hours=old_hours)).isoformat()
    with sqlite3.connect(path) as connection:
        connection.execute("""CREATE TABLE provider_health_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT, provider_name TEXT,
                status TEXT, response_time_ms REAL, uptime_percent REAL,
                rate_limit_percent REAL, last_check_timestamp TEXT,
                error_message TEXT, consecutive_failures INTEGER,
                created_at TEXT)""")
        connection.executemany(
            "INSERT INTO provider_health_checks(provider_name,status,response_time_ms,uptime_percent,rate_limit_percent,last_check_timestamp,error_message,consecutive_failures) VALUES (?,?,?,?,?,?,?,?)",
            [
                ("alpha", "healthy", 100.0, 99.0, 1.0, recent, "", 0),
                ("beta", "degraded", 900.0, 80.0, 2.0, old, "slow", 1),
                ("gamma", "down", 0.0, 0.0, 0.0, old, "failed", 3),
            ],
        )
    return path


def test_status_snapshot_history_best_health_and_csv(tmp_path):
    store = ProviderStatusStore(status_database(tmp_path), stale_after_seconds=60)
    current = store.current_status()
    assert current["read_only"] is True and current["source"] == "provider_status.db"
    assert current["providers"]["alpha"]["stale"] is False
    assert current["providers"]["beta"]["stale"] is True
    assert current["providers"]["alpha"]["provider_name"] == "alpha"
    assert store.single("alpha")["status"] == "healthy"
    assert store.single("missing") is None
    assert store.best()["best_provider"] == "alpha"
    assert len(store.history("beta", 24)) == 1
    assert store.all_history(24)["alpha"]
    health = store.health()
    assert health["status"] == "healthy" and health["read_only"] is True
    csv_text = store.export_csv()
    assert csv_text.startswith("provider_name,status,")
    assert "alpha,healthy," in csv_text


def test_stale_snapshot_is_marked_degraded(tmp_path):
    path = status_database(tmp_path, old_hours=2)
    old = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE provider_health_checks SET last_check_timestamp=? WHERE provider_name='alpha'",
            (old,),
        )
    store = ProviderStatusStore(path, stale_after_seconds=60)
    assert store.health()["status"] == "degraded"
    assert store.best()["stale"] is True
    assert all(item["stale"] for item in store.current_status()["providers"].values())


@contextmanager
def running(server):
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}
    )
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def request(server, path, auth=True):
    connection = HTTPConnection(*server.server_address, timeout=3)
    headers = {"Authorization": "Bearer test-client"} if auth else {}
    try:
        connection.request("GET", path, headers=headers)
        response = connection.getresponse()
        return response.status, response.read().decode(), dict(response.getheaders())
    finally:
        connection.close()


def test_http_provider_status_routes_are_read_only(tmp_path):
    ledger, state = QuotaLedger(), RouterState()
    runtime = Runtime(ledger, state, "http://127.0.0.1:8002/v1", "test-relay")
    store = ProviderStatusStore(status_database(tmp_path), stale_after_seconds=60)
    try:
        with running(
            Server(("127.0.0.1", 0), runtime, "test-client", provider_status=store)
        ) as server:
            assert request(server, "/api/providers/status", auth=False)[0] == 401
            status, body, _ = request(server, "/api/providers/status")
            assert (
                status == 200
                and json.loads(body)["providers"]["alpha"]["status"] == "healthy"
            )
            assert request(server, "/api/providers/status/alpha")[0] == 200
            assert request(server, "/api/providers/status/missing")[0] == 404
            assert request(server, "/api/providers/best")[0] == 200
            assert request(server, "/api/providers/history?hours=24")[0] == 200
            assert request(server, "/api/providers/history/alpha?hours=24")[0] == 200
            assert request(server, "/api/providers/history/alpha?hours=0")[0] == 400
            status, body, headers = request(server, "/api/providers/export/csv")
            assert status == 200 and headers["Content-Type"].startswith("text/csv")
            assert body.startswith("provider_name,status,")
            assert request(server, "/api/providers/health")[0] == 200
    finally:
        state.close()
        ledger.close()


def test_missing_status_database_is_explicit(tmp_path):
    ledger, state = QuotaLedger(), RouterState()
    runtime = Runtime(ledger, state, "http://127.0.0.1:8002/v1", "test-relay")
    try:
        with running(Server(("127.0.0.1", 0), runtime, "test-client")) as server:
            status, body, _ = request(server, "/api/providers/status")
            assert status == 503 and "not configured" in body
    finally:
        state.close()
        ledger.close()
