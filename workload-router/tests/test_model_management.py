from contextlib import contextmanager
from http.client import HTTPConnection
import json
import threading

from model_management import ModelSessionStore, detect_intent
from newapi_runtime import MODEL_ALIASES
from newapi_server import Server
from newapi_runtime import Runtime
from quota_ledger import QuotaLedger
from router_state import RouterState


def test_session_store_is_bounded_and_expires():
    store = ModelSessionStore(MODEL_ALIASES, max_sessions=2, ttl_seconds=10)
    assert store.get("a", now=0) == "auto-chat"
    assert store.set("a", "coding-elite", now=0) is None
    assert store.set("a", "chat-fast", now=1) == "coding-elite"
    assert store.set("b", "chat-rp", now=2) is None
    assert store.set("c", "chat-smart", now=3) is None
    assert store.count() == 2
    assert store.get("c", now=3) == "chat-smart"
    assert store.get("a", now=20) == "auto-chat"


def test_intent_heuristic_stays_within_sidecar_aliases():
    assert (
        detect_intent([{"role": "user", "content": "hi"}]).suggested_model
        == "chat-fast"
    )
    coding = detect_intent(
        [{"role": "user", "content": "implement a FastAPI architecture endpoint"}]
    )
    assert coding.suggested_model == "coding-elite"
    rp = detect_intent([{"role": "user", "content": "continue roleplay *smiles*"}])
    assert rp.suggested_model == "chat-rp"
    reasoning = detect_intent(
        [{"role": "user", "content": "analyze the tradeoffs in detail"}]
    )
    assert reasoning.suggested_model == "chat-smart"
    assert (
        detect_intent(
            [{"role": "user", "content": "a plain statement"}],
            current_model="auto-chat",
        ).suggested_model
        == "auto-chat"
    )
    assert (
        detect_intent(
            [{"role": "system", "content": "you are in character"}]
        ).suggested_model
        == "chat-rp"
    )


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


def request(server, path, *, method="GET", payload=None, auth=True):
    body = None if payload is None else json.dumps(payload)
    headers = {"Authorization": "Bearer test-client"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if not auth:
        headers = {}
    connection = HTTPConnection(*server.server_address, timeout=3)
    try:
        connection.request(method, path, body, headers)
        response = connection.getresponse()
        return response.status, response.read().decode()
    finally:
        connection.close()


def test_model_session_switch_and_intent_http_routes(tmp_path):
    ledger, state = QuotaLedger(), RouterState()
    runtime = Runtime(ledger, state, "http://127.0.0.1:8002/v1", "test-relay")
    try:
        with running(Server(("127.0.0.1", 0), runtime, "test-client")) as server:
            status, body = request(server, "/v1/model/session/demo")
            assert status == 200 and json.loads(body)["current_model"] == "auto-chat"
            status, body = request(
                server,
                "/v1/model/switch",
                method="POST",
                payload={"model": "coding-elite", "session_id": "demo"},
            )
            switched = json.loads(body)
            assert status == 200 and switched["previous_model"] == "auto-chat"
            assert switched["current_model"] == "coding-elite"
            status, body = request(server, "/v1/model/session/demo")
            assert json.loads(body)["current_model"] == "coding-elite"
            assert (
                request(
                    server,
                    "/v1/model/switch",
                    method="POST",
                    payload={"model": "not-real"},
                )[0]
                == 400
            )
            assert (
                request(
                    server,
                    "/v1/model/switch",
                    method="POST",
                    payload={"model": "auto", "session_id": "x" * 513},
                )[0]
                == 400
            )
            status, body = request(
                server,
                "/v1/model/detect-intent",
                method="POST",
                payload={
                    "messages": [
                        {
                            "role": "user",
                            "content": "write a system architecture unit test",
                        }
                    ]
                },
            )
            detected = json.loads(body)
            assert status == 200 and detected["suggested_model"] == "coding-elite"
            assert request(server, "/v1/model/session/" + "x" * 513)[0] == 400
            status, body = request(server, "/v1/model-info/stats")
            stats = json.loads(body)
            assert status == 200 and stats["ready"] is False
            assert stats["configured_route_count"] == 0
            assert stats["alias_count"] == 22
            assert request(server, "/v1/model/session/demo", auth=False)[0] == 401
    finally:
        state.close()
        ledger.close()
