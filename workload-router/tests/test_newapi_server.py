"""Local HTTP tests; no real credentials, provider requests or gateway instance."""

from contextlib import contextmanager
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler
import json
import socket
import threading
import time

import pytest

from newapi_runtime import Runtime
from newapi_server import Server, channel_key_proof_from_command, strict_json
from provider_adapter import ProviderCompletion, ProviderError
from quota_ledger import QuotaLedger, Route
from router_state import RouterState


@contextmanager
def running(server):
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture
def sidecar(monkeypatch):
    ledger, state = QuotaLedger(), RouterState()
    runtime = Runtime(ledger, state, "http://127.0.0.1:8002/v1", "test-relay", explore=False)
    route = Route("local", "fixture", "local-model", "opaque", "sc.local", 1,
                  quota_type="unlimited", context_window=262144, output_limit=32768,
                  input_price=0, output_price=0, stream=True, tool_call=True)
    ledger.configure(route)
    ledger.configure(Route("local-embed", "fixture", "embed-model", "opaque", "sc.embed", 2,
                           quota_type="unlimited", input_price=0, output_price=0))
    calls = []
    scripts = []
    class Adapter:
        def complete(self, deployment, data):
            calls.append(data.to_payload(deployment.model, False))
            if "tool" in scripts:
                return ProviderCompletion(
                    {"role": "assistant", "content": None,
                     "tool_calls": [{"id": "call_1", "type": "function",
                                     "function": {"name": "lookup", "arguments": "{\"q\": \"x\"}"}}]},
                    "tool_calls", {"prompt_tokens": 10, "completion_tokens": 2})
            return ProviderCompletion({"role": "assistant", "content": "ok"}, "stop",
                                      {"prompt_tokens": 10, "completion_tokens": 2})

        def stream(self, deployment, data):
            calls.append(data.to_payload(deployment.model, True))
            yield json.dumps({"choices": [{"index": 0, "delta": {"content": "ok"}}]})
            if scripts:
                raise ProviderError("private", error_class="timeout")
            yield json.dumps({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 2}})
            yield "[DONE]"
    monkeypatch.setattr(runtime, "adapter", lambda _remaining: Adapter())
    try:
        with running(Server(("127.0.0.1", 0), runtime, "test-client",
                            embedding_aliases=frozenset({"sc.embed"}))) as server:
            yield server, runtime, calls, scripts
    finally:
        state.close()
        ledger.close()


def call(server, path="/v1/chat/completions", *, method="POST", auth=True, raw=None, headers=None, **options):
    payload = {"model": "auto-chat", "messages": [{"role": "user", "content": "hello"}], **options}
    body = json.dumps(payload) if raw is None else raw
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    if auth:
        hdrs["Authorization"] = "Bearer test-client"
    connection = HTTPConnection(*server.server_address, timeout=3)
    try:
        connection.request(method, path, body if method == "POST" else None, hdrs)
        response = connection.getresponse()
        return response.status, response.read().decode(), dict(response.getheaders())
    finally:
        connection.close()


@pytest.mark.parametrize("raw", [b"[]", b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'])
def test_strict_json_rejects_ambiguous_objects(raw):
    with pytest.raises(ValueError):
        strict_json(raw)


def test_channel_key_proof_helper_uses_stdin_and_validates_output(monkeypatch):
    seen = {}

    class Result:
        stdout = "proof-token\n"

    def run(argv, **kwargs):
        seen["argv"] = argv
        seen.update(kwargs)
        return Result()

    monkeypatch.setattr("newapi_server.subprocess.run", run)
    proof = channel_key_proof_from_command("/usr/local/bin/proof-helper --quiet")
    assert proof(42) == "proof-token"
    assert seen["argv"] == ["/usr/local/bin/proof-helper", "--quiet"]
    assert seen["input"] == "42\n"
    assert seen["capture_output"] is True and seen["check"] is True


@pytest.mark.parametrize("stdout", ["", "two\nlines\n", "bad\x00token"])
def test_channel_key_proof_helper_rejects_invalid_output(monkeypatch, stdout):
    class Result:
        pass

    Result.stdout = stdout
    monkeypatch.setattr("newapi_server.subprocess.run", lambda *args, **kwargs: Result())
    with pytest.raises((RuntimeError, ValueError), match="invalid"):
        channel_key_proof_from_command("/usr/local/bin/proof-helper")(42)


def test_auth_models_health_and_diagnostics(sidecar):
    server, runtime, calls, _ = sidecar
    status, body, _ = call(server, "/", method="GET", auth=False)
    assert status == 200 and json.loads(body)["Status"] == "New-API sidecar is running"
    assert call(server, "/healthz", method="GET", auth=False)[0] == 200
    assert call(server, "/v1/models", method="GET", auth=False)[0] == 401
    status, body, headers = call(server, "/v1/models", method="GET")
    models = json.loads(body)["data"]
    ids = {item["id"] for item in models}
    deprecated = {item["id"] for item in models if item.get("deprecated")}
    assert status == 200 and {"auto-chat", "auto-code", "auto-coding"} <= ids
    assert {"auto", "coding-elite", "agent-oracle"} <= ids
    assert len(deprecated) == 19 and len(ids) == 22
    assert headers["Cache-Control"] == "no-store"
    assert call(server, "/v1/router/diagnostics", method="GET")[0] == 200
    assert not calls and not runtime.ledger.diagnostics()["attempts"]


def test_route_preview_does_not_reserve_or_dispatch(sidecar):
    server, runtime, calls, _ = sidecar
    status, body, _ = call(server, "/v1/router/route")
    assert status == 200 and json.loads(body)["candidates"][0]["route_id"] == "local"
    assert not calls and not runtime.ledger.diagnostics()["attempts"]


def test_tps_stats_are_ledger_backed_and_bounded(sidecar):
    server, runtime, calls, _ = sidecar
    route = next(row for row in runtime.ledger.routes() if row.route_id == "local")
    for _ in range(2):
        identity = runtime.ledger.reserve(
            route, 10, 100, profile="auto-chat", step="chat", streaming=True, now=time.time())
        runtime.ledger.start(identity, now=time.time())
        runtime.ledger.finish(
            identity, route, success=True,
            usage={"prompt_tokens": 10, "completion_tokens": 100},
            latency_ms=1100, ttft_ms=100, now=time.time())
    status, body, _ = call(server, "/v1/tps-stats?days=1&min_samples=1", method="GET")
    result = json.loads(body)
    assert status == 200 and result["count"] == 1
    entry = result["data"][0]
    assert entry["provider_name"] == "fixture" and entry["model"] == "local-model"
    assert entry["sample_count"] == 2 and entry["median_tps"] == 100.0
    assert entry["avg_ttft_ms"] == 100.0
    assert call(server, "/v1/tps-stats?days=0", method="GET")[0] == 400
    assert call(server, "/v1/tps-stats?export=true", method="GET")[0] == 400
    assert not calls


def test_cost_estimate_is_ledger_backed_and_read_only(sidecar):
    server, runtime, calls, _ = sidecar
    runtime.ledger.configure(Route(
        "paid", "fixture", "paid-model", "opaque", "sc.paid", 3,
        quota_type="paid", input_price=1, output_price=2))
    status, body, _ = call(server, "/v1/cost-estimate", raw=json.dumps({
        "model": "sc.paid", "prompt_tokens": 1_000_000, "completion_tokens": 500_000}))
    result = json.loads(body)
    assert status == 200 and result["model"] == "sc.paid" and result["cost"] == 2.0
    assert result["pricing"] == {"input_cost_per_token": 0.000001,
                                "output_cost_per_token": 0.000002}
    assert result["source"] == "sidecar_ledger"
    assert result["free_class"] == "paid" and result["free"] is False
    status, body, _ = call(server, "/v1/cost-estimate", raw=json.dumps({
        "model": "sc.local", "prompt_tokens": 1_000_000, "completion_tokens": 500_000}))
    free = json.loads(body)
    assert status == 200 and free["cost"] == 0.0
    assert free["free_class"] == "free_unlimited" and free["free"] is True
    assert free["source"] == "free_tier"
    status, body, _ = call(server, "/v1/cost-estimate", raw=json.dumps({
        "model": "missing", "prompt_tokens": 1, "completion_tokens": 1}))
    result = json.loads(body)
    assert status == 200 and result["cost"] is None
    assert result["free_class"] == "unknown" and result["free"] is False
    assert call(server, "/v1/cost-estimate", raw=json.dumps({
        "model": "sc.paid", "prompt_tokens": -1}))[0] == 400
    assert not calls and runtime.ledger.diagnostics()["attempts"] == {}
    assert runtime.ledger.diagnostics()["free_classes"] == {
        "free_unlimited": 2, "paid": 1}


def test_token_count_is_a_labeled_upper_bound_and_never_dispatches(sidecar):
    server, runtime, calls, _ = sidecar
    status, body, _ = call(server, "/v1/token-count", raw=json.dumps({
        "model": "auto-chat",
        "messages": [{"role": "user", "content": "hello tokenizer"}],
        "tools": [{"type": "function", "function": {"name": "x", "parameters": {"type": "object"}}}],
    }))
    result = json.loads(body)
    assert status == 200 and result["token_count"] > 0
    assert result["estimated"] is True
    assert result["method"] == "utf8_byte_upper_bound"
    assert call(server, "/v1/token-count", raw=json.dumps({
        "model": "auto-chat", "messages": []}))[0] == 400
    assert call(server, "/v1/token-count", raw=json.dumps({
        "model": "auto-chat", "messages": [{"role": "user", "content": [{"type": "image_url"}]}]}))[0] == 400
    assert not calls and runtime.ledger.diagnostics()["attempts"] == {}


def test_model_management_list_status_and_command_parse(sidecar):
    server, runtime, calls, _ = sidecar
    status, body, _ = call(server, "/v1/model/list", method="GET")
    listing = json.loads(body)
    assert status == 200 and len(listing["models"]) == 22
    assert "auto" in listing["descriptions"]

    status, body, _ = call(server, "/v1/model/status", method="GET")
    model_status = json.loads(body)
    assert status == 200 and model_status["current_model"] == "auto-chat"
    assert len(model_status["available_models"]) == 22
    assert model_status["auto_routing_enabled"] is True

    status, body, _ = call(server, "/v1/model/parse-command",
                           raw=json.dumps({"message": "/model coding-elite please"}))
    parsed = json.loads(body)
    assert status == 200 and parsed == {
        "command_found": True, "requested_model": "coding-elite",
        "canonical_model": "coding-elite", "remaining_message": "please", "valid": True}
    status, body, _ = call(server, "/v1/model/parse-command",
                           raw=json.dumps({"message": "/model not-real"}))
    assert status == 200 and json.loads(body)["valid"] is False
    status, body, _ = call(server, "/v1/model/parse-command",
                           raw=json.dumps({"message": "/coding"}))
    assert status == 200 and json.loads(body)["canonical_model"] == "coding-elite"
    assert call(server, "/v1/model/parse-command", raw=json.dumps({"message": "hello"}))[0] == 200
    assert call(server, "/v1/model/parse-command", auth=False,
                raw=json.dumps({"message": "/coding"}))[0] == 401
    assert not calls and runtime.ledger.diagnostics()["attempts"] == {}


def test_model_detail_and_provider_list_are_read_only(sidecar):
    server, _, calls, _ = sidecar
    status, body, _ = call(server, "/v1/models/auto", method="GET")
    model = json.loads(body)
    assert status == 200 and model["id"] == "auto"
    assert model["object"] == "model" and model["owned_by"] == "sidecar"
    assert model["deprecated"] is True and isinstance(model["created"], int)

    status, body, _ = call(server, "/v1/models/openrouter/some-model", method="GET")
    unknown = json.loads(body)
    assert status == 200 and unknown["owned_by"] == "openrouter"
    assert "deprecated" not in unknown

    status, body, _ = call(server, "/v1/providers", method="GET")
    assert status == 200 and json.loads(body) == ["fixture"]
    assert call(server, "/v1/models/auto", method="GET", auth=False)[0] == 401
    assert not calls


@pytest.mark.parametrize("raw", ["[]", '{"model":"auto-chat","model":"auto-code"}',
                                  '{"model":"auto-chat","messages":[{"content":"x"}],"temperature":NaN}'])
def test_invalid_http_json_never_dispatches(sidecar, raw):
    server, _, calls, _ = sidecar
    assert call(server, raw=raw)[0] == 400 and not calls


def test_large_body_rejected_before_dispatch(sidecar):
    server, _, calls, _ = sidecar
    # Do not race a megabyte upload against the server's early close/rejection.
    connection = HTTPConnection(*server.server_address, timeout=3)
    connection.putrequest("POST", "/v1/chat/completions")
    connection.putheader("Authorization", "Bearer test-client")
    connection.putheader("Content-Length", str(1024 * 1024 + 1))
    connection.endheaders()
    response = connection.getresponse()
    assert response.status == 413
    response.read()
    connection.close()
    assert not calls


@pytest.mark.parametrize("model", ["auto-chat", "auto-code", "auto-coding"])
def test_completion_preserves_alias_and_settles(sidecar, model):
    server, runtime, calls, _ = sidecar
    status, body, headers = call(server, model=model, headers={"X-Session-Id": "http-session"})
    result = json.loads(body)
    assert status == 200 and result["model"] == model
    assert result["usage"]["completion_tokens"] == 2
    assert calls[0]["model"] == "sc.local" and calls[0]["max_tokens"] == 2048
    assert runtime.ledger.decision(headers["X-Router-Decision"])["status"] == "success"


def test_native_stream_complete_and_late_failure(sidecar):
    server, runtime, calls, scripts = sidecar
    status, body, headers = call(server, stream=True)
    assert status == 200 and body.endswith("data: [DONE]\n\n")
    assert '"model":"auto-chat"' in body
    assert runtime.ledger.decision(headers["X-Router-Decision"])["status"] == "success"
    scripts.append("late failure")
    status, body, headers = call(server, stream=True)
    assert status == 200 and "ok" in body and "[DONE]" not in body and "private" not in body
    assert runtime.ledger.decision(headers["X-Router-Decision"])["status"] == "failure"
    assert len(calls) == 2


def test_wrong_method_returns_json_405(sidecar):
    server, _, calls, _ = sidecar
    status, body, headers = call(server, method="PUT")
    assert status == 405 and json.loads(body)["error"] == "method not allowed"
    assert headers["Allow"] == "GET, POST" and not calls


def responses_call(server, payload):
    return call(server, "/v1/responses", raw=json.dumps(payload))


def test_responses_non_streaming_translates_and_echoes(sidecar):
    server, runtime, calls, _ = sidecar
    status, body, headers = responses_call(server, {
        "model": "auto-chat", "input": "hello", "instructions": "be terse",
        "max_output_tokens": 16})
    response = json.loads(body)
    assert status == 200 and response["object"] == "response"
    assert response["output_text"] == "ok" and response["model"] == "auto-chat"
    assert response["usage"]["input_tokens"] == 10
    assert response["usage"]["output_tokens"] == 2
    assert calls[-1]["messages"] == [{"role": "system", "content": "be terse"},
                                     {"role": "user", "content": "hello"}]
    assert calls[-1]["max_tokens"] == 16
    assert runtime.ledger.decision(headers["X-Router-Decision"])["status"] == "success"


def test_responses_translates_function_tools(sidecar):
    server, _, calls, scripts = sidecar
    scripts.append("tool")
    status, body, _ = responses_call(server, {
        "model": "auto-chat", "input": "find x",
        "tools": [{"type": "function", "name": "lookup",
                   "description": "search", "parameters": {"type": "object"}}],
        "tool_choice": "required"})
    response = json.loads(body)
    assert status == 200
    assert calls[-1]["tools"] == [{"type": "function", "function": {
        "name": "lookup", "description": "search", "parameters": {"type": "object"}}}]
    assert calls[-1]["tool_choice"] == "required"
    assert response["output"][0]["type"] == "function_call"
    assert response["output"][0]["name"] == "lookup"
    assert response["output_text"] == ""


def test_responses_rejects_unsupported_shapes(sidecar):
    server, _, calls, _ = sidecar
    assert responses_call(server, {"model": "auto-chat", "input": "hi",
                                   "previous_response_id": "resp_1"})[0] == 400
    assert responses_call(server, {"model": "auto-chat", "input": "hi", "stream": True,
                                   "tools": [{"type": "function", "name": "lookup"}]})[0] == 400
    assert responses_call(server, {"model": "auto-chat", "input": [
        {"type": "function_call_output", "call_id": "c1", "output": "x"}]})[0] == 400
    assert not calls


def test_responses_streaming_text_events(sidecar):
    server, runtime, _, _ = sidecar
    status, body, headers = responses_call(server, {"model": "auto-chat", "input": "hi",
                                                   "stream": True})
    assert status == 200
    events = [json.loads(line[5:]) for line in body.splitlines() if line.startswith("data:")]
    types = [event["type"] for event in events]
    assert types[0] == "response.created" and types[-1] == "response.completed"
    assert "response.output_text.delta" in types
    assert "".join(event["delta"] for event in events
                   if event["type"] == "response.output_text.delta") == "ok"
    assert events[-1]["response"]["usage"]["output_tokens"] == 2
    assert runtime.ledger.decision(headers["X-Router-Decision"])["status"] == "success"


def test_embeddings_forward_and_account(sidecar, monkeypatch):
    server, runtime, calls, _ = sidecar
    seen = {}

    def post(base_url, api_key, payload, timeout):
        seen.update(base_url=base_url, api_key=api_key, payload=payload)
        return {"object": "list", "data": [{"object": "embedding", "index": 0,
                                             "embedding": [0.5, 0.25]}],
                "model": "embed-model"}

    monkeypatch.setattr("newapi_runtime._embedding_post", post)
    status, body, headers = call(server, "/v1/embeddings",
                                 raw=json.dumps({"model": "sc.embed", "input": ["a", "b"]}))
    result = json.loads(body)
    assert status == 200 and result["data"][0]["embedding"] == [0.5, 0.25]
    assert result["model"] == "sc.embed"
    assert seen["payload"] == {"model": "sc.embed", "input": ["a", "b"]}
    assert seen["base_url"] == "http://127.0.0.1:8002/v1"
    decision = runtime.ledger.decision(headers["X-Router-Decision"])
    assert decision["status"] == "success" and decision["output_tokens"] == 0
    assert not calls


def test_embeddings_reject_unknown_alias_and_bad_input(sidecar, monkeypatch):
    server, runtime, _, _ = sidecar
    monkeypatch.setattr("newapi_runtime._embedding_post",
                        lambda *args: pytest.fail("must not dispatch"))
    assert call(server, "/v1/embeddings",
                raw=json.dumps({"model": "auto-chat", "input": "a"}))[0] == 400
    assert call(server, "/v1/embeddings",
                raw=json.dumps({"model": "sc.embed", "input": []}))[0] == 400
    assert call(server, "/v1/embeddings",
                raw=json.dumps({"model": "sc.embed", "input": [1, 2]}))[0] == 400
    assert not runtime.ledger.diagnostics()["attempts"]


def test_request_threads_bounded_before_handler_creation():
    entered, release = threading.Event(), threading.Event()
    class Holding(BaseHTTPRequestHandler):
        def do_GET(self):
            entered.set()
            release.wait(2)
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
        def log_message(self, *_args):
            pass
    ledger, state = QuotaLedger(), RouterState()
    runtime = Runtime(ledger, state, "http://127.0.0.1:8002/v1", "test-relay")
    server = Server(("127.0.0.1", 0), runtime, "test-client", workers=1)
    server.RequestHandlerClass = Holding
    try:
        with running(server):
            first = HTTPConnection(*server.server_address, timeout=3)
            first.request("GET", "/hold")
            assert entered.wait(1)
            with socket.create_connection(server.server_address, timeout=2) as second:
                second.sendall(b"GET /healthz HTTP/1.1\r\nHost: localhost\r\n\r\n")
                assert b"503 Service Unavailable" in second.recv(4096)
            release.set()
            assert first.getresponse().status == 200
            first.close()
    finally:
        release.set()
        state.close()
        ledger.close()
