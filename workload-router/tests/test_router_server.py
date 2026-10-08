import json
import threading
from dataclasses import replace
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

from router_core import HealthRegistry
from router_server import RouterHandler
from provider_adapter import ProviderCompletion, ProviderError


def request(server, path, payload, headers=None):
    connection = HTTPConnection(*server.server_address)
    body = json.dumps(payload)
    connection.request("POST", path, body, {"Content-Type": "application/json", **(headers or {})})
    response = connection.getresponse()
    result = json.loads(response.read())
    connection.close()
    return response.status, result, dict(response.getheaders())


def get(server, path):
    connection = HTTPConnection(*server.server_address)
    connection.request("GET", path)
    response = connection.getresponse()
    result = json.loads(response.read())
    connection.close()
    return response.status, result


def test_http_route_and_cooldown_behavior():
    server = ThreadingHTTPServer(("127.0.0.1", 0), RouterHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, result, response_headers = request(
            server,
            "/v1/router/route",
            {"messages": [{"role": "user", "content": "answer"}]},
            {"X-Session-Id": "http-test"},
        )
        assert status == 200
        assert result["deployment"] == "general"
        assert response_headers["X-Router-Deployment"] == "general"
        assert response_headers["Cache-Control"] == "no-store"

        for _ in range(3):
            status, result, _ = request(
                server,
                "/v1/router/route",
                {"messages": [{"role": "user", "content": "answer"}]},
                {"X-Session-Id": "stall-test", "X-Router-Action": "pytest", "X-Router-Verification": "failed"},
            )
        assert status == 200
        assert result["escalation"] == 2

        status, result, _ = request(
            server,
            "/v1/router/outcome",
            {"deployment": "general", "outcome": "operational_failure", "error_class": "timeout"},
            {"X-Session-Id": "http-test"},
        )
        assert status == 200
        assert result["recorded"] == "operational_failure"

        status, result, _ = request(
            server,
            "/v1/router/route",
            {"messages": [{"role": "user", "content": "answer"}]},
            {"X-Session-Id": "http-test-2"},
        )
        assert status == 503
        assert "no eligible deployment" in result["error"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_models_endpoint_lists_stable_aliases():
    server = ThreadingHTTPServer(("127.0.0.1", 0), RouterHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, result = get(server, "/v1/models")
        assert status == 200
        assert result["object"] == "list"
        assert [item["id"] for item in result["data"]] == ["auto", "auto-chat", "auto-code"]
        assert all(item["owned_by"] == "ai-workload-router" for item in result["data"])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_rejects_empty_messages():
    server = ThreadingHTTPServer(("127.0.0.1", 0), RouterHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, result, _ = request(server, "/v1/router/route", {"messages": []})
        assert status == 400
        assert "non-empty list" in result["error"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_rejects_non_object_json_body():
    server = ThreadingHTTPServer(("127.0.0.1", 0), RouterHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for path in ("/v1/router/route", "/v1/chat/completions", "/v1/router/outcome"):
            status, result, _ = request(server, path, ["not", "an", "object"])
            assert status == 400
            assert "JSON object" in result["error"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_health_and_diagnostics_are_prompt_free():
    server = ThreadingHTTPServer(("127.0.0.1", 0), RouterHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, result = get(server, "/healthz")
        assert status == 200
        assert result == {"status": "ok", "service": "ai-workload-router", "version": "0.1.0"}
        status, result = get(server, "/v1/router/diagnostics")
        assert status == 200
        assert result["status"] == "ok"
        assert result["deployments"] == 5
        assert "successful_routes" in result
        assert "cache_hits" in result
        assert "prompt" not in json.dumps(result).lower()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class MockAdapter:
    def complete(self, deployment, prompt):
        return "mock response"


class ForwardingAdapter:
    def __init__(self):
        self.request = None

    def complete(self, deployment, provider_request):
        self.request = provider_request
        return ProviderCompletion(
            message={
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call-1", "type": "function"}],
            },
            finish_reason="tool_calls",
            usage={"prompt_tokens": 12, "completion_tokens": 3},
        )


def test_chat_completions_forwards_structured_provider_request():
    previous = RouterHandler.adapters
    previous_health = RouterHandler.health
    capture = ForwardingAdapter()
    previous_deployments = RouterHandler.deployments
    RouterHandler.deployments = [replace(deployment, structured_output=True) for deployment in previous_deployments]
    RouterHandler.adapters = {
        deployment.deployment_id: capture for deployment in RouterHandler.deployments
    }
    RouterHandler.health = HealthRegistry()
    server = ThreadingHTTPServer(("127.0.0.1", 0), RouterHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, result, _ = request(
            server,
            "/v1/chat/completions",
            {
                "model": "auto",
                "messages": [
                    {"role": "system", "content": "Use tools"},
                    {"role": "user", "content": "look this up"},
                ],
                "tools": [{"type": "function", "function": {"name": "lookup"}}],
                "tool_choice": "auto",
                "response_format": {"type": "json_object"},
                "temperature": 0,
                "context_tokens": 42,
            },
        )
        assert status == 200
        assert capture.request.messages[0]["role"] == "system"
        assert capture.request.messages[1]["content"] == "look this up"
        assert capture.request.options["tools"][0]["function"]["name"] == "lookup"
        assert capture.request.options["response_format"] == {"type": "json_object"}
        assert capture.request.options["temperature"] == 0
        assert "context_tokens" not in capture.request.options
        assert result["choices"][0]["finish_reason"] == "tool_calls"
        assert result["choices"][0]["message"]["tool_calls"][0]["id"] == "call-1"
        assert result["usage"]["prompt_tokens"] == 12
    finally:
        RouterHandler.adapters = previous
        RouterHandler.deployments = previous_deployments
        RouterHandler.health = previous_health
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class NativeStreamAdapter(MockAdapter):
    def stream(self, deployment, prompt):
        yield json.dumps({"id": "upstream", "object": "chat.completion.chunk",
                          "choices": [{"index": 0, "delta": {"content": "native"},
                                        "finish_reason": None}]})
        yield "[DONE]"


def test_chat_completions_returns_openai_shape():
    previous = RouterHandler.adapters
    previous_health = RouterHandler.health
    RouterHandler.adapters = {deployment.deployment_id: MockAdapter() for deployment in RouterHandler.deployments}
    RouterHandler.health = HealthRegistry()
    server = ThreadingHTTPServer(("127.0.0.1", 0), RouterHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, result, headers = request(
            server,
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hello"}]},
        )
        assert status == 200
        assert result["object"] == "chat.completion"
        assert result["model"] == "auto"
        assert result["choices"][0]["message"]["content"] == "mock response"
        assert headers["X-Router-Deployment"] == "general"
        status, result, _ = request(
            server,
            "/v1/chat/completions",
            {"model": "auto-code", "messages": [{"role": "user", "content": "make a change"}]},
        )
        assert status == 200
        assert result["router"]["deployment"] == "implement"
    finally:
        RouterHandler.adapters = previous
        RouterHandler.health = previous_health
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_chat_provider_failure_records_bounded_operational_telemetry():
    previous = RouterHandler.adapters
    previous_health = RouterHandler.health

    class FailingAdapter:
        def complete(self, deployment, provider_request):
            raise ProviderError("upstream details must not be persisted", error_class="upstream_failure", status_code=503)

    RouterHandler.adapters = {"general": FailingAdapter()}
    RouterHandler.health = HealthRegistry()
    server = ThreadingHTTPServer(("127.0.0.1", 0), RouterHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, result, _ = request(
            server,
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hello"}]},
        )
        assert status == 503
        assert "upstream details" not in result["error"]
        before = get(server, "/v1/router/diagnostics")[1]["operational_failures"]
        status, result, _ = request(
            server,
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hello"}]},
        )
        assert status == 503
        after = get(server, "/v1/router/diagnostics")[1]["operational_failures"]
        assert after == before
    finally:
        RouterHandler.adapters = previous
        RouterHandler.health = previous_health
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_chat_completions_rejects_unknown_model_alias():
    previous = RouterHandler.adapters
    RouterHandler.adapters = {deployment.deployment_id: MockAdapter() for deployment in RouterHandler.deployments}
    server = ThreadingHTTPServer(("127.0.0.1", 0), RouterHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, result, _ = request(server, "/v1/chat/completions", {"model": "unknown", "messages": [{"content": "hello"}]})
        assert status == 400
        assert "model" in result["error"]
    finally:
        RouterHandler.adapters = previous
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_chat_completions_returns_sse_stream():
    expected = "def example():\n    return  42\n\n" + "x" * 600

    class CodeAdapter:
        def complete(self, deployment, prompt):
            return expected

    previous = RouterHandler.adapters
    previous_health = RouterHandler.health
    RouterHandler.adapters = {deployment.deployment_id: CodeAdapter() for deployment in RouterHandler.deployments}
    RouterHandler.health = HealthRegistry()
    server = ThreadingHTTPServer(("127.0.0.1", 0), RouterHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection(*server.server_address)
        connection.request("POST", "/v1/chat/completions", json.dumps({"messages": [{"content": "hello"}], "stream": True}), {"Content-Type": "application/json"})
        response = connection.getresponse()
        body = response.read().decode()
        connection.close()
        assert response.status == 200
        assert response.getheader("Content-Type") == "text/event-stream"
        assert "chat.completion.chunk" in body
        assert "[DONE]" in body
        events = [json.loads(line[6:]) for line in body.splitlines()
                  if line.startswith("data: ") and line != "data: [DONE]"]
        reconstructed = "".join(event["choices"][0]["delta"].get("content", "") for event in events)
        assert reconstructed == expected
        assert events[-1]["choices"][0]["finish_reason"] == "stop"
    finally:
        RouterHandler.adapters = previous
        RouterHandler.health = previous_health
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_chat_completions_passes_through_native_provider_stream():
    previous = RouterHandler.adapters
    previous_health = RouterHandler.health
    RouterHandler.adapters = {deployment.deployment_id: NativeStreamAdapter() for deployment in RouterHandler.deployments}
    RouterHandler.health = HealthRegistry()
    server = ThreadingHTTPServer(("127.0.0.1", 0), RouterHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection(*server.server_address)
        connection.request("POST", "/v1/chat/completions", json.dumps({
            "messages": [{"content": "hello"}], "stream": True,
        }), {"Content-Type": "application/json"})
        response = connection.getresponse()
        body = response.read().decode()
        connection.close()
        assert response.status == 200
        assert '"id": "upstream"' in body
        assert body.endswith("data: [DONE]\n\n")
    finally:
        RouterHandler.adapters = previous
        RouterHandler.health = previous_health
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_malformed_native_stream_closes_without_fake_success_response():
    previous = RouterHandler.adapters
    previous_health = RouterHandler.health

    class MalformedStreamAdapter:
        def complete(self, deployment, provider_request):
            return "unused"

        def stream(self, deployment, provider_request):
            yield json.dumps({"choices": [{"delta": {"content": "partial"}}]})
            yield "not-json"

    RouterHandler.adapters = {
        deployment.deployment_id: MalformedStreamAdapter() for deployment in RouterHandler.deployments
    }
    RouterHandler.health = HealthRegistry()
    server = ThreadingHTTPServer(("127.0.0.1", 0), RouterHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        before = get(server, "/v1/router/diagnostics")[1]["operational_failures"]
        connection = HTTPConnection(*server.server_address)
        connection.request(
            "POST",
            "/v1/chat/completions",
            json.dumps({"messages": [{"content": "hello"}], "stream": True}),
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        body = response.read().decode()
        connection.close()
        after = get(server, "/v1/router/diagnostics")[1]["operational_failures"]
        assert response.status == 200
        assert "partial" in body
        assert "not-json" not in body
        assert "[DONE]" not in body
        assert after == before + 1
    finally:
        RouterHandler.adapters = previous
        RouterHandler.health = previous_health
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_chat_completions_validates_response_format():
    server = ThreadingHTTPServer(("127.0.0.1", 0), RouterHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, result, _ = request(server, "/v1/chat/completions", {"messages": [{"content": "hello"}], "response_format": "json"})
        assert status == 400
        assert "response_format" in result["error"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_rejects_oversized_body():
    server = ThreadingHTTPServer(("127.0.0.1", 0), RouterHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection(*server.server_address)
        connection.putrequest("POST", "/v1/router/route")
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", str(RouterHandler.max_body_bytes + 1))
        connection.endheaders()
        response = connection.getresponse()
        assert response.status == 413
        assert "1 MiB" in json.loads(response.read())["error"]
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
