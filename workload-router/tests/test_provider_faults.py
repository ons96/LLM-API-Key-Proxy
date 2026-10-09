"""Exercise real HTTP transport and router state using local scripted providers."""

from collections import defaultdict, deque
from contextlib import contextmanager
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time

import pytest

from provider_adapter import OpenAICompatibleAdapter
from router_core import Capability, Deployment, HealthRegistry
from router_server import RouterHandler
from router_state import RouterState


@contextmanager
def serving(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture
def chain_router():
    scripts = defaultdict(deque)
    calls = []

    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            model = payload["model"]
            calls.append(payload)
            action, headers = scripts[model].popleft() if scripts[model] else ("ok", {})
            try:
                if action == "timeout":
                    time.sleep(0.15)
                if isinstance(action, int):
                    body = b'{"error":"private upstream message"}'
                    self.send_response(action)
                    for name, value in headers.items():
                        self.send_header(name, value)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if payload.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    if action == "empty":
                        return
                    if action == "malformed":
                        self.wfile.write(b"data: not-json\n\n")
                        return
                    event = {
                        "id": "local-fixture",
                        "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": None}],
                    }
                    self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                    self.wfile.flush()
                    if action == "truncated":
                        return
                    if action == "late-malformed":
                        self.wfile.write(b"data: not-json\n\n")
                        return
                    if action == "late-timeout":
                        time.sleep(0.15)
                    event["choices"] = []
                    event["usage"] = {"prompt_tokens_details": {"cached_tokens": 5000}}
                    self.wfile.write(f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n".encode())
                    return
                body = json.dumps({
                    "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens_details": {"cached_tokens": 5000}},
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Expected when the client times out against a scripted delay.

        def log_message(self, *_args):
            pass

    with serving(Upstream) as upstream:
        base = f"http://{upstream.server_address[0]}:{upstream.server_address[1]}/v1"

        class IsolatedRouter(RouterHandler):
            deployments = [
                Deployment(name, Capability.FAST_GENERAL, 100000, tools=True,
                           structured_output=True, model=name, provider="fixture",
                           group=group, chain_priority=priority)
                for name, group, priority in (
                    ("first", "chat-fast", 1), ("second", "chat-fast", 2),
                    ("foreign", "chat-elite", 1),
                )
            ]
            adapters = {item.deployment_id: OpenAICompatibleAdapter("fixture", timeout=0.05, base_url=base)
                        for item in deployments}
            health = HealthRegistry()
            state = RouterState()
            stall_states = {}

        with serving(IsolatedRouter) as router:
            yield router, IsolatedRouter, scripts, calls
        IsolatedRouter.state.close()


def chat(server, session_id="fault-smoke", **options):
    payload = {"model": "auto-chat", "model_group": "chat-fast",
               "messages": [{"role": "user", "content": "hello"}], **options}
    connection = HTTPConnection(*server.server_address, timeout=3)
    connection.request("POST", "/v1/chat/completions", json.dumps(payload),
                       {"Content-Type": "application/json", "X-Session-Id": session_id})
    response = connection.getresponse()
    body = response.read().decode()
    result = response.status, body, dict(response.getheaders())
    connection.close()
    return result


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("failure,category", [
    (429, "rate_limit"), (503, "upstream_failure"), (401, "provider_rejection"),
    ("timeout", "timeout"),
])
def test_http_faults_fail_over_within_group_and_skip_cooldown(chain_router, failure, category, stream):
    server, router, scripts, calls = chain_router
    scripts["first"].append((failure, {"Retry-After": "120"}))
    status, body, headers = chat(server, stream=stream)
    assert status == 200
    assert headers["X-Router-Deployment"] == "second"
    assert [call["model"] for call in calls] == ["first", "second"]
    assert router.health.error_class("first") == category
    assert router.health.retry_after("first") >= 119 if failure != "timeout" else True
    assert router.state.outcome_count("operational_failure") == 1
    assert router.state.outcome_count("success") == 1
    assert "private upstream message" not in body
    assert chat(server, stream=stream)[0] == 200
    assert [call["model"] for call in calls] == ["first", "second", "second"]
    router.health.mark_success("first")
    # A different session starts at the top after recovery; the original
    # session may correctly prefer the surviving deployment's warm cache.
    assert chat(server, session_id="fresh-session", stream=stream, cached_tokens=0)[0] == 200
    assert calls[-1]["model"] == "first"


@pytest.mark.parametrize("stream", [False, True])
def test_exhausted_chain_returns_bounded_failure_without_other_group(chain_router, stream):
    server, router, scripts, calls = chain_router
    scripts["first"].append((503, {}))
    scripts["second"].append((429, {"Retry-After": "120"}))
    status, body, headers = chat(server, stream=stream)
    assert status == 503
    assert headers["X-Router-Error-Class"] == "rate_limit"
    assert headers["Retry-After"] == "120"
    assert "private upstream message" not in body
    assert [call["model"] for call in calls] == ["first", "second"]
    assert router.state.outcome_count("success") == 0
    assert chat(server, stream=stream)[0] == 503
    assert len(calls) == 2


@pytest.mark.parametrize("stream", [False, True])
def test_quota_reset_wins_over_short_retry_hint_and_preserves_cooldown(chain_router, stream):
    server, router, scripts, calls = chain_router
    router.state.record_cache("fault-smoke", "first", "hello", 5000, 0)
    scripts["first"].append((429, {"Retry-After": "0.01", "X-RateLimit-Reset": str(time.time() + 600)}))
    assert chat(server, stream=stream)[0] == 200
    assert [call["model"] for call in calls] == ["first", "second"]
    assert router.health.retry_after("first") > 590


@pytest.mark.parametrize("stream", [False, True])
def test_short_warm_retry_recovers_without_cold_fallback(chain_router, stream):
    server, router, scripts, calls = chain_router
    router.state.record_cache("fault-smoke", "first", "hello", 5000, 0)
    scripts["first"].append((429, {"Retry-After": "0.01"}))
    status, _, headers = chat(server, stream=stream)
    assert status == 200
    assert headers["X-Router-Deployment"] == "first"
    assert [call["model"] for call in calls] == ["first", "first"]
    assert router.health.is_healthy("first")
    assert router.state.outcome_count("operational_failure") == 1
    assert router.state.outcome_count("success") == 1


def test_router_controls_do_not_leak_into_real_upstream_request(chain_router):
    server, _, _, calls = chain_router
    status, _, _ = chat(server, continuation=True, cache_ttl_seconds=45, cached_tokens=20,
                        tools=[{"type": "function", "function": {"name": "lookup"}}],
                        response_format={"type": "json_object"}, temperature=0)
    assert status == 200
    forwarded = calls[0]
    assert not {"model_group", "continuation", "cache_ttl_seconds", "cached_tokens"} & forwarded.keys()
    assert forwarded["tools"][0]["function"]["name"] == "lookup"
    assert forwarded["response_format"] == {"type": "json_object"}
    assert forwarded["temperature"] == 0


@pytest.mark.parametrize("failure,category", [("malformed", "invalid_response"), ("empty", "empty_stream")])
def test_invalid_first_stream_event_fails_over_before_headers(chain_router, failure, category):
    server, router, scripts, calls = chain_router
    scripts["first"].append((failure, {}))
    status, body, headers = chat(server, stream=True)
    assert status == 200
    assert headers["X-Router-Deployment"] == "second"
    assert body.endswith("data: [DONE]\n\n")
    assert router.health.error_class("first") == category
    assert len(calls) == 2


@pytest.mark.parametrize("failure,category", [
    ("truncated", "invalid_response"), ("late-malformed", "invalid_response"),
    ("late-timeout", "timeout"),
])
def test_partial_stream_never_retries_or_records_success(chain_router, failure, category):
    server, router, scripts, calls = chain_router
    scripts["first"].append((failure, {}))
    status, body, _ = chat(server, stream=True)
    assert status == 200  # Headers/content were already committed.
    assert "ok" in body and "[DONE]" not in body and "not-json" not in body
    assert len(calls) == 1
    assert router.state.outcome_count("success") == 0
    assert router.state.outcome_count("operational_failure") == 1
    assert router.health.error_class("first") == category


def test_native_usage_updates_cache_observations(chain_router):
    server, router, _, _ = chain_router
    assert chat(server, stream=True)[0] == 200
    hints = router.state.cache_hints("fault-smoke", "hello")
    assert hints["first"].cached_tokens == 5000
