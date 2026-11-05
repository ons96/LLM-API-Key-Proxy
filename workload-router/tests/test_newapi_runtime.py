"""Provider-free accounting and stream lifecycle regressions."""

from dataclasses import replace
import json

import pytest

from newapi_runtime import Runtime, prepare, validate_endpoint
from provider_adapter import ProviderCompletion, ProviderError
from quota_ledger import QuotaLedger, Route
from router_state import RouterState


def route(name="a", **changes):
    base = Route(name, "local-provider", name, "opaque-key", "sc." + name,
                 1 if name == "a" else 2, quota_type="rate_limited",
                 input_price=0, output_price=0, context_window=262144,
                 output_limit=32768, tool_call=True, structured_output=True,
                 stream=True, limits={"rpm": 100}, latency_ms=2000)
    return replace(base, **changes)


def event(delta=None, usage=None):
    result = {"choices": [] if delta is None else [{"index": 0, "delta": delta}]}
    if usage is not None:
        result["usage"] = usage
    return json.dumps(result)


class Source:
    def __init__(self, values):
        self.values = iter(values)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        value = next(self.values)
        if isinstance(value, Exception):
            raise value
        return value

    def close(self):
        self.closed = True


@pytest.fixture
def runtime():
    ledger, state = QuotaLedger(), RouterState()
    instance = Runtime(ledger, state, "http://127.0.0.1:8002/v1", "test-only-relay",
                       explore=False)
    ledger.configure(route())
    ledger.configure(route("b"))
    try:
        yield instance
    finally:
        state.close()
        ledger.close()


def request(**options):
    return prepare({"model": "auto-chat", "messages": [{"role": "user", "content": "hello"}],
                    **options})


def rows(runtime):
    return [dict(row) for row in runtime.ledger.db.execute("SELECT * FROM routing_decisions ORDER BY created_at")]


def install(runtime, monkeypatch, actions, calls):
    class Adapter:
        def operation(self, deployment, data):
            calls.append(data.to_payload(deployment.model, stream=False))
            action = actions[deployment.model]
            if isinstance(action, Exception):
                raise action
            return action

        complete = operation
        stream = operation

    monkeypatch.setattr(runtime, "adapter", lambda _remaining: Adapter())


@pytest.mark.parametrize("options", [
    {"model": []}, {"stream": 1}, {"max_tokens": True}, {"max_tokens": 0},
    {"max_tokens": 10, "max_completion_tokens": 11}, {"n": 2}, {"best_of": 3},
    {"stream": True, "stream_options": []},
    {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]},
    {"temperature": float("nan")},
])
def test_invalid_request_limits(options):
    with pytest.raises(ValueError):
        request(**options)


@pytest.mark.parametrize("endpoint", ["http://remote.invalid/v1", "https://name:pass@remote.invalid/v1",
                                      "https://remote.invalid:bad/v1", "https://remote.invalid/v1?key=secret"])
def test_endpoint_boundary(endpoint):
    with pytest.raises(ValueError):
        validate_endpoint(endpoint)


def test_prepare_bounds_tools_and_preserves_output_cap():
    tools = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]
    data = request(tools=tools, stream=True)
    forwarded = data.request.to_payload("sc.a", True)
    assert forwarded["max_tokens"] == data.work.output_tokens == 2048
    assert forwarded["tools"] == tools
    assert forwarded["stream_options"] == {"include_usage": True}
    assert data.work.input_tokens > len("hello")
    assert "read" in data.prefix


def test_429_once_each_and_conservative_failed_charge(runtime, monkeypatch):
    calls = []
    install(runtime, monkeypatch, {"sc.a": ProviderError("private", error_class="rate_limit", status_code=429,
                                                      limit_hints={"rpm": 2}),
                                  "sc.b": ProviderCompletion({"content": "ok"}, "stop",
                                                           {"prompt_tokens": 10, "completion_tokens": 2})}, calls)
    result = runtime.complete(request())
    assert result.route.route_id == "b" and len(result.attempts) == 2
    assert [call["model"] for call in calls] == ["sc.a", "sc.b"]
    failed, success = rows(runtime)
    assert failed["status"] == "failure" and failed["usage_estimated"] == 1
    assert failed["output_tokens"] == 2048
    assert success["status"] == "success" and success["quality"] is None
    assert success["ttft_ms"] is None and success["tokens_per_second"] is None
    assert not runtime.ledger.capacity(route(), 1, 1).feasible
    assert runtime.ledger.db.execute("SELECT learned FROM quota_limits WHERE learned IS NOT NULL").fetchone()[0] == 2


def test_grader_fault_settles_then_raises(runtime, monkeypatch):
    install(runtime, monkeypatch, {"sc.a": ProviderCompletion({"content": "ok"})}, [])
    def broken(_result):
        raise RuntimeError("broken grader")
    with pytest.raises(RuntimeError, match="broken grader"):
        runtime.complete(request(), grader=broken)
    assert rows(runtime)[0]["status"] == "success"
    assert rows(runtime)[0]["quality"] is None


def test_setup_fault_is_cancelled_without_dispatch(runtime, monkeypatch):
    def broken(_remaining):
        raise ValueError("adapter setup")
    monkeypatch.setattr(runtime, "adapter", broken)
    with pytest.raises(ProviderError):
        runtime.complete(request())
    assert all(row["status"] == "cancelled" for row in rows(runtime))
    assert runtime.ledger.db.execute("SELECT SUM(amount) FROM quota_windows").fetchone()[0] == 0


def test_graded_failure_fails_over_and_updates_evidence(runtime, monkeypatch):
    calls = []
    install(runtime, monkeypatch, {"sc.a": ProviderCompletion({"content": "wrong"}),
                                  "sc.b": ProviderCompletion({"content": "ok"})}, calls)
    result = runtime.complete(request(), grader=lambda output: output.content == "ok")
    assert result.route.route_id == "b"
    assert [row["quality"] for row in rows(runtime)] == [0, 1]
    assert runtime.ledger.evidence(route(), "chat")[:2] == (2, 3)


def test_stream_buffers_once_and_separates_text_from_reasoning(runtime, monkeypatch):
    values = [event({"role": "assistant"}), event({"reasoning_content": "thinking"}),
              event({"content": "ok"}), event(usage={"prompt_tokens": 10, "completion_tokens": 5}), "[DONE]"]
    source = Source(values)
    install(runtime, monkeypatch, {"sc.a": source}, [])
    stream = runtime.open_stream(request(stream=True))
    assert list(stream) == values
    stream.close()
    decision = rows(runtime)[0]
    assert decision["status"] == "success" and decision["usage_estimated"] == 0
    assert decision["reasoning_ttft_ms"] <= decision["ttft_ms"]
    assert decision["tokens_per_second"] > 0 and source.closed


@pytest.mark.parametrize("failure", [Source([]), Source(["[DONE]"]),
                                     Source([event({"role": "assistant"}), "not-json"])])
def test_pre_content_stream_failure_can_fail_over(runtime, monkeypatch, failure):
    calls = []
    install(runtime, monkeypatch, {"sc.a": failure, "sc.b": Source([event({"content": "ok"}), "[DONE]"])}, calls)
    stream = runtime.open_stream(request(stream=True))
    assert stream.route.route_id == "b"
    assert list(stream)[-1] == "[DONE]" and failure.closed
    assert [row["status"] for row in rows(runtime)] == ["failure", "success"]


def test_late_error_keeps_full_charge_and_never_fails_over(runtime, monkeypatch):
    calls = []
    source = Source([event({"content": "ok"}), event(usage={"prompt_tokens": 1, "completion_tokens": 1}),
                     ProviderError("private", error_class="timeout")])
    install(runtime, monkeypatch, {"sc.a": source}, calls)
    stream = runtime.open_stream(request(stream=True))
    with pytest.raises(ProviderError):
        list(stream)
    stream.close()
    decision = rows(runtime)[0]
    assert decision["status"] == "failure" and decision["output_tokens"] == 2048
    assert decision["usage_estimated"] == 1 and len(calls) == 1 and source.closed


def test_consumer_disconnect_settles_once(runtime, monkeypatch):
    source = Source([event({"content": "ok"}), "[DONE]"])
    install(runtime, monkeypatch, {"sc.a": source}, [])
    stream = runtime.open_stream(request(stream=True))
    stream.close()
    stream.close()
    assert rows(runtime)[0]["status"] == "failure" and source.closed


def test_coding_affinity_not_applied_to_chat(runtime, monkeypatch):
    install(runtime, monkeypatch, {"sc.a": ProviderCompletion({"content": "ok"})}, [])
    coding = prepare({"model": "auto-coding", "messages": [{"role": "user", "content": "code"}]}, "session")
    runtime.complete(coding)
    assert runtime.state.preferred("session", "auto-coding") == "a"
    chat = prepare({"model": "auto-chat", "messages": [{"role": "user", "content": "chat"}]}, "session")
    monkeypatch.setattr(runtime.state, "preferred", lambda *_: pytest.fail("chat used affinity"))
    monkeypatch.setattr(runtime.state, "cache_hints", lambda *_: pytest.fail("chat used cache"))
    assert runtime.ranked(chat)
