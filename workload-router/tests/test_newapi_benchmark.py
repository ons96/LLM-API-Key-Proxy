"""Deterministic grading and ranking evidence using provider-free adapters."""

import base64
from dataclasses import replace
import json

import pytest

from newapi_benchmark import CASES, benchmark, collect, percentile
from newapi_runtime import Runtime
from provider_adapter import ProviderCompletion
from quota_ledger import QuotaLedger, Route
from router_state import RouterState
from tools.verify_newapi import totp


def output(case, correct=True):
    if case.tool:
        args = case.expected if correct else {"path": "wrong", "start": 0, "count": 1}
        return ProviderCompletion({"role": "assistant", "content": None, "tool_calls": [{
            "id": "test-call", "type": "function", "function": {
                "name": "read_file", "arguments": json.dumps(args)}}]})
    return ProviderCompletion({"role": "assistant", "content": json.dumps(case.expected if correct else {"wrong": True})})


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_graders_accept_only_expected_semantics(case):
    assert case.grade(output(case))
    assert not case.grade(output(case, False))
    assert not case.grade(ProviderCompletion({"content": "not-json"}))


def test_grader_does_not_confuse_boolean_with_integer():
    assert not CASES[1].grade(ProviderCompletion({"content": '{"timeout_seconds":8,"retries":false}'}))


@pytest.mark.parametrize("stream", [False, True])
def test_grades_change_rankings_without_dispatching_paid_routes(monkeypatch, stream):
    ledger, state = QuotaLedger(), RouterState()
    runtime = Runtime(ledger, state, "http://127.0.0.1:8002/v1", "test-relay", explore=False)
    a = Route("a", "fixture", "wrong", "a-key", "sc.a", 1,
              quota_type="unlimited", context_window=131072, output_limit=4096,
              input_price=0, output_price=0, tool_call=True, structured_output=True, stream=True)
    b = replace(a, route_id="b", model_id="correct", key_id="b-key", alias="sc.b", channel_id=2)
    paid = replace(a, route_id="paid", alias="sc.paid", channel_id=3, quota_type="paid", input_price=1, output_price=1)
    routes = [a, b, paid]
    for route in routes:
        ledger.configure(route)
    calls = []
    class Adapter:
        def complete(self, deployment, request):
            calls.append(deployment.model)
            case = next(case for case in CASES if case.prompt == request.messages[0]["content"])
            result = output(case, correct=deployment.model == "sc.b")
            return ProviderCompletion(result.message, "stop", {"prompt_tokens": 50, "completion_tokens": 20})

        def stream(self, deployment, request):
            result = self.complete(deployment, request)
            message = result.message
            delta = {"tool_calls": [{**message["tool_calls"][0], "index": 0}]} if message.get("tool_calls") else {"content": message["content"]}
            yield json.dumps({"choices": [{"index": 0, "delta": delta}]})
            yield json.dumps({"choices": [], "usage": result.usage})
            yield "[DONE]"
    monkeypatch.setattr(runtime, "adapter", lambda _remaining: Adapter())
    try:
        report = benchmark(runtime, routes, seed=7, stream=stream)
        assert len(calls) == 10 and "sc.paid" not in calls
        assert report["metrics"]["a"]["correct"] == 0
        assert report["metrics"]["b"]["correct"] == 5
        assert report["metrics"]["paid"]["attempts"] == 0
        assert report["ranking_changed"]
        assert all(ids[0] == "b" for ids in report["after"].values())
        assert report["metrics"]["b"]["p50_latency_ms"] > 0
        assert report["metrics"]["b"]["p50_time_to_correct_ms"] > 0
        assert (report["metrics"]["b"]["p50_ttft_ms"] is not None) == stream
        saved = json.loads(ledger.db.execute("SELECT summary FROM benchmark_runs WHERE id=?", (report["id"],)).fetchone()[0])
        assert saved == report and "Task plan" not in json.dumps(saved)
    finally:
        state.close()
        ledger.close()


def test_benchmark_budget_rejects_before_any_dispatch():
    ledger, state = QuotaLedger(), RouterState()
    runtime = Runtime(ledger, state, "http://127.0.0.1:8002/v1", "test-relay")
    route = Route("a", "p", "m", "k", "sc.a", 1)
    try:
        with pytest.raises(ValueError, match="request budget"):
            benchmark(runtime, [route], max_requests=1)
        assert not ledger.diagnostics()["attempts"]
    finally:
        state.close()
        ledger.close()


def test_collect_reassembles_fragmented_tool_arguments():
    parts = [{"index": 0, "id": "c", "type": "function", "function": {"name": "read_file", "arguments": '{"path":'}},
             {"index": 0, "function": {"arguments": '"src/app.py","start":12,"count":4}'}}]
    class Stream:
        usage = None
        closed = False
        def __iter__(self):
            for part in parts:
                yield json.dumps({"choices": [{"index": 0, "delta": {"tool_calls": [part]}}]})
            yield "[DONE]"
        def close(self):
            self.closed = True
    source = Stream()
    assert CASES[3].grade(collect(source)) and source.closed


def test_percentile_interpolates_small_samples():
    assert percentile([], 0.5) is None
    assert percentile([4], 0.95) == 4
    assert percentile([4, 2], 0.5) == 3


def test_disposable_totp_matches_rfc6238_vector():
    secret = base64.b32encode(b"12345678901234567890").decode().rstrip("=")
    assert totp(secret, 59, digits=8) == "94287082"
