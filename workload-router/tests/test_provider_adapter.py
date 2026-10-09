import time

import pytest

from provider_adapter import ProviderError, execute, stream_execute
from router_core import CacheHint, Capability, Deployment, HealthRegistry, RequestFeatures


class Adapter:
    def __init__(self, result=None, fails=False):
        self.result = result
        self.fails = fails

    def complete(self, deployment, prompt):
        if self.fails:
            raise ProviderError("timeout")
        return self.result


def test_execution_fails_over_within_same_capability():
    deployments = [
        Deployment("first", Capability.FAST_GENERAL, 4096, latency_ms=1, success_rate=1),
        Deployment("second", Capability.FAST_GENERAL, 4096, latency_ms=2, success_rate=1),
    ]
    result = execute(
        RequestFeatures("answer"),
        deployments,
        {"first": Adapter(fails=True), "second": Adapter(result="ok")},
        HealthRegistry(),
        "answer",
    )
    assert (result.deployment_id, result.content, result.attempts) == ("second", "ok", 2)


def test_execution_does_not_fallback_to_different_capability():
    with pytest.raises(ProviderError, match="all eligible deployments"):
        execute(
            RequestFeatures("answer"),
            [Deployment("general", Capability.FAST_GENERAL, 4096)],
            {"general": Adapter(fails=True)},
            HealthRegistry(),
            "answer",
        )


def test_execution_reports_failed_attempts_without_leaking_error_text():
    deployments = [
        Deployment("first", Capability.FAST_GENERAL, 4096, latency_ms=1, success_rate=1),
        Deployment("second", Capability.FAST_GENERAL, 4096, latency_ms=2, success_rate=1),
    ]
    failures = []

    class RateLimitedAdapter(Adapter):
        def complete(self, deployment, provider_request):
            raise ProviderError("secret upstream detail", error_class="rate_limit", status_code=429)

    result = execute(
        RequestFeatures("answer"),
        deployments,
        {"first": RateLimitedAdapter(), "second": Adapter(result="ok")},
        HealthRegistry(),
        "answer",
        on_failure=lambda deployment, error_class: failures.append((deployment, error_class)),
    )
    assert result.deployment_id == "second"
    assert result.failures == (("first", "rate_limit"),)
    assert failures == [("first", "rate_limit")]
    assert "secret" not in str(result)


def test_execution_retries_short_rate_limit_when_cache_is_warm():
    deployment = Deployment("warm", Capability.FAST_GENERAL, 4096, group="chat-fast", chain_priority=1)

    class FlakyAdapter:
        def __init__(self):
            self.calls = 0

        def complete(self, deployment, provider_request):
            self.calls += 1
            if self.calls == 1:
                raise ProviderError("temporarily limited", error_class="rate_limit", retry_after_seconds=0.01)
            return "ok"

    adapter = FlakyAdapter()
    result = execute(
        RequestFeatures("answer", model_group="chat-fast"),
        [deployment],
        {"warm": adapter},
        HealthRegistry(),
        "answer",
        cache_hints={"warm": CacheHint("warm", 5000, observed_at=time.time())},
    )
    assert adapter.calls == 2
    assert result.deployment_id == "warm"
    assert result.attempts == 2
    assert result.failures == (("warm", "rate_limit"),)


@pytest.mark.parametrize("exception, error_class", [
    (OSError("connection reset"), "network_error"),
    (ValueError("bad provider result"), "invalid_response"),
])
def test_unexpected_completion_exceptions_fail_over_with_bounded_category(exception, error_class):
    deployments = [
        Deployment("first", Capability.FAST_GENERAL, 4096, latency_ms=1, success_rate=1),
        Deployment("second", Capability.FAST_GENERAL, 4096, latency_ms=2, success_rate=1),
    ]

    class Broken:
        def complete(self, deployment, provider_request):
            raise exception

    result = execute(
        RequestFeatures("answer"),
        deployments,
        {"first": Broken(), "second": Adapter(result="ok")},
        HealthRegistry(),
        "answer",
    )
    assert result.deployment_id == "second"


@pytest.mark.parametrize("exception, error_class", [
    (OSError("connection reset"), "network_error"),
    (ValueError("bad stream"), "invalid_response"),
])
def test_unexpected_stream_exceptions_fail_over_before_headers(exception, error_class):
    deployments = [
        Deployment("first", Capability.FAST_GENERAL, 4096, latency_ms=1, success_rate=1),
        Deployment("second", Capability.FAST_GENERAL, 4096, latency_ms=2, success_rate=1),
    ]

    class Broken:
        def stream(self, deployment, provider_request):
            def events():
                raise exception
                yield "unreachable"
            return events()

    class Working:
        def stream(self, deployment, provider_request):
            yield '{"choices": []}'
            yield "[DONE]"

    health = HealthRegistry()
    selected, events = stream_execute(
        RequestFeatures("answer"),
        deployments,
        {"first": Broken(), "second": Working()},
        health,
        "answer",
    )
    assert selected.deployment_id == "second"
    assert list(events) == ['{"choices": []}', "[DONE]"]
    assert health.error_class("first") == error_class


def test_pre_header_stream_failure_closes_acquired_iterator():
    deployments = [
        Deployment("first", Capability.FAST_GENERAL, 4096, latency_ms=1, success_rate=1),
        Deployment("second", Capability.FAST_GENERAL, 4096, latency_ms=2, success_rate=1),
    ]
    closed = []

    class Broken:
        def stream(self, deployment, provider_request):
            class Events:
                def __iter__(self):
                    return self

                def __next__(self):
                    raise OSError("connection reset")

                def close(self):
                    closed.append(deployment.deployment_id)

            return Events()

    class Working:
        def stream(self, deployment, provider_request):
            yield '{"choices": []}'
            yield "[DONE]"

    selected, events = stream_execute(
        RequestFeatures("answer"),
        deployments,
        {"first": Broken(), "second": Working()},
        HealthRegistry(),
        "answer",
    )
    assert selected.deployment_id == "second"
    assert list(events) == ['{"choices": []}', "[DONE]"]
    assert closed == ["first"]
