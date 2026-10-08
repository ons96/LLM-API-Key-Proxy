from provider_adapter import ProviderError, stream_execute
from router_core import Capability, Deployment, HealthRegistry, RequestFeatures


class FailingStream:
    def stream(self, deployment, prompt):
        raise ProviderError("stream failed")
        yield "unreachable"


class WorkingStream:
    def stream(self, deployment, prompt):
        yield "first"
        yield "[DONE]"


def test_stream_execution_fails_over_before_first_event():
    deployments = [
        Deployment("first", Capability.FAST_GENERAL, 1000, latency_ms=1, success_rate=1),
        Deployment("second", Capability.FAST_GENERAL, 1000, latency_ms=2, success_rate=1),
    ]
    selected, events = stream_execute(
        RequestFeatures("hello"), deployments,
        {"first": FailingStream(), "second": WorkingStream()}, HealthRegistry(), "hello"
    )
    assert selected.deployment_id == "second"
    assert list(events) == ["first", "[DONE]"]
