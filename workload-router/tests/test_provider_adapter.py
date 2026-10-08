import pytest

from provider_adapter import ProviderError, execute
from router_core import Capability, Deployment, HealthRegistry, RequestFeatures


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
