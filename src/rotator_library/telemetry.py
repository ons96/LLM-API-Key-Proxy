import os
from typing import Optional


try:
    from prometheus_client import Counter, Histogram

    _PROMETHEUS_AVAILABLE = True
except Exception:  # pragma: no cover
    Counter = None  # type: ignore[assignment]
    Histogram = None  # type: ignore[assignment]
    _PROMETHEUS_AVAILABLE = False


_METRICS_ENABLED = os.getenv("METRICS_ENABLED", "true").lower() == "true"


if _PROMETHEUS_AVAILABLE and _METRICS_ENABLED:
    PROVIDER_REQUESTS_TOTAL = Counter(
        "rotator_provider_requests_total",
        "Total LLM requests attempted by provider",
        labelnames=("provider", "operation", "outcome"),
    )
    PROVIDER_ERRORS_TOTAL = Counter(
        "rotator_provider_errors_total",
        "Total provider errors (classified) by provider",
        labelnames=("provider", "operation", "error_type"),
    )
    PROVIDER_REQUEST_DURATION_SECONDS = Histogram(
        "rotator_provider_request_duration_seconds",
        "Provider request duration in seconds",
        labelnames=("provider", "operation"),
        buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 40, 80),
    )
    PROVIDER_FAILOVERS_TOTAL = Counter(
        "rotator_provider_failovers_total",
        "Total provider failovers (fallback transitions)",
        labelnames=("from_provider", "to_provider", "operation"),
    )
    CREDENTIAL_USES_TOTAL = Counter(
        "rotator_credential_uses_total",
        "Total credential selections by provider and priority",
        labelnames=("provider", "priority"),
    )
else:  # pragma: no cover
    PROVIDER_REQUESTS_TOTAL = None
    PROVIDER_ERRORS_TOTAL = None
    PROVIDER_REQUEST_DURATION_SECONDS = None
    PROVIDER_FAILOVERS_TOTAL = None
    CREDENTIAL_USES_TOTAL = None


def record_provider_request(
    *,
    provider: str,
    operation: str,
    outcome: str,
    duration_seconds: Optional[float] = None,
) -> None:
    if not PROVIDER_REQUESTS_TOTAL:
        return

    PROVIDER_REQUESTS_TOTAL.labels(provider=provider, operation=operation, outcome=outcome).inc()
    if duration_seconds is not None and PROVIDER_REQUEST_DURATION_SECONDS:
        PROVIDER_REQUEST_DURATION_SECONDS.labels(provider=provider, operation=operation).observe(duration_seconds)


def record_provider_error(*, provider: str, operation: str, error_type: str) -> None:
    if not PROVIDER_ERRORS_TOTAL:
        return
    PROVIDER_ERRORS_TOTAL.labels(provider=provider, operation=operation, error_type=error_type).inc()


def record_failover(*, from_provider: str, to_provider: str, operation: str) -> None:
    if not PROVIDER_FAILOVERS_TOTAL:
        return
    PROVIDER_FAILOVERS_TOTAL.labels(from_provider=from_provider, to_provider=to_provider, operation=operation).inc()


def record_credential_use(*, provider: str, priority: Optional[int]) -> None:
    if not CREDENTIAL_USES_TOTAL:
        return
    label = str(priority) if priority is not None else "unknown"
    CREDENTIAL_USES_TOTAL.labels(provider=provider, priority=label).inc()
