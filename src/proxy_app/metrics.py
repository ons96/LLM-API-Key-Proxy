import os
import time
from dataclasses import dataclass
from typing import Optional

from fastapi import Request


try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

    _PROMETHEUS_AVAILABLE = True
except Exception:  # pragma: no cover
    CONTENT_TYPE_LATEST = "text/plain"
    Counter = None  # type: ignore[assignment]
    Histogram = None  # type: ignore[assignment]
    generate_latest = None  # type: ignore[assignment]
    _PROMETHEUS_AVAILABLE = False


_METRICS_ENABLED = os.getenv("METRICS_ENABLED", "true").lower() == "true"


if _PROMETHEUS_AVAILABLE and _METRICS_ENABLED:
    HTTP_REQUESTS_TOTAL = Counter(
        "proxy_http_requests_total",
        "HTTP requests handled by the proxy",
        labelnames=("method", "route", "status"),
    )

    HTTP_REQUEST_DURATION_SECONDS = Histogram(
        "proxy_http_request_duration_seconds",
        "HTTP request duration in seconds",
        labelnames=("method", "route"),
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
    )
else:  # pragma: no cover
    HTTP_REQUESTS_TOTAL = None
    HTTP_REQUEST_DURATION_SECONDS = None


@dataclass
class RequestTimer:
    start: float


def start_timer() -> RequestTimer:
    return RequestTimer(start=time.perf_counter())


def observe_http_request(method: str, route: str, status: int, elapsed_seconds: float) -> None:
    if not HTTP_REQUESTS_TOTAL:
        return

    HTTP_REQUESTS_TOTAL.labels(method=method, route=route, status=str(status)).inc()
    if HTTP_REQUEST_DURATION_SECONDS:
        HTTP_REQUEST_DURATION_SECONDS.labels(method=method, route=route).observe(elapsed_seconds)


def get_route_template(request: Request) -> str:
    route = request.scope.get("route")
    if route and hasattr(route, "path"):
        return str(route.path)
    return request.url.path


def metrics_payload() -> Optional[bytes]:
    if not _PROMETHEUS_AVAILABLE or not generate_latest:
        return None
    return generate_latest()
