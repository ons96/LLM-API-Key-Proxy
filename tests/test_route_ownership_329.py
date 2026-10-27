"""Regression tests for #329: no duplicate routes shadow each other.

``enhanced_proxy`` imports the shared FastAPI app from ``proxy_app.main`` and
used to also register POST /v1/chat/completions (plus GET /v1/models and
/health) on it. FastAPI matches the first registered route, so main.py's
handlers always won: enhanced_proxy's copies were dead at the HTTP layer while
tests importing ``enhanced_proxy.app`` believed they were exercising them.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proxy_app import enhanced_proxy  # noqa: E402  (path setup above)


def _routes(path: str, method: str):
    return [
        route
        for route in enhanced_proxy.app.routes
        if getattr(route, "path", None) == path
        and method in getattr(route, "methods", set())
    ]


def test_chat_completions_route_has_single_owner():
    routes = _routes("/v1/chat/completions", "POST")
    assert len(routes) == 1, "duplicate /v1/chat/completions registrations (#329)"
    assert routes[0].endpoint.__module__ == "proxy_app.main"


def test_models_and_health_routes_have_single_owner():
    for path in ("/v1/models", "/health"):
        routes = _routes(path, "GET")
        assert len(routes) == 1, f"duplicate {path} registrations (#329)"
        assert routes[0].endpoint.__module__ == "proxy_app.main"


def test_enhanced_unique_routes_still_registered():
    for path, method in (
        ("/v1/search", "POST"),
        ("/v1/health", "GET"),
        ("/v1/router/status", "GET"),
        ("/v1/router/metrics", "GET"),
        ("/v1/router/refresh", "POST"),
    ):
        assert _routes(path, method), f"{method} {path} missing from enhanced_proxy"


def test_enhanced_chat_handler_stays_importable_and_callable():
    # main.py owns the HTTP route; the enhanced handler stays available for
    # direct use (tests/test_error_detail_leak_rest.py calls it).
    assert callable(enhanced_proxy.enhanced_chat_completions)
