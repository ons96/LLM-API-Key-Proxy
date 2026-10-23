"""`POST /v1/search` — regression for #261, real implementation for #1123.

#261: the endpoint validated the query and then answered with a fake success
payload (`"status": "placeholder"`), which a client cannot distinguish from a
real search. It was changed to return 501 rather than lie.

#1123: search now actually runs through the gateway's configured provider
ladder and returns cited results. The honesty property #261 protects is kept:
with no provider configured the endpoint answers 503, and a provider failure
answers 502 -- neither ever fabricates a result list.

The module also had to become importable for any of this to be testable: it used
`Request`, `HTTPException`, `StreamingResponse` and `Depends` without importing
them, and its chat handler annotation is not a valid FastAPI response model.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastapi.testclient import TestClient

from proxy_app import enhanced_proxy


@pytest.fixture(scope="module")
def client():
    """HTTP client without app lifespan.

    Entering TestClient as a context manager runs startup/shutdown; shutdown
    waits on the provider status tracker, which makes live upstream calls. Those
    fail intermittently in CI and turn a passing test into a teardown error.
    Routes are registered at import time, so the lifespan is not needed here.
    """
    return TestClient(enhanced_proxy.app)


class FakeCore:
    """Stands in for RouterCore, holding one search provider."""

    search_providers = {"fake": object()}

    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error
        self.calls = []

    async def search(self, query, max_results=5):
        self.calls.append((query, max_results))
        if self.error:
            raise self.error
        return self.results[:max_results]


class FakeWrapper:
    def __init__(self, core):
        self.router_integration = type("Integration", (), {"router": core})()


@pytest.fixture
def wire(monkeypatch):
    def _wire(core):
        from proxy_app import router_wrapper

        monkeypatch.setattr(router_wrapper, "_router_wrapper", FakeWrapper(core))
        return core

    return _wire


# ---------------------------------------------------------------------------
# input validation (unchanged by #1123)
# ---------------------------------------------------------------------------


def test_empty_query_is_still_rejected(client):
    response = client.post("/v1/search", json={"query": ""})

    assert response.status_code == 400
    assert response.json()["detail"] == "Query is required"


def test_missing_query_is_still_rejected(client):
    response = client.post("/v1/search", json={})

    assert response.status_code == 400


# ---------------------------------------------------------------------------
# honesty properties (#261)
# ---------------------------------------------------------------------------


def test_no_providers_is_a_503_not_a_fake_result(client, monkeypatch):
    from proxy_app import router_wrapper

    monkeypatch.setattr(router_wrapper, "_router_wrapper", None)

    response = client.post("/v1/search", json={"query": "free coding models"})

    assert response.status_code == 503
    assert response.json()["detail"] == "No search providers are configured"
    assert "placeholder" not in response.text


def test_provider_failure_is_sanitized_and_not_a_result(client, wire):
    secret = "sk-live-SECRET-abcdef123456"
    wire(FakeCore(error=RuntimeError(f"upstream 500 for key {secret} at internal-gw.corp")))

    response = client.post("/v1/search", json={"query": "anything"})

    assert response.status_code == 502
    assert secret not in response.text
    assert "internal-gw.corp" not in response.text


# ---------------------------------------------------------------------------
# real search (#1123)
# ---------------------------------------------------------------------------


def test_configured_provider_returns_results_and_citations(client, wire):
    core = wire(
        FakeCore(
            results=[
                {"title": "A", "url": "https://example.org/a", "content": "alpha"},
                {"title": "B", "url": "https://example.org/b", "content": "beta"},
                {"title": "A again", "url": "https://example.org/a", "content": "alpha"},
            ]
        )
    )

    response = client.post("/v1/search", json={"query": "free models"})

    assert response.status_code == 200
    body = response.json()
    assert [r["title"] for r in body["results"]] == ["A", "B", "A again"]
    assert body["citations"] == ["https://example.org/a", "https://example.org/b"]
    assert body["degraded"] is False
    assert core.calls == [("free models", 5)]


def test_max_results_is_passed_through(client, wire):
    core = wire(
        FakeCore(results=[{"title": str(n), "url": f"https://e/{n}", "content": ""} for n in range(9)])
    )

    response = client.post("/v1/search", json={"query": "q", "max_results": 2})

    assert response.status_code == 200
    assert len(response.json()["results"]) == 2
    assert core.calls == [("q", 2)]


def test_empty_provider_result_is_marked_degraded_not_ok(client, wire):
    wire(FakeCore(results=[]))

    response = client.post("/v1/search", json={"query": "q"})

    assert response.status_code == 200
    body = response.json()
    assert body["results"] == [] and body["citations"] == []
    assert body["degraded"] is True


# ---------------------------------------------------------------------------
# module wiring
# ---------------------------------------------------------------------------


def test_module_imports_cleanly():
    """The endpoint module must be importable at all (it was not, before)."""
    assert enhanced_proxy.app is not None
    paths = {getattr(route, "path", None) for route in enhanced_proxy.app.routes}
    assert "/v1/search" in paths
