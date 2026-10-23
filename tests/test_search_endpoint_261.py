"""Regression tests for LLM-API-Key-Proxy issue #261.

`POST /v1/search` used to validate the query and then answer with a fake
success payload (`"status": "placeholder"`), which clients cannot distinguish
from a real search. It must now return 501 while keeping the empty-query 400.

The module also had to become importable for this to be testable: it used
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


def test_empty_query_is_still_rejected(client):
    response = client.post("/v1/search", json={"query": ""})

    assert response.status_code == 400
    assert response.json()["detail"] == "Query is required"


def test_missing_query_is_still_rejected(client):
    response = client.post("/v1/search", json={})

    assert response.status_code == 400


def test_search_returns_501_not_placeholder_success(client):
    response = client.post("/v1/search", json={"query": "free coding models"})

    assert response.status_code == 501
    body = response.json()
    assert body["detail"] == "Search functionality is not implemented yet"


def test_placeholder_payload_is_gone(client):
    response = client.post("/v1/search", json={"query": "anything"})

    assert response.status_code == 501
    text = response.text
    assert "placeholder" not in text
    assert "will be implemented with real providers" not in text
    assert "example.com" not in text


def test_module_imports_cleanly():
    """The endpoint module must be importable at all (it was not, before)."""
    assert enhanced_proxy.app is not None
    paths = {getattr(route, "path", None) for route in enhanced_proxy.app.routes}
    assert "/v1/search" in paths
