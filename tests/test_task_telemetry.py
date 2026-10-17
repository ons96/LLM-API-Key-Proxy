"""
Task-class telemetry + outcome API tests (routing brain Phase 3).
"""

import os
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from proxy_app import task_telemetry
from proxy_app.outcomes_api import router as outcomes_router


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    db = tmp_path / "task_telemetry.db"
    monkeypatch.setattr(task_telemetry, "DB_PATH", str(db))
    # outcomes_api calls task_telemetry functions, which read DB_PATH at
    # call time via _connect().
    task_telemetry.init_db()
    yield db


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(outcomes_router)
    return TestClient(app)


class TestTaskTelemetry:
    def test_record_and_export_outcomes(self):
        task_telemetry.record_outcome(
            task_class="agentic-medium", success=True, model_id="glm-4.6",
            provider="nvidia", total_tokens=41000, turns=7,
            source="test", observed_at="2026-08-16T10:00:00Z",
        )
        task_telemetry.record_outcome(
            task_class="agentic-medium", success=False, model_id="glm-4.6",
            source="test", observed_at="2026-08-16T11:00:00Z",
        )
        out = task_telemetry.export_outcomes()
        assert len(out) == 2
        assert out[0]["model_id"] == "glm-4.6" and out[0]["success"] is True
        assert out[1]["success"] is False

    def test_export_since_filter(self):
        task_telemetry.record_outcome(
            task_class="chat", success=True, observed_at="2026-08-15T00:00:00Z",
            source="test",
        )
        task_telemetry.record_outcome(
            task_class="chat", success=True, observed_at="2026-08-16T12:00:00Z",
            source="test",
        )
        out = task_telemetry.export_outcomes(since="2026-08-16T00:00:00Z")
        assert len(out) == 1
        assert out[0]["observed_at"] == "2026-08-16T12:00:00Z"

    def test_record_outcome_requires_task_class(self):
        with pytest.raises(ValueError):
            task_telemetry.record_outcome(task_class="", success=True)

    def test_gateway_request_logging_never_raises(self, tmp_db):
        before = tmp_db.exists()
        task_telemetry.record_gateway_request(
            request_id="req_1", virtual_model="auto/agentic-medium",
            task_class="agentic-medium", task_id="task_1",
            concrete_provider="nvidia", concrete_model="glm/glm-4.6",
            stream=False, status="success",
        )
        reqs = task_telemetry.export_gateway_requests()
        assert len(reqs) == 1
        assert reqs[0]["task_class"] == "agentic-medium"
        assert reqs[0]["status"] == "success"
        assert before


class TestOutcomeAPI:
    def test_post_outcome_roundtrip(self, client):
        resp = client.post(
            "/api/task-outcome",
            json={
                "task_class": "quick-edit",
                "success": True,
                "model_id": "qwen3-coder",
                "provider": "openrouter",
                "total_tokens": 9000,
                "turns": 2,
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["ok"] is True and body["id"] >= 1

        out = client.get("/api/task-outcomes/export").json()
        assert out["outcomes"][0]["model_id"] == "qwen3-coder"
        assert out["outcomes"][0]["total_tokens"] == 9000

    def test_post_outcome_validation(self, client):
        assert client.post("/api/task-outcome", json={"success": True}).status_code == 422
        assert (
            client.post("/api/task-outcome", json={"task_class": "chat", "success": "yes"}).status_code
            == 422
        )
        assert (
            client.post("/api/task-outcome", json={"task_class": "chat", "success": True, "turns": -3}).status_code
            == 422
        )

    def test_export_include_requests(self, client):
        task_telemetry.record_gateway_request(
            request_id="r1", virtual_model="auto/chat", task_class="chat",
            task_id="t1", concrete_provider=None, concrete_model=None,
            stream=True, status="success",
        )
        payload = client.get("/api/task-outcomes/export", params={"include_requests": "true"}).json()
        assert payload["gateway_requests"][0]["request_id"] == "r1"

    def test_auth_enforced_when_key_set(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PROXY_API_KEY", "sekrit")
        app = FastAPI()
        app.include_router(outcomes_router)
        c = TestClient(app)
        assert c.post("/api/task-outcome", json={"task_class": "chat", "success": True}).status_code == 401
        ok = c.post(
            "/api/task-outcome",
            json={"task_class": "chat", "success": True},
            headers={"Authorization": "Bearer sekrit"},
        )
        assert ok.status_code == 200
        monkeypatch.delenv("PROXY_API_KEY")
