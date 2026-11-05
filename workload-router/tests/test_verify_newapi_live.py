"""Free-only live verification policy; no real network or credentials are used."""

from contextlib import contextmanager
import io
import json
import socket
import sqlite3
import urllib.error

import pytest

from newapi_catalog import CatalogError
from provider_adapter import ProviderCompletion, ProviderError
from quota_ledger import Route
from tools.verify_newapi_live import (
    Broker,
    OUTPUT_CAP,
    Target,
    frame_read,
    frame_send,
    free_model,
    load_targets,
    private_key,
    verify_failover,
)


def test_free_model_rejects_stale_nonfree_and_unknown_metadata():
    model = {
        "id": "model:free",
        "pricing": {"prompt": "0", "completion": "0"},
        "context_length": 32768,
        "supported_parameters": ["tools", "tool_choice"],
    }
    assert free_model("openrouter", "model:free", [model], {})["context"] == 32768
    for change in (
        {"pricing": {"prompt": "1", "completion": "0"}},
        {"pricing": {"prompt": "NaN", "completion": "0"}},
        {"supported_parameters": []},
        {"context_length": None},
    ):
        with pytest.raises(ValueError):
            free_model("openrouter", "model:free", [{**model, **change}], {})
    with pytest.raises(ValueError):
        free_model("openrouter", "absent:free", [model], {})
    with pytest.raises(ValueError):
        free_model("groq", "model", [{"id": "model"}], {})


def test_private_key_requires_owner_only_file_in_secret_store(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.verify_newapi_live.Path.home", lambda: tmp_path)
    directory = tmp_path / ".secrets"
    directory.mkdir(mode=0o700)
    key = directory / "test"
    key.write_text("fake-private-key")
    key.chmod(0o600)
    assert private_key(key) == "fake-private-key"
    key.chmod(0o644)
    with pytest.raises(ValueError, match="owner-only"):
        private_key(key)
    link = directory / "link"
    link.symlink_to(key)
    with pytest.raises(OSError):
        private_key(link)
    with pytest.raises(ValueError, match="private secret store"):
        private_key(tmp_path / "outside")


@pytest.mark.parametrize(
    "targets",
    [
        [],
        [{"provider": "paid", "model": "m", "key_file": "file"}],
        [{"provider": "groq", "model": "m", "key_file": "file"}] * 2,
    ],
)
def test_plan_rejects_missing_unapproved_or_repeated_targets_before_network(
    tmp_path, targets
):
    plan = tmp_path / "targets.json"
    plan.write_text(json.dumps({"targets": targets}))
    with pytest.raises(ValueError):
        load_targets(tmp_path / "unused.db", plan)


@pytest.mark.parametrize("source", ["file", "env", "wrong_env", "missing_env"])
def test_catalog_preflight_keeps_canonical_db_read_only(tmp_path, monkeypatch, source):
    monkeypatch.setattr("tools.verify_newapi_live.Path.home", lambda: tmp_path)
    store = tmp_path / ".secrets"
    store.mkdir(mode=0o700)
    key = store / "openrouter"
    key.write_text("fixture-only-key")
    key.chmod(0o600)
    database = tmp_path / "providers.db"
    with sqlite3.connect(database) as db:
        db.execute(
            "CREATE TABLE providers (id, key_name, base_url, env_var, no_api_key_required, enabled, free_tier, free_one_time, rate_limit_rpm)"
        )
        db.execute(
            "INSERT INTO providers VALUES (1,'openrouter','https://provider.invalid/v1','KEY',0,1,1,0,20)"
        )
        db.execute(
            "CREATE TABLE models (provider_id, model_id, context_window, free_tier, capabilities)"
        )
    plan = tmp_path / "targets.json"
    credentials = (
        {"key_file": str(key)}
        if source == "file"
        else {"key_env": "OTHER_KEY" if source == "wrong_env" else "KEY"}
    )
    monkeypatch.setenv("KEY", "fixture-only-key" if source != "missing_env" else "")
    plan.write_text(
        json.dumps(
            {"targets": [{"provider": "openrouter", "model": "m:free", **credentials}]}
        )
    )
    before = database.read_bytes()
    calls = []

    def fetch(opener, endpoint, method, headers):
        calls.append((endpoint, method))
        assert headers["Authorization"] == "Bearer fixture-only-key"
        return {
            "data": [
                {
                    "id": "m:free",
                    "context_length": 32768,
                    "supported_parameters": ["tools", "tool_choice"],
                    "pricing": {"prompt": "0", "completion": "0"},
                }
            ]
        }

    monkeypatch.setattr("tools.verify_newapi_live._request", fetch)
    targets, reports = load_targets(database, plan)
    assert database.read_bytes() == before
    if source in {"wrong_env", "missing_env"}:
        assert not targets and not reports[0]["ok"] and not calls
        return
    assert len(targets) == 1 and reports[0]["ok"]
    assert targets[0].rpm == 4
    assert "fixture-only-key" not in repr(targets) + repr(reports)
    assert calls == [("https://provider.invalid/v1/models", "GET")]

    def denied(*_args):
        raise CatalogError("catalog HTTP status 403")

    monkeypatch.setattr("tools.verify_newapi_live._request", denied)
    targets, reports = load_targets(database, plan)
    assert not targets and not reports[0]["ok"]
    assert database.read_bytes() == before


def test_free_model_accepts_only_active_free_tool_groq_metadata():
    canonical = {"free_tier": 1, "capabilities": "chat,tools", "context_window": 32768}
    assert (
        free_model("groq", "m", [{"id": "m", "active": True}], canonical)["context"]
        == 32768
    )
    with pytest.raises(ValueError):
        free_model("groq", "m", [{"id": "m", "active": False}], canonical)
    with pytest.raises(ValueError):
        free_model("groq", "m", [{"id": "m"}], {**canonical, "free_tier": 0})


def test_mistral_needs_live_tools_and_canonical_free_context():
    model = {
        "id": "codestral-latest",
        "max_context_length": 256000,
        "deprecation": None,
        "capabilities": {"completion_chat": True, "function_calling": True},
    }
    canonical = {"free_tier": 1, "capabilities": "chat,tools", "context_window": 200000}
    assert free_model("mistral", model["id"], [model], canonical)["context"] == 200000
    for change in (
        {"deprecation": "retired"},
        {"capabilities": {"completion_chat": True}},
        {"max_context_length": None},
    ):
        with pytest.raises(ValueError):
            free_model("mistral", model["id"], [{**model, **change}], canonical)
    with pytest.raises(ValueError):
        free_model("mistral", model["id"], [model], {**canonical, "free_tier": 0})


@pytest.mark.parametrize("fault_mode", ["none", "hidden_retry", "missing_usage"])
def test_failover_is_independently_accounted_in_both_directions(
    tmp_path, monkeypatch, fault_mode
):
    routes = [
        Route(
            name,
            name,
            "m",
            "key-" + name,
            "sc." + name,
            index,
            quota_type="rate_limited",
            context_window=32768,
            output_limit=128,
            input_price=0,
            output_price=0,
            stream=True,
            limits={"rpd": 2},
        )
        for index, name in enumerate(("a", "b"), 1)
    ]
    faults, attempts = set(), []

    class Adapter:
        def complete(self, deployment, payload):
            attempts.append(deployment.provider)
            if deployment.provider in faults:
                raise ProviderError(
                    "fixture", error_class="rate_limit", retry_after_seconds=120
                )
            if fault_mode == "hidden_retry":
                attempts.append(deployment.provider)
            return ProviderCompletion(
                {"role": "assistant", "content": "ok"},
                "stop",
                (
                    None
                    if fault_mode == "missing_usage"
                    else {"prompt_tokens": 10, "completion_tokens": 2}
                ),
            )

        def stream(self, deployment, payload):
            result = self.complete(deployment, payload)
            yield json.dumps({"choices": [{"index": 0, "delta": {"content": "ok"}}]})
            yield json.dumps({"choices": [], "usage": result.usage})
            yield "[DONE]"

    monkeypatch.setattr(
        "tools.verify_newapi_live.Runtime.adapter", lambda *_args: Adapter()
    )
    if fault_mode != "none":
        match = (
            "hidden retry" if fault_mode == "hidden_retry" else "final usage is missing"
        )
        with pytest.raises(CatalogError, match=match):
            verify_failover(
                tmp_path,
                "http://127.0.0.1:8002",
                "fixture-relay",
                routes,
                faults,
                attempts,
            )
        assert not faults
        return
    result = verify_failover(
        tmp_path, "http://127.0.0.1:8002", "fixture-relay", routes, faults, attempts
    )
    assert [row["attempt_order"] for row in result] == [["a", "b"], ["b", "a"]]
    assert all(
        row["hidden_retries"] == 0
        and row["cooldown_persisted"]
        and row["failed_reservation_retained"]
        and row["reservation_persisted"]
        for row in result
    )
    assert not faults


def target():
    return Target(
        "groq", "model", "https://provider.invalid/v1", "private-test-value", 32768, 4
    )


def payload(**options):
    return {
        "model": "model",
        "max_tokens": OUTPUT_CAP,
        "messages": [{"role": "user", "content": "ok"}],
        **options,
    }


@contextmanager
def successful_response():
    response = io.BytesIO(b'{"choices":[{"message":{"content":"ok"}}]}')
    response.status = 200
    response.headers = {"Content-Type": "application/json"}
    yield response


def drain(channel):
    frames = []
    while True:
        message = frame_read(channel)
        frames.append(message)
        if message.get("end"):
            return frames


@pytest.mark.parametrize("failover", [False, True])
def test_bridge_budget_prevents_hidden_extra_generation(monkeypatch, failover):
    broker = Broker([target()], failover=failover)
    attempts = []

    def open_request(req, **kwargs):
        attempts.append(json.loads(req.data))
        return successful_response()

    monkeypatch.setattr(broker.opener, "open", open_request)
    left, right = socket.socketpair()
    try:
        for _ in range(3):
            broker.dispatch(left, {"provider": "groq", "payload": payload()})
            frames = drain(right)
        assert len(attempts) == (1 if failover else 2)
        assert frames[0]["status"] == 429
        assert "private-test-value" not in repr(broker.events)
    finally:
        left.close()
        right.close()


@pytest.mark.parametrize(
    "changes", [{"model": "paid"}, {"max_tokens": 129}, {"max_tokens": True}, {"n": 2}]
)
def test_bridge_rejects_nonallowlisted_or_oversized_generation(changes):
    broker = Broker([target()])
    left, right = socket.socketpair()
    try:
        with pytest.raises(ValueError):
            broker.dispatch(left, {"provider": "groq", "payload": payload(**changes)})
        assert broker.calls["groq"] == 0
    finally:
        left.close()
        right.close()


@pytest.mark.parametrize("status", [401, 403, 429])
def test_auth_and_quota_failure_are_not_retried(monkeypatch, status):
    broker = Broker([target()])
    attempts = []

    def denied(req, **kwargs):
        attempts.append(req)
        raise urllib.error.HTTPError(req.full_url, status, "rejected", {}, io.BytesIO())

    monkeypatch.setattr(broker.opener, "open", denied)
    left, right = socket.socketpair()
    try:
        for _ in range(2):
            broker.dispatch(left, {"provider": "groq", "payload": payload()})
            drain(right)
        assert len(attempts) == 1
        assert "groq" in broker.blocked
    finally:
        left.close()
        right.close()


def test_bridge_rejects_credential_echo_and_blocks_later_calls(monkeypatch):
    broker = Broker([target()])

    @contextmanager
    def echoed_response():
        response = io.BytesIO(b"private-test-value")
        response.status = 200
        response.headers = {"Content-Type": "application/json"}
        yield response

    monkeypatch.setattr(
        broker.opener, "open", lambda *_args, **_kwargs: echoed_response()
    )
    left, right = socket.socketpair()
    try:
        broker.dispatch(left, {"provider": "groq", "payload": payload()})
        frames = drain(right)
        assert "private-test-value" not in repr(frames)
        assert "groq" in broker.blocked
    finally:
        left.close()
        right.close()


def test_tool_loop_has_a_distinct_three_request_budget(monkeypatch):
    broker = Broker([target()], tool_loop=True)
    monkeypatch.setattr(
        broker.opener, "open", lambda *_args, **_kwargs: successful_response()
    )
    left, right = socket.socketpair()
    try:
        for _ in range(4):
            broker.dispatch(
                left, {"provider": "groq", "payload": payload(max_tokens=512)}
            )
            frames = drain(right)
        assert broker.calls["groq"] == 3
        assert frames[0]["status"] == 429
    finally:
        left.close()
        right.close()


def test_transport_failure_returns_a_bounded_error_and_blocks_target(monkeypatch):
    broker = Broker([target()])

    def unavailable(*_args, **_kwargs):
        raise urllib.error.URLError("private upstream diagnostic")

    monkeypatch.setattr(broker.opener, "open", unavailable)
    left, right = socket.socketpair()
    try:
        broker.dispatch(left, {"provider": "groq", "payload": payload()})
        frames = drain(right)
        assert frames[0]["status"] == 502
        assert "private upstream diagnostic" not in repr(frames)
        assert "groq" in broker.blocked
    finally:
        left.close()
        right.close()


def test_frame_roundtrip_is_bounded():
    left, right = socket.socketpair()
    try:
        frame_send(left, {"value": "ok"})
        assert frame_read(right) == {"value": "ok"}
        with pytest.raises(ValueError, match="exceeds"):
            frame_send(left, {"value": "x" * (1024 * 1024)})
    finally:
        left.close()
        right.close()
