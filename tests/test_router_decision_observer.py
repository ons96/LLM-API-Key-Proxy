from types import SimpleNamespace
from unittest.mock import AsyncMock
import json

import pytest

from src.proxy_app.router_core import ProviderCandidate, ProviderMetrics, RouterCore
from src.proxy_app.routing.tier_classifier import Tier

from src.proxy_app.routing.decision_observer import (
    build_decision_record,
    cutover_enabled,
    log_decision,
)


def _result():
    return SimpleNamespace(
        chain=[
            SimpleNamespace(
                provider="free-provider",
                model="fast-model",
                priority=1,
                reason="telemetry",
                e_time_ms=120,
                confidence=0.91,
                source="ewma",
            ),
            SimpleNamespace(
                provider="backup-provider",
                model="backup-model",
                priority=2,
                reason="chain",
            ),
        ],
        escalation=[
            SimpleNamespace(
                provider="deep-provider",
                model="deep-model",
                priority=3,
                reason="escalation",
            )
        ],
        static_fallback=False,
    )


def test_decision_record_is_bounded_and_does_not_retain_session_value():
    record = build_decision_record(
        _result(),
        "router/chat-fast",
        2,
        request={"session_id": "private-session-value", "phase": "implement"},
        request_id="request-123",
        env={"ROUTER_CUTOVER_ENABLED": "0"},
    )

    assert record["mode"] == "dry_run"
    assert record["cutover_enabled"] is False
    assert record["selected"]["provider"] == "free-provider"
    assert record["reason"] == "telemetry"
    assert record["phase"] == "implement"
    assert record["session"]
    assert "private-session-value" not in str(record)
    assert len(record["chain"]) == 2
    assert len(record["escalation"]) == 1


def test_decision_record_normalizes_non_finite_candidate_metadata():
    result = _result()
    result.chain[0].priority = float("nan")
    result.chain[0].confidence = float("inf")
    result.chain[0].e_time_ms = float("-inf")

    record = build_decision_record(result, "router/chat-fast", 1)

    assert record["selected"]["priority"] is None
    assert record["selected"]["confidence"] is None
    assert record["selected"]["e_time_ms"] is None
    json.dumps(record, allow_nan=False)


def test_cutover_flag_is_explicitly_default_off():
    assert cutover_enabled({}) is False
    assert cutover_enabled({"ROUTER_CUTOVER_ENABLED": "false"}) is False
    assert cutover_enabled({"ROUTER_CUTOVER_ENABLED": "on"}) is True


def test_logging_can_be_disabled_and_cutover_mode_is_recorded(caplog):
    with caplog.at_level("INFO", logger="src.proxy_app.routing.decision_observer"):
        assert (
            log_decision(
                _result(),
                "router/chat-fast",
                1,
                env={"ROUTER_DRY_RUN_LOG": "0"},
            )
            is None
        )
        record = log_decision(
            _result(),
            "router/chat-fast",
            1,
            env={"ROUTER_CUTOVER_ENABLED": "1", "ROUTER_DRY_RUN_LOG": "1"},
        )

    assert record is not None
    assert record["mode"] == "cutover"
    assert record["cutover_enabled"] is True
    assert any("router_decision" in message for message in caplog.messages)


@pytest.mark.parametrize(
    ("cutover", "expected_provider"),
    [(False, "first"), (True, "second")],
)
@pytest.mark.asyncio
async def test_router_boundary_observes_and_gates_reordering(
    monkeypatch, cutover, expected_provider
):
    """The observer is non-blocking and the flag controls candidate ordering."""
    monkeypatch.setenv("ROUTER_DRY_RUN", "0")
    if cutover:
        monkeypatch.setenv("ROUTER_CUTOVER_ENABLED", "1")
    else:
        monkeypatch.delenv("ROUTER_CUTOVER_ENABLED", raising=False)

    router = RouterCore("/nonexistent/config.yaml")
    router.free_only_mode = False
    router.config = {"providers": {}}
    candidates = [
        ProviderCandidate("first", "first-model", priority=1),
        ProviderCandidate("second", "second-model", priority=2),
    ]
    router._get_candidates = AsyncMock(return_value=candidates)
    router._get_metrics = lambda _provider, _model: ProviderMetrics()
    router._prompt_cache_lookup = lambda _prefix: None
    router._execute_single_candidate = AsyncMock(return_value={"choices": []})

    class FakeSelector:
        def select(self, *_args, **_kwargs):
            return SimpleNamespace(
                chain=[
                    SimpleNamespace(
                        provider="second",
                        model="second-model",
                        priority=1,
                        reason="smart-chain",
                    ),
                    SimpleNamespace(
                        provider="first",
                        model="first-model",
                        priority=2,
                        reason="fallback",
                    ),
                ],
                escalation=[],
                static_fallback=False,
            )

    observed = []

    def observer(*_args, **kwargs):
        observed.append(kwargs)
        raise RuntimeError("observation must not affect routing")

    router._smart_fx = (
        lambda _request: SimpleNamespace(task_class="chat"),
        lambda _features, header_override=None: Tier.T1,
        lambda _result, _model: "smart-chain",
    )
    router._smart_selectors = {}
    router._smart_default = FakeSelector()
    router._smart_output_est = SimpleNamespace(estimate=lambda _task: 200)
    router._smart_log_decision = observer

    response = await router._route_request_inner(
        {
            "model": "router/test",
            "messages": [{"role": "user", "content": "hello"}],
            "session_id": "private-session",
        },
        "request-1",
    )

    assert response == {"choices": []}
    assert observed[0]["request"]["session_id"] == "private-session"
    assert router._execute_single_candidate.await_args.args[0].provider == expected_provider
