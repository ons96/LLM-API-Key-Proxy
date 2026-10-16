"""Routing engine correctness suite (LLM-API-Key-Proxy issue #265).

Issue #265 asks for automated coverage of provider failover, rate limiting,
model discovery and the streaming vs non-streaming paths. Most of that already
exists (see `test_fallback_logic.py`, `test_integration_router.py`,
`test_free_only_mode.py`); this module covers the parts that did not:

- streaming and non-streaming must try the same candidates in the same order,
  and must forward the same resolved request, so a client cannot get a
  different (or worse) provider just by asking for a stream;
- the failure surface must stay bounded: a total outage answers with the
  router's own 503, and the model-discovery payload must never carry
  credentials.
"""

import json
import re
from typing import Any, Dict, List
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from src.proxy_app.router_core import RouterCore, ProviderCandidate
from tests.fixtures.provider_mocks import MockProviderResponse, TimeoutError
from tests.fixtures.scenarios import create_request, ensure_providers_enabled


@pytest.fixture
def mock_router_config(tmp_path):
    """A small deterministic chain: one model, three providers, manual order."""
    import yaml

    config = ensure_providers_enabled(
        {
            "free_only_mode": True,
            "router_models": {
                "coding-smart": {
                    "description": "Best coding models with fallback chain",
                    "candidates": [
                        {"provider": "provider_a", "model": "gpt-4o", "priority": 1, "free_tier_only": True},
                        {"provider": "provider_b", "model": "gpt-4o", "priority": 2, "free_tier_only": True},
                        {"provider": "provider_c", "model": "gpt-4o", "priority": 3, "free_tier_only": True},
                    ],
                    "auto_order": False,
                }
            },
            "routing": {"default_cooldown_seconds": 60, "rate_limit_cooldown_seconds": 300},
        }
    )
    # Give the provider block a free_tier_models list so discovery has
    # provider-backed entries to report alongside the virtual model.
    for provider in ("provider_a", "provider_b", "provider_c"):
        config["providers"][provider]["free_tier_models"] = ["gpt-4o"]

    config_file = tmp_path / "router_config.yaml"
    with open(config_file, "w") as handle:
        yaml.dump(config, handle)
    return str(config_file)


@pytest.fixture
def make_router(mock_router_config):
    """Factory for fresh routers (cooldowns are per-instance state)."""
    return lambda: RouterCore(mock_router_config)


def _record(attempt_log: List[Dict[str, Any]], candidate: ProviderCandidate, request: Dict[str, Any]) -> None:
    attempt_log.append(
        {
            "provider": candidate.provider,
            "model": candidate.model,
            "forwarded_model": request.get("model"),
            "stream_flag": bool(request.get("stream")),
            "message_count": len(request.get("messages", [])),
        }
    )


class TestStreamingParity:
    """The transport mode must not change the routing decision."""

    @pytest.mark.asyncio
    async def test_stream_and_non_stream_try_same_candidates_in_same_order(self, make_router):
        failing = {"provider_a", "provider_b"}
        non_stream_attempts: List[Dict[str, Any]] = []
        stream_attempts: List[Dict[str, Any]] = []

        async def fake_execute(candidate, request, request_id):
            _record(non_stream_attempts, candidate, request)
            if candidate.provider in failing:
                raise TimeoutError()
            return MockProviderResponse(model=candidate.model, content="ok")

        async def fake_execute_stream(candidate, request, request_id):
            _record(stream_attempts, candidate, request)
            if candidate.provider in failing:
                raise TimeoutError()
            yield {"choices": [{"delta": {"content": "ok"}}]}

        router = make_router()
        with patch.object(router, "_execute_single_candidate", side_effect=fake_execute):
            plain_result = await router.route_request(
                create_request(model="coding-smart", stream=False), request_id="parity-plain"
            )

        # Fresh instance so the cooldowns recorded by the first run cannot
        # silently remove candidates from the second run's chain.
        stream_router = make_router()
        with patch.object(stream_router, "_execute_single_candidate_stream", side_effect=fake_execute_stream):
            stream_generator = await stream_router.route_request(
                create_request(model="coding-smart", stream=True), request_id="parity-stream"
            )
            chunks = [chunk async for chunk in stream_generator]

        assert plain_result is not None
        assert chunks, "streaming path must yield at least one chunk"

        assert [a["provider"] for a in non_stream_attempts] == ["provider_a", "provider_b", "provider_c"]
        assert [a["provider"] for a in stream_attempts] == [a["provider"] for a in non_stream_attempts], (
            "streaming and non-streaming must fall back through the same chain in the same order"
        )
        assert [a["model"] for a in stream_attempts] == [a["model"] for a in non_stream_attempts]

        # The request handed to the executor is identical in both modes: the
        # requested model and the message history are forwarded unchanged, so
        # streaming cannot see a different payload than non-streaming.
        assert len({a["forwarded_model"] for a in non_stream_attempts}) == 1
        assert [a["forwarded_model"] for a in stream_attempts] == [a["forwarded_model"] for a in non_stream_attempts]
        assert [a["message_count"] for a in stream_attempts] == [a["message_count"] for a in non_stream_attempts]

    @pytest.mark.asyncio
    async def test_stream_flag_is_the_only_difference_between_the_two_paths(self, make_router):
        """Only `stream` may differ in the forwarded request."""
        captured: Dict[str, List[Dict[str, Any]]] = {"plain": [], "stream": []}

        async def fake_execute(candidate, request, request_id):
            captured["plain"].append(dict(request))
            return MockProviderResponse(model=candidate.model, content="ok")

        async def fake_execute_stream(candidate, request, request_id):
            captured["stream"].append(dict(request))
            yield {"choices": [{"delta": {"content": "ok"}}]}

        plain_router = make_router()
        with patch.object(plain_router, "_execute_single_candidate", side_effect=fake_execute):
            await plain_router.route_request(create_request(model="coding-smart", stream=False), request_id="diff-plain")

        stream_router = make_router()
        with patch.object(stream_router, "_execute_single_candidate_stream", side_effect=fake_execute_stream):
            generator = await stream_router.route_request(
                create_request(model="coding-smart", stream=True), request_id="diff-stream"
            )
            _ = [chunk async for chunk in generator]

        plain_request = captured["plain"][0]
        stream_request = captured["stream"][0]

        stream_request_without_flag = dict(stream_request)
        stream_request_without_flag.pop("stream", None)
        plain_request_without_flag = dict(plain_request)
        plain_request_without_flag.pop("stream", None)

        assert stream_request.get("stream") is True
        assert plain_request.get("stream") in (False, None)
        assert stream_request_without_flag == plain_request_without_flag


class TestBoundedFailureSurface:
    """A total outage must answer with the router's own bounded error."""

    @pytest.mark.asyncio
    async def test_all_candidates_cooling_down_returns_bounded_503(self, make_router):
        router = make_router()
        for provider in ("provider_a", "provider_b", "provider_c"):
            router._get_metrics(provider, "gpt-4o").set_cooldown(300)

        with pytest.raises(HTTPException) as exc_info:
            await router.route_request(create_request(model="coding-smart"), request_id="all-cooldown")

        assert exc_info.value.status_code == 503
        assert exc_info.value.detail == "No healthy providers available for the request"
        # The router's own message, not provider text.
        assert len(exc_info.value.detail) < 200

    @pytest.mark.asyncio
    async def test_unknown_model_is_a_bounded_400(self, make_router):
        router = make_router()

        with pytest.raises(HTTPException) as exc_info:
            await router.route_request(create_request(model="not-a-real-model"), request_id="unknown-model")

        assert exc_info.value.status_code == 400
        assert "not-a-real-model" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_failover_does_not_swallow_the_success(self, make_router):
        """Earlier failures must not hide a later success (issue #265)."""
        attempts: List[str] = []

        async def fake_execute(candidate, request, request_id):
            attempts.append(candidate.provider)
            if candidate.provider != "provider_c":
                raise TimeoutError()
            return MockProviderResponse(model=candidate.model, content="recovered")

        router = make_router()
        with patch.object(router, "_execute_single_candidate", side_effect=fake_execute):
            result = await router.route_request(create_request(model="coding-smart"), request_id="recover")

        assert attempts == ["provider_a", "provider_b", "provider_c"]
        assert result is not None


class TestModelDiscoveryContract:
    """/v1/models data must be complete, unique and credential-free."""

    @pytest.mark.asyncio
    async def test_model_list_has_virtual_and_provider_entries(self, make_router):
        models = make_router().get_model_list()
        ids = [entry["id"] for entry in models]

        assert "coding-smart" in ids, "the configured virtual model must be discoverable"
        assert "provider_c/gpt-4o" in ids, "provider-backed free models must be discoverable"

    def test_model_ids_are_unique_and_strings(self, make_router):
        models = make_router().get_model_list()
        ids = [entry["id"] for entry in models]

        assert all(isinstance(model_id, str) and model_id for model_id in ids)
        assert len(ids) == len(set(ids)), "duplicate model ids would make /v1/models ambiguous"

    def test_model_list_contains_no_credentials(self, make_router):
        payload = json.dumps(make_router().get_model_list())

        forbidden = [
            r"sk-[A-Za-z0-9]{8,}",
            r"ghp_[A-Za-z0-9]{8,}",
            r"Bearer\s+[A-Za-z0-9]",
            r"api[_-]?key[\"']?\s*[:=]",
            r"https?://[^/\s]*:[^@\s]+@",  # credentials embedded in a URL
        ]
        for pattern in forbidden:
            assert re.search(pattern, payload, re.IGNORECASE) is None, (
                f"model discovery leaked something matching {pattern!r}"
            )

    def test_disabled_providers_are_not_advertised(self, tmp_path, mock_router_config):
        import yaml

        with open(mock_router_config) as handle:
            config = yaml.safe_load(handle)
        config["providers"]["provider_b"]["enabled"] = False

        edited = tmp_path / "router_config_disabled.yaml"
        with open(edited, "w") as handle:
            yaml.dump(config, handle)

        ids = [entry["id"] for entry in RouterCore(str(edited)).get_model_list()]

        assert "provider_b/gpt-4o" not in ids
        assert "provider_a/gpt-4o" in ids
