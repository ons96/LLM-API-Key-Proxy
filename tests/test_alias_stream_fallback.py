"""
Alias streaming fallback tests.

Regression tests for the alias chain: for streaming requests, route_request
returns an async generator that has not started yet, so provider errors
surface during iteration — after the alias loop has returned. The
alias stream wrapper must still fail over to the next candidate.
"""

import inspect

import pytest
import yaml

from src.proxy_app.router_core import RouterCore


@pytest.fixture
def alias_router(tmp_path):
    config = {
        "free_only_mode": True,
        "routing": {
            "default_cooldown_seconds": 1,
            "rate_limit_cooldown_seconds": 1,
        },
    }
    config_file = tmp_path / "router_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)

    router = RouterCore(str(config_file))
    router.aliases = {
        "coding": {
            "description": "test alias",
            "candidates": [
                {"provider": "provider_a", "model": "model-a"},
                {"provider": "provider_b", "model": "model-b"},
            ],
        }
    }
    return router


async def _failing_stream():
    yield {"delta": "partial-a"}
    raise RuntimeError("provider A exploded mid-stream")


async def _working_stream():
    yield {"delta": "hello-from-b"}
    yield {"delta": "-done"}


def _make_route_request(call_log):
    async def fake_route_request(request, request_id):
        model = request.get("model", "")
        call_log.append(model)
        if model.startswith("provider_a/"):
            return _failing_stream()
        if model.startswith("provider_b/"):
            return _working_stream()
        raise AssertionError(f"unexpected model routed: {model}")

    return fake_route_request


class TestAliasStreamingFallback:
    @pytest.mark.asyncio
    async def test_stream_fails_over_to_next_alias_candidate(self, alias_router):
        call_log = []
        request = {"model": "coding", "stream": True, "messages": [{"role": "user", "content": "hi"}]}

        # Iterate inside the patch scope: the wrapper resolves later
        # candidates lazily, after _route_request_inner has returned.
        with patch_route_request(alias_router, _make_route_request(call_log)):
            result = await alias_router._route_request_inner(request, "req-stream")
            assert inspect.isasyncgen(result), "streaming alias must return an async generator"
            chunks = [c async for c in result]

        # Candidate A yields one chunk before dying; the wrapper forwards it,
        # then fails over to B — same partial-output semantics as
        # _stream_with_fallback for the virtual-model path. Before the fix,
        # the mid-stream error propagated to the client and B was never tried.
        assert chunks == [
            {"delta": "partial-a"},
            {"delta": "hello-from-b"},
            {"delta": "-done"},
        ]
        assert call_log == ["provider_a/model-a", "provider_b/model-b"], (
            "must try candidate A, then fail over to candidate B"
        )

    @pytest.mark.asyncio
    async def test_stream_raises_when_all_alias_candidates_fail(self, alias_router):
        call_log = []

        async def all_failing(request, request_id):
            call_log.append(request.get("model"))

            async def stream():
                raise RuntimeError("dead provider")
                yield  # noqa: unreachable — makes this an async generator

            return stream()

        with patch_route_request(alias_router, all_failing):
            result = await alias_router._route_request_inner(
                {"model": "coding", "stream": True}, "req-all-fail"
            )
            with pytest.raises(RuntimeError, match="dead provider"):
                _ = [c async for c in result]
        assert len(call_log) == 2, "both alias candidates must be tried"

    @pytest.mark.asyncio
    async def test_non_streaming_alias_unchanged(self, alias_router):
        """Non-streaming alias resolution returns the first successful dict."""
        call_log = []

        async def fake_route_request(request, request_id):
            call_log.append(request.get("model"))
            return {"content": "ok-from-a"}

        with patch_route_request(alias_router, fake_route_request):
            result = await alias_router._route_request_inner({"model": "coding"}, "req-nonstream")

        assert result == {"content": "ok-from-a"}
        assert call_log == ["provider_a/model-a"]


def patch_route_request(router, fake):
    from unittest.mock import patch

    return patch.object(router, "route_request", side_effect=fake)
