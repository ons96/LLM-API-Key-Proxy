"""
Fault-injection harness for smart-router validation (#486).

Covers the two untested smoke-matrix cases from #481:

  case 6: first chain candidate returns 429 -> next candidate serves (200)
  case 7: first chain candidate fails 422 (capability) -> escalation to a
          stronger tier candidate serves

These drive the REAL RouterCore fallback loop (not just the pure on_error()
unit function): candidates are fixed via a patched _get_candidates, then
HTTP-status failures are injected into _execute_single_candidate, and the test
asserts which candidate actually served.

Also carries regression tests for the #486 AC1 config fix:
  - `cerebras/glm-5` (a model cerebras does not serve) must not appear in any
    virtual-model chain (it caused HTTP 404 during smoke case 3 execution)
  - vision-capable chains must contain a model whose capabilities metadata
    marks it VISION so the request can actually return 200

Hermetic: no network, no litellm (execution mocked). Pure router-loop tests.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from src.proxy_app.router_core import ProviderCandidate, RouterCore  # noqa: E402
from tests.fixtures.provider_mocks import MockProviderResponse  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_router_config(tmp_path):
    """RouterCore loads this config, but candidates come from _get_candidates."""
    import yaml

    config = {
        "free_only_mode": False,
        "router_models": {
            "coding-smart": {
                "description": "fault-injection test chain",
                "candidates": [
                    {
                        "provider": "fast_provider",
                        "model": "llama-3.1-8b-instant",
                        "priority": 1,
                        "free_tier_only": True,
                    },
                    {
                        "provider": "strong_provider",
                        "model": "gpt-oss-120b",
                        "priority": 2,
                        "free_tier_only": True,
                    },
                    {
                        "provider": "fallback_provider",
                        "model": "claude-3.5-sonnet",
                        "priority": 3,
                        "free_tier_only": True,
                    },
                ],
            }
        },
        "routing": {
            "default_cooldown_seconds": 60,
            "rate_limit_cooldown_seconds": 300,
        },
    }
    config_file = tmp_path / "router_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)
    return str(config_file)


@pytest.fixture
def router(mock_router_config):
    """RouterCore instance for fault-injection runs."""
    return RouterCore(mock_router_config)


def _candidates():
    """Fixed chain: fast -> strong -> fallback. Mirrors the mock config."""
    return [
        ProviderCandidate(provider="fast_provider", model="llama-3.1-8b-instant", priority=1),
        ProviderCandidate(provider="strong_provider", model="gpt-oss-120b", priority=2),
        ProviderCandidate(provider="fallback_provider", model="claude-3.5-sonnet", priority=3),
    ]


def _request(model="coding-smart"):
    return {
        "model": model,
        "messages": [{"role": "user", "content": "test fault injection"}],
        "max_tokens": 32,
    }


async def _inject(router, request, injections):
    """Run route_request with fixed candidates + a fault table.

    injections: dict keyed by provider name -> either an exception instance
    to raise, or a MockProviderResponse to return. Providers absent from the
    table return a default success (so only the targeted candidate fails).
    """
    fixed = _candidates()

    async def fixed_candidates(model_id, requirements):
        return list(fixed)

    async def injected_execution(candidate, req, request_id):
        fault = injections.get(candidate.provider)
        if isinstance(fault, BaseException):
            raise fault
        if fault is not None:
            return fault
        return MockProviderResponse(model=f"{candidate.provider}/{candidate.model}")

    with patch.object(router, "_get_candidates", side_effect=fixed_candidates), patch.object(
        router, "_execute_single_candidate", side_effect=injected_execution
    ):
        return await router.route_request(request, request_id="fault-inject")


def _served_model(result):
    """Extract the model that actually served from a router response."""
    if isinstance(result, dict):
        return result.get("model", "")
    return getattr(result, "model", "")


# ---------------------------------------------------------------------------
# Case 6: injected 429 -> next candidate serves
# ---------------------------------------------------------------------------


class TestCase6RateLimitFallback:
    """Smoke case 6: first chain candidate returns 429 -> next candidate serves."""

    @pytest.mark.asyncio
    async def test_429_on_first_candidate_falls_to_second(self, router):
        result = await _inject(
            router,
            _request(),
            {
                "fast_provider": HTTPException(status_code=429, detail="rate limit hit"),
                "strong_provider": MockProviderResponse(
                    model="strong_provider/gpt-oss-120b", content="served by #2"
                ),
            },
        )
        assert _served_model(result) == "strong_provider/gpt-oss-120b", (
            f"expected 2nd candidate to serve, got {_served_model(result)!r}"
        )

    @pytest.mark.asyncio
    async def test_429_retry_after_header_used(self, router):
        """429 with a Retry-After hint must not abort the loop (cooldown only)."""
        exc = HTTPException(status_code=429, detail="quota exceeded")
        exc.headers = {"retry-after": "5"}
        result = await _inject(
            router,
            _request(),
            {
                "fast_provider": exc,
                "strong_provider": MockProviderResponse(
                    model="strong_provider/gpt-oss-120b", content="ok"
                ),
            },
        )
        assert _served_model(result) == "strong_provider/gpt-oss-120b"

    @pytest.mark.asyncio
    async def test_all_429_raises_last_error(self, router):
        """Every candidate 429 -> the loop must surface an error, not hang."""
        with pytest.raises(HTTPException):
            await _inject(
                router,
                _request(),
                {
                    "fast_provider": HTTPException(status_code=429, detail="rl"),
                    "strong_provider": HTTPException(status_code=429, detail="rl"),
                    "fallback_provider": HTTPException(status_code=429, detail="rl"),
                },
            )


# ---------------------------------------------------------------------------
# Case 7: injected 422 capability failure -> escalation to stronger tier
# ---------------------------------------------------------------------------


class TestCase7CapabilityEscalation:
    """Smoke case 7: 422 capability failure -> stronger tier serves."""

    @pytest.mark.asyncio
    async def test_422_capability_failure_escalates(self, router):
        """First candidate rejects the modality (422) -> stronger candidate serves."""
        result = await _inject(
            router,
            _request(),
            {
                "fast_provider": HTTPException(
                    status_code=422,
                    detail="model does not support vision capability",
                ),
                "strong_provider": MockProviderResponse(
                    model="strong_provider/gpt-oss-120b", content="escalated ok"
                ),
            },
        )
        assert _served_model(result) == "strong_provider/gpt-oss-120b", (
            f"expected escalation to stronger tier, got {_served_model(result)!r}"
        )

    @pytest.mark.asyncio
    async def test_400_unsupported_also_escalates(self, router):
        """400-class capability errors (unsupported modality) also escalate."""
        result = await _inject(
            router,
            _request(),
            {
                "fast_provider": HTTPException(
                    status_code=400, detail="bad request: unsupported input modality"
                ),
                "strong_provider": MockProviderResponse(
                    model="strong_provider/gpt-oss-120b", content="ok"
                ),
            },
        )
        assert _served_model(result) == "strong_provider/gpt-oss-120b"

    @pytest.mark.asyncio
    async def test_404_model_not_found_falls_through(self, router):
        """404 (model missing on upstream) must try the next candidate, not fail."""
        result = await _inject(
            router,
            _request(),
            {
                "fast_provider": HTTPException(
                    status_code=404, detail="model not found on upstream"
                ),
                "strong_provider": MockProviderResponse(
                    model="strong_provider/gpt-oss-120b", content="ok"
                ),
            },
        )
        assert _served_model(result) == "strong_provider/gpt-oss-120b"

    @pytest.mark.asyncio
    async def test_all_capability_failures_exhaust(self, router):
        """All candidates capability-fail -> raise (no silent empty 200)."""
        with pytest.raises(HTTPException):
            await _inject(
                router,
                _request(),
                {
                    "fast_provider": HTTPException(status_code=422, detail="nope"),
                    "strong_provider": HTTPException(status_code=422, detail="nope"),
                    "fallback_provider": HTTPException(status_code=422, detail="nope"),
                },
            )


# ---------------------------------------------------------------------------
# AC1 config regression: cerebras/glm-5 removed + vision candidate present
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parents[1]
VIRTUAL_MODELS = REPO_ROOT / "config" / "virtual_models.yaml"
PROVIDERS_DB = REPO_ROOT / "config" / "providers_database.yaml"


class TestVisionCandidateConfig:
    """#486 AC1: vision-capable chains must have a working vision model."""

    def test_no_cerebras_glm5_in_any_chain(self):
        """cerebras/glm-5 is not served by cerebras (caused 404 in smoke #481)."""
        text = VIRTUAL_MODELS.read_text(encoding="utf-8")
        assert "provider: cerebras\n      model: glm-5" not in text

    def test_cerebras_entries_use_served_models(self):
        """cerebras chain entries must be models the provider actually serves."""
        import yaml

        data = yaml.safe_load(VIRTUAL_MODELS.read_text(encoding="utf-8"))
        served = {
            "llama-3.3-70b",
            "llama3.1-8b",
            "qwen-3-32b",
            "qwen-3-235b-a22b-instruct-2507",
            "gpt-oss-120b",
            "zai-glm-4.7",
            "zai-glm-4.6",
        }
        bad = []
        for vm, spec in (data.get("virtual_models") or {}).items():
            for entry in spec.get("fallback_chain") or []:
                if entry.get("provider") == "cerebras" and entry.get("model") not in served:
                    bad.append((vm, entry.get("model")))
        assert not bad, f"cerebras entries with unserved models: {bad}"

    def test_vision_capable_chains_have_vision_candidate(self):
        """Per #486 AC1: a vision request needs a reachable vision-capable model."""
        import yaml

        from proxy_app.routing.chain_selector import load_capabilities
        from proxy_app.routing.request_features import Capabilities

        data = yaml.safe_load(VIRTUAL_MODELS.read_text(encoding="utf-8"))
        caps = load_capabilities(str(PROVIDERS_DB), None)
        missing = []
        for vm, spec in (data.get("virtual_models") or {}).items():
            chain = spec.get("fallback_chain") or []
            if not chain:
                continue
            if not any(
                caps.get(f"{e['provider']}/{e['model']}", Capabilities.TEXT) & Capabilities.VISION
                for e in chain
            ):
                missing.append(vm)
        # coding-smart / coding-fast must carry vision after the #486 fix.
        assert "coding-smart" not in missing, (
            "coding-smart chain lost its vision-capable candidate"
        )
        assert "coding-fast" not in missing, (
            "coding-fast chain lost its vision-capable candidate"
        )

    def test_providers_db_marks_maverick_vision(self):
        """The chosen vision candidate must be vision-tagged in the DB."""
        import yaml

        data = yaml.safe_load(PROVIDERS_DB.read_text(encoding="utf-8"))
        for prov in data.get("providers", []):
            if prov.get("id") != "groq":
                continue
            models = {
                (m.get("id"), tuple(m.get("capabilities") or []))
                for m in prov.get("free_models") or []
            }
            assert any(
                "meta-llama/llama-4-maverick-17b-128e-instruct" in mid and "vision" in caps
                for mid, caps in models
            ), "groq llama-4-maverick must be vision-tagged in providers_database.yaml"
            return
        pytest.fail("groq provider missing from providers_database.yaml")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
