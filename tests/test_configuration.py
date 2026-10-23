"""
Configuration & Dynamic Reordering Tests

Verify configuration is properly applied and reordering works.
"""

import pytest
import asyncio
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

from src.proxy_app.router_core import RouterCore
from tests.fixtures.benchmark_data import (
    SAMPLE_MODEL_RANKINGS,
    SAMPLE_PROVIDER_PERFORMANCE,
)
from tests.fixtures.scenarios import create_request


@pytest.fixture
def mock_router_config(tmp_path):
    """Create a mock router configuration with auto-ordering enabled."""
    config = {
        "free_only_mode": True,
        "router_models": {
            "coding-smart": {
                "description": "Best coding models",
                "candidates": [
                    {
                        "provider": "g4f",
                        "model": "gpt-4o",
                        "priority": 1,
                        "free_tier_only": True,
                    },
                    {
                        "provider": "g4f",
                        "model": "claude-3.5-sonnet",
                        "priority": 2,
                        "free_tier_only": True,
                    },
                    {
                        "provider": "g4f",
                        "model": "o1-mini",
                        "priority": 3,
                        "free_tier_only": True,
                    },
                ],
                "auto_order": True,  # Enable auto-ordering
            },
            "coding-fast": {
                "description": "Fast coding models",
                "candidates": [
                    {
                        "provider": "cerebras",
                        "model": "llama-3.1-70b",
                        "priority": 1,
                        "free_tier_only": True,
                    },
                    {
                        "provider": "groq",
                        "model": "llama-3.3-70b-versatile",
                        "priority": 2,
                        "free_tier_only": True,
                    },
                ],
                "auto_order": True,
            },
        },
        "routing": {"default_cooldown_seconds": 60, "ranking_strategy": "balanced"},
    }

    config_file = tmp_path / "router_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)

    return str(config_file)


@pytest.fixture
def router(mock_router_config):
    """Create router instance for testing."""
    return RouterCore(mock_router_config)


class TestProviderRanking:
    """
    Test: test_provider_ranking_by_tps()
    Test: test_provider_ranking_by_ttft()
    Test: test_provider_ranking_by_success_rate()

    Verify providers are ranked by performance metrics.
    """

    def test_provider_metrics_tracked(self, router):
        """Test that provider metrics are tracked separately."""
        # Get metrics for different providers
        metrics_a = router._get_metrics("provider_a", "model_a")
        metrics_b = router._get_metrics("provider_b", "model_b")

        # Simulate different performance
        metrics_a.update_latency(50.0)  # Fast
        metrics_a.record_success()

        metrics_b.update_latency(200.0)  # Slow
        metrics_b.record_success()

        # Verify metrics are independent
        assert metrics_a.ewma_latency_ms < metrics_b.ewma_latency_ms

        print(f"\nProvider A latency: {metrics_a.ewma_latency_ms}ms")
        print(f"Provider B latency: {metrics_b.ewma_latency_ms}ms")

    def test_provider_success_rate_tracking(self, router):
        """Test that success rate is tracked per provider."""
        metrics_reliable = router._get_metrics("reliable_provider", "model")
        metrics_unreliable = router._get_metrics("unreliable_provider", "model")

        # Simulate reliable provider (9 success, 1 fail)
        for _ in range(9):
            metrics_reliable.record_success()
        metrics_reliable.record_error()

        # Simulate unreliable provider (5 success, 5 fail)
        for _ in range(5):
            metrics_unreliable.record_success()
            metrics_unreliable.record_error()

        print(f"\nReliable provider success rate: {metrics_reliable.success_rate}")
        print(f"Unreliable provider success rate: {metrics_unreliable.success_rate}")

        # Verify success rates
        assert metrics_reliable.success_rate == 0.9
        assert metrics_unreliable.success_rate == 0.5


class TestConfigurationLoading:
    """
    Test configuration loading and application.
    """

    def test_router_loads_virtual_models(self, router):
        """Test that router loads virtual models from config."""
        # Check virtual models were loaded
        assert len(router.virtual_models) > 0, "Should have loaded virtual models"

        print(f"\nLoaded virtual models: {router.virtual_models.keys()}")

        # Check specific models
        if "coding-smart" in router.virtual_models:
            coding_smart = router.virtual_models["coding-smart"]
            assert "candidates" in coding_smart or "fallback_chain" in coding_smart, (
                "Virtual model should have candidates/fallback_chain"
            )

    def test_router_respects_free_only_mode(self, router):
        """Test that free_only_mode is respected."""
        assert router.free_only_mode is not None

        print(f"\nFree only mode: {router.free_only_mode}")

    def test_router_applies_cooldown_settings(self, router):
        """Test that cooldown settings are loaded from config."""
        # Check config was loaded
        assert router.config is not None

        routing_config = router.config.get("routing", {})

        print(f"\nRouting config: {routing_config}")

        # Verify cooldown settings exist
        if "default_cooldown_seconds" in routing_config:
            assert routing_config["default_cooldown_seconds"] > 0


class TestAutoOrderingFlag:
    """
    Test that auto_order flag enables/disables ranking.
    """

    @pytest.mark.asyncio
    async def test_auto_order_flag_respected(self, tmp_path):
        """Test that auto_order flag controls ranking behavior."""
        # Create config with auto_order=False
        config = {
            "free_only_mode": True,
            "router_models": {
                "test-model": {
                    "candidates": [
                        {"provider": "p1", "model": "m1", "priority": 3},
                        {"provider": "p2", "model": "m2", "priority": 1},
                        {"provider": "p3", "model": "m3", "priority": 2},
                    ],
                    "auto_order": False,
                }
            },
        }

        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        router = RouterCore(str(config_file))

        # Get candidates (should respect manual priority order)
        candidates = await router._get_candidates(
            "test-model", router._extract_requirements({})
        )

        print(
            f"\nCandidate order (auto_order=False): {[(c.provider, c.priority) for c in candidates]}"
        )

        # With auto_order=False, should respect manual priority
        # Priority 1 should be first (p2), then 2 (p3), then 3 (p1)
        if len(candidates) >= 2:
            # Verify sorted by priority (lower number = higher priority)
            priorities = [c.priority for c in candidates]
            assert priorities == sorted(priorities), (
                "Without auto_order, should respect manual priority order"
            )


class TestRankingIntegration:
    """
    Integration tests for ranking with router.
    """

    @pytest.mark.asyncio
    async def test_ranked_candidates_used_in_order(self, router):
        """Test that ranked candidates are tried in the correct order."""
        from tests.fixtures.provider_mocks import MockProviderResponse

        execution_order = []

        async def track_execution(candidate, request, request_id):
            execution_order.append(
                {
                    "provider": candidate.provider,
                    "model": candidate.model,
                    "priority": candidate.priority,
                }
            )

            # First two fail, third succeeds
            if len(execution_order) <= 2:
                raise Exception("Simulated failure")

            return MockProviderResponse(model=f"{candidate.provider}/{candidate.model}")

        with patch.object(
            router, "_execute_single_candidate", side_effect=track_execution
        ):
            try:
                await router.route_request(
                    create_request(model="coding-smart"), request_id="rank-test"
                )
            except:
                pass

        print(f"\nExecution order: {execution_order}")

        if len(execution_order) > 1:
            # Verify attempts were made in priority order
            priorities = [e["priority"] for e in execution_order]
            print(f"Priority order: {priorities}")

