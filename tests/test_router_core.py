"""
Tests for Router Core Functionality
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
import yaml
from pathlib import Path

from src.proxy_app.router_core import (
    RouterCore, CapabilityRequirements, ProviderMetrics, 
    ProviderCandidate, ErrorCategory
)


@pytest.fixture
def mock_config_file(tmp_path):
    """Create a mock configuration file for testing."""
    config = {
        "free_only_mode": True,
        "providers": {
            "groq": {
                "enabled": True,
                "env_var": "GROQ_API_KEY",
                "free_tier_models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
            },
            "gemini": {
                "enabled": True,
                "env_var": "GEMINI_API_KEY", 
                "free_tier_models": ["gemini-1.5-flash"]
            }
        },
        "router_models": {
            "router/best-coding": {
                "description": "Best for coding",
                "candidates": [
                    {"provider": "groq", "model": "llama-3.3-70b-versatile", "priority": 1},
                    {"provider": "gemini", "model": "gemini-1.5-flash", "priority": 2}
                ]
            }
        },
        "routing": {
            "default_cooldown_seconds": 60,
            "rate_limit_cooldown_seconds": 300
        }
    }
    
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    
    return str(config_file)


@pytest.fixture
def router(mock_config_file):
    """Create router instance for testing."""
    return RouterCore(mock_config_file)


class TestCapabilityRequirements:
    """Test capability requirements extraction."""
    
    def test_basic_requirements(self):
        """Test basic capability requirements."""
        req = CapabilityRequirements()
        assert not req.needs_tools
        assert not req.needs_vision
        assert not req.needs_structured_output
        assert not req.streaming
        assert not req.moe_mode
    
    def test_tools_requirements(self):
        """Test tools requirement detection."""
        request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "tools": [{"type": "function", "function": {"name": "test", "description": "test"}}]
        }
        
        router = RouterCore()
        req = router._extract_requirements(request)
        
        assert req.needs_tools


class TestProviderMetrics:
    """Test provider metrics tracking."""
    
    def test_initial_state(self):
        """Test initial metrics state."""
        metrics = ProviderMetrics()
        assert metrics.success_rate == 1.0
        assert metrics.consecutive_failures == 0
        assert metrics.total_requests == 0
        assert metrics.is_healthy()
    
    def test_success_tracking(self):
        """Test success tracking."""
        metrics = ProviderMetrics()
        metrics.record_success()
        
        assert metrics.total_requests == 1
        assert metrics.consecutive_failures == 0
        assert metrics.success_rate == 1.0
    
    def test_error_tracking(self):
        """Test error tracking."""
        metrics = ProviderMetrics()
        metrics.record_error()
        
        assert metrics.total_requests == 1
        assert metrics.total_errors == 1
        assert metrics.consecutive_failures == 1
        assert metrics.success_rate == 0.0
    
    def test_cooldown(self):
        """Test cooldown functionality."""
        metrics = ProviderMetrics()
        metrics.set_cooldown(60)
        
        assert not metrics.is_healthy()
        assert metrics.cooldown_until > time.time()
    
    def test_latency_update(self):
        """Test latency tracking with EWMA."""
        metrics = ProviderMetrics()
        
        # First update
        metrics.update_latency(100.0)
        assert metrics.ewma_latency_ms == 100.0
        
        # Second update
        metrics.update_latency(200.0, alpha=0.5)
        assert metrics.ewma_latency_ms == 150.0  # (100 + 200) / 2


class TestProviderCandidate:
    """Test provider candidate matching."""
    
    def test_capability_matching(self):
        """Test capability matching."""
        candidate = ProviderCandidate(
            provider="test",
            model="test-model",
            capabilities={"tools", "vision", "long_context"}
        )
        
        # Should match
        req = CapabilityRequirements(needs_tools=True)
        assert candidate.matches_requirements(req)
        
        req = CapabilityRequirements(needs_tools=True, needs_vision=True)
        assert candidate.matches_requirements(req)
        
        # Should not match
        req = CapabilityRequirements(needs_tools=False, needs_vision=True, needs_structured_output=True)
        candidate_no_structured = ProviderCandidate(
            provider="test",
            model="test-model",
            capabilities={"tools", "vision"}
        )
        assert not candidate_no_structured.matches_requirements(req)


class TestErrorClassification:
    """Test error classification logic."""
    
    @pytest.mark.asyncio
    async def test_rate_limit_error(self):
        """Test rate limit error classification."""
        router = RouterCore()
        
        # Mock rate limit exception
        mock_error = Exception("Rate limit exceeded")
        mock_error.response = Mock()
        mock_error.response.headers = {"retry-after": "60"}
        
        category, retry_after = await router._classify_error(mock_error)
        
        assert category == ErrorCategory.RATE_LIMIT
        assert retry_after == 60
    
    @pytest.mark.asyncio
    async def test_auth_error(self):
        """Test authentication error classification."""
        router = RouterCore()
        
        mock_error = Exception("Unauthorized: Invalid API key")
        
        category, retry_after = await router._classify_error(mock_error)
        
        assert category == ErrorCategory.AUTH_ERROR
        assert retry_after is None
    
    @pytest.mark.asyncio
    async def test_transient_error(self):
        """Test transient error classification."""
        router = RouterCore()
        
        mock_error = Exception("Connection timeout")
        
        category, retry_after = await router._classify_error(mock_error)
        
        assert category == ErrorCategory.TRANSIENT
        assert retry_after is None


class TestRouterConfiguration:
    """Test router configuration loading."""
    
    def test_default_config(self):
        """Test loading default config when file doesn't exist."""
        router = RouterCore("/nonexistent/config.yaml")
        
        assert router.free_only_mode is True
        assert "default_cooldown_seconds" in router.config.get("routing", {})
    
    def test_custom_config(self, mock_config_file):
        """Test loading custom configuration."""
        router = RouterCore(mock_config_file)
        
        assert router.free_only_mode is True
        assert "groq" in router.config.get("providers", {})
        assert "router/best-coding" in router.virtual_models


class TestModelResolution:
    """Test model resolution logic."""
    
    @pytest.fixture
    def simple_router(self):
        """Router with simple configuration."""
        router = RouterCore()
        router.virtual_models = {
            "router/test": {
                "candidates": [
                    {"provider": "groq", "model": "test-model-1", "priority": 1, "capabilities": ["tools"]},
                    {"provider": "gemini", "model": "test-model-2", "priority": 2}
                ]
            }
        }
        router.config = {
            "free_only_mode": True,
            "safety": {"forbidden_providers_under_free_mode": ["openai"]}
        }
        return router
    
    def test_virtual_model_resolution(self, simple_router):
        """Test resolving virtual model to candidates."""
        req = CapabilityRequirements()
        candidates = simple_router._get_candidates("router/test", req)
        
        assert len(candidates) == 2
        assert candidates[0].provider == "groq"  # Higher priority first
        assert candidates[1].provider == "gemini"
    
    def test_direct_model_resolution(self, simple_router):
        """Test resolving direct model reference."""
        req = CapabilityRequirements()
        candidates = simple_router._get_candidates("groq/test-model", req)
        
        assert len(candidates) == 1
        assert candidates[0].provider == "groq"
        assert candidates[0].model == "test-model"


class TestFreeOnlyModeEnforcement:
    """Test FREE_ONLY_MODE enforcement."""
    
    @pytest.fixture
    def strict_router(self):
        """Router with strict FREE_ONLY_MODE."""
        router = RouterCore()
        router.free_only_mode = True
        router.config = {
            "free_only_mode": True,
            "providers": {
                "groq": {
                    "enabled": True,
                    "free_tier_models": ["free-model"]
                },
                "openai": {
                    "enabled": True,
                    "free_tier_models": []  # No free models
                }
            },
            "safety": {
                "forbidden_providers_under_free_mode": ["openai", "anthropic"]
            }
        }
        return router
    
    def test_free_tier_enforcement(self, strict_router):
        """Test that only free tier models are allowed in FREE_ONLY_MODE."""
        req = CapabilityRequirements()
        
        # Mock metrics to be healthy
        strict_router.provider_metrics = {
            ("groq", "free-model"): ProviderMetrics(),
            ("groq", "paid-model"): ProviderMetrics()
        }
        
        # Add some candidates
        with patch.object(strict_router, '_get_candidates', return_value=[
            ProviderCandidate("groq", "free-model", 1),
            ProviderCandidate("groq", "paid-model", 2),
            ProviderCandidate("openai", "any-model", 3)
        ]):
            candidates = strict_router._get_candidates("test", req)
            available = []
            
            for candidate in candidates:
                if candidate.provider in strict_router.config.get("safety", {}).get("forbidden_providers_under_free_mode", []):
                    continue
                provider_config = strict_router.config.get("providers", {}).get(candidate.provider, {})
                free_models = provider_config.get("free_tier_models", [])
                
                if strict_router.free_only_mode and free_models and candidate.model not in free_models:
                    continue
                
                available.append(candidate)
            
            # Should only get the free model, not paid or forbidden providers
            assert len(available) >= 1
            for candidate in available:
                assert candidate.provider != "openai"  # Forbidden provider


class TestMoEMode:
    """Test MoE (Mixture of Experts) mode functionality."""
    
    @pytest.fixture
    def moe_router(self):
        """Router with MoE configuration."""
        router = RouterCore()
        router.virtual_models = {
            "router/best-coding-moe": {
                "moe_mode": True,
                "max_experts": 3,
                "aggregator_model": "groq/llama-3.3-70b-versatile",
                "candidates": [
                    {"provider": "groq", "model": "expert-1", "role": "expert-architect"},
                    {"provider": "gemini", "model": "expert-2", "role": "expert-secure"},
                    {"provider": "groq", "model": "expert-3", "role": "expert-optimize"}
                ]
            }
        }
        return router
    
    def test_moe_requirements_extraction(self, moe_router):
        """Test that MoE mode is detected from virtual model."""
        request = {
            "model": "router/best-coding-moe",
            "messages": [{"role": "user", "content": "test"}]
        }
        
        req = moe_router._extract_requirements(request)
        assert req.moe_mode is True
    
    def test_expert_candidate_selection(self, moe_router):
        """Test expert candidate selection for MoE."""
        req = CapabilityRequirements(moe_mode=True)
        candidates = moe_router._get_candidates("router/best-coding-moe", req)
        
        # Should get expert candidates
        experts = [c for c in candidates if c.role and "expert" in c.role]
        assert len(experts) > 0
        assert all("expert" in e.role for e in experts)


# Integration Tests (require mocking)

@pytest.mark.asyncio
async def test_router_initialization():
    """Test router initialization with real config file."""
    config_path = "config/router_config.yaml"
    
    if not Path(config_path).exists():
        pytest.skip("Config file not found")
    
    router = RouterCore(config_path)
    
    # Should initialize successfully
    assert router.config is not None
    assert isinstance(router.free_only_mode, bool)
    assert isinstance(router.virtual_models, dict)


@pytest.mark.asyncio
async def test_model_list_generation():
    """Test model list generation."""
    router = RouterCore()
    
    # Mock some virtual models
    router.virtual_models = {
        "router/test": {
            "description": "Test virtual model"
        }
    }
    
    models = router.get_model_list()
    
    # Should include virtual models
    virtual_models = [m for m in models if m["id"].startswith("router/")]
    assert len(virtual_models) > 0
    
    for model in virtual_models:
        assert "id" in model
        assert "object" in model
        assert "created" in model
        assert "owned_by" in model


@pytest.mark.asyncio
async def test_health_status():
    """Test health status generation."""
    router = RouterCore()
    
    # Mock some metrics
    router.provider_metrics = {
        ("groq", "test-model"): ProviderMetrics()
    }
    
    health = router.get_health_status()
    
    assert "free_only_mode" in health
    assert "providers" in health
    assert "search_providers" in health
    assert "timestamp" in health


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])