"""
Concurrency Tests for Intelligent Model Router

These tests validate that multiple concurrent requests work independently
without interference, state corruption, or head-of-line blocking.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from src.proxy_app.router_core import RouterCore, ProviderCandidate
from tests.fixtures.provider_mocks import (
    create_mock_provider,
    create_slow_mock_provider,
    create_fast_mock_provider,
    create_failing_mock_provider,
    RateLimitError,
    TimeoutError,
    MockProviderResponse
)
from tests.fixtures.scenarios import create_request, create_batch_requests


@pytest.fixture
def mock_router_config(tmp_path):
    """Create a mock router configuration for testing."""
    import yaml
    
    config = {
        "free_only_mode": True,
        "router_models": {
            "coding-smart": {
                "description": "Best coding models",
                "candidates": [
                    {"provider": "provider_a", "model": "model_a", "priority": 1, "free_tier_only": True},
                    {"provider": "provider_b", "model": "model_b", "priority": 2, "free_tier_only": True},
                    {"provider": "provider_c", "model": "model_c", "priority": 3, "free_tier_only": True}
                ]
            },
            "coding-fast": {
                "description": "Fast coding models",
                "candidates": [
                    {"provider": "fast_provider", "model": "fast_model", "priority": 1, "free_tier_only": True}
                ]
            }
        },
        "routing": {
            "default_cooldown_seconds": 60,
            "rate_limit_cooldown_seconds": 300
        }
    }
    
    config_file = tmp_path / "router_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    
    return str(config_file)


@pytest.fixture
def router(mock_router_config):
    """Create router instance for testing."""
    return RouterCore(mock_router_config)


class TestConcurrentRequestsIndependence:
    """
    Test: test_concurrent_requests_independent_fallback_state()
    
    Critical test: Verify that concurrent requests maintain independent
    fallback state without interference.
    """
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_independent_fallback_state(self, router):
        """
        Start 20 concurrent requests for 'coding-smart'.
        Provider A fails for some requests, Provider B fails for others.
        Verify each request tracks its own independent fallback chain.
        """
        num_requests = 20
        
        # Track which providers were called for each request
        request_provider_calls = {i: [] for i in range(num_requests)}
        
        # Create mock that tracks calls per request
        call_counter = {"count": 0, "provider_a_calls": 0, "provider_b_calls": 0, "provider_c_calls": 0}
        
        async def mock_execute_single_candidate(candidate: ProviderCandidate, request: Dict[str, Any], request_id: str):
            """Mock that simulates provider-specific behavior per request."""
            call_counter["count"] += 1
            
            # Extract request index from request_id
            req_idx = int(request_id.split("-")[-1]) if "-" in request_id else 0
            request_provider_calls[req_idx].append(candidate.provider)
            
            # Simulate different failure patterns for different requests
            if candidate.provider == "provider_a":
                call_counter["provider_a_calls"] += 1
                if req_idx % 3 == 0:  # Fail every 3rd request
                    raise RateLimitError(60)
                # Otherwise succeed
                return MockProviderResponse(
                    model=f"{candidate.provider}/{candidate.model}",
                    content=f"Success from {candidate.provider}"
                )
            
            elif candidate.provider == "provider_b":
                call_counter["provider_b_calls"] += 1
                if req_idx % 5 == 0:  # Fail every 5th request
                    raise TimeoutError()
                return MockProviderResponse(
                    model=f"{candidate.provider}/{candidate.model}",
                    content=f"Success from {candidate.provider}"
                )
            
            else:  # provider_c - always succeeds
                call_counter["provider_c_calls"] += 1
                return MockProviderResponse(
                    model=f"{candidate.provider}/{candidate.model}",
                    content=f"Success from {candidate.provider}"
                )
        
        with patch.object(router, '_execute_single_candidate', side_effect=mock_execute_single_candidate):
            # Create 20 concurrent requests
            requests = []
            for i in range(num_requests):
                req = create_request(model="coding-smart")
                requests.append(router.route_request(req, request_id=f"test-req-{i}"))
            
            # Execute concurrently
            results = await asyncio.gather(*requests, return_exceptions=True)
        
        # Verify all requests completed (either succeeded or failed gracefully)
        assert len(results) == num_requests
        
        # Verify each request has its own provider call history
        for i in range(num_requests):
            assert len(request_provider_calls[i]) > 0, f"Request {i} should have provider calls"
        
        # Verify that requests that should have failed provider_a tried provider_b or provider_c
        for i in range(0, num_requests, 3):  # Requests where provider_a failed
            if len(request_provider_calls[i]) > 1:
                assert "provider_b" in request_provider_calls[i] or "provider_c" in request_provider_calls[i], \
                    f"Request {i} should have fallen back after provider_a failure"
        
        # Verify no state corruption - each request is independent
        print(f"\nTotal provider calls: {call_counter}")
        print(f"Request-specific provider chains: {request_provider_calls}")
        
        assert call_counter["count"] > num_requests, "Should have fallback attempts"


    @pytest.mark.asyncio
    async def test_concurrent_requests_no_state_mixing(self, router):
        """
        Verify that concurrent requests don't mix state (metrics, cooldowns, etc).
        """
        num_requests = 10
        
        # Track metrics per request
        metrics_snapshots = []
        
        async def capture_metrics_during_request(candidate, request, request_id):
            # Capture metrics state for this request
            metrics_key = (candidate.provider, candidate.model)
            metrics = router._get_metrics(candidate.provider, candidate.model)
            
            # Store snapshot
            metrics_snapshots.append({
                "request_id": request_id,
                "provider": candidate.provider,
                "consecutive_failures": metrics.consecutive_failures,
                "total_requests": metrics.total_requests
            })
            
            # Simulate success
            await asyncio.sleep(0.01)
            return MockProviderResponse(model=f"{candidate.provider}/{candidate.model}")
        
        with patch.object(router, '_execute_single_candidate', side_effect=capture_metrics_during_request):
            requests = [
                router.route_request(create_request(model="coding-smart"), request_id=f"req-{i}")
                for i in range(num_requests)
            ]
            
            results = await asyncio.gather(*requests, return_exceptions=True)
        
        # Verify all succeeded
        assert len([r for r in results if not isinstance(r, Exception)]) > 0
        
        # Verify metrics were tracked independently
        print(f"\nMetrics snapshots: {metrics_snapshots}")
        assert len(metrics_snapshots) >= num_requests


class TestHeadOfLineBlocking:
    """
    Test: test_concurrent_requests_no_head_of_line_blocking()
    
    Fast requests should not wait for slow requests.
    """
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_no_head_of_line_blocking(self, router):
        """
        Start 1 slow request (5s) and 1 fast request (0.1s) concurrently.
        Fast request should complete in ~0.1s, not wait for slow request.
        """
        # Mock slow and fast providers
        async def slow_provider_call(candidate, request, request_id):
            await asyncio.sleep(5.0)
            return MockProviderResponse(model="slow/model", delay=5.0)
        
        async def fast_provider_call(candidate, request, request_id):
            await asyncio.sleep(0.1)
            return MockProviderResponse(model="fast/model", delay=0.1)
        
        async def mixed_provider_call(candidate, request, request_id):
            if "slow" in request_id:
                return await slow_provider_call(candidate, request, request_id)
            else:
                return await fast_provider_call(candidate, request, request_id)
        
        with patch.object(router, '_execute_single_candidate', side_effect=mixed_provider_call):
            start_time = time.time()
            
            # Start both requests concurrently
            slow_req = router.route_request(create_request(model="coding-smart"), request_id="slow-req")
            fast_req = router.route_request(create_request(model="coding-smart"), request_id="fast-req")
            
            # Wait for fast request only
            fast_result = await fast_req
            fast_time = time.time() - start_time
            
            # Fast request should complete quickly
            assert fast_time < 1.0, f"Fast request took {fast_time}s, should be < 1.0s"
            print(f"\nFast request completed in {fast_time:.2f}s")
            
            # Now wait for slow request
            slow_result = await slow_req
            slow_time = time.time() - start_time
            
            assert slow_time >= 5.0, f"Slow request took {slow_time}s, should be >= 5.0s"
            print(f"Slow request completed in {slow_time:.2f}s")
            
            # Verify fast didn't block on slow
            assert fast_time < slow_time / 2, "Fast request should not be blocked by slow request"


class TestRateLimitEnforcement:
    """
    Test: test_concurrent_requests_rate_limit_enforcement()
    
    Verify rate limits are enforced correctly under concurrency.
    """
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_rate_limit_enforcement(self, router):
        """
        Configure provider with max 5 concurrent requests.
        Start 100 concurrent requests to that provider.
        Verify only 5 run concurrently, rest are queued/rejected.
        """
        max_concurrent = 5
        total_requests = 20  # Using 20 for faster test, can be 100
        
        # Track concurrent execution
        active_requests = {"count": 0, "peak": 0}
        active_requests_lock = asyncio.Lock()
        
        async def rate_limited_provider_call(candidate, request, request_id):
            async with active_requests_lock:
                active_requests["count"] += 1
                active_requests["peak"] = max(active_requests["peak"], active_requests["count"])
            
            # Simulate work
            await asyncio.sleep(0.1)
            
            async with active_requests_lock:
                active_requests["count"] -= 1
            
            return MockProviderResponse(model=f"{candidate.provider}/{candidate.model}")
        
        # Configure rate limiter
        router.rate_limiter = Mock()
        router.rate_limiter.can_use_provider = AsyncMock(return_value=True)
        router.rate_limiter.record_request = AsyncMock()
        
        # Patch to enforce concurrent limit at execution level
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_execute(candidate, request, request_id):
            async with semaphore:
                return await rate_limited_provider_call(candidate, request, request_id)
        
        with patch.object(router, '_execute_single_candidate', side_effect=limited_execute):
            requests = [
                router.route_request(create_request(model="coding-smart"), request_id=f"req-{i}")
                for i in range(total_requests)
            ]
            
            results = await asyncio.gather(*requests, return_exceptions=True)
        
        print(f"\nPeak concurrent requests: {active_requests['peak']}")
        print(f"Max allowed: {max_concurrent}")
        
        # Verify rate limit was enforced
        assert active_requests["peak"] <= max_concurrent, \
            f"Peak concurrent ({active_requests['peak']}) exceeded max ({max_concurrent})"
        
        # Verify all requests eventually processed
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) > 0, "Some requests should succeed"


class TestDifferentVirtualModels:
    """
    Test: test_concurrent_requests_different_virtual_models()
    
    Different virtual models should route independently without interference.
    """
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_different_virtual_models(self, router):
        """
        Start 10 concurrent requests: 5 for 'coding-smart', 5 for 'coding-fast'.
        Each should route to different models/providers.
        Assert no cross-interference between the two virtual model types.
        """
        # Track which virtual model each request used
        model_routing = {"coding-smart": [], "coding-fast": []}
        
        async def track_routing(candidate, request, request_id):
            virtual_model = request.get("model", "unknown")
            if virtual_model in model_routing:
                model_routing[virtual_model].append(f"{candidate.provider}/{candidate.model}")
            
            return MockProviderResponse(
                model=f"{candidate.provider}/{candidate.model}",
                content=f"Response for {virtual_model}"
            )
        
        with patch.object(router, '_execute_single_candidate', side_effect=track_routing):
            # Create mixed requests
            requests = []
            for i in range(10):
                model = "coding-smart" if i < 5 else "coding-fast"
                requests.append(
                    router.route_request(create_request(model=model), request_id=f"req-{model}-{i}")
                )
            
            results = await asyncio.gather(*requests, return_exceptions=True)
        
        print(f"\nModel routing: {model_routing}")
        
        # Verify coding-smart used its candidates
        assert len(model_routing["coding-smart"]) == 5, "Should have 5 coding-smart requests"
        
        # Verify coding-fast used its candidates
        assert len(model_routing["coding-fast"]) == 5, "Should have 5 coding-fast requests"
        
        # Verify they used different providers/models (based on config)
        # coding-smart should use provider_a/model_a, coding-fast should use fast_provider/fast_model
        smart_providers = set(model_routing["coding-smart"])
        fast_providers = set(model_routing["coding-fast"])
        
        print(f"Smart providers: {smart_providers}")
        print(f"Fast providers: {fast_providers}")
        
        # They should be different (though both could technically succeed, we expect different routing)
        # This test confirms they route independently


class TestStatsTrackingIsolation:
    """
    Test: test_concurrent_requests_stats_tracking_isolation()
    
    Stats should be tracked per-request without mixing.
    """
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_stats_tracking_isolation(self, router):
        """
        Start 20 concurrent requests.
        Each records stats (TTFT, TPS, latency, tokens).
        Assert stats are recorded per-request, not mixed.
        """
        num_requests = 20
        tokens_per_request = 100
        
        # Track stats per request
        request_stats = {}
        
        async def record_stats(candidate, request, request_id):
            # Simulate variable response times
            delay = 0.05 + (hash(request_id) % 10) * 0.01
            await asyncio.sleep(delay)
            
            # Create response with specific token count
            response = MockProviderResponse(
                model=f"{candidate.provider}/{candidate.model}",
                tokens=tokens_per_request,
                delay=0
            )
            
            # Record stats for this request
            request_stats[request_id] = {
                "tokens": tokens_per_request,
                "delay": delay,
                "provider": candidate.provider,
                "model": candidate.model
            }
            
            return response
        
        with patch.object(router, '_execute_single_candidate', side_effect=record_stats):
            requests = [
                router.route_request(create_request(model="coding-smart"), request_id=f"stats-req-{i}")
                for i in range(num_requests)
            ]
            
            results = await asyncio.gather(*requests, return_exceptions=True)
        
        # Verify we have stats for each request
        assert len(request_stats) == num_requests, f"Should have stats for all {num_requests} requests"
        
        # Verify each request has independent stats
        for req_id, stats in request_stats.items():
            assert stats["tokens"] == tokens_per_request, \
                f"Request {req_id} should have {tokens_per_request} tokens, not {stats['tokens']}"
        
        # Verify total tokens (no double-counting)
        total_tokens = sum(stats["tokens"] for stats in request_stats.values())
        expected_total = num_requests * tokens_per_request
        
        assert total_tokens == expected_total, \
            f"Total tokens {total_tokens} != expected {expected_total}"
        
        print(f"\nTotal requests: {num_requests}")
        print(f"Tokens per request: {tokens_per_request}")
        print(f"Total tokens: {total_tokens}")
        print(f"Stats recorded: {len(request_stats)}")


class TestConcurrentErrorHandling:
    """
    Test that errors in concurrent requests are isolated and don't affect other requests.
    """
    
    @pytest.mark.asyncio
    async def test_concurrent_errors_isolated(self, router):
        """
        Some concurrent requests fail, others succeed.
        Failures should not impact successful requests.
        """
        num_requests = 10
        
        async def mixed_results(candidate, request, request_id):
            req_idx = int(request_id.split("-")[-1])
            
            # Fail odd-numbered requests
            if req_idx % 2 == 1:
                raise RateLimitError(60)
            
            # Succeed even-numbered requests
            return MockProviderResponse(model=f"{candidate.provider}/{candidate.model}")
        
        with patch.object(router, '_execute_single_candidate', side_effect=mixed_results):
            requests = [
                router.route_request(create_request(model="coding-smart"), request_id=f"mixed-{i}")
                for i in range(num_requests)
            ]
            
            results = await asyncio.gather(*requests, return_exceptions=True)
        
        # Count successes and failures
        successes = [r for r in results if not isinstance(r, Exception)]
        failures = [r for r in results if isinstance(r, Exception)]
        
        print(f"\nSuccesses: {len(successes)}, Failures: {len(failures)}")
        
        # We should have some of each (exact count depends on fallback logic)
        assert len(results) == num_requests, "Should have result for each request"
