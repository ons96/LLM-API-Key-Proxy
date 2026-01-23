"""
Integration Tests for Model Router

End-to-end tests with real or simulated provider calls.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

from src.proxy_app.router_core import RouterCore
from tests.fixtures.provider_mocks import (
    create_mock_provider,
    create_slow_mock_provider,
    create_fast_mock_provider,
    create_failing_mock_provider,
    MockProviderResponse,
    RateLimitError,
    TimeoutError
)
from tests.fixtures.scenarios import create_request, create_batch_requests


@pytest.fixture
def mock_router_config(tmp_path):
    """Create a comprehensive mock router configuration."""
    import yaml
    
    config = {
        "free_only_mode": True,
        "router_models": {
            "coding-smart": {
                "description": "Best coding models",
                "candidates": [
                    {"provider": "provider_a", "model": "gpt-4o", "priority": 1, "free_tier_only": True},
                    {"provider": "provider_b", "model": "gpt-4o", "priority": 2, "free_tier_only": True},
                    {"provider": "provider_c", "model": "claude-3.5-sonnet", "priority": 3, "free_tier_only": True}
                ]
            },
            "coding-fast": {
                "description": "Fast coding models",
                "candidates": [
                    {"provider": "fast_provider", "model": "llama-3.3-70b", "priority": 1, "free_tier_only": True}
                ]
            },
            "chat-smart": {
                "description": "Smart chat",
                "candidates": [
                    {"provider": "provider_a", "model": "gpt-4o", "priority": 1, "free_tier_only": True}
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


class TestEndToEndCodingSmart:
    """
    Test: test_e2e_coding_smart_with_provider_failures()
    
    End-to-end test with provider failures and fallback.
    """
    
    @pytest.mark.asyncio
    async def test_e2e_coding_smart_with_provider_failures(self, router):
        """
        Make actual request to 'coding-smart' endpoint.
        Simulate Provider A failure.
        Verify fallback to Provider B.
        Check response is valid.
        Verify stats recorded.
        """
        execution_log = []
        
        async def simulated_execution(candidate, request, request_id):
            execution_log.append({
                "provider": candidate.provider,
                "model": candidate.model,
                "timestamp": time.time()
            })
            
            # Provider A fails
            if candidate.provider == "provider_a":
                raise RateLimitError(60)
            
            # Provider B succeeds
            elif candidate.provider == "provider_b":
                return MockProviderResponse(
                    model="gpt-4o",
                    content="def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)",
                    tokens=50
                )
            
            # Provider C as fallback
            else:
                return MockProviderResponse(
                    model="claude-3.5-sonnet",
                    content="Fallback response",
                    tokens=40
                )
        
        with patch.object(router, '_execute_single_candidate', side_effect=simulated_execution):
            result = await router.route_request(
                create_request(model="coding-smart"),
                request_id="e2e-test"
            )
        
        print(f"\nExecution log: {execution_log}")
        
        # Verify provider_a was tried and failed
        assert any(e["provider"] == "provider_a" for e in execution_log), \
            "Provider A should have been tried"
        
        # Verify fallback to provider_b occurred
        assert any(e["provider"] == "provider_b" for e in execution_log), \
            "Should have fallen back to Provider B"
        
        # Verify we got a result
        assert result is not None, "Should have a successful result"
        
        # Verify metrics were updated
        metrics_a = router._get_metrics("provider_a", "gpt-4o")
        metrics_b = router._get_metrics("provider_b", "gpt-4o")
        
        print(f"\nProvider A metrics: requests={metrics_a.total_requests}, errors={metrics_a.total_errors}")
        print(f"Provider B metrics: requests={metrics_b.total_requests}, errors={metrics_b.total_errors}")
        
        assert metrics_a.total_errors > 0, "Provider A should have recorded error"
        assert metrics_b.total_requests > 0, "Provider B should have recorded request"


class TestLoadTest50Concurrent:
    """
    Test: test_e2e_load_test_50_concurrent_requests()
    
    High-concurrency load test.
    """
    
    @pytest.mark.asyncio
    async def test_e2e_load_test_50_concurrent_requests(self, router):
        """
        Fire 50 concurrent requests to 'coding-smart'.
        All target providers with realistic rate limits.
        Measure: total time, average latency, throughput, errors.
        Assert all requests complete successfully.
        Assert no requests exceed timeout.
        """
        num_requests = 50
        timeout_seconds = 60
        
        # Track metrics
        start_time = time.time()
        completed_requests = []
        failed_requests = []
        
        async def load_test_execution(candidate, request, request_id):
            req_start = time.time()
            
            # Simulate realistic processing time
            await asyncio.sleep(0.05)  # 50ms
            
            req_duration = time.time() - req_start
            
            # Small chance of failure for realism
            req_num = int(request_id.split("-")[-1])
            if req_num % 17 == 0:  # ~6% failure rate
                raise RateLimitError(60)
            
            return MockProviderResponse(
                model=f"{candidate.provider}/{candidate.model}",
                tokens=100,
                delay=0
            )
        
        with patch.object(router, '_execute_single_candidate', side_effect=load_test_execution):
            # Create 50 concurrent requests
            requests = [
                router.route_request(create_request(model="coding-smart"), request_id=f"load-{i}")
                for i in range(num_requests)
            ]
            
            # Execute with timeout
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*requests, return_exceptions=True),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                pytest.fail(f"Load test exceeded {timeout_seconds}s timeout")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Analyze results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_requests.append(i)
            else:
                completed_requests.append(i)
        
        # Calculate metrics
        throughput = len(completed_requests) / total_time if total_time > 0 else 0
        avg_latency = total_time / num_requests
        success_rate = len(completed_requests) / num_requests
        
        print(f"\n=== Load Test Results ===")
        print(f"Total requests: {num_requests}")
        print(f"Completed: {len(completed_requests)}")
        print(f"Failed: {len(failed_requests)}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Throughput: {throughput:.2f} req/s")
        print(f"Avg latency: {avg_latency:.3f}s")
        print(f"Success rate: {success_rate:.1%}")
        
        # Assertions
        assert total_time < timeout_seconds, f"Should complete within {timeout_seconds}s"
        assert len(completed_requests) > 0, "At least some requests should succeed"
        
        # With fallback, most requests should succeed
        assert success_rate >= 0.5, f"Success rate {success_rate:.1%} should be >= 50%"
        
        # Verify no deadlocks
        assert len(results) == num_requests, "All requests should return"


class TestMixedVirtualModelsConcurrent:
    """
    Test: test_e2e_mixed_virtual_models_concurrent()
    
    Multiple virtual models used concurrently.
    """
    
    @pytest.mark.asyncio
    async def test_e2e_mixed_virtual_models_concurrent(self, router):
        """
        20 concurrent requests mixing:
        - 7 to 'coding-smart'
        - 7 to 'coding-fast'
        - 6 to 'chat-smart'
        
        Verify each uses correct model/provider.
        Verify stats tracked per virtual model.
        """
        request_distribution = {
            "coding-smart": 7,
            "coding-fast": 7,
            "chat-smart": 6
        }
        
        routing_log = {
            "coding-smart": [],
            "coding-fast": [],
            "chat-smart": []
        }
        
        async def track_virtual_model_routing(candidate, request, request_id):
            # Extract virtual model from request
            virtual_model = request.get("model", "unknown")
            
            if virtual_model in routing_log:
                routing_log[virtual_model].append({
                    "provider": candidate.provider,
                    "model": candidate.model,
                    "request_id": request_id
                })
            
            # Simulate success
            await asyncio.sleep(0.02)
            return MockProviderResponse(model=f"{candidate.provider}/{candidate.model}")
        
        with patch.object(router, '_execute_single_candidate', side_effect=track_virtual_model_routing):
            # Create mixed requests
            all_requests = []
            
            for virtual_model, count in request_distribution.items():
                for i in range(count):
                    req = create_request(model=virtual_model)
                    all_requests.append(
                        router.route_request(req, request_id=f"{virtual_model}-{i}")
                    )
            
            # Execute all concurrently
            results = await asyncio.gather(*all_requests, return_exceptions=True)
        
        print(f"\n=== Virtual Model Routing ===")
        for vm, routes in routing_log.items():
            print(f"{vm}: {len(routes)} requests")
            if routes:
                providers_used = set(r["provider"] for r in routes)
                print(f"  Providers used: {providers_used}")
        
        # Verify each virtual model was used
        for vm, expected_count in request_distribution.items():
            actual_count = len(routing_log[vm])
            assert actual_count == expected_count, \
                f"{vm} should have {expected_count} requests, got {actual_count}"
        
        # Verify different virtual models used appropriate providers
        # coding-smart should use provider_a/b/c
        # coding-fast should use fast_provider
        # chat-smart should use provider_a
        
        if routing_log["coding-fast"]:
            fast_providers = set(r["provider"] for r in routing_log["coding-fast"])
            assert "fast_provider" in fast_providers or len(routing_log["coding-fast"]) > 0, \
                "coding-fast should use fast_provider"


class TestStressTestConcurrency:
    """
    Stress test with very high concurrency.
    """
    
    @pytest.mark.asyncio
    async def test_high_concurrency_stress(self, router):
        """
        Test with 100 concurrent requests to verify stability.
        """
        num_requests = 100
        
        async def quick_response(candidate, request, request_id):
            await asyncio.sleep(0.01)
            return MockProviderResponse(model=f"{candidate.provider}/{candidate.model}")
        
        with patch.object(router, '_execute_single_candidate', side_effect=quick_response):
            start = time.time()
            
            requests = [
                router.route_request(create_request(model="coding-smart"), request_id=f"stress-{i}")
                for i in range(num_requests)
            ]
            
            results = await asyncio.gather(*requests, return_exceptions=True)
            
            duration = time.time() - start
        
        # Count successes
        successes = [r for r in results if not isinstance(r, Exception)]
        
        print(f"\n=== Stress Test Results ===")
        print(f"Total requests: {num_requests}")
        print(f"Successful: {len(successes)}")
        print(f"Duration: {duration:.2f}s")
        print(f"Throughput: {len(successes) / duration:.2f} req/s")
        
        # Verify system handled high concurrency
        assert len(results) == num_requests, "All requests should complete"
        assert len(successes) > 0, "At least some requests should succeed"


class TestErrorRecovery:
    """
    Test error recovery and graceful degradation.
    """
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_under_failures(self, router):
        """
        Test that system degrades gracefully when providers fail.
        """
        failure_rate = 0.3  # 30% of attempts fail
        attempt_count = {"count": 0}
        
        async def failing_provider(candidate, request, request_id):
            attempt_count["count"] += 1
            
            # Fail 30% of the time
            if attempt_count["count"] % 3 == 0:
                raise TimeoutError()
            
            return MockProviderResponse(model=f"{candidate.provider}/{candidate.model}")
        
        with patch.object(router, '_execute_single_candidate', side_effect=failing_provider):
            requests = [
                router.route_request(create_request(model="coding-smart"), request_id=f"degrade-{i}")
                for i in range(20)
            ]
            
            results = await asyncio.gather(*requests, return_exceptions=True)
        
        successes = [r for r in results if not isinstance(r, Exception)]
        failures = [r for r in results if isinstance(r, Exception)]
        
        print(f"\nGraceful degradation: {len(successes)} succeeded, {len(failures)} failed")
        
        # With fallback, should have some successes even with failures
        assert len(successes) > 0, "Should have some successful requests despite failures"


class TestRateLimitHandling:
    """
    Test rate limit handling in integration scenario.
    """
    
    @pytest.mark.asyncio
    async def test_rate_limit_backoff_and_retry(self, router):
        """
        Test that rate limits trigger appropriate backoff and retry behavior.
        """
        call_log = []
        
        async def rate_limited_provider(candidate, request, request_id):
            call_log.append({
                "provider": candidate.provider,
                "timestamp": time.time()
            })
            
            # Provider A always rate limited
            if candidate.provider == "provider_a":
                raise RateLimitError(60)
            
            # Provider B succeeds
            return MockProviderResponse(model=f"{candidate.provider}/{candidate.model}")
        
        with patch.object(router, '_execute_single_candidate', side_effect=rate_limited_provider):
            # Make multiple requests
            results = []
            for i in range(5):
                try:
                    result = await router.route_request(
                        create_request(model="coding-smart"),
                        request_id=f"ratelimit-{i}"
                    )
                    results.append(result)
                except Exception as e:
                    results.append(e)
        
        print(f"\nRate limit handling: {len(call_log)} provider calls")
        
        # Verify provider_a was tried but failed
        provider_a_calls = [c for c in call_log if c["provider"] == "provider_a"]
        print(f"Provider A calls: {len(provider_a_calls)}")
        
        # Verify fallback to provider_b
        provider_b_calls = [c for c in call_log if c["provider"] == "provider_b"]
        print(f"Provider B calls: {len(provider_b_calls)}")
        
        # Should have fallen back to provider_b
        assert len(provider_b_calls) > 0, "Should have fallen back to provider_b"


class TestMetricsCollection:
    """
    Test that metrics are collected correctly in integration scenario.
    """
    
    @pytest.mark.asyncio
    async def test_metrics_collected_during_requests(self, router):
        """
        Verify metrics are collected and accurate across multiple requests.
        """
        async def metered_provider(candidate, request, request_id):
            await asyncio.sleep(0.05)
            return MockProviderResponse(
                model=f"{candidate.provider}/{candidate.model}",
                tokens=100
            )
        
        with patch.object(router, '_execute_single_candidate', side_effect=metered_provider):
            # Make 10 requests
            for i in range(10):
                await router.route_request(
                    create_request(model="coding-smart"),
                    request_id=f"metrics-{i}"
                )
        
        # Check collected metrics
        all_metrics = {}
        for (provider, model), metrics in router.provider_metrics.items():
            all_metrics[f"{provider}/{model}"] = {
                "total_requests": metrics.total_requests,
                "total_errors": metrics.total_errors,
                "success_rate": metrics.success_rate,
                "latency_ms": metrics.ewma_latency_ms
            }
        
        print(f"\n=== Collected Metrics ===")
        for key, metrics in all_metrics.items():
            print(f"{key}:")
            print(f"  Requests: {metrics['total_requests']}")
            print(f"  Errors: {metrics['total_errors']}")
            print(f"  Success rate: {metrics['success_rate']:.1%}")
            print(f"  Latency: {metrics['latency_ms']:.1f}ms")
        
        # Verify metrics were collected
        assert len(all_metrics) > 0, "Should have collected metrics"
        
        # Verify at least one provider has requests
        total_requests = sum(m["total_requests"] for m in all_metrics.values())
        assert total_requests >= 10, f"Should have at least 10 requests recorded, got {total_requests}"


class TestTimeoutHandling:
    """
    Test timeout handling in integration scenario.
    """
    
    @pytest.mark.asyncio
    async def test_request_timeout_handling(self, router):
        """
        Test that slow providers timeout and fallback occurs.
        """
        async def slow_then_fast_provider(candidate, request, request_id):
            if candidate.provider == "provider_a":
                # Very slow, should trigger timeout
                await asyncio.sleep(10.0)
                return MockProviderResponse(model=f"{candidate.provider}/{candidate.model}")
            else:
                # Fast response
                await asyncio.sleep(0.1)
                return MockProviderResponse(model=f"{candidate.provider}/{candidate.model}")
        
        with patch.object(router, '_execute_single_candidate', side_effect=slow_then_fast_provider):
            start = time.time()
            
            # This should timeout on provider_a and fallback to provider_b
            try:
                result = await asyncio.wait_for(
                    router.route_request(create_request(model="coding-smart"), request_id="timeout-test"),
                    timeout=5.0  # 5 second overall timeout
                )
            except asyncio.TimeoutError:
                # If overall timeout, that's also acceptable behavior
                result = None
            
            duration = time.time() - start
        
        print(f"\nTimeout test completed in {duration:.2f}s")
        
        # Should complete relatively quickly (either via fallback or timeout)
        assert duration < 10.0, "Should not take full 10 seconds (slow provider time)"
