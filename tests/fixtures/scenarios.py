"""Test scenarios for router testing."""

from typing import Dict, Any, List
from .provider_mocks import (
    RateLimitError,
    TimeoutError,
    AuthError,
    ConnectionError,
    InvalidRequestError
)


# Scenario 1: Multi-provider fallback with mixed failures
SCENARIO_MIXED_FAILURES = {
    "description": "Provider A rate limited, Provider B timeout, Provider C succeeds",
    "providers": ["provider_a", "provider_b", "provider_c"],
    "provider_a": {"error": RateLimitError, "retry_after": 60},
    "provider_b": {"error": TimeoutError},
    "provider_c": {"success": True, "content": "Success after fallback"}
}

# Scenario 2: All providers of one model fail, fallback to next model
SCENARIO_MODEL_FALLBACK = {
    "description": "All providers for model1 fail, fallback to model2",
    "model1_providers": ["provider_a", "provider_b"],
    "model2_providers": ["provider_c"],
    "provider_a": {"error": RateLimitError},
    "provider_b": {"error": RateLimitError},
    "provider_c": {"success": True, "content": "Fallback model success"}
}

# Scenario 3: Concurrent requests with independent failures
SCENARIO_CONCURRENT_INDEPENDENT = {
    "description": "20 concurrent requests with varying failures",
    "num_requests": 20,
    "failure_pattern": [
        # Request index -> failure info
        {"request": 0, "provider_a": "fail", "provider_b": "success"},
        {"request": 5, "provider_a": "success"},
        {"request": 10, "provider_a": "fail", "provider_b": "fail", "provider_c": "success"},
        {"request": 15, "provider_a": "success"}
    ]
}

# Scenario 4: Rate limit enforcement
SCENARIO_RATE_LIMIT_ENFORCEMENT = {
    "description": "Test rate limit with 5 max concurrent, 100 total requests",
    "max_concurrent": 5,
    "total_requests": 100,
    "provider": "rate_limited_provider",
    "rpm_limit": 60,
    "expected_behavior": "Only 5 concurrent, rest queued"
}

# Scenario 5: Head of line blocking test
SCENARIO_NO_HOL_BLOCKING = {
    "description": "Fast request should not wait for slow request",
    "slow_request": {"delay": 5.0, "provider": "slow_provider"},
    "fast_request": {"delay": 0.1, "provider": "fast_provider"},
    "expected": "Fast completes in ~0.1s, slow in ~5s"
}

# Scenario 6: Stats isolation
SCENARIO_STATS_ISOLATION = {
    "description": "20 concurrent requests with separate stats",
    "num_requests": 20,
    "tokens_per_request": 100,
    "expected_total_tokens": 2000,
    "expected_entries": 20
}

# Scenario 7: Provider ranking by performance
SCENARIO_PROVIDER_RANKING = {
    "description": "Providers ranked by TPS: A(150) > B(100) > C(50)",
    "providers": {
        "provider_a": {"tps": 150, "ttft_ms": 50, "success_rate": 0.99},
        "provider_b": {"tps": 100, "ttft_ms": 100, "success_rate": 0.95},
        "provider_c": {"tps": 50, "ttft_ms": 200, "success_rate": 0.85}
    },
    "expected_order": ["provider_a", "provider_b", "provider_c"]
}

# Scenario 8: Model ranking by benchmark
SCENARIO_MODEL_RANKING = {
    "description": "Models ranked by coding performance",
    "models": {
        "gpt-4o": {"swe_bench": 38.1, "humaneval": 90.2, "priority": 1},
        "claude-3.5-sonnet": {"swe_bench": 49.0, "humaneval": 92.0, "priority": 2},
        "o1-mini": {"swe_bench": 34.7, "humaneval": 87.5, "priority": 3}
    },
    "expected_order": ["claude-3.5-sonnet", "gpt-4o", "o1-mini"]
}

# Scenario 9: Different virtual models concurrent
SCENARIO_MIXED_VIRTUAL_MODELS = {
    "description": "10 concurrent: 5 coding-smart, 5 coding-fast",
    "coding_smart_count": 5,
    "coding_fast_count": 5,
    "coding_smart_expected_models": ["gpt-4o", "claude-3.5-sonnet"],
    "coding_fast_expected_models": ["llama-3.3-70b-versatile", "llama-3.1-8b"]
}

# Scenario 10: Error type tracking
SCENARIO_ERROR_TRACKING = {
    "description": "Various error types tracked correctly",
    "errors": [
        {"type": "rate_limit", "count": 3},
        {"type": "timeout", "count": 2},
        {"type": "auth", "count": 1},
        {"type": "connection", "count": 1}
    ],
    "expected_total_errors": 7
}


def create_request(
    model: str = "coding-smart",
    messages: List[Dict[str, str]] = None,
    stream: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a test request.
    
    Args:
        model: Model identifier
        messages: Chat messages
        stream: Enable streaming
        **kwargs: Additional request parameters
    
    Returns:
        Request dictionary
    """
    if messages is None:
        messages = [
            {"role": "user", "content": "Write a Python function to reverse a string"}
        ]
    
    request = {
        "model": model,
        "messages": messages,
        "stream": stream,
        **kwargs
    }
    
    return request


def create_batch_requests(
    count: int,
    model: str = "coding-smart",
    vary_model: bool = False
) -> List[Dict[str, Any]]:
    """
    Create a batch of test requests.
    
    Args:
        count: Number of requests
        model: Base model identifier
        vary_model: Whether to vary models across requests
    
    Returns:
        List of request dictionaries
    """
    requests = []
    models = ["coding-smart", "coding-fast", "chat-smart", "chat-fast"]
    
    for i in range(count):
        if vary_model:
            current_model = models[i % len(models)]
        else:
            current_model = model
        
        requests.append(create_request(
            model=current_model,
            messages=[
                {"role": "user", "content": f"Test request {i+1}"}
            ]
        ))
    
    return requests


def create_failure_scenario(
    num_attempts: int,
    failures: List[type],
    final_success: bool = True
) -> Dict[str, Any]:
    """
    Create a failure scenario configuration.
    
    Args:
        num_attempts: Number of attempts
        failures: List of exception types for each attempt
        final_success: Whether final attempt succeeds
    
    Returns:
        Scenario configuration
    """
    return {
        "num_attempts": num_attempts,
        "failures": failures,
        "final_success": final_success,
        "expected_attempts": num_attempts + (1 if final_success else 0)
    }


# Performance test scenarios

PERFORMANCE_LOAD_TEST_50 = {
    "description": "50 concurrent requests load test",
    "num_requests": 50,
    "models": ["coding-smart", "coding-fast"],
    "expected_max_time_seconds": 30,
    "expected_min_throughput": 10  # requests per second
}

PERFORMANCE_LOAD_TEST_100 = {
    "description": "100 concurrent requests load test",
    "num_requests": 100,
    "models": ["coding-smart"],
    "expected_max_time_seconds": 60,
    "expected_min_throughput": 15
}

# Configuration test scenarios

CONFIG_RANKING_STRATEGIES = {
    "best-performance": {
        "description": "Prioritize quality over speed",
        "weight_quality": 0.8,
        "weight_speed": 0.2,
        "expected_first": "claude-3.5-sonnet"  # Best SWE-Bench score
    },
    "fastest": {
        "description": "Prioritize speed over quality",
        "weight_quality": 0.2,
        "weight_speed": 0.8,
        "expected_first": "cerebras/llama-3.1-70b"  # Highest TPS
    },
    "balanced": {
        "description": "Balance quality and speed",
        "weight_quality": 0.6,
        "weight_speed": 0.4,
        "expected_first": "gpt-4o"  # Good balance
    }
}
