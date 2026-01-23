># Router Test Suite

Comprehensive test suite for the Intelligent Model Router implementation.

## Test Organization

### Core Test Files

#### `test_concurrency.py` - **CRITICAL PRIORITY**
Tests for concurrent request handling:
- `test_concurrent_requests_independent_fallback_state()` - Verify independent fallback chains
- `test_concurrent_requests_no_head_of_line_blocking()` - Verify no HOL blocking
- `test_concurrent_requests_rate_limit_enforcement()` - Verify rate limits enforced
- `test_concurrent_requests_different_virtual_models()` - Verify independent routing
- `test_concurrent_requests_stats_tracking_isolation()` - Verify stat isolation

**Purpose**: Ensure multiple concurrent requests work independently without interference.

#### `test_fallback_logic.py`
Tests for multi-provider and multi-model fallback:
- `test_fallback_tries_all_providers_for_model()` - Sequential provider fallback
- `test_fallback_moves_to_next_model_after_all_providers_fail()` - Model-level fallback
- `test_fallback_respects_benchmark_ranking_order()` - Benchmark-based ordering
- `test_fallback_provider_order_by_performance()` - Performance-based ordering
- `test_fallback_tracks_failure_reasons()` - Error tracking

**Purpose**: Validate sequential fallback logic across providers and models.

#### `test_performance_tracking.py`
Tests for performance metrics collection:
- `test_stats_tracked_per_request()` - Per-request stat tracking
- `test_stats_accurate_under_concurrency()` - Concurrent stat accuracy
- `test_error_types_tracked()` - Error type tracking
- `test_request_timestamp_accurate()` - Timestamp accuracy
- `test_stats_persistence_survives_restart()` - Persistence (future)

**Purpose**: Ensure accurate performance metrics collection and tracking.

#### `test_configuration.py`
Tests for configuration and dynamic reordering:
- `test_reordering_updates_model_order()` - Dynamic reordering
- `test_ranking_strategy_best_performance()` - Quality-based ranking
- `test_ranking_strategy_fastest()` - Speed-based ranking
- `test_ranking_strategy_balanced()` - Balanced ranking
- `test_provider_ranking_by_tps()` - TPS-based provider ranking
- `test_manual_reorder_endpoint()` - Manual reorder trigger

**Purpose**: Verify configuration loading and ranking strategies.

#### `test_virtual_models.py`
Tests for virtual model routing:
- `test_virtual_model_coding_smart_uses_best_models()` - coding-smart routing
- `test_virtual_model_coding_fast_uses_fast_models()` - coding-fast routing
- `test_virtual_model_chat_smart_uses_conversation_models()` - chat-smart routing
- `test_virtual_model_aliases_resolved()` - Alias resolution

**Purpose**: Validate virtual model definitions and routing behavior.

#### `test_integration_router.py`
End-to-end integration tests:
- `test_e2e_coding_smart_with_provider_failures()` - E2E with failures
- `test_e2e_load_test_50_concurrent_requests()` - 50 concurrent load test
- `test_e2e_mixed_virtual_models_concurrent()` - Mixed virtual models
- High concurrency stress tests

**Purpose**: Comprehensive end-to-end testing with realistic scenarios.

### Fixtures Directory

#### `fixtures/benchmark_data.py`
Sample benchmark data for testing:
- `SAMPLE_MODEL_RANKINGS` - Model benchmark scores
- `SAMPLE_PROVIDER_PERFORMANCE` - Provider performance metrics
- `SAMPLE_VIRTUAL_MODELS` - Virtual model configurations

#### `fixtures/provider_mocks.py`
Mock provider responses and error scenarios:
- `MockProviderResponse` - Simulated provider response
- `create_mock_provider()` - Generic mock provider
- `create_failing_mock_provider()` - Provider with failures
- Error classes: `RateLimitError`, `TimeoutError`, `AuthError`, etc.

#### `fixtures/scenarios.py`
Predefined test scenarios:
- `SCENARIO_MIXED_FAILURES` - Mixed failure patterns
- `SCENARIO_CONCURRENT_INDEPENDENT` - Concurrent independence
- `SCENARIO_RATE_LIMIT_ENFORCEMENT` - Rate limit scenarios
- Helper functions: `create_request()`, `create_batch_requests()`

### Configuration Files

#### `conftest.py`
Shared pytest configuration and fixtures:
- Pytest configuration and markers
- Path setup for imports
- Common request fixtures
- Asyncio configuration

## Running Tests

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Category
```bash
# Concurrency tests (most critical)
pytest tests/test_concurrency.py -v

# Fallback logic tests
pytest tests/test_fallback_logic.py -v

# Performance tracking tests
pytest tests/test_performance_tracking.py -v

# Configuration tests
pytest tests/test_configuration.py -v

# Virtual model tests
pytest tests/test_virtual_models.py -v

# Integration tests
pytest tests/test_integration_router.py -v
```

### Run Specific Test
```bash
pytest tests/test_concurrency.py::TestConcurrentRequestsIndependence::test_concurrent_requests_independent_fallback_state -v
```

### Run with Coverage
```bash
pytest tests/ --cov=src/proxy_app --cov-report=html --cov-report=term
```

### Run Fast Tests Only (Exclude Slow)
```bash
pytest tests/ -v -m "not slow"
```

### Run Load Tests
```bash
pytest tests/ -v -m "load"
```

### Run Integration Tests
```bash
pytest tests/ -v -m "integration"
```

## Test Markers

Tests are marked with the following markers:
- `@pytest.mark.asyncio` - Async test requiring event loop
- `@pytest.mark.slow` - Slow test (long duration)
- `@pytest.mark.integration` - Integration test (E2E)
- `@pytest.mark.load` - Load/stress test

## Expected Failures

**Some tests are EXPECTED TO FAIL initially.** This is intentional - the tests serve as:
1. Validation of correct functionality
2. Documentation of expected behavior
3. Identification of gaps that need fixing

### Common Expected Failures

1. **Concurrency isolation failures** - If state is shared between requests
2. **Rate limit enforcement failures** - If limits aren't properly enforced
3. **Fallback logic failures** - If fallback chains don't work correctly
4. **Stats tracking failures** - If metrics aren't isolated per request
5. **Ranking failures** - If dynamic reordering doesn't work

## Test Development Guidelines

### Adding New Tests

1. **Choose the right file** - Place test in appropriate category file
2. **Use fixtures** - Leverage existing fixtures from `conftest.py` and `fixtures/`
3. **Mock providers** - Use `provider_mocks.py` for consistent mocking
4. **Document purpose** - Add clear docstrings explaining what the test validates
5. **Use scenarios** - Leverage predefined scenarios from `scenarios.py`

### Test Structure

```python
@pytest.mark.asyncio
async def test_feature_name(self, router):
    """
    Clear description of what this test validates.
    
    Expected behavior:
    - X should happen
    - Y should be tracked
    - Z should not occur
    """
    # Arrange
    # ... setup
    
    # Act
    # ... execute
    
    # Assert
    # ... verify
    
    # Debug output
    print(f"\\nDebug info: {info}")
```

### Mocking Best Practices

```python
# Use context manager for patches
with patch.object(router, '_execute_single_candidate', side_effect=mock_fn):
    result = await router.route(request)

# Track calls in mocks
call_log = []

async def tracking_mock(candidate, request, request_id):
    call_log.append({...})
    return MockProviderResponse(...)
```

## Debugging Failed Tests

### Enable Verbose Output
```bash
pytest tests/test_concurrency.py -v -s
```

### Run Single Test with Full Output
```bash
pytest tests/test_concurrency.py::TestClassName::test_method -v -s --tb=long
```

### Use pytest debugger
```bash
pytest tests/test_concurrency.py --pdb
```

### Check Coverage
```bash
pytest tests/ --cov=src/proxy_app --cov-report=term-missing
```

## Test Coverage Goals

Target coverage by module:
- `router_core.py` - 85%+
- `model_ranker.py` - 90%+
- `rate_limiter.py` - 90%+
- `provider_adapter.py` - 80%+

Critical paths (must be 90%+):
- Fallback logic
- Concurrency handling
- Stats tracking
- Rate limit enforcement

## Continuous Integration

Tests are designed to run in CI/CD pipelines:
- Fast tests complete in < 1 minute
- Load tests complete in < 5 minutes
- All tests complete in < 10 minutes

### CI Configuration

```yaml
# Example GitHub Actions
- name: Run fast tests
  run: pytest tests/ -v -m "not slow and not load"

- name: Run load tests
  run: pytest tests/ -v -m "load"
  
- name: Check coverage
  run: pytest tests/ --cov=src/proxy_app --cov-fail-under=80
```

## Performance Benchmarks

Expected test performance:
- Single test: < 1 second
- Test file: < 30 seconds
- Full suite: < 5 minutes (without load tests)
- With load tests: < 10 minutes

## Troubleshooting

### Import Errors
- Ensure `PYTHONPATH` includes project root
- Check `conftest.py` path setup
- Verify `__init__.py` files exist

### Async Errors
- Ensure `pytest-asyncio` is installed
- Use `@pytest.mark.asyncio` decorator
- Check event loop configuration

### Mock Errors
- Verify patch targets are correct
- Check mock side effects
- Use `return_exceptions=True` in `gather()`

### Timeout Errors
- Increase `asyncio.wait_for()` timeout
- Check for deadlocks in code
- Verify semaphore/lock usage

## Contributing

When adding tests:
1. Follow existing test patterns
2. Add docstrings explaining purpose
3. Use descriptive test names
4. Add debug output for failures
5. Update this README if adding new test categories

## References

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest-Asyncio](https://pytest-asyncio.readthedocs.io/)
- [unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
