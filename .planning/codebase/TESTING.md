# Testing

- **Framework**: Pytest (async-heavy). Tests live in `tests/`; fixtures in `tests/fixtures/`.
- **Commands**:
  - Run all: `pytest tests/ -v`
  - Common markers: `-m "not slow"`, `-m "integration"`, `-m "load"`.
  - Per-file examples: `pytest tests/test_concurrency.py -v`, `pytest tests/test_fallback_logic.py -v`, `pytest tests/test_virtual_models.py -v`.
  - Coverage (suggested): `pytest tests/ --cov=src/proxy_app --cov-report=term`.
- **Dependencies**: `tests/requirements-test.txt` (pytest, pytest-asyncio, etc.).
- **Focus areas (per tests/README.md)**: concurrency isolation, fallback ordering, performance tracking, configuration/ranking, virtual models, integration router flows, G4F resilience, responses endpoint.
- **Expected behaviors**: Some tests marked expected-fail to document gaps (rate limits, fallback completeness, stats isolation).
- **Test data/mocks**: `tests/fixtures/benchmark_data.py`, `provider_mocks.py`, `scenarios.py` for predefined inputs and failure patterns.
