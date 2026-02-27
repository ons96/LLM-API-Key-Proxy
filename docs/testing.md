# Testing & Coverage

## Running Tests

Run all tests:
```bash
pytest -c config/pytest.ini
```

Run with coverage report:
```bash
pytest -c config/pytest.ini --cov-report=html
```

## Coverage Configuration

Coverage is configured via `config/coverage.ini`.

Key settings:
- **Branch coverage**: Enabled to catch missed branches in conditionals
- **Source**: `src/proxy_app` and `src/rotator_library`
- **Omitted files**: Entry points (main.py), TUI launcher, and standalone scripts
- **Reports**: 
  - Terminal (with missing line numbers)
  - HTML (`htmlcov/index.html`)
  - XML (`coverage.xml`) for CI integration

## Viewing Coverage Reports

After running tests:
- Open `htmlcov/index.html` in a browser for detailed HTML report
- Check `coverage.xml` for CI/CD integration (Codecov, Coveralls, etc.)

## CI/CD Integration

For GitHub Actions, add:
```yaml
- name: Run tests with coverage
  run: pytest -c config/pytest.ini
- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    files: coverage.xml
```
