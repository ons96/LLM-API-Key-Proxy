# Contributing to LLM API Proxy

Thank you for your interest in contributing to the LLM API Proxy project! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Code Style](#code-style)
- [Testing](#testing)
- [Commit Messages](#commit-messages)
- [Pull Requests](#pull-requests)
- [Adding New Providers](#adding-new-providers)

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/LLM-API-Key-Proxy.git`
3. Create a branch: `git checkout -b feature/my-feature`
4. Make your changes
5. Push to your fork: `git push origin feature/my-feature`
6. Open a Pull Request

## Development Setup

### Prerequisites

- Python 3.10+
- pip (Python package manager)
- Git

### Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# (Optional) Install development dependencies
pip install black isort flake8 mypy pytest

# Run the server
python src/proxy_app/main.py
```

### Verify Installation

```bash
# Check server is running
curl http://localhost:8000/v1/models

# Test a chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "coding-fast", "messages": [{"role": "user", "content": "test"}]}'
```

## Making Changes

### Branch Naming

Use descriptive branch names:

- `feature/add-new-provider` - New features
- `fix/rate-limit-handling` - Bug fixes
- `docs/improve-readme` - Documentation changes
- `refactor/router-logic` - Code refactoring

### Code Style

This project follows these conventions:

#### Python

- **Formatter**: Black (88-char line limit)
- **Imports**: isort (Black-compatible)
- **Linting**: flake8
- **Type hints**: Required for function signatures

```bash
# Format code
black src/

# Sort imports
isort src/

# Check linting
flake8 src/

# Type check
mypy src/
```

#### Docstrings

Use Google-style docstrings:

```python
def process_request(model: str, messages: List[dict]) -> dict:
    """Process a chat completion request.
    
    Args:
        model: The model name (e.g., "coding-elite").
        messages: List of message dictionaries.
    
    Returns:
        The completion response as a dictionary.
    
    Raises:
        ValueError: If model is not found.
        HTTPException: If the request fails.
    """
    pass
```

#### Type Hints

All public functions must have type hints:

```python
# Good
def get_provider(name: str) -> Optional[ProviderInterface]:
    ...

# Bad
def get_provider(name):
    ...
```

### Configuration Files

- **YAML files**: 2-space indentation
- **JSON files**: 2-space indentation
- **Python files**: 4-space indentation

## Testing

### Run Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_router.py

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_router.py::test_free_only_mode
```

### Writing Tests

Place tests in the `tests/` directory:

```python
import pytest

class TestMyFeature:
    """Test my feature."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        return TestClient()
    
    def test_feature_works(self, client):
        """Test that the feature works."""
        result = client.do_something()
        assert result.success is True
```

### Test Coverage

- Aim for >80% coverage on new code
- Test edge cases and error conditions
- Mock external API calls

## Commit Messages

Follow conventional commits format:

```
type(scope): brief description

Longer explanation if needed.

Fixes #issue
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting (no code change)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

### Examples

```
feat(router): add support for Mistral provider

Add Mistral API integration with credential rotation
and rate limit handling.

Fixes #112
```

```
fix(g4f): handle complex model IDs correctly

Strip version suffixes from model IDs before sending
to G4F backend.

Fixes #125
```

## Pull Requests

### Before Submitting

- [ ] Code is formatted (`black src/`)
- [ ] Imports are sorted (`isort src/`)
- [ ] Tests pass (`pytest`)
- [ ] Type checks pass (`mypy src/`)
- [ ] Commit messages follow convention

### PR Template

```markdown
## Summary
Brief description of changes.

## Changes Made
- Change 1
- Change 2

## Testing
How to test these changes.

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] CHANGELOG updated (if applicable)
```

### Review Process

1. At least one approval required
2. All CI checks must pass
3. No merge conflicts
4. Branch must be up to date with main

## Adding New Providers

### Step 1: Create Provider File

Create `src/rotator_library/providers/my_provider.py`:

```python
from typing import List
import httpx
from .provider_interface import ProviderInterface

class MyProvider(ProviderInterface):
    """Adapter for MyProvider API."""
    
    provider_env_name = "my_provider"
    
    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """Fetch available models from MyProvider."""
        response = await client.get(
            "https://api.myprovider.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        models = response.json().get("data", [])
        return [f"my_provider/{model['id']}" for model in models]
```

### Step 2: Add Configuration

Add to `config/router_config.yaml`:

```yaml
providers:
  my_provider:
    enabled: true
    free: true  # or false if paid
    free_tier_models:
      - my_provider/model-1
      - my_provider/model-2
```

### Step 3: Add Environment Variable

Update `.env.example`:

```env
MY_PROVIDER_API_KEY=your-api-key
```

### Step 4: Write Tests

Create `tests/test_my_provider.py`:

```python
import pytest
from src.rotator_library.providers.my_provider import MyProvider

class TestMyProvider:
    def test_strip_provider_prefix(self):
        provider = MyProvider()
        # Add tests
```

### Step 5: Document

Update documentation:
- `README.md` - Add to provider list
- `docs/PROVIDER_ARCHITECTURE.md` - Add to registry table
- `AGENTS.md` - Update if needed

## Getting Help

- **Issues**: https://github.com/ons96/LLM-API-Key-Proxy/issues
- **Documentation**: See `docs/` directory

Thank you for contributing!
