# AGENTS.md - Unified Model Mapping Across Providers

## 1. Role/Mission

### Role
You are an autonomous coding agent responsible for implementing a unified model mapping system that identifies, maps, and combines identical AI models from different providers.

### Mission
Your mission is to create a robust system that:
- Identifies when different providers offer the same underlying AI model
- Normalizes provider-specific model names to a unified internal identifier
- Creates a comprehensive mapping that allows users to access equivalent models through any provider
- Handles edge cases such as partial matches, version differences, and family relationships

### Decision Authority
You have full authority to:
- Choose implementation approaches and algorithms
- Determine mapping confidence thresholds
- Decide how to handle ambiguous matches
- Structure the project as you see fit, within the constraints

### Questions Protocol
If you encounter unclear requirements, blockers, or need clarification:
1. First attempt to make a reasonable independent decision
2. Document any significant questions in `QUESTIONS.md` with your proposed approach
3. Continue working on other aspects while awaiting answers

---

## 2. Technical Stack

### Core Language
- **Python 3.10+** - Primary implementation language

### Dependencies (Free Tier)
- `pyyaml` - For configuration and model definitions
- `requests` - For potential provider API testing
- `pytest` - For testing framework
- `pytest-cov` - For coverage reporting
- `click` or `typer` - For CLI interface

### Data Storage
- **YAML files** - For model definitions and mappings (human-readable, no database required)
- **JSON files** - For API responses and cached data

### Version Control
- **Git** - Version control
- **GitHub Actions** - For CI/CD automation

### No External Paid Services
- All data sources must be publicly available or mock-able
- No paid API keys required for development

---

## 3. Requirements (Numbered)

### R1: Model Fingerprinting System
- Create a function that generates a unique fingerprint for any AI model based on:
  - Model family (e.g., "gpt", "claude", "llama")
  - Parameter count or size tier (e.g., "7b", "70b")
  - Version/iteration (e.g., "v1", "v2", "2024-01")
  - Fine-tuning variant (if applicable)
- Fingerprint must be deterministic and hashable

### R2: Provider Normalization
- Implement normalization for at least 3 major providers:
  - OpenAI (GPT models)
  - Anthropic (Claude models)
  - Meta/Official (Llama models)
  - Plus any others you choose to add
- Normalize model names to a standard format: `{family}-{size}-{version}`

### R3: Unified Model List
- Create a comprehensive YAML file containing:
  - All known models from each provider
  - Their normalized names
  - Their fingerprints
  - Mapping to equivalent models from other providers

### R4: Match Detection Algorithm
- Implement algorithm to detect:
  - **Exact matches**: Same family, size, version
  - **Family matches**: Same family, different sizes
  - **Partial matches**: Similar but not identical
- Support confidence scoring (100%, 90%, 75%, etc.)

### R5: Query Interface
- Create a CLI tool or Python API that:
  - Accepts a model name from any provider
  - Returns all known equivalent models
  - Provides confidence level of match
  - Lists all providers offering the model

### R6: Extensibility Framework
- Design system to easily add new providers:
  - Provider plugin structure
  - Configuration-based additon
  - Clear documentation for adding providers

### R7: Validation Utilities
- Implement validation functions to:
  - Check for duplicate mappings
  - Verify fingerprint uniqueness
  - Detect missing provider mappings

---

## 4. File Structure

```
unified-model-mapper/
├── LICENSE
├── README.md
├── AGENTS.md
├── QUESTIONS.md
├── pyproject.toml
├── requirements.txt
├── setup.py
│
├── unified_model_mapper/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── fingerprint.py      # Model fingerprinting logic
│   │   ├── normalizer.py      # Provider name normalization
│   │   ├── matcher.py          # Match detection algorithm
│   │   └── validator.py       # Data validation utilities
│   │
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py             # Base provider class
│   │   ├── openai.py           # OpenAI provider
│   │   ├── anthropic.py        # Anthropic provider
│   │   ├── meta.py             # Meta/Llama provider
│   │   └── registry.py         # Provider registry
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── models.yaml         # Master model definitions
│   │   ├── mappings.yaml       # Cross-provider mappings
│   │   └── provider_config.yamlProvider configs
│   │
│   ├── cli/
│   │   ├── __init__.py
│   │   └── main.py             # CLI entry point
│   │
│   └── utils/
│       ├── __init__.py
│       └── logger.py           # Logging utilities
│
├── tests/
│   ├── __init__.py
│   ├── test_fingerprint.py
│   ├── test_normalizer.py
│   ├── test_matcher.py
│   ├── test_providers/
│   │   ├── __init__.py
│   │   ├── test_openai.py
│   │   ├── test_anthropic.py
│   │   └── test_meta.py
│   ├── test_validator.py
│   ├── test_cli.py
│   └── fixtures/
│       ├── __init__.py
│       ├── models_sample.yaml
│       └── sample_queries.json
│
├── docs/
│   ├── adding_provider.md
│   ├── api_reference.md
│   └── architecture.md
│
└── scripts/
    ├── generate_fingerprints.py
    ├── validate_mappings.py
    └── export_mappings.py
```

---

## 5. Testing Requirements

### Test Coverage Expectations
- Minimum **80% code coverage** required
- All core functions must have unit tests
- All provider implementations must have tests

### Test Categories

#### Unit Tests
- `test_fingerprint.py`: Test fingerprint generation for various model types
- `test_normalizer.py`: Test name normalization for all providers
- `test_matcher.py`: Test match detection with various confidence levels
- `test_validator.py`: Test validation utilities

#### Integration Tests
- `test_providers/test_*.py`: Test each provider implementation
- `test_cli.py`: Test CLI commands end-to-end

#### Test Data
- Use `tests/fixtures/` for sample data
- Include edge cases: unknown providers, partial matches, etc.

### Running Tests
```bash
# Run all tests with coverage
pytest --cov=unified_model_mapper --cov-report=html

# Run specific test file
pytest tests/test_fingerprint.py

# Run with verbose output
pytest -v
```

### CI Integration
- GitHub Actions should run tests on every push
- Coverage report should be generated and attached
- Tests must pass before merge

---

## 6. Git Protocol

### Branch Strategy
- **Main branch**: `main` - Production-ready code only
- **Development branch**: `develop` - Integration branch
- **Feature branches**: `feature