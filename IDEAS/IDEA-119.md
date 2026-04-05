# AGENTS.md - Multi-Provider Integration for Mirrowel

---

## 1. Role/Mission

**Role:** Autonomous Coding Agent

**Mission:** Implement a flexible, multi-provider API integration system for Mirrowel that allows the application to seamlessly use multiple LLM providers (including g4f, puter.js, and additional third-party APIs) through a unified interface. The agent must ensure all integrations are configurable, extensible, and operational without requiring paid resources—leveraging free tiers, trial accounts, and open-source solutions exclusively.

**Primary Objectives:**

- Integrate g4f (Go-g4f) as the primary multi-provider wrapper
- Evaluate and integrate puter.js or leverage its API capabilities via g4f
- Identify and configure additional free-tier LLM providers
- Create a unified provider abstraction layer for easy provider switching
- Ensure all configurations can be managed via environment variables or config files
- Maintain backward compatibility with any existing provider implementations
- Document all provider configurations and usage instructions

**Decision-Making Authority:**

- The agent may autonomously determine provider implementations, code structure, and testing approaches
- The agent may create new modules/files as needed to achieve the objectives
- The agent may skip non-essential features if free resources are unavailable
- The agent must save any blockers or clarification questions to QUESTIONS.md

---

## 2. Technical Stack

**Core Technologies:**

- **Language:** Python 3.9+ (primary backend)
- **Package Manager:** pip / Poetry (to be determined from existing project)
- **LLM Provider Wrapper:** g4f (https://github.com/xtekky/g4f)
- **JavaScript Integration:** puter.js (https://github.com/HeyPuter/puter) - if needed for JS-based AI features
- **HTTP Client:** httpx (async) or requests (sync)
- **Configuration Management:** python-dotenv for environment variables
- **Logging:** Python standard logging or loguru
- **Testing:** pytest with pytest-asyncio for async tests

**Provider Focus (Free Resources):**

- g4f built-in providers (OpenRouter, Liaobots, etc.)
- OpenAI free tier (if available)
- Anthropic (free tier trials when available)
- Google Gemini (free tier)
- Local/Offline alternatives (LLM locally via ollama or similar) as fallback
- Any other freely accessible LLM APIs discovered during research

**Version Constraints:**

- Minimum Python: 3.9
- g4f: Latest stable version compatible with Python 3.9+
- All dependencies must have MIT/Apache-2.0 compatible licenses

---

## 3. Requirements (numbered)

### Phase 1: Provider Research & Selection

1. **Research g4f Documentation**
   - Review g4f capabilities, supported providers, and configuration options
   - Identify all free-tier providers available through g4f
   - Document provider limitations (rate limits, model availability)

2. **Research Additional Free Providers**
   - Investigate other freely accessible LLM APIs beyond g4f
   - Evaluate puter.js capabilities and determine integration approach
   - Create a list of priority providers ranked by reliability and features

3. **Assess Current Mirrowel Architecture**
   - Review existing provider implementation (if any)
   - Identify integration points and potential refactoring needs
   - Determine backward compatibility requirements

### Phase 2: Core Implementation

4. **Create Unified Provider Abstraction Layer**
   - Design and implement a `ProviderInterface` abstract base class
   - Define standard methods: `chat_completion()`, `embeddings()`, `get_models()`, `health_check()`
   - Ensure consistent response format across all providers

5. **Implement g4f Integration**
   - Create `G4FProvider` class extending ProviderInterface
   - Implement async/sync support based on existing codebase patterns
   - Add proper error handling and retry logic
   - Configure default models for each provider

6. **Implement puter.js Integration**
   - Evaluate puter.js as standalone or via g4f
   - Create `PuterProvider` class if standalone integration needed
   - Verify API compatibility and feature parity

7. **Implement Additional Providers**
   - Add at least 2-3 additional providers from the free provider list
   - Ensure provider fallback mechanisms (if one fails, try another)
   - Document provider-specific configurations

8. **Create Configuration System**
   - Implement environment variable-based configuration
   - Support config file (YAML/JSON) for provider settings
   - Create `.env.example` with all required configuration keys
   - Add provider priority/fallback ordering configuration

### Phase 3: Operational Requirements

9. **Error Handling & Resilience**
   - Implement provider health checking with automatic failover
   - Add rate limit detection and backoff strategies
   - Create proper logging for all provider interactions
   - Handle authentication errors gracefully

10. **Provider Management Features**
    - Create CLI or API endpoint to list available providers
    - Implement provider status monitoring
    - Add dynamic provider switching capability
    - Create usage tracking per provider

11. **Documentation**
    - Add docstrings to all new classes and methods
    - Create PROVIDERS.md with usage instructions for each provider
    - Document configuration options and environment variables
    - Add examples for common use cases

### Phase 4: Polish & Testing

12. **Unit Tests**
    - Write unit tests for all new provider classes
    - Mock external API calls in tests (use responses library or similar)
    - Achieve minimum 70% code coverage on new modules

13. **Integration Tests**
    - Test provider failover behavior
    - Verify configuration loading
    - Test error handling paths

14. **Performance & Resource Testing**
    - Verify no memory leaks during provider switching
    - Test concurrent request handling
    - Verify timeout configurations work correctly

---

## 4. File Structure

**Proposed Directory Structure:**

```
mirrowel/
├── src/
│   └── mirrowel/
│       ├── __init__.py
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── base.py              # Abstract provider interface
│       │   ├── g4f_provider.py      # g4f implementation
│       │   ├── puter_provider.py    # puter.js integration
│       │   ├── anthropic_provider.py
│       │   ├── google_provider.py
│       │   ├── openrouter_provider.py
│       │   └── factory.py           # Provider factory
│       ├── config/
│       │   ├── __init__.py
│       │   ├── settings.py           # Configuration management
│       │   └── provider_config.py   # Provider-specific config
│       ├── core/
│       │   ├── __init__.py
│       │   ├── client.py             # Main client class
│       │   └── router.py             # Provider routing/failover
│       └── utils/
│           ├── __init__.py
│           ├── logging.py           # Logging utilities
│           └── validators.py        # Input validators
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_providers/
│   │   │   ├── __init__.py
│   │   │   ├── test_base.py
│   │   │   ├── test_g4f_provider.py
│   │   │   └── test_factory.py
│   │   └── test_config/
│   │       ├── __init__.py
│   │       └── test_settings.py
│   └── integration/
│       ├── __init__.py
│       ├── test_provider_failover.py
│       └── test_integration.py
├── .env.example
├── pyproject.toml
└── README.md
```

**Key Files to Create/Modify:**

| File | Purpose | Action |
|------|---------|--------|
| `src/mirrowel/providers/base.py` | Abstract base class for all providers | CREATE |
| `src/mirrowel/providers/g4f_provider.py` | g4f wrapper implementation | CREATE |
| `src/mirrowel/providers/factory.py` | Provider instantiation factory | CREATE |
| `src/mirrowel/config/settings.py` | Configuration management | CREATE |
| `src/mirrowel/core/client.py` | Main client with multi-provider support | MODIFY |
| `src/mirrowel/core/router.py` | Provider routing and failover logic | CREATE |
| `.env.example` | Example environment configuration | CREATE |
| `PROVIDERS.md` | Provider-specific documentation | CREATE |
| `QUESTIONS.md` | Questions for human review | CREATE |

---

## 5. Testing Requirements

**Overview:**

- Use pytest as the testing framework
- All tests must pass on GitHub Actions (free tier)
- Minimize external API calls in tests by using mocks
- Prioritize fast, reliable tests over comprehensive coverage

**Test Naming Convention:**

```
test