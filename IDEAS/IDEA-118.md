# AGENTS.md - Mirrowel Configuration with Parallel Agents and Fallbacks

## 1. Role/Mission

**Mission:** Configure Mirrowel with parallel agent execution capabilities, automatic fallbacks for rate/usage limits, and establish a custom model group containing agentic coding models prioritized from best to worst performance.

**Goal:** Enable OpenCode to reliably execute coding tasks using free LLM resources with graceful degradation when primary models are unavailable or rate-limited.

**Key Objectives:**
- Set up parallel agent infrastructure for concurrent task execution
- Implement intelligent fallback system that redirects to backup models when rate/usage limits are hit
- Create ordered model groups with agentic coding capabilities
- Ensure resilience and reliability for autonomous coding operations on GitHub Actions

---

## 2. Technical Stack

**Core Technologies:**
- **Mirrowel**: Configuration wrapper for multi-LLM provider routing
- **Python**: Primary implementation language (3.10+)
- **aiohttp/asyncio**: Async HTTP requests for parallel agent execution
- **GitHub Actions**: CI/CD and autonomous agent runtime environment

**LLM Providers (Free Tier Focus):**
- OpenAI (with free tier limits)
- Anthropic (with fallback tiers)
- Ollama (local self-hosted option)
- OpenRouter (aggregated free tier access)

**Key Libraries:**
- `mirrowel`: Main configuration library
- `asyncio`: Concurrent execution
- `tenacity`: Retry/fallback logic
- `pydantic`: Configuration validation
- `pytest`: Testing framework
- `pytest-asyncio`: Async test support
- `httpx`: Async HTTP client

---

## 3. Requirements (Numbered)

### 3.1 Core Configuration

1. **Install and configure Mirrowel**
   - Set up `mirrowel` package with proper configuration file
   - Configure provider credentials via environment variables
   - Validate installation with basic connectivity test

2. **Create custom model group definition**
   - Define `agentic_coding` model group
   - Order models: Best → Worst performance/capability
   - Include at least 3-5 models with clear hierarchy
   - Prioritize free tier availability

3. **Implement parallel agent configuration**
   - Configure 2-4 parallel agents for concurrent execution
   - Set up proper async/await patterns
   - Define task distribution logic

### 3.2 Fallback System

4. **Implement rate limit detection**
   - Add middleware to detect 429 status codes
   - Create rate limit tracking per provider
   - Implement exponential backoff logic

5. **Implement usage limit handling**
   - Track token usage per model/group
   - Detect usage quota exhaustion
   - Trigger automatic fallback on limit hits

6. **Create fallback chain logic**
   - Define ordered fallback sequence
   - Implement state preservation across retries
   - Add logging for fallback decisions

### 3.3 Async Execution

7. **Configure async request handling**
   - Set up asyncio event loops
   - Implement concurrent request handling
   - Add proper timeout configurations

8. **Implement parallel execution flow**
   - Create parallel agent task queue
   - Add result aggregation logic
   - Implement error handling per agent

### 3.4 Integration

9. **Integrate with OpenCode**
   - Ensure Mirrowel config works within OpenCode context
   - Test actual coding task execution
   - Validate full pipeline functionality

10. **Add monitoring and logging**
    - Create usage tracking metrics
    - Add fallback event logging
    - Implement observability for debugging

---

## 4. File Structure

```
mirrowel-config/
├── AGENTS.md                          # This file
├── QUESTIONS.md                      # Questions for human review
├── pyproject.toml                    # Project configuration
├── .env.example                     # Environment variables template
├── README.md                        # Documentation
│
├── src/
│   └── mirrowel_config/
│       ├── __init__.py
│       ├── config.py                # Main configuration
│       ├── models.py               # Model group definitions
│       ├── fallback.py             # Fallback logic
│       ├── parallel.py             # Parallel agent execution
│       └── client.py              # Mirrowel client wrapper
│
├── configs/
│   ├── mirrowel.yaml               # Primary Mirrowel config
│   ├── model_groups.yaml          # Model group definitions
│   └── providers.yaml             # Provider configurations
│
├── tests/
│   ├── __init__.py
│   ├── test_config.py             # Config tests
│   ├── test_fallback.py           # Fallback tests
│   ├── test_parallel.py           # Parallel execution tests
│   └── test_integration.py        # Integration tests
│
└── scripts/
    ├── setup.py                   # Setup script
    ├── validate.py               # Validation script
    └── demo.py                   # Demo script
```

### 4.1 Configuration Files Detail

**.env.example:**
```
# OpenAI Configuration
OPENAI_API_KEY=sk-...
OPENAI_ORG_ID=org-...

# Anthropic Configuration
ANTHROPIC_API_KEY=sk-...

# OpenRouter Configuration
OPENROUTER_API_KEY=...

# Ollama Configuration (local)
OLLAMA_BASE_URL=http://localhost:11434

# Mirrowel Settings
MIRROWEL_DEFAULT_MODEL_GROUP=agentic_coding
MIRROWEL_MAX_PARALLEL_AGENTS=3
MIRROWEL_RATE_LIMIT_RETRIES=3
MIRROWEL_TIMEOUT=120
```

**configs/mirrowel.yaml:**
```yaml
version: "1.0"

# Default settings
defaults:
  timeout: 120
  max_retries: 3
  retry_delay: 2
  
# Enable parallel execution
parallel:
  enabled: true
  max_agents: 3
  task_timeout: 300
  
# Model group to use
model_group: agentic_coding

# Fallback configuration
fallback:
  enabled: true
  retry_on_rate_limit: true
  retry_on_usage_limit: true
  max_fallback_attempts: 3
```

**configs/model_groups.yaml:**
```yaml
model_groups:
  agentic_coding:
    name: "Agentic Coding Models"
    description: "Ordered list of models for coding tasks, best to worst"
    models:
      - name: "claude-sonnet-4-20250514"
        provider: "anthropic"
        priority: 1
        capabilities: ["coding", "reasoning", "tool_use"]
      - name: "gpt-4.1"
        provider: "openai"
        priority: 2
        capabilities: ["coding", "reasoning"]
      - name: "gpt-4o"
        provider: "openai"
        priority: 3
        capabilities: ["coding", "vision"]
      - name: "claude-3-haiku-20240307"
        provider: "anthropic"
        priority: 4
        capabilities: ["fast_coding"]
      - name: "qwen-coder-32b"
        provider: "ollama"
        priority: 5
        capabilities: ["local", "coding"]
        
  fast_coding:
    name: "Fast Coding Models"
    description: "Quick response models for simple tasks"
    models:
      - name: "claude-3-haiku-20240307"
        provider: "anthropic"
      - name: "gpt-4o-mini"
        provider: "openai"
```

---

## 5. Testing Requirements

### 5.1 Unit Tests

**test_config.py:**
```python
# Test configuration loading
def test_load_mirrowel_config():
    """Test that Mirrowel config loads correctly"""
    pass

def test_validate_model_groups():
    """Test model group validation"""
    pass

def test_environment_variables():
    """Test env var loading and defaults"""
    pass
```

**test_fallback.py:**
```python
# Test fallback logic
@pytest.mark.asyncio
async def test_rate_limit_detection():
    """Test detection of 429 responses"""
    pass

@pytest.mark.asyncio
async def test_usage_limit_detection():
    """Test detection of usage quota exhaustion"""
    pass

@pytest.mark.asyncio
async def test_fallback_chain_execution():
    """Test fallback chain proceeds correctly"""
    pass

@pytest.mark.asyncio
async def test_exponential_backoff():
    """Test backoff timing"""
    pass
```

### 5.2 Integration Tests

**test_parallel.py:**
```python
# Test parallel execution
@pytest.mark.asyncio
async def test_parallel_agent_spawn():
    """Test spawning multiple agents"""
    pass

@pytest.mark.asyncio
async def test_parallel_result_aggregation():
    """Test results are aggregated correctly"""
    pass

@pytest.mark.asyncio
async def test_parallel_error_handling():
    """Test errors in parallel are handled"""
    pass
```

**test_integration.py:**
```python
# Integration tests
@pytest.mark.asyncio
async def test_full_mirrowel_execution():
    """Test complete Mirrowel execution flow"""
    pass

@pytest.mark.asyncio
async def test_fallback_integration():
    """Test fallback works in real scenario"""
    pass

@pytest.mark.asyncio
async def test_opencode_integration():
    """Test integration with OpenCode context"""
    pass
```

### 5.3 Test Execution

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_fallback.py -v

# Run with coverage
pytest tests/ --cov=mirrowel_config --cov-report=html

# Run async tests
pytest tests/ -v --asyncio-mode=auto
```

---

## 6. Git Protocol

### 6.1 Branch Strategy

```
main                    # Stable configuration
├── dev                 # Development branch
│   ├── feature/parallel-agents
│   ├── feature/fallback-system
│   └── feature/model-groups
└── release            # Release candidates
```

### 6.2 Commit Messages

**Format:** `type(scope): description`

**Types:**
- `feat`: New feature implementation
- `fix`: Bug fix
- `config`: Configuration changes
- `test`: Test additions/changes
- `docs`: Documentation updates
- `refactor`: Code refactoring

**Examples:**
```
feat(config): Add agentic_coding model group definition
feat(fallback): Implement rate limit detection middleware
feat(parallel): Add parallel agent execution support
fix(fallback): Correct backoff timing calculation
config(env): Update .env.example with new variables
test(fallback): Add fallback chain tests
```

### 6.3 Pull Request Process

1. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature
   ```

2. **Make changes and commit**
   ```bash
   git add .
   git commit -m "feat(scope): description"
   ```

3. **Push and create PR**
   ```bash
   git push -u origin feature/your-feature
   gh pr create --title "Feature: description"
   ```

4. **Request review**
   - Tag for review
   - Link related issues
   - Add testing notes

### 6.4 Important Files to Commit

**Always commit:**
- `configs/*.yaml` - Configuration files
- `src/` - Source code
- `tests/` - Test files
- `pyproject.toml` - Project config

**Potentially skip (add to .gitignore):**
- `.env` - Actual environment variables
- `__pycache__/` - Python cache
- `.pytest_cache/` - Test cache
- `*.pyc` - Compiled Python
- `htmlcov/` - Coverage reports

---

## 7. Completion Criteria

### 7.1 Configuration Complete

- [ ] Mirrowel package installed and validated
- [ ] YAML configuration files created and valid
- [ ] Environment variables template (.env.example) created
- [ ] Model groups defined with 3+ models ordered best to worst
- [ ] Provider credentials configured for free tier access

### 7.2 Fallback System Complete

- [ ] Rate limit detection functioning (detects 429 responses)
- [ ] Usage limit tracking implemented
- [ ] Automatic fallback triggers on limit hits
- [ ] Exponential backoff working correctly
- [ ] Fallback chain logs decisions

### 7.3 Parallel Execution Complete

- [ ] Parallel agent configuration set (2-4 agents)
- [ ] Async execution working
- [ ] Task distribution functioning
- [ ] Result aggregation implemented
- [ ] Proper timeout handling in place

### 7.4 Testing Complete

- [ ] Unit tests for configuration pass
- [ ] Unit tests for fallback logic pass
- [ ] Unit tests for parallel execution pass
- [ ] Integration tests pass
- [ ] Test coverage > 70%

### 7.5 Integration Complete

- [ ] Works within OpenCode context
- [ ] Free tier resources prioritized
- [ ] Actual coding task execution working
- [ ] No hard dependencies requiring paid access
- [ ] Graceful degradation on failures

### 7.6 Documentation Complete

- [ ] AGENTS.md complete (this file)
- [ ] QUESTIONS.md created for clarification
- [ ] README.md updated with setup instructions
- [ ] Configuration comments clear
- [ ] Git protocol followed

### 7.7 Success Indicators

**Functional Validation:**
```bash
# Test configuration loads
python -c "from mirrowel_config import config; print('Config loaded')"

# Test parallel execution
python scripts/demo.py

# Run full test suite
pytest tests/ -v

# Check code coverage
pytest tests/ --cov=mirrowel_config --cov-report=term-missing
```

**Expected Outputs:**
- All tests pass (100% pass rate)
- No 429 errors in normal operation
- Automatic fallback occurs when limits hit
- Parallel agents execute concurrently
- Configuration creates valid requests

---

## Notes for Autonomous Agent

1. **Free Resources Only**: Prioritize free tier providers (Anthropic, Ollama, OpenRouter free tiers). Do NOT require paid API access.

2. **Independent Decisions**: Make configuration decisions based on availability and reliability. Save questions to QUESTIONS.md if unclear.

3. **Resilience**: Ensure system degrades gracefully - when one model fails, automatically try next in chain.

4. **GitHub Actions**: All execution happens in GitHub Actions context. Consider resource limits and timeouts appropriate for that environment.

5. **Testing**: Write tests that can run in CI/CD without external dependencies beyond configured APIs.

6. **Verification**: Validate each component works independently before integration testing.