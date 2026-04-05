# AGENTS.md - Smart LLM Auto-Selector Based on Query Complexity

## 1. Role/Mission

**Role**: Autonomous AI Coding Agent
**Project**: Smart LLM Auto-Selector

**Mission**: Build an intelligent routing system that analyzes user prompts for complexity and automatically selects the most appropriate LLM from a pool of available models. The system should optimize for cost-efficiency and latency while maintaining response quality.

**Key Objectives**:
- Analyze incoming prompts to determine complexity score (1-100 scale)
- Route simple prompts to fast/cheap models (e.g., GPT-4o-mini, Claude-3-haiku)
- Route complex prompts to reasoning models (e.g., GPT-4o, Claude-3.5-sonnet, Gemini-Advanced)
- Handle multi-step reasoning tasks with capable models
- Minimize response latency where possible
- Track and log routing decisions for analysis

---

## 2. Technical Stack

**Language**: Python 3.10+
**Cloud Platform**: Free tier eligible (details below)
**APIs Used**: OpenAI API (free tier credits), Anthropic API (free tier credits), Google Gemini API (free tier)

**Core Dependencies**:
- `fastapi` - REST API server (free tier hosting)
- `uvicorn` - ASGI server
- `pydantic` - Data validation
- `httpx` - HTTP client
- `python-dotenv` - Environment variable management
- `scikit-learn` - Complexity scoring model (optional, can use heuristic)
- `pytest` - Testing framework
- `pytest-asyncio` - Async testing

**Free Resources**:
- OpenAI: $5 free credit for new accounts (use for testing)
- Anthropic: $10 free credit for new accounts (use for testing)
- Google AI Studio: Free tier with generous limits
- Render/Vercel/Fly.io: Free tier for hosting API (max 1 instance)

---

## 3. Requirements (Numbered)

### Core Requirements

1. **Complexity Analyzer Module**
   - Implement a function `analyze_complexity(prompt: str) -> int` that returns a score 1-100
   - Use heuristic analysis counting: words, punctuation, nested instructions, technical terms
   - Include keyword detection for reasoning indicators: "explain", "why", "analyze", "compare", "derive"

2. **Model Registry**
   - Create a configuration file `models.json` containing:
     - Model ID, provider name, capabilities, speed tier, cost per 1K tokens
     - Categorization: "fast" (simple tasks) vs "reasoning" (complex tasks)
   - Support minimum 3 models from different providers:
     - Fast: gpt-4o-mini, claude-3-haiku, or gemini-1.5-flash
     - Reasoning: gpt-4o, claude-3.5-sonnet, or gemini-1.5-pro

3. **Router Engine**
   - Implement `select_model(complexity_score: int, models: list) -> Model`
   - Threshold logic: score 0-30 → fast model, 30-70 → balanced, 70-100 → reasoning model
   - Allow override via explicit user request

4. **API Endpoint**
   - POST `/api/chat` - Accept prompt, return model response with routing info
   - GET `/api/models` - List available models
   - GET `/api/health` - Health check endpoint

5. **API Client Wrapper**
   - Create unified client that interfaces with OpenAI, Anthropic, Google APIs
   - Handle authentication via environment variables
   - Implement fallback logic if primary model fails

6. **Request Logging**
   - Log each request: timestamp, prompt length, complexity score, selected model, response time
   - Store logs in JSON format for analysis

### Infrastructure Requirements

7. **Environment Configuration**
   - Use `.env` file for API keys (DO NOT commit to version control)
   - Create `.env.example` template with placeholder values
   - Validate all required env vars on startup

8. **Error Handling**
   - Implement retry logic with exponential backoff (max 3 retries)
   - Graceful fallback between models of similar capability
   - Proper error responses with meaningful messages

9. **Configuration Management**
   - All thresholds and settings in `config.yaml`
   - No hardcoded values in business logic

---

## 4. File Structure

```
smart-llm-selector/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI workflow
├── src/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Configuration loading
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_registry.py   # Model configurations
│   │   └── model_types.py     # Pydantic models
│   ├── analyzer/
│   │   ├── __init__.py
│   │   └── complexity.py       # Complexity scoring logic
│   ├── router/
│   │   ├── __init__.py
│   │   └── engine.py           # Model selection logic
│   ├── clients/
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract LLM client
│   │   ├── openai_client.py    # OpenAI implementation
│   │   ├── anthropic_client.py # Anthropic implementation
│   │   ├── google_client.py    # Google Gemini implementation
│   │   └── unified.py          # Unified client with fallback
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py           # API endpoints
│   └── utils/
│       ├── __init__.py
│       └── logger.py           # Logging utilities
├── tests/
│   ├── __init__.py
│   ├── test_analyzer.py       # Complexity analyzer tests
│   ├── test_router.py         # Router engine tests
│   ├── test_clients.py       # Client wrapper tests
│   └── test_api.py           # API endpoint tests
├── config/
│   └── config.yaml            # Configuration file
├── data/
│   ├── models.json           # Model registry data
│   └── logs/                 # Request logs directory
├── .env.example              # Environment template
├── .gitignore                # Git ignore rules
├── requirements.txt          # Python dependencies
├── Dockerfile                # Container definition
└── README.md                 # Project documentation
```

---

## 5. Testing Requirements

**Test Coverage Goal**: 80%+ code coverage

### Unit Tests

1. **Complexity Analyzer Tests**
   - Test simple prompts (score should be low: 0-30)
   - Test complex prompts with reasoning keywords
   - Test edge cases: empty string, very long prompts, special characters

2. **Router Engine Tests**
   - Test threshold boundaries (30, 70)
   - Test selection matches expected model category
   - Test override functionality

3. **Client Tests**
   - Test mock API responses
   - Test error handling and retries
   - Test fallback logic

### Integration Tests

4. **API Tests**
   - Test `/api/health` returns 200
   - Test `/api/models` returns model list
   - Test `/api/chat` processes prompt and returns response

### Running Tests

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_analyzer.py -v

# Run with verbose output
pytest -vv
```

---

## 6. Git Protocol

**Branch Strategy**: Simple single-branch workflow

### Workflow

1. **Start**: Create working branch from main
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/your-feature-name
   ```

2. **Work**: Make commits incrementally
   ```bash
   git add changed_files.py
   git commit -m "Description of changes"
   ```

3. **Before Push**: Review changes and run tests
   ```bash
   pytest -v  # Ensure all tests pass
   ```

4. **Push**: Push to remote
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Pull Request**: Create PR via GitHub UI (not CLI)
   - Title: Brief description
   - Description: What was implemented
   - Link any related issues

### Commit Message Format

```
