# AGENTS.md - AI-Driven Prompt-to-Model Router

## 1. Role/Mission

**Mission:** Build an intelligent routing system that analyzes incoming prompts using an LLM, determines task complexity and type, and automatically routes each prompt to the most appropriate model for the job—balancing speed, capability, and cost efficiency.

**Agent Responsibilities:**
- Analyze prompt complexity using available LLM resources
- Classify task types (simple queries, coding tasks, complex reasoning, creative writing, etc.)
- Maintain a model capability registry with routing logic
- Implement decision trees for model selection
- Optimize for free tier usage while meeting task requirements
- Make autonomous decisions about model routing without user intervention
- Log all routing decisions for future optimization

---

## 2. Technical Stack

**Core Technologies:**
- **Language:** Python 3.10+
- **LLM Analysis:** Uses free tier APIs or local models (avoid paid services)
- **API Framework:** FastAPI for the routing endpoint
- **Configuration:** YAML-based model registry
- **Testing:** pytest, pytest-asyncio

**Free Resources Strategy:**
- Primary: Free tier LLM APIs (Anthropic, OpenAI free credits, etc.)
- Fallback: Local model via Ollama or similar self-hosted solution
- Analysis caching to minimize API calls

**Dependencies:**
```
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
pyyaml>=6.0
httpx>=0.24.0
pytest>=7.4.0
pytest-asyncio>=0.21.0
python-dotenv>=1.0.0
aiofiles>=23.0.0
```

---

## 3. Requirements (numbered)

1. **Prompt Classification Engine**
   - Build an LLM-based classifier that analyzes prompt complexity (1-5 scale)
   - Detect task categories: simple_qa, coding, reasoning, creative, analysis, translation
   - Identify required capabilities: mathematical, logical, creative, factual, multilingual
   - Return confidence scores for each classification

2. **Model Capability Registry**
   - Create YAML configuration defining available models
   - Include model attributes: name, provider, capabilities, speed tier, cost tier, context window
   - Support at least 3 model tiers: fast/light, balanced, smart/capable

3. **Routing Decision System**
   - Implement decision logic mapping task complexity to model selection
   - Factor in explicit user preferences (if provided)
   - Support override flags for forced model selection
   - Add fallback logic when primary model unavailable

4. **API Endpoint**
   - POST /route endpoint accepting prompt + optional parameters
   - GET /models endpoint listing available models
   - GET /health endpoint for status
   - Return routed response with model info and classification

5. **Caching Layer**
   - Cache classification results for repeated/similar prompts
   - Implement TTL-based cache expiration
   - Support cache invalidation

6. **Logging & Observability**
   - Log all routing decisions with reasoning
   - Track model response times and success rates
   - Implement basic metrics (requests routed, complexity distribution)

7. **Configuration Management**
   - Environment-based configuration for API keys
   - Model registry loaded from YAML
   - Support for model enable/disable flags

---

## 4. File Structure

```
prompt-router/
├── README.md
├── requirements.txt
├── .env.example
├── config/
│   ├── models.yaml          # Model capability registry
│   └── settings.yaml       # Application settings
├── src/
│   ├── __init__.py
│   ├── main.py             # FastAPI application
│   ├── router.py           # Core routing logic
│   ├── classifier.py       # Prompt classification
│   ├── models/
│   │   ├── __init__.py
│   │   ├── registry.py     # Model registry management
│   │   └── types.py        # Model/data types
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm_client.py   # LLM API client
│   │   └── cache.py        # Caching service
│   └── utils/
│       ├── __init__.py
│       └── logging.py     # Logging configuration
├── tests/
│   ├── __init__.py
│   ├── test_classifier.py
│   ├── test_router.py
│   ├── test_registry.py
│   └── test_integration.py
├── QUESTIONS.md            # Agent questions for human review
└── docs/
    └── api.md              # API documentation
```

---

## 5. Testing Requirements

**Unit Tests:**
- Test prompt classification accuracy with known inputs
- Test model registry loading and validation
- Test routing decision logic for all complexity levels
- Test caching functionality

**Integration Tests:**
- Test full /route endpoint flow
- Test model fallback behavior
- Test configuration loading

**Test Coverage Goals:**
- Minimum 80% code coverage
- All routing logic branches tested
- Mock external LLM calls for reproducible tests

**Test Data:**
- Create fixture files with sample prompts of varying complexity
- Include edge cases: empty prompts, very long prompts, ambiguous tasks

---

## 6. Git Protocol

**Branch Strategy:**
- `main` - Production-ready code only
- `feature/*` - New features and improvements
- `fix/*` - Bug fixes
- `test/*` - Test additions

**Commit Message Format:**
```
