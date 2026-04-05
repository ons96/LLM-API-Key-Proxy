# AGENTS.md - Prompt-Based LLM Router

**Project:** Prompt-Based LLM Router  
**Purpose:** Intelligent prompt routing system that analyzes incoming requests and selects the optimal LLM API based on task complexity and requirements

---

## 1. Role/Mission

### Mission Statement
Build an intelligent gateway that automatically analyzes incoming prompts and routes them to the most appropriate LLM API endpoint. The router uses a lightweight LLM to classify task complexity, then selects between faster/cheaper models for simple tasks and more powerful models for complex planning, reasoning, or code generation tasks.

### Core Objectives
- **Prompt Analysis**: Parse and analyze incoming prompts to determine complexity, intent, and resource requirements
- **Intelligent Routing**: Use LLM-based classification to select the optimal target API
- **Cost Efficiency**: Route simple tasks to fast, cheap models; reserve powerful models for complex tasks
- **Fallback Handling**: Gracefully handle API failures with retry logic and fallback routing
- **Free Tier Compliance**: Operate entirely using free LLM API tiers (no paid services required)

### Target Users
- AI coding assistants and autonomous agents
- Developer tools requiring multi-tier LLM routing
- Any system wanting to optimize LLM API costs while maintaining quality

---

## 2. Technical Stack

### Language & Runtime
- **Language**: Python 3.10+
- **Runtime**: Standard Python with async support via `asyncio`
- **Package Manager**: `uv` (modern, fast Python package manager)

### Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| `httpx` | ^0.27.0 | Async HTTP client for API calls |
| `pydantic` | ^2.0.0 | Data validation and settings |
| `python-dotenv` | ^1.0.0 | Environment variable loading |
| `aiofiles` | ^23.0.0 | Async file operations |
| `pytest` | ^8.0.0 | Testing framework |
| `pytest-asyncio` | ^0.23.0 | Async test support |

### LLM APIs (Free Tier)
| API | Tier | Use Case |
|-----|------|----------|
| **Groq** | Free (no auth required for limited) | Fast inference, classification |
| **GitHub Models** | Free tier | Primary routing decisions |
| **Ollama** (local) | Free (self-hosted) | Fallback/local classification |

### External Services
- **GitHub Actions**: CI/CD and autonomous execution
- **GitHub Models API**: Primary LLM for classification (free tier available)

---

## 3. Requirements (Numbered)

### Phase 1: Core Infrastructure (Requirements 1-5)

1. **Prompt Analyzer Module**
   - Create `src/analyzer/prompt_classifier.py` with function `classify_prompt(prompt: str) -> TaskClassification`
   - Implement task complexity scoring (1-10 scale)
   - Detect task type: `simple_qa`, `coding`, `planning`, `reasoning`, `analysis`
   - Use keyword matching + basic heuristics as first-pass filter

2. **LLM-Based Router**
   - Create `src/router/llm_router.py` with function `select_target_api(classification: TaskClassification) -> APIConfig`
   - Implement `RouterLLM` class using GitHub Models or Groq for classification
   - Define routing logic: simple → fast API, complex → powerful API
   - Cache routing decisions with TTL to reduce API calls

3. **API Client Registry**
   - Create `src/clients/base.py` with abstract `BaseAPIClient` class
   - Implement `OpenAIClient`, `AnthropicClient`, `GroqClient` adapters
   - Standardize response format across all providers
   - Include timeout, retry, and rate-limit handling

4. **Configuration System**
   - Create `src/config/settings.py` using Pydantic Settings
   - Define `APIConfig` with endpoint, model, api_key, priority
   - Support environment variable overrides
   - Define routing rules in YAML or JSON config

5. **Main Router Gateway**
   - Create `src/gateway/router.py` with async `route_prompt(prompt: str) -> LLMResponse`
   - Orchestrate: analyze → classify → select API → call → return
   - Implement logging and error tracking

### Phase 2: Intelligence & Optimization (Requirements 6-10)

6. **Dynamic Routing Logic**
   - Implement cost-aware routing rules in `src/router/rules.py`
   - Add latency-based routing (faster API for time-sensitive tasks)
   - Support custom routing rules via config file

7. **Request Deduplication & Caching**
   - Create `src/cache/response_cache.py` with TTL-based caching
   - Hash prompts to enable exact-match caching
   - Implement cache invalidation based on config

8. **Metrics & Logging**
   - Create `src/metrics/collector.py` tracking:
     - Request count by task type
     - API selection distribution
     - Latency per API
     - Cost estimation (based on token counts)
   - Output structured JSON logs

9. **Error Handling & Fallbacks**
   - Implement retry logic with exponential backoff
   - Create fallback chain: primary → secondary → tertiary
   - Handle rate limits gracefully
   - Log all failures for analysis

10. **CLI Interface**
    - Create `cli.py` with commands:
      - `classify