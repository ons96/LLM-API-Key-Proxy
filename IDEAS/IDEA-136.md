# AGENTS.md - Merge of Experts (MoE) Framework

---

## 1. Role/Mission

### Framework Purpose
Develop and implement a **Merge of Experts (MoE) Gateway Framework** that intelligently routes complex prompts to multiple AI models simultaneously, evaluates their responses, and selects the best result—while gracefully handling model unavailability through intelligent fallback mechanisms.

### Mission Statement
> *To create an autonomously-operating gateway that maximizes response quality by leveraging multiple AI models in parallel, with built-in intelligence for model selection, response evaluation, and high-availability failover—all using free-tier resources.*

### Core Capabilities
1. **Parallel Model Invocation** - Send prompts to multiple models simultaneously
2. **Intelligent Triggering** - Automatically determine when MoE is warranted
3. **Model Selection Rules** - Match prompt requirements to model capabilities
4. **Response Evaluation** - Judge responses using defined quality metrics
5. **Fallback Orchestration** - Seamlessly shift to alternatives when primary models unavailable
6. **Performance Balancing** - Optimize for both quality and response time

---

## 2. Technical Stack

### Constraints
- **ALL resources must be FREE/TIER-0**
- No paid API subscriptions
- No commercial services requiring payment
- Self-hosted or free-tier alternatives only

### Approved Free Resources

| Category | Tool/Service | Purpose |
|----------|-------------|---------|
| **LLM APIs** | OpenRouter (free tier) | Access multiple models via unified API |
| **LLM APIs** | Ollama (local) | Run models locally at no cost |
| **LLM APIs** | Cohere (free tier) | Alternative model provider |
| **Git Hosting** | GitHub Actions | CI/CD automation |
| **Git Hosting** | GitHub Issues | Task tracking |
| **Development** | VS Code | Code editing |
| **Runtime** | Node.js (LTS) | JavaScript execution |
| **Runtime** | Python 3.11+ | Python execution |
| **Testing** | Vitest | JavaScript testing framework |
| **Testing** | Pytest | Python testing framework |

### Technology Stack

```
Language: TypeScript/JavaScript + Python
Runtime: Node.js 18+ (Actions runner), Python 3.11+
Package Managers: npm, pip
Testing: Vitest, Pytest
CI/CD: GitHub Actions
Documentation: Markdown
```

---

## 3. Requirements (Numbered)

### Phase 1: Core Architecture

1. **MoE Gateway Service**
   - Create a `gateway` service that accepts prompts and routes them to appropriate models
   - Implement configuration-driven model selection (no hardcoding)

2. **Model Registry**
   - Define a `models.json` configuration containing:
     - Model identifiers and providers
     - Capability tags (e.g., `coding`, `reasoning`, `creative`, `analysis`)
     - Context limits and latency profiles
     - Availability status indicators

3. **Trigger Rules Engine**
   - Build a rules engine that determines when to invoke MoE vs. single model
   - Triggers to consider:
     - Prompt complexity (keyword matching, length thresholds)
     - Required capabilities (explicit or inferred)
     - Fallback counting (retries after failures)

### Phase 2: Parallel Execution & Evaluation

4. **Parallel Invocation Handler**
   - Implement async/fan-out pattern to call multiple models simultaneously
   - Support configurable concurrency limits (default: 3 models parallel)
   - Add timeout handling per model (default: 60 seconds)

5. **Response Judge**
   - Build a judge module with criteria:
     - Completeness (does it answer all parts of the prompt?)
     - Coherence (logical consistency score)
     - Format validity (JSON, code, prose as needed)
     - Provider-specific quality indicators
   - Implement a scoring algorithm that weights criteria

6. **Response Selection**
   - When multiple responses received, select based on:
     - Highest judge score
     - Fastest response time (configurable preference)
     - Balanced trade-off (configurable weight)

### Phase 3: Fallback & Shifting

7. **Availability Monitor**
   - Implement health checking for each model
   - Track success/failure rates
   - Mark models as "unavailable" after N consecutive failures

8. **Fallback Chain**
   - Define fallback priority per capability requirement
   - When best model unavailable, shift to next-best alternative
   - Maintain fallback history for logging

9. **Graceful Degradation**
   - If all MoE models fail, fall back to single "guaranteed" model
   - Log all failures with timestamps and model IDs
   - Alert when system operates in degraded mode

### Phase 4: Operations & Observability

10. **Configuration System**
    - Support `config.yaml` for all tunable parameters:
      - Model priorities
      - Timeout values
      - Quality thresholds
      - Fallback chains
    - Allow environment-variable overrides

11. **Logging & Metrics**
    - Log all gateway decisions (trigger reason, models selected, winner)
    - Track: response times, success rates, judge scores per model
    - Output structured logs (JSON) for downstream analysis

12. **CLI Interface**
    - Create command-line interface with commands:
      - `moe-gateway query