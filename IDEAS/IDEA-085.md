# AGENTS.md - LLM Benchmark Score Integration with Model Ordering

## 1. Role/Mission

### Role
You are an autonomous coding agent responsible for implementing benchmark score integration for LLM model ordering within an LLM API proxy gateway system.

### Mission
Implement a system that fetches benchmark scores from the [llm-leaderboard](https://github.com/ons96/llm-leaderboard) repository and uses them, combined with runtime metrics (token speed, latency, rate limits, usage limits), to calculate an optimal fallback order for LLM model providers.

**Core Responsibilities:**
- Fetch and parse benchmark data from lmarena, livebench, and artificial analysis sources
- Normalize scores across different benchmarks into a unified scoring system
- Integrate runtime metrics (token speed, latency, rate limits) into the ordering calculation
- Implement a dynamic scoring formula that weights benchmark performance against operational reliability
- Create a model ordering service that can be queried by the proxy gateway for fallback decisions
- Make independent decisions about scoring weights and formulas based on best practices
- Document any technical questions or architecture decisions in QUESTIONS.md
- Only use free resources (no paid APIs, no costing inference)

## 2. Technical Stack

### Core Technologies
- **Language**: TypeScript/JavaScript (Node.js environment)
- **Benchmark Source**: [llm-leaderboard](https://github.com/ons96/llm-leaderboard) - JSON data files
- **HTTP Client**: Native fetch or axios (free tier)
- **Caching**: In-memory cache or local JSON file storage
- **Testing**: Vitest or Jest
- **CI/CD**: GitHub Actions

### Key Dependencies (Free Resources Only)
- `axios` - HTTP requests to fetch benchmark data
- `vitest` - Testing framework
- `jsonfile` or native `fs` - Local data storage
- `dotenv` - Environment configuration

### Supported Benchmark Sources
- **LM Arena** (`lmarena`) - Human preference rankings
- **LiveBench** (`livebench`) - Real-world task performance
- **Artificial Analysis** (`artificial-analysis`) - Automated evaluation scores

### Runtime Metrics Sources
- Token speed (tokens/second)
- Latency (ms per request)
- Rate limits (requests/minute)
- Usage limits (tokens/day)

## 3. Requirements (Numbered)

### 3.1 Data Fetching & Integration
- [ ] Fetch benchmark JSON data from the llm-leaderboard repository
- [ ] Parse and extract relevant model scores from each benchmark source
- [ ] Implement a configurable refresh interval for benchmark data (default: every 6 hours)
- [ ] Cache benchmark data locally to avoid excessive API calls
- [ ] Handle network failures gracefully with exponential backoff

### 3.2 Score Normalization
- [ ] Create a normalization function that converts different score scales to a 0-100 scale
- [ ] Implement weighted averaging across multiple benchmarks
- [ ] Allow configurable weights for each benchmark source
- [ ] Handle missing benchmark data for models (use available data only)

### 3.3 Runtime Metrics Integration
- [ ] Create a metrics collection module for tracking token speed, latency, rate limits
- [ ] Implement a rolling average calculation for latency and token speed
- [ ] Track rate limit usage and remaining quota
- [ ] Calculate a "reliability score" based on recent performance

### 3.4 Ordering Calculation Formula
- [ ] Implement the master scoring formula:
  ```
  Final_Score = (Normalized_Benchmark_Score * Benchmark_Weight) +
                (Speed_Score * Speed_Weight) +
                (Reliability_Score * Reliability_Weight) +
                (Availability_Score * Availability_Weight)
  ```
- [ ] Default weights: Benchmark 50%, Speed 20%, Reliability 20%, Availability 10%
- [ ] Allow weights to be configurable via environment variables
- [ ] Sort models by Final_Score descending to determine fallback order

### 3.5 Model Provider Configuration
- [ ] Support mapping virtual model names to actual provider endpoints
- [ ] Store provider configuration in a JSON file
- [ ] Support multiple providers per virtual model (for chaining)
- [ ] Allow manual override of calculated order

### 3.6 API Service
- [ ] Expose a REST API endpoint to get current model order
- [ ] Expose endpoints to get individual model scores
- [ ] Expose an endpoint to refresh benchmark data
- [ ] Return both the ordered list and individual score breakdowns

### 3.7 Persistence & Monitoring
- [ ] Persist calculated orders to local JSON storage
- [ ] Log all scoring calculations for debugging
- [ ] Track historical scores for trend analysis

## 4. File Structure

```
llm-benchmark-ordering/
├── src/
│   ├── index.ts                    # Main entry point
│   ├── config/
│   │   └── config.ts             # Configuration management
│   ├── models/
│   │   ├── types.ts              # Type definitions
│   │   └── ModelProvider.ts      # Model provider class
│   ├── services/
│   │   ├── BenchmarkService.ts   # Fetch & parse benchmark data
│   │   ├── ScoreNormalizer.ts    # Score normalization logic
│   │   ├── MetricsService.ts     # Runtime metrics collection
│   │   ├── OrderingService.ts    # Calculate fallback order
│   │   └── ProviderService.ts    # Provider management
│   ├── api/
│   │   └── server.ts             # REST API server
│   ├── utils/
│   │   ├── http.ts               # HTTP utilities
│   │   └── logger.ts             # Logging utilities
│   └── data/
│       ├── benchmarks.json       # Cached benchmark data
│       ├── providers.json        # Provider configuration
│       └── ordering.json         # Current ordering
├── tests/
│   ├── ScoreNormalizer.test.ts
│   ├── OrderingService.test.ts
│   ├── MetricsService.test.ts
│   └── integration.test.ts
├── .env.example                  # Environment template
├── package.json
├── tsconfig.json
└── README.md
```

### Key Files Description

| File | Purpose |
|------|---------|
| `src/services/BenchmarkService.ts` | Fetches data from llm-leaderboard, caches locally |
| `src/services/ScoreNormalizer.ts` | Normalizes scores to 0-100 scale with weights |
| `src/services/MetricsService.ts` | Tracks runtime performance metrics |
| `src/services/OrderingService.ts` | Main scoring formula & ordering logic |
| `src/models/types.ts` | TypeScript interfaces for all data structures |

## 5. Testing Requirements

### 5.1 Unit Tests
- [ ] Test score normalization with mock data
- [ ] Test ordering calculation with known inputs
- [ ] Test metrics aggregation functions
- [ ] Test edge cases (missing data, zero scores, negative values)

### 5.2 Integration Tests
- [ ] Test full ordering flow with mock benchmark API responses
- [ ] Test API endpoints return correct data
- [ ] Test configuration loading and overrides
- [ ] Test caching behavior

### 5.3 Test Coverage Minimum
- **Statement Coverage**: 80%
- **Critical Paths**: 100% (ordering calculation, score normalization)
- All scoring functions must have explicit test cases

### 5.4 Test Data
- Use mock JSON files in `/tests/fixtures/` for benchmark data
- Include edge case fixtures: missing values, different scales, extreme scores

## 6. Git Protocol

### 6.1 Branch Strategy
- Work on feature branches: `feature/benchmark-integration`, `feature/score-normalizer`
- Make small, focused commits with clear messages
- Create a PR for review before merging to main

### 6.2 Commit Messages
- Use conventional commits: `feat:`, `fix:`, `test:`, `refactor:`, `docs:`
- Example: `feat: add benchmark fetching from lmarena source`

### 6.3 Documentation
- Update README.md with setup instructions
- Document scoring formula in code comments
- Save architectural questions to QUESTIONS.md

### 6.4 Questions & Decisions
- Create `QUESTIONS.md` for unresolved questions
- Document design decisions with rationale in PR description
- If blocked for > 30 minutes, document the blocker and propose alternatives

### 6.5 Free Resources Constraint
- **ALWAYS** use free/public APIs only
- Do not use any paid services without explicit approval
- Use mock data for any external API calls in tests

## 7. Completion Criteria

### 7.1 Functional Requirements
- [ ] Benchmark data successfully fetched from llm-leaderboard
- [ ] All three benchmarks (lmarena, livebench, artificial-analysis) integrated
- [ ] Scores normalized to unified 0-100 scale
- [ ] Runtime metrics tracking implemented
- [ ] Ordering calculation produces deterministic results
- [ ] API returns correct fallback order

### 7.2 Non-Functional Requirements
- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Code follows TypeScript best practices
- [ ] No lint errors
- [ ] Documentation complete

### 7.3 Validation Checklist
- [ ] Run `npm test` - all tests pass
- [ ] Run `npm run build` - compiles without errors
- [ ] Verify ordering.json is generated correctly
- [ ] Test API endpoint returns expected data
- [ ] Questions documented in QUESTIONS.md (if any)

### 7.4 Delivery
- Create a pull request with all changes
- Include summary of implementation in PR description
- Update README.md with usage instructions
- Mark all completed items in this AGENTS.md

---

## Important Notes for Autonomous Agent

1. **Independence**: Make decisions about scoring weights and formula internals based on your understanding of LLM benchmarking best practices. Document your rationale.

2. **Free Resources**: Do not integrate any paid APIs. Use only free, public data sources.

3. **Questions**: If you encounter ambiguity or need architectural decisions, save them to QUESTIONS.md instead of waiting.

4. **Testing**: Prioritize test coverage for the scoring formula - this is the core business logic.

5. **Simplicity**: Keep the implementation straightforward. Prefer clear, maintainable code over clever optimizations.

6. **Completion**: Do not stop until all requirements are met and tests pass.