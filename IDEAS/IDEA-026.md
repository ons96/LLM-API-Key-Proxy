# AGENTS.md

## Dynamic Provider Performance Optimizer

---

## 1. Role/Mission

### Role
You are an autonomous software architect and developer tasked with implementing a **Dynamic Provider Performance Optimizer** for AI provider management.

### Mission
Your mission is to build a performance tracking system that:
- Continuously monitors AI provider performance metrics (time to first token, total response time, token counts)
- Uses weighted average algorithms to analyze provider speed over time
- Dynamically reorders providers based on performance
- Adapts to changing provider speeds through periodic resets and recent-data weighting
- Helps optimize cost and latency by selecting the fastest providers for each request

### Success Conditions
- Create a functioning system that tracks and analyzes provider performance
- Implement dynamic provider reordering based on performance metrics
- Ensure the system can handle provider speed fluctuations
- Provide clear APIs for integration into existing applications

---

## 2. Technical Stack

### Programming Language
- **TypeScript** (primary) - For type safety and better maintainability
- **Node.js** - Runtime environment

### Key Libraries/Dependencies
- **Node.js built-in utilities**: `perf_hooks` for timing measurements
- **uuid** or `nanoid`: For unique metric IDs
- **otlp** (optional): For observability/tracing (if needed)

### No External Services Required
This implementation uses only **free, built-in Node.js functionality** where possible to minimize dependencies and ensure portability.

### Development Tools
- **npm**: Package management
- **Jest** or Node.js built-in test runner: For unit testing
- **ESLint**: For code quality (optional)

---

## 3. Requirements (Numbered)

### 3.1 Core Functionality

1. **Metric Collection System**
   - Track time to first token (TTFT) for each provider response
   - Track total response time from request sent to response complete
   - Track prompt token count and response token count
   - Store metrics with timestamps for temporal analysis

2. **Performance Calculation Engine**
   - Implement weighted average algorithm that emphasizes recent performance data
   - Calculate composite performance score based on:
     - Time to first token (responsiveness)
     - Total response time (throughput)
     - Tokens per second (calculation: response_tokens / total_time)
   - Apply exponential weight decay (more recent = higher weight)

3. **Dynamic Provider Reordering**
   - Maintain ordered list of providers ranked by performance score
   - Automatically reorder providers when scores change significantly
   - Expose API to get current best provider

4. **Periodic Reset Mechanism**
   - Implement configurable reset interval (default: every 24 hours or 1000 requests)
   - Option to disable auto-reset and use pure weighted averages
   - Manual reset capability via API

### 3.2 Configuration & Extensibility

5. **Configurable Weight Parameters**
   - Allow customization of recency weight factor (default: 0.8 for 80% recent emphasis)
   - Allow setting minimum sample size before ranking (default: 3 requests)
   - Support custom performance weights for TTFT vs total time vs throughput

6. **Provider Management**
   - Support adding/removing providers dynamically
   - Store provider metadata (name, endpoint, optional API key reference)
   - Handle provider timeout and error tracking (reduce score on failure)

### 3.3 Data Persistence

7. **In-Memory Storage with Optional Persistence**
   - Default: Store metrics in memory (resets on app restart)
   - Optional: Implement file-based persistence for metrics history
   - Keep configurable history window (default: last 100 requests per provider)

### 3.4 API Design

8. **Public API**
   - `recordMetric(providerId, metrics)`: Record a new performance measurement
   - `getBestProvider()`: Return the current best-performing provider
   - `getProviderRanking()`: Return all providers ordered by performance
   - `getProviderStats(providerId)`: Return detailed stats for a provider
   - `resetMetrics()`: Reset all metrics and rankings
   - `addProvider(provider)`: Add a new provider
   - `removeProvider(providerId)`: Remove a provider

### 3.5 Error Handling

9. **Error Handling & Edge Cases**
   - Handle zero or missing token counts gracefully
   - Prevent division by zero in performance calculations
   - Handle concurrent metric recordings (thread-safety where applicable)
   - Timeout handling: reduce provider score on timeout

---

## 4. File Structure

```
dynamic-provider-optimizer/
├── package.json
├── tsconfig.json
├── jest.config.js (if using Jest)
├── README.md
├── src/
│   ├── index.ts                 # Main entry point, exports public API
│   ├── types.ts                 # TypeScript interfaces and types
│   ├── config.ts                # Configuration defaults and validation
│   ├── provider/
│   │   ├── ProviderManager.ts   # Provider CRUD operations
│   │   └── types.ts             # Provider-specific types
│   ├── metrics/
│   │   ├── MetricCollector.ts   # Metrics recording logic
│   │   └── types.ts             # Metric types
│   ├── performance/
│   │   ├── PerformanceCalculator.ts  # Weighted average calculations
│   │   └── ProviderRanker.ts    # Ranking and reordering logic
│   ├── storage/
│   │   ├── MemoryStore.ts      # In-memory storage implementation
│   │   └── FileStore.ts       # Optional file-based persistence
│   └── utils/
│       ├── time.ts             # Timing helper functions
│       └── math.ts             # Math helper functions (weighted avg)
├── tests/
│   ├── unit/
│   │   ├── PerformanceCalculator.test.ts
│   │   ├── MetricCollector.test.ts
│   │   ├── ProviderRanker.test.ts
│   │   └── ProviderManager.test.ts
│   └── integration/
│       └── full-flow.test.ts   # Full system integration tests
└── QUESTIONS.md                # Agent questions for human review
```

---

## 5. Testing Requirements

### 5.1 Unit Tests (Required)

Test the following components individually:

1. **PerformanceCalculator**
   - Test weighted average with different weight factors
   - Test handling of zero values
   - Test recency emphasis calculation

2. **MetricCollector**
   - Test metric recording
   - Test metric retrieval
   - Test history window enforcement

3. **ProviderRanker**
   - Test ranking order based on scores
   - Test reordering triggers
   - Test minimum sample size enforcement

4. **ProviderManager**
   - Test adding provider
   - Test removing provider
   - Test duplicate prevention

### 5.2 Integration Tests (Required)

5. **Full Flow Test**
   - Record multiple metrics from multiple providers
   - Verify provider reordering occurs correctly
   - Verify periodic reset functionality

### 5.3 Test Coverage Targets

- **Minimum 80% code coverage** for core algorithm files
- All public API methods must have tests

### 5.4 Test Execution

Run tests with:
```bash
npm test
```

---

## 6. Git Protocol

### Branch Strategy
- Work on a feature branch: `feature/performance-optimizer`
- Create PR for review: **Do not self-approve merges**

### Commit Messages
Use conventional commits format:
```
