# AGENTS.md

## Intelligent Provider Sorting Algorithm

---

## 1. Role/Mission

**Role:** Autonomous Coding Agent

**Mission:** Implement an intelligent provider sorting system that monitors API provider rate/usage limits and automatically sorts providers by their expiry times and aggregated performance scores to enable optimal fallback selection. The algorithm should compute effective expiry times using max functions, calculate performance scores from historical usage data, and maintain a sorted data structure that allows the system to always select the best available provider first.

**Autonomy Level:** High. The agent is expected to independently analyze the codebase, implement the sorting algorithm, create supporting data structures, and verify functionality without requiring human intervention. The agent should only save clarifying questions to QUESTIONS.md if critical business logic is ambiguous.

---

## 2. Technical Stack

- **Language:** TypeScript/JavaScript (Node.js runtime)
- **Version Control:** Git
- **CI/CD Platform:** GitHub Actions
- **Data Storage:** JSON files (local filesystem)
- **Package Manager:** npm
- **Linting:** ESLint with TypeScript support
- **Testing:** Jest
- **Free Resources Only:** No paid services, no API credits consumed beyond minimal testing, no external paid dependencies

---

## 3. Requirements

1. **Provider Data Load**
   - Load existing API provider configuration from a JSON file (e.g., `providers.json` or similar)
   - Parse provider entries containing at minimum: provider name, rateLimitExpiry, usageLimitExpiry, and performance metrics

2. **Effective Expiry Computation**
   - For each provider row, compute the effective expiry using `max(rateLimitExpiry, usageLimitExpiry)`
   - Store the computed value in a new column named `effectiveExpiry` or similar
   - Handle timestamps in a consistent format (ISO 8601 recommended)

3. **Performance Score Aggregation**
   - Calculate an aggregated performance score for each provider based on available metrics
   - Metrics to consider include: success rate, average latency, error rate, and historical uptime
   - Normalize scores to a consistent range (0-100 scale recommended)

4. **Provider Sorting**
   - Sort all providers by effective expiry (ascending - soonest expiring first)
   - Secondary sort by performance score (descending - highest score first)
   - Maintain a sorted data structure in memory or persisted to file

5. **Fallback Selection Logic**
   - Implement a function to get the optimal provider for a given request
   - Function should return the first available provider (not expired) with the highest performance score
   - Handle edge cases: all providers expired, single provider available

6. **Data Persistence**
   - Save the sorted provider list to a file for later reference
   - Include computed effectiveExpiry and performanceScore in the output
   - Use consistent formatting and indentation

7. **Configuration Management**
   - Support configurable sorting parameters (primary sort field, secondary sort field)
   - Allow weights for performance score calculation to be configurable
   - Support excluding certain providers from sorting

8. **Logging and Diagnostics**
   - Log sorting operations with timestamps
   - Log provider selection decisions with reasoning
   - Provide summary statistics after each sort operation

---

## 4. File Structure

```
/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions workflow
├── src/
│   ├── index.ts                # Main entry point
│   ├── types/
│   │   └── provider.ts        # TypeScript interfaces
│   ├── services/
│   │   ├── providerLoader.ts  # Load provider data
│   │   ├── expiryCalculator.ts # Compute effective expiry
│   │   ├── performanceScorer.ts # Calculate scores
│   │   ├── providerSorter.ts   # Sorting logic
│   │   └── fallbackSelector.ts # Selection logic
│   ├── utils/
│   │   └── logger.ts          # Logging utilities
│   └── config/
│       └── settings.ts        # Configuration
├── data/
│   ├── providers.json         # Raw provider data (input)
│   └── providers-sorted.json  # Sorted provider data (output)
├── tests/
│   ├── providerLoader.test.ts
│   ├── expiryCalculator.test.ts
│   ├── performanceScorer.test.ts
│   ├── providerSorter.test.ts
│   └── fallbackSelector.test.ts
├── package.json
├── tsconfig.json
├── jest.config.js
├── eslint.config.js
├── .gitignore
└── README.md
```

---

## 5. Testing Requirements

1. **Unit Tests (Mandatory)**
   - Each service function must have at least 3 unit tests
   - Unit tests must cover: happy path, edge cases, and error conditions
   - Minimum 80% code coverage required

2. **Integration Tests**
   - Test end-to-end flow: load → compute → sort → select
   - Verify output file matches expected sorted order

3. **Test Data**
   - Create test provider data with varied expiry times
   - Include providers with identical expiry times (test secondary sort)
   - Include edge cases: expired providers, missing fields

4. **Test Execution**
   - Tests must run on every PR/commit via GitHub Actions
   - Tests must complete within 2 minutes
   - No external API calls during testing (mock all external dependencies)

5. **Test Output**
   - Generate coverage report after test run
   - Attach coverage percentage to PR comment

---

## 6. Git Protocol

1. **Branch Strategy**
   - Work on feature branch: `feature/provider-sorting`
   - Create PRs against `main` branch
   - Squash commits before merging

2. **Commit Messages**
   - Format: `type(scope): description`
   - Types: `feat`, `fix`, `test`, `docs`, `refactor`
   - Example: `feat(providerSorter): add secondary sort by performance score`

3. **PR Requirements**
   - All tests must pass
   - Minimum 80% code coverage
   - No eslint errors
   - Update CHANGELOG.md with changes

4. **Questions Protocol**
   - If any clarification is needed, create `QUESTIONS.md` in the repository root
   - Format questions clearly with context
   - Do not block on questions - implement reasonable defaults and note in QUESTIONS.md

---

## 7. Completion Criteria

1. **Functional Criteria**
   - [ ] Provider data loads successfully from JSON file
   - [ ] Effective expiry computed correctly using max function
   - [ ] Performance scores calculated and normalized (0-100)
   - [ ] Providers sorted by expiry (primary) and score (secondary)
   - [ ] Fallback selector returns optimal available provider
   - [ ] Output file written with sorted providers

2. **Code Quality Criteria**
   - [ ] All code compiles without TypeScript errors
   - [ ] ESLint passes with no errors or warnings
   - [ ] 80%+ test coverage achieved
   - [ ] All unit tests pass
   - [ ] All integration tests pass

3. **Automation Criteria**
   - [ ] GitHub Actions workflow triggers on push/PR
   - [ ] Tests run automatically on CI
   - [ ] Build succeeds without manual intervention

4. **Documentation Criteria**
   - [ ] README.md updated with usage instructions
   - [ ] CODE comments explain complex logic
   - [ ] TypeScript types/infaces documented

5. **Deliverable Criteria**
   - [ ] All source files in `/src` directory as specified
   - [ ] Test files in `/tests` directory
   - [ ] Configuration file for sorting parameters
   - [ ] Sample input/output data files

---

**End of AGENTS.md**