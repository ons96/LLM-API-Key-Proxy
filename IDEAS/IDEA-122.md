# AGENTS.md - DeepSeek V3.2 Integration via Nvidia and iFlow Providers

## 1. Role/Mission

You are an autonomous software integration agent tasked with integrating DeepSeek V3.2 model access into the existing free LLM gateway system through two new provider endpoints: Nvidia API and iFlow API.

**Your Mission:**
- research available documentation for both Nvidia and iFlow APIs
- implement provider adapters that conform to the existing gateway architecture
- ensure all integrations use free tier resources only
- validate end-to-end functionality through testing
- document your implementation and decisions

**Decision-Making Guidelines:**
- If you encounter ambiguous requirements, make a reasonable architectural decision and document it in your commit
- If you need clarification on provider-specific implementation details, research independently first before saving to QUESTIONS.md
- Prioritize stability and backward compatibility with the existing gateway system
- Always prefer free resources; do not implement paid tier dependencies

---

## 2. Technical Stack

### Core Technologies
- **Runtime:** Node.js (LTS version as specified in project config)
- **Package Manager:** npm
- **Testing Framework:** Jest
- **HTTP Client:** Native fetch or axios (per existing project convention)

### Provider SDKs/Libraries
- **Nvidia API:** Use official Nvidia NIM SDK or REST API with appropriate authentication
- **iFlow API:** Use official iFlow REST API client patterns
- **DeepSeek Model:** DeepSeek V3.2 (released version)

### Gateway Dependencies
- Reference existing provider adapter patterns in the codebase
- Use shared configuration management system
- Follow existing retry/circuit breaker patterns

### Version Control
- **Git:** Required
- **Hosting:** GitHub
- **CI/CD:** GitHub Actions (per existing workflow)

---

## 3. Requirements (Numbered)

### 3.1 Research Phase
1. Research Nvidia API documentation for model access, including:
   - Authentication method (API keys, OAuth, etc.)
   - Request/response format
   - Rate limits for free tier
   - Available DeepSeek model variants

2. Research iFlow API documentation for model access, including:
   - Authentication method
   - Request/response format
   - Rate limits for free tier
   - Available DeepSeek model variants

3. Review existing gateway provider adapter patterns in the codebase

### 3.2 Implementation Phase
4. Create Nixd adapter module at `src/providers/nvidia.js` that:
   - Implements the standard provider interface (init, chat, validateKey methods)
   - Handles Nvidia-specific authentication
   - Maps requests to Nvidia API format
   - Handles streaming and non-streaming responses

5. Create iFlow adapter module at `src/providers/iflow.js` that:
   - Implements the standard provider interface
   - Handles iFlow-specific authentication
   - Maps requests to iFlow API format
   - Handles streaming and non-streaming responses

6. Add provider configurations to config file:
   - Nvidia: endpoint URL, free tier limits, model mapping
   - iFlow: endpoint URL, free tier limits, model mapping

7. Update provider registry/index to include both new providers

8. Implement free-tier usage tracking for each provider

### 3.3 Integration Phase
9. Ensure adapters handle rate limiting gracefully (retry with backoff)

10. Add proper error handling for both providers:
    - API errors with meaningful messages
    - Timeout handling
    - Network error recovery

11. Implement environment variable configuration:
    - NVIDIA_API_KEY
    - IFLOW_API_KEY
    - Optional override endpoints

### 3.4 Documentation
12. Add API documentation for new providers in docs/
13. Update README with provider setup instructions

---

## 4. File Structure

```
llm-gateway/
├── src/
│   ├── providers/
│   │   ├── nvidia.js          # NEW - Nvidia adapter
│   │   ├── iflow.js           # NEW - iFlow adapter
│   │   ├── index.js           # Existing - provider registry
│   │   └── base.js            # Existing - base adapter class
│   ├── config/
│   │   ├── providers.js       # MODIFY - add new provider configs
│   │   └── index.js
│   ├── utils/
│   │   ├── rate-limiter.js    # Existing - shared rate limiting
│   │   └── logger.js
│   └── gateway/
│       └── router.js
├── tests/
│   ├── providers/
│   │   ├── nvidia.test.js    # NEW
│   │   └── iflow.test.js     # NEW
│   └── integration/
│       └── providers.test.js # MODIFY - add integration tests
├── docs/
│   ├── providers/
│   │   ├── nvidia.md         # NEW
│   │   └── iflow.md          # NEW
│   └── api.md
├── scripts/
│   └── setup-keys.sh         # MODIFY - add env var instructions
├── .env.example              # MODIFY - add new env vars
├── config.yaml               # MODIFY - add provider configs
└── AGENTS.md                 # This file
```

---

## 5. Testing Requirements

### 5.1 Unit Tests (Required)
1. Test Nvidia adapter initialization and key validation
2. Test iFlow adapter initialization and key validation
3. Test request mapping for both providers
4. Test error handling paths
5. Test rate limit detection and handling

### 5.2 Integration Tests (Required)
6. Test full request/response cycle with mock API responses
7. Test streaming response handling
8. Test provider fallback behavior

### 5.3 Manual Verification (Required)
9. Verify free tier API access works (may require actual API keys)
10. Verify response format matches gateway standard

### 5.4 Test Coverage Requirements
- Minimum 80% code coverage on new adapter modules
- All public methods must have test coverage
- Error paths must be tested

### 5.5 Test Environment
- Use environment variables for API keys in tests
- Mock external API responses for CI/CD reliability
- Do not depend on live API availability for passing tests

---

## 6. Git Protocol

### 6.1 Branch Strategy
- Create feature branch: `feature/deepseek-v32-nvidia-iflow`
- Commit frequency: At minimum, after each requirement item completion
- One commit per logical change

### 6.2 Commit Message Format
```
