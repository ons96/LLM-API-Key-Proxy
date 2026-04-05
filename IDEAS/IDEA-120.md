# AGENTS.md - G4F Gateway Router Accounting

---

## 1. Role/Mission

**Mission:** Implement a comprehensive usage accounting system for g4f (a multi-provider AI gateway) that tracks and reports its own operational costs when it acts as an intermediate router/fallback mechanism. The system must accurately attribute usage to g4f itself when it performs nested routing decisions, fallback executions, or any intermediate operations between providers.

**Role:** You are an autonomous coding agent responsible for implementing a nested usage tracking system that captures:
- Internal routing decisions made by g4f
- Fallback chain executions
- Provider selection logic
- Latency introduced by fallback mechanisms
- Cost allocation when g4f intermediates between providers

---

## 2. Technical Stack

| Component | Technology | Notes |
|-----------|------------|-------|
| **Language** | Python 3.10+ | Core implementation |
| **Framework** | g4f (v0.2.x) | Base gateway framework |
| **Storage** | SQLite (via aiosqlite) | Local usage logs |
| **Async** | asyncio / aiohttp | Non-blocking operations |
| **Testing** | pytest + pytest-asyncio | Unit and integration tests |
| **Mocking** | unittest.mock | Provider Simulation |
| **CI/CD** | GitHub Actions | Free tier usage only |

---

## 3. Requirements

1. **Requirement 1: Nested Usage Tracking**
   - Track when g4f initiates a fallback to another provider
   - Record the original request that triggered the fallback
   - Capture which provider was attempted first and why it failed

2. **Requirement 2: Fallback Chain Logging**
   - Log each step in the fallback chain with timestamps
   - Track latency added by each intermediate step
   - Record success/failure status for each provider in the chain

3. **Requirement 3: Gateway Self-Accounting**
   - Attribute usage to "g4f-gateway" for internal routing operations
   - Calculate "intermediation cost" (compute time spent in routing logic)
   - Track tokens processed through fallback mechanisms

4. **Requirement 4: Cost Attribution**
   - Implement cost calculation for g4f's own processing
   - Support configurable cost weights for different fallback strategies
   - Generate usage reports showing g4f's internal vs. provider costs

5. **Requirement 5: Provider Metrics Collection**
   - Collect per-provider metrics (latency, success rate, token count)
   - Track which provider fulfilled the final request
   - Record original provider vs. fallback provider statistics

6. **Requirement 6: API/Interface**
   - Create a simple interface to query usage statistics
   - Support filtering by date range, provider, status
   - Export functionality for usage reports

7. **Requirement 7: Configuration**
   - Allow configuration of fallback strategies
   - Support enabling/disabling accounting per provider
   - Environment variable support for settings

8. **Requirement 8: Data Persistence**
   - Implement SQLite-based storage for usage logs
   - Support log rotation to prevent unbounded growth
   - Provide cleanup utilities for old records

---

## 4. File Structure

```
g4f-accounting/
├── .github/
│   └── workflows/
│       └── test.yml           # GitHub Actions CI
├── src/
│   └── g4f_accounting/
│       ├── __init__.py        # Package initialization
│       ├── core.py            # Core accounting logic
│       ├── models.py          # Data models
│       ├── storage.py          # SQLite storage layer
│       ├── providers.py       # Provider metrics collection
│       ├── fallback.py        # Fallback tracking
│       ├── api.py              # Query interface
│       └── config.py           # Configuration
├── tests/
│   ├── __init__.py
│   ├── test_core.py           # Core functionality tests
│   ├── test_storage.py        # Storage layer tests
│   ├── test_fallback.py       # Fallback tracking tests
│   └── test_integration.py   # Integration tests
├── .gitignore
├── requirements.txt          # Dependencies
├── pyproject.toml            # Project config
├── README.md                 # Documentation
└── AGENTS.md                # This file
```

---

## 5. Testing Requirements

| Test Type | Coverage Target | Framework |
|-----------|-----------------|------------|
| **Unit Tests** | Core tracking logic, models, storage | pytest |
| **Integration Tests** | Provider fallback simulation | pytest + mocks |
| **Mock Tests** | Simulate provider failures/responses | unittest.mock |

**Minimum Coverage:** 80% code coverage required

**Test Scenarios to Implement:**
- Test tracking of single fallback event
- Test tracking of multi-provider fallback chain
- Test cost calculation for intermediation
- Test usage query interface
- Test data persistence and retrieval
- Test log rotation functionality

---

## 6. Git Protocol

**Branch Strategy:** Single branch (`main`) with feature flags for development

**Commit Messages:** Conventional commits format
```
