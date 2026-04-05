# AGENTS.md - Multi-Key API Handler with Auto-Deranking

## 1. Role/Mission

**Mission:** Build an autonomous system that manages multiple API keys, automatically validates their functionality, tracks errors, and intelligently deranks non-working keys while prioritizing reliable ones.

**Core Capabilities:**
- Accept multiple API keys from various providers
- Automatically validate each key through test requests
- Track success/failure rates per key over time
- Implement smart deranking logic (move failed keys to bottom of priority list)
- Provide a clean interface for retrieving the best available key
- Persist key state across runs using free storage solutions

## 2. Technical Stack

| Component | Technology | Notes |
|-----------|------------|-------|
| Runtime | Node.js 20.x | Free, widely supported |
| Package Manager | npm | Built into Node.js |
| Storage | Local JSON files | Simple, persistent, free |
| Testing | Built-in fetch/axios | Use native fetch for zero deps |
| Scheduling | GitHub Actions Cron | Free tier available |
| Error Handling | Custom wrapper | No external dependencies |

**Constraint:** Use only free, native solutions. No paid APIs, no external paid services, no premium dependencies.

## 3. Requirements (Numbered)

1. **Key Ingestion System**
   - Accept API keys via environment variable (JSON array format)
   - Accept keys via configuration file
   - Support adding keys dynamically at runtime
   - Each key must have: `id`, `key`, `provider`, `createdAt`, `lastTestedAt`

2. **Key Validation Engine**
   - Implement test endpoint calling for common providers (generic HTTP check)
   - Allow custom validation endpoint configuration per provider
   - Support both GET and POST test methods
   - Configurable timeout (default 10 seconds)
   - Retry failed validations up to 2 times before marking as failing

3. **Error Tracking System**
   - Track consecutive failures per key
   - Track total requests per key
   - Track success rate percentage
   - Store last error message per key
   - timestamps for all state changes

4. **Deranking Logic**
   - Keys with 3+ consecutive failures move to "deranked" status
   - Deranked keys tried last in priority
   - Keys can auto-revalidate after 1 hour of being deranked
   - Permanently disable keys after 10 consecutive failures
   - Working keys sorted by: success rate → least recently tested

5. **Priority Queue Management**
   - Always return highest-priority working key
   - Fallback to next available key if primary fails
   - Automatic rotation to distribute load
   - Manual override capability (force use specific key)

6. **Persistence Layer**
   - Save key state to JSON file after every mutation
   - Load state on system initialization
   - Handle corrupted state files gracefully (reset to defaults)

7. **API Interface**
   - `getWorkingKey()` - returns best available key
   - `addKey(keyData)` - add new key to pool
   - `removeKey(keyId)` - remove key entirely
   - `testKey(keyId)` - manually trigger validation
   - `getStatus()` - return all keys with their status
   - `resetKey(keyId)` - reset deranked key to try again

## 4. File Structure

```
.
├── AGENTS.md                    # This file
├── QUESTIONS.md                # Questions for human review
├── package.json                # Project metadata
├── README.md                   # Usage documentation
├── src/
│   ├── index.js               # Main entry point
│   ├── KeyManager.js           # Core key management class
│   ├── KeyValidator.js         # Validation engine
│   ├── Deranker.js            # Deranking logic
│   ├── PriorityQueue.js      # Key prioritization
│   ├── StateStore.js          # JSON persistence
│   └── utils/
│       ├── logger.js          # Logging utility
│       └── errors.js         # Custom error classes
├── config/
│   └── default.json          # Default configuration
├── data/
│   └── keys.json             # Persisted key state (gitignored)
├── scripts/
│   └── validate-all.js       # Standalone validation script
└── tests/
    ├── KeyManager.test.js
    ├── KeyValidator.test.js
    ├── Deranker.test.js
    └── integration.test.js
```

## 5. Testing Requirements

**Unit Tests:**
- [ ] Test KeyManager initialization with empty keys
- [ ] Test KeyManager initialization with pre-existing keys
- [ ] Test addKey() adds key with correct metadata
- [ ] Test removeKey() removes key by ID
- [ ] Test getWorkingKey() returns highest priority key

**Validation Tests:**
- [ ] Test KeyValidator handles successful response (2xx)
- [ ] Test KeyValidator handles failure response (4xx/5xx)
- [ ] Test KeyValidator handles network timeout
- [ ] Test KeyValidator handles invalid endpoints
- [ ] Test retry logic (2 retries before failure)

**Deranking Tests:**
- [ ] Test key moves to deranked after 3 consecutive failures
- [ ] Test key reverts from deranked after successful revalidation
- [ ] Test key permanently disabled after 10 failures
- [ ] Test priority queue correctly orders working keys

**Integration Tests:**
- [ ] Test full flow: add keys → validate → get working key
- [ ] Test state persistence across process restarts
- [ ] Test concurrent validation requests

**Coverage Goal:** Minimum 80% code coverage required.

## 6. Git Protocol

**Branch Strategy:**
- Feature branches: `feature/*` or `feat/*`
- Bugfix branches: `fix/*` or `bugfix/*`
- No direct commits to `main`

**Commit Messages:**
Follow Conventional Commits format:
```
