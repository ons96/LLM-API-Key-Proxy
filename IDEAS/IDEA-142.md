# AGENTS.md - Multi-Provider API Integration for Gateway

## 1. Role/Mission

You are an autonomous software agent tasked with implementing a unified gateway that integrates multiple LLM API providers to maximize available AI model options while maintaining reliability through intelligent fallback mechanisms.

**Your Mission:**
- Build a wrapper/gateway layer that aggregates g4f, puter.js, and additional free LLM API providers
- Implement intelligent provider fallback logic treating g4f as a meta-gateway (since g4f itself routes to multiple providers)
- Create a unified API interface that abstracts provider differences
- Ensure all integrations use only free resources (no paid API keys required)
- Operate autonomously on GitHub Actions without manual intervention
- Document any blocking questions in QUESTIONS.md

**Core Philosophy:**
- Maximize availability: If one provider fails, automatically try alternatives
- Simplify consumption: Single API interface regardless of underlying provider
- Treat g4f as a first-class provider while respecting its router capabilities

---

## 2. Technical Stack

### Programming Language
- **Runtime:** Node.js 18+ (LTS)
- **Language:** TypeScript 5.x

### Core Dependencies
```json
{
  "g4f": "^0.3.x",           // Meta-gateway with multiple provider integrations
  "puter": "^2.x",           //puter.js - Free AI API client
  "express": "^4.18.x",      // HTTP server for gateway API
  "axios": "^1.6.x",        // HTTP client for direct API calls
  "dotenv": "^16.x",         // Environment configuration
  "winston": "^3.x",         // Structured logging
  "zod": "^3.x",             // Schema validation
  "undici": "^6.x"           // Fast HTTP client
}
```

### Development Dependencies
```json
{
  "typescript": "^5.x",
  "ts-node": "^10.x",
  "vitest": "^1.x",          // Testing framework
  "supertest": "^6.x",       // HTTP testing
  "eslint": "^8.x",
  "prettier": "^3.x"
}
```

### Testing & CI Stack
- **CI Platform:** GitHub Actions
- **Testing:** Vitest with coverage
- **Code Quality:** ESLint + Prettier

---

## 3. Requirements (Numbered)

### Core Requirements

1. **Multi-Provider Gateway Architecture**
   - Create a unified gateway service with standardized interface
   - Support at minimum: g4f, puter.js, and 2-3 additional free providers
   - All providers must work without paid API keys

2. **Provider Abstraction Layer**
   - Define standard interface: `chat(completion: ChatRequest): Promise