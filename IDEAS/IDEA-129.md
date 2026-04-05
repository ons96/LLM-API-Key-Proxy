# AGENTS.md

## LLM API Gateway with Dynamic Fallback System

---

## 1. Role/Mission

**Role:** Senior Software Architect & Full-Stack Engineer

**Mission:** Build an intelligent LLM API gateway that serves as a proxy layer between client applications and multiple LLM providers (OpenAI, Anthropic, Google, etc.). The system must:

- **Route requests intelligently** to available providers based on real-time availability
- **Fail over instantly** when encountering rate limits or failures—do not block waiting on unavailable endpoints
- **Track performance metrics** continuously for each provider/model combination
- **Dynamically reorder provider priority** based on measured latency, success rate, and rate limit status
- **Expire stale rate limit data** using timestamp-based expiry tracking to ensure accurate state

The gateway should present a unified API to consumers while handling all complexity of multi-provider orchestration internally.

---

## 2. Technical Stack

| Component | Technology | Justification |
|-----------|------------|----------------|
| **Runtime** | Node.js 20.x (LTS) | Async I/O excellent for concurrent API calls; extensive LLM provider SDKs |
| **Language** | TypeScript 5.x | Type safety for complex multi-provider interfaces |
| **Configuration** | YAML (js-yaml) | Human-readable config format; easy to version control |
| **Storage** | In-memory with file persistence (JSON) | Free, zero-dependency; sufficient for single-instance gateway |
| **HTTP Client** | Native fetch / undici | Built-in in Node 20+; no additional dependencies |
| **Testing** | Vitest | Fast, modern test runner with excellent TypeScript support |
| **Mock Servers** | MSW (Mock Service Worker) | In-memory mocking for integration tests |
| **Logging** | pino | High-performance JSON logging |

**Free Tier Compatible Providers:**
- OpenAI (GPT-4o mini - free tier available)
- Anthropic (Claude - pay-as-you-go, no free tier but low cost)
- Google Gemini (free tier)
- Groq (free tier with generous limits)
- Ollama (local, free)

---

## 3. Requirements

### Core Functionality

1. **Unified Gateway API**
   - Expose single endpoint that accepts requests for any model
   - Accept provider hints but allow system to override for reliability
   - Return standardized response format regardless of underlying provider

2. **Provider Configuration Management**
   - Load provider credentials from config file
   - Support multiple API keys per provider
   - Define per-provider rate limits and usage quotas in configuration

3. **Rate Limit Expiry Tracking**
   - Track `rateLimitExpiry` (datetime) for each provider—when rate limit resets
   - Track `usageLimitExpiry` (datetime) for daily/monthly quotas
   - Use `max(rateLimitExpiry, usageLimitExpiry)` to determine true availability
   - Implement timestamp-based expiry that auto-clears when time passes

4. **Instant Failover Logic**
   - When a provider returns 429 (rate limit), immediately skip without retry delay
   - Use "try next" strategy rather than waiting
   - Queue failed requests briefly only for transient errors (5xx), not rate limits

5. **Performance Metrics Collection**
   - Track latency (time to first token / total response time)
   - Track success/failure counts per provider
   - Calculate performance score: `(successes / totalRequests) * (1 / avgLatency)`
   - Update metrics on each request completion

6. **Dynamic Provider Reordering**
   - Sort available providers by: `expiryTime ASC` then `performanceScore DESC`
   - Prioritize providers with nearest expiry (soonest available)
   - Break ties using performance score
   - Re-evaluate sort order after each request

7. **Graceful Degradation**
   - If all providers exhausted, return 503 with clear error message
   - Include wait times in response when rate limits are known

### Data Models

8. **Provider State Schema**
   ```typescript
   interface ProviderState {
     providerId: string;
     modelId: string;
     rateLimitExpiry: Date | null;      // When 429 will clear
     usageLimitExpiry: Date | null;     // When quota resets
     requestCount: number;
     successCount: number;
     failureCount: number;
     totalLatencyMs: number;
     lastUsedAt: Date;
     isHealthy: boolean;
   }
   ```

9. **Sorted Provider Logic**
   - Compute `availableAt = max(rateLimitExpiry, usageLimitExpiry)`
   - Filter to only providers where `availableAt