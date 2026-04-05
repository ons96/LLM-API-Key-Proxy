# AGENTS.md - Provider Availability & Limit Tracking System

## 1. Role/Mission

**Mission:** Create an autonomous system that tracks API provider availability, distinguishes between temporary rate limit issues versus longer outages, and implements intelligent time-based recovery mechanisms for provider selection.

**Core Capabilities:**
- Monitor and track the health status of multiple API providers
- Classify failures as either temporary (rate limits) or persistent (outages/config issues)
- Maintain historical availability data for each provider
- Implement time-based de-prioritization: temporary limits retry on next call, longer issues remain de-prioritized based on historical patterns
- Provide a clean API for querying available providers ranked by readiness

**Autonomous Behavior:**
- The agent must make independent decisions about provider selection without user intervention
- If resources are exhausted or unavailable, the agent should save questions to QUESTIONS.md rather than prompting
- Use only free tier resources (no paid APIs unless explicitly provided)

---

## 2. Technical Stack

**Language:** Python 3.10+

**Core Dependencies:**
- `pytest` - Testing framework
- `pyyaml` - Configuration storage
- `requests` - HTTP client for health checks (if needed)

**Storage:**
- JSON files for persistent state (provider health history, current status)
- YAML for configuration

**Execution Environment:**
- GitHub Actions (for CI/CD and scheduled tasks)
- Local execution support for development

**No External Services Required:**
- All tracking is done in-memory or with local file storage
- No database dependencies

---

## 3. Requirements

### 3.1 Provider Data Model

1. **Provider Registry**
   - Each provider must have: `id`, `name`, `base_priority`, `current_status`, `failure_count`, `last_failure_time`, `failure_type`, `recovery_config`

2. **Failure Classification**
   - `TEMPORARY`: Rate limit errors (HTTP 429) - should retry quickly
   - `PERSISTENT`: Configuration errors, usage limits, extended outages - should remain de-prioritized longer
   - `UNKNOWN`: Unclassified failures

3. **Health State Tracking**
   - Track: `available`, ` degraded`, `unavailable`, `unknown`

### 3.2 Time-Based Recovery Logic

4. **Temporary Failure Recovery**
   - Temporary failures (rate limits) should be reconsidered on the next provider selection call
   - Maximum de-prioritization window: 60 seconds after last failure

5. **Persistent Failure Recovery**
   - Persistent failures use exponential backoff based on historical failure frequency
   - Minimum cooldown: 5 minutes between availability checks
   - Recovery time increases with consecutive failures (up to 1 hour max)

6. **Historical Availability Scoring**
   - Maintain rolling 24-hour availability percentage per provider
   - Providers with >95% historical availability get priority boost
   - Providers with