# AGENTS.md: Cost-Aware AI Model Selection Router

## 1. Role/Mission

**Role:** Cost-Aware AI Model Selection Router

**Mission:** Build an intelligent routing system that dynamically selects AI models based on token costs, multipliers, and performance characteristics while actively managing daily token quota consumption. The router must maximize coding capability when quota permits, and automatically transition to cost-efficient models when quota depletion is predicted within the billing period.

**Core Responsibilities:**
- Track token consumption against daily quota limits in real-time
- Calculate price/performance ratios for available models using cost multipliers
- Predict quota exhaustion timelines based on current consumption rates
- Automatically select optimal models balancing capability and cost efficiency
- Handle quota reset timing and usage pattern awareness
- Provide transparent logging and recommendations for model switches

---

## 2. Technical Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Runtime** | Node.js 20.x | Execution environment |
| **Language** | TypeScript 5.x | Type-safe implementation |
| **Configuration** | JSON/YAML | Model definitions, quota settings |
| **Storage** | Local JSON files | Quota state, usage history |
| **Testing** | Vitest | Unit and integration testing |
| **Linting** | ESLint + Prettier | Code quality |
| **CI/CD** | GitHub Actions | Automation |

---

## 3. Requirements

### 3.1 Token Cost Tracking

1. **Model Configuration**: Define a configuration file containing all supported models with their attributes:
   - Model identifier (e.g., `o3-pro`, `o3`, `glm-4.5-air`)
   - Token cost multiplier (e.g., `1x`, `8x`, `0x`)
   - Performance rating (1-10 scale for agentic coding capability)
   - Cost tier classification (`premium`, `standard`, `free`)

2. **Cost Calculation Engine**: Implement a function that calculates effective token cost for any operation:
   - Input: base token count, model multiplier
   - Output: actual tokens consumed
   - Handle edge cases (zero-cost models, invalid multipliers)

3. **Multi-model Comparison**: Provide utilities to compare price/performance across models:
   - Calculate price/performance ratio (cost per performance point)
   - Rank models by efficiency
   - Identify best model for specific budget constraints

### 3.2 Quota Monitoring

4. **Quota State Management**: Track current quota status:
   - Total daily quota (configurable, default 500000)
   - Tokens consumed (current period)
   - Tokens remaining
   - Quota reset timestamp
   - Usage rate (tokens per hour)

5. **Smart Reset Handling**: Implement reset logic that accounts for:
   - Fixed reset time (e.g., midnight UTC) OR rolling 24-hour window
   - Configurable reset strategy in settings
   - Proper state reset with history archival

6. **Consumption Tracking**: Record every token operation:
   - Timestamp of request
   - Model used
   - Token count (with multiplier applied)
   - Running total for period

### 3.3 Predictive Analysis

7. **Usage Rate Calculation**: Compute rolling consumption metrics:
   - Tokens per minute/hour rolling average
   - Weighted recent usage (emphasize last hour)
   - Support configurable averaging window

8. **Quota Exhaustion Prediction**: Calculate predicted timeline:
   - Estimated hours/minutes until quota depletion
   - Confidence level based on usage variance
   - Threshold detection (e.g., "at risk" when