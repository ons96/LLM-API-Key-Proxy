# Future Project Ideas and Improvements

This file documents future project ideas, planned improvements, and research tasks based on your requirements.

## 🚀 Enhancements to Existing Projects

### 📊 **LLM Leaderboards & Analytics**
**Target Projects:** `llm-leaderboard`, `llm-leaderboard-aggregate`

*   **Agentic Coding Dashboard:**
    *   **Goal:** Build a user-friendly view/table specifically for "Best Agentic Coding Models" (aggregated & normalized).
    *   **Use Case:** Quick reference for selecting models in OpenCode until the gateway is fully autonomous.
*   **Advanced Aggregation Logic:**
    *   **Goal:** Create specialized leaderboards for specific capabilities:
        *   Reasoning
        *   Chat / Knowledge
        *   Agentic Coding (Coding capability)
        *   Speed (Fastest for coding vs. fastest for chat)
        *   Reliability (Lowest hallucination rates)
    *   **Data Sources:** Scrape/integrate frequently updated sources like Artificial Analysis, LiveBench, Arena.ai, llm-stats.com.
    *   **Provider Speed:** Incorporate API provider latency/speed data (e.g., from Artificial Analysis) to rank the "fastest and best" combinations.

### 🌐 **LLM Gateway & Routing**
**Target Projects:** `agentic_gateway` (or `agentic_gateway_clean`), `Autonomous_LLM_Router`

*   **Provider Verification:**
    *   **Task:** Confirm that **Gemini CLI** and **Antigravity API** providers function correctly within your fallback-enabled gateway code (hosted on Oracle Cloud).
*   **Smart Fallbacks & Auto-Switching:**
    *   **Goal:** Improve the solution for changing models/providers in OpenCode.
    *   **Feature:** Implement advanced auto-fallback logic triggered by rate limits or usage caps.
    *   **Implementation:** Enhance the gateway's router to handle these failures transparently, serving as a robust backend for OpenCode.
*   **Adaptive Error Handling (G4F Focus):**
    *   **Problem:** G4F models often return "API call failed, you exceeded rate limit..." or token limit errors.
    *   **Solution:**
        1.  **Log Analysis:** Capture specific error messages associated with Model + Provider combinations.
        2.  **Dynamic Configuration:** Have an LLM (or logic script) parse these logs to update a "Capabilities/Limits" config file.
        3.  **Behavior Adjustment:** Automatically downgrade rankings or enforce stricter parameters (e.g., "Max Output: 10k tokens") for providers that historically fail.

### 💬 **Chat & AI Interaction**
**Target Projects:** `LibreChat`, `flask_app` (Trae Agent)

*   **Mixture of Experts / AI Counsel:**
    *   **Idea:** Create a "Counsel Mode" chatbot.
    *   **Implementation:** A frontend (potentially a LibreChat plugin or a new interface) that queries the LLM Gateway to synthesize responses from multiple models/experts.

### 🔍 **Deep Research**
**Target Projects:** `free_research_agent`

*   **Free Deep Research Stack:**
    *   **Goal:** Build the most powerful "Deep Research" tool using $0 cost components.
    *   **LLMs:** Free hosted LLMs (via G4F or free tiers).
    *   **Search APIs:**
        *   **Tavily:** (Has free tier) - Assess for "Deep Research" suitability.
        *   **DuckDuckGo:** (Free, unlimited) - Assess usage limits and relevance.
        *   **Brave Search:** Assess free tier suitability.
    *   **Task:** Benchmark these combinations to find the best free "Research Agent" setup.

---

## 💡 New Project / Research Tasks

### 🛡️ **Agentic Sandbox Environment**
**Status:** *New Research Project*

*   **Safety Analysis:**
    *   **Question:** Is it safe to use Agentic AI in a non-virtual environment?
    *   **Risk:** Agents deleting codebases, executing malicious scripts, or making irreversible system changes.
*   **Optimal Setup:**
    *   **Goal:** Determine the best, fastest, and most efficient **Free Sandbox/Coding Environment** for local agentic coding (Windows 11 or WSL).
    *   **Candidates to Explore:** Docker containers, shallow VMs, Windows Sandbox, or specific "Agent Sandbox" tools.

### 🔄 **Standalone Model Switcher (Backup)**
**Status:** *Potential New Tool / Script*

*   **Goal:** If the `agentic_gateway` cannot be used (e.g., downtime, connectivity), create a local script/tool for OpenCode that facilitates easy model swapping and basic fallbacks locally.

### 📈 **Polymarket Prediction Market Arbitrage Bot**
**Status:** *New Research Project / Bot Implementation*

*   **Core Concept:** Develop an automated trading bot to identify and exploit arbitrage opportunities within prediction markets (specifically Polymarket) and cross-platform differences.
*   **Key Features:**
    *   **Intra-Event Arbitrage:** Detect temporary discrepancies within a single event (e.g., live sports like Superbowl) where implied probabilities sum to < 100% (guaranteed profit).
    *   **Cross-Platform Arbitrage:** Compare Polymarket odds against other prediction markets or sportsbooks.
    *   **Risk-Free Execution:**
        *   Check order books for depth.
        *   Place simultaneous small orders across outcomes to lock in guaranteed profit (Atomic transactions concept).
        *   Recalculate and repeat in high-frequency loops.
    *   **Profitability Analysis:**
        *   Must account for **Gas Fees** (Polygon/Ethereum) and **Platform Fees**.
        *   Metrics: Highest Profit/Hour, Highest Annualized ROI, Profit per Minute.
*   **Investigation Tasks:**
    *   Assess API rate limits and latency for Polymarket.
    *   Investigate smart contract interactions for "atomic" execution (revert if not all orders fill).
