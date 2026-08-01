# Comprehensive Project Overview & Guide

## 1. LLM API Gateway & Router
**Directory**: `agentic_gateway`, `LLM-API-Key-Proxy`, `Autonomous_LLM_Router`
**Repo**: [Check local git status]

### What it does
Acts as a central "Brain" that accepts OpenAI-compatible API requests and routes them to the best free provider (Cerebras, Groq, Puter.js, etc.).

### Key Features
- **3-Tier Failover**: Primary -> Secondary -> Fallback.
- **Smart Routing**: Routes to the fastest or smartest model based on request type.
- **Unified Billing**: (Virtual) - tracks usage across free providers.

### How to use
```bash
# Run the proxy
cd LLM-API-Key-Proxy
pip install -r requirements.txt
python app.py
# Endpoint: http://localhost:8000/v1
```

---

## 2. Free Research Agent (Perplexity Replacement)
**Directory**: `free_research_agent`
**Repo**: `free_research_agent` (Needs push)

### What it does
An AI-powered search engine that performs "Deep Research" (multi-step investigation) and shopping analysis.

### Key Features
- **Deep Research Mode**: Recursively searches, reads, and plans follow-up queries.
- **Shopping Mode**: Extracts product prices using LLM for accurate comparison.
- **Council Mode**: Queries top 3 models from the leaderboard and synthesizes answers.
- **Auto-Detect**: Automatically switches modes based on your query.

### How to use
```bash
cd free_research_agent
pip install -r requirements.txt
uvicorn app.main:app --reload
# Go to http://localhost:8000
```

---

## 3. LLM Leaderboard Aggregator
**Directory**: `llm-leaderboard`
**Repo**: `llm-leaderboard` (Needs push)

### What it does
Scrapes data from 5+ major leaderboards (LMSYS, Aider, LiveBench, UGI), normalizes scores, and ranks models.

### Key Features
- **UGI Integration**: Includes "Uncensored" scores.
- **Speed Metrics**: Merges data from Artificial Analysis (tokens/sec).
- **Viewer**: `viewer.html` for easy sorting/filtering.

### How to use
```bash
cd llm-leaderboard
python llm_aggregated_leaderboard.py
# Open viewer.html in browser
```

---

## 4. LLM Provider Status Tracker
**Directory**: `llm-provider-tracker`
**Repo**: `llm-provider-tracker` (Needs push)

### What it does
Monitors the health and speed (TTFT/TPS) of free API providers (Cerebras, Groq, OpenRouter).

### Key Features
- **Real-time Polling**: Checks endpoints every minute.
- **Dashboard**: Visualizes latency history.

### How to use
```bash
cd llm-provider-tracker
python poller.py & # Run in background
streamlit run dashboard.py
```

---

## 5. Uncensored AI (BlackBerry RP)
**Directory**: `AI_RP_app`
**Repo**: `AI_RP_app` (Needs push)

### What it does
A lightweight chat app optimized for BlackBerry Classic/older devices, focusing on Roleplay.

### Key Features
- **Jailbreaks**: Built-in prompts to bypass filters.
- **Character Cards**: Supports SillyTavern-style cards.
- **Lightweight**: Works on old browsers.

---

## 6. Utilities
### eSIM Finder
**Directory**: `esimdb_scraper`
Finds cheapest eSIM plans.
`python optimize_itinerary_cli.py --countries "Germany,France"`

### Steam Region Optimizer
**Directory**: `steam-region-optimizer`
Finds fastest Steam download servers.
`python steam_optimizer.py`

### Oracle Automation
**Directory**: `oracle-freetier-automation`
Script to retry creating Always Free instances.
`python oracle_instance_retry.py`

---

## Git Repositories Status
*Note: Run `git init` and `git push` in directories that are currently local-only.*
