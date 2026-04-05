# AGENTS.md - g4f Working Model Status Checker

---

## 1. Role/Mission

**Role:** Autonomous Monitoring Agent for AI Model Availability

**Mission:** Build and maintain a scraper that monitors the g4f-working GitHub repository to identify and track currently working AI models (agentic coding, reasoning, and chat models). The tool must provide real-time model availability information to prevent errors in downstream tools that depend on g4f model availability.

**Core Objectives:**
- Scrape the g4f-working GitHub repository for model status data
- Identify working vs. non-working models
- Provide real-time availability status for downstream consumption
- Run autonomously on GitHub Actions with minimal human intervention

---

## 2. Technical Stack

| Component | Technology | Notes |
|-----------|------------|-------|
| **Language** | Python 3.10+ | Stable, well-supported |
| **HTTP Client** | `requests` | Simple, reliable for HTTP calls |
| **HTML Parsing** | `BeautifulSoup4` | Robust scraping capabilities |
| **CI/CD** | GitHub Actions | Free tier, autonomous execution |
| **Data Storage** | JSON files in repo | No external DB required |
| **Scheduling** | GitHub Actions Cron | Built-in scheduling |
| **Logging** | Python `logging` | Standard library |

**Why This Stack:**
- All tools are free and require no external accounts
- Python ecosystem provides excellent scraping libraries
- GitHub Actions provides free compute for scheduled jobs
- JSON is human-readable and easy to parse downstream

---

## 3. Requirements (Numbered)

### 3.1 Core Functionality

1. **Scrape g4f-working Repository**
   - Access the g4f-working repository via GitHub API or web scraping
   - Identify model status information from repository contents
   - Parse working model list from README or dedicated files

2. **Model Status Validation**
   - Each run must validate at minimum 10 models
   - Categorize models by type: coding, reasoning, chat
   - Track historical availability (last 7 runs)

3. **Real-Time Status Output**
   - Generate JSON status file with current model availability
   - Include timestamp, model name, status (working/broken), category
   - Output file must be machine-parseable

4. **Scheduled Execution**
   - Run automatically every 6 hours via GitHub Actions cron
   - Produce deterministic results for same repository state

### 3.2 Operational Requirements

5. **Error Handling**
   - Gracefully handle GitHub API rate limits (use authenticated requests if needed)
   - Retry failed requests up to 3 times with exponential backoff
   - Log all errors with sufficient context for debugging

6. **Notification System**
   - Create GitHub Issue when new models become available/broken
   - Include diff from previous run in notification

7. **Data Persistence**
   - Save current status as JSON in repository
   - Maintain previous run data for comparison
   - Keep last 7 days of history

### 3.3 Autonomous Behavior

8. **Independent Decision Making**
   - Agent may update scripts, fix bugs, and improve functionality without prior approval
   - Agent must save any clarifying questions to QUESTIONS.md instead of asking humans
   - Agent may create branches and pull requests autonomously

9. **Free Resources Only**
   - Use only free tier services
   - No paid API keys unless absolutely required (document in QUESTIONS.md if needed)
   - Use GitHub's built-in Actions minutes ( generous free tier)

---

## 4. File Structure

```
g4f-working-status/
├── .github/
│   └── workflows/
│       └── status-check.yml      # Main automated job
├── src/
│   ├── __init__.py
│   ├── scraper.py                # GitHub scraping logic
│   ├── parser.py                 # Model status parsing
│   ├── validator.py              # Model validation
│   └── status.py                 # Status output generation
├── data/
│   ├── current_status.json       # Latest model status
│   └── history/                  # Historical status files
│       └── YYYY-MM-DD.json
├── tests/
│   ├── test_scraper.py
│   ├── test_parser.py
│   └── test_validator.py
├── scripts/
│   └── run_check.py              # Local execution script
├── QUESTIONS.md                   # Agent questions for humans
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
└── .gitignore
```

**Purpose of Each File:**

| File | Purpose |
|------|---------|
| `.github/workflows/status-check.yml` | GitHub Actions workflow definition |
| `src/scraper.py` | Scrapes g4f-working repository |
| `src/parser.py` | Parses raw data into model information |
| `src/validator.py` | Validates model working status |
| `src/status.py` | Generates JSON status output |
| `data/current_status.json` | Latest model availability data |
| `data/history/*.json` | Historical runs for comparison |
| `tests/*.py` | Unit tests for all modules |
| `QUESTIONS.md` | Agent questions requiring human input |
| `requirements.txt` | pip dependencies |

---

## 5. Testing Requirements

### 5.1 Unit Tests

**Required Test Coverage:**

| Module | Test Cases |
|--------|------------|
| `parser.py` | Parse valid model list, handle malformed input, extract categories |
| `validator.py` | Validate model working status, handle empty responses |
| `scraper.py` | Mock HTTP responses, handle 404/500 errors, test rate limiting |

**Coverage Target:** Minimum 80% code coverage

### 5.2 Integration Tests

1. **End-to-End Test**
   - Run full pipeline with mocked GitHub API response
   - Verify JSON output format matches specification
   - Verify history files are created correctly

2. **Failure Mode Tests**
   - Test behavior when repository is unavailable
   - Test behavior when API rate limited
   - Test behavior with empty model list

### 5.3 Validation Tests

3. **Output Schema Validation**
   - Verify JSON output matches expected schema
   - Verify timestamp is ISO 8601 format
   - Verify all required fields present

---

## 6. Git Protocol

### 6.1 Branch Strategy

| Branch | Purpose | Who Creates |
|--------|---------|-------------|
| `main` or `master` | Production code | Agent or humans |
| `feature/*` | New features | Agent |
| `fix/*` | Bug fixes | Agent |
| `automation/*` | CI/CD improvements | Agent |

### 6.2 Commit Convention

**Format:**