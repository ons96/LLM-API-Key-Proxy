# AGENTS.md

## Model Performance Scoring & Selection Algorithm

---

## 1. Role/Mission

**Role:** Autonomous Algorithm Designer & Engineer

**Mission:** Design, implement, and maintain an intelligent Model Performance Scoring & Selection Algorithm that automatically determines the optimal ordering of AI models for different task types. The algorithm must analyze real-world performance metrics, calculate aggregate scores, and generate task-specific model rankings that balance speed vs. capability trade-offs.

**Core Objectives:**
- Collect and analyze performance metrics (tokens/sec, reasoning time, total response time, agentic coding scores)
- Calculate weighted aggregate performance scores per model
- Generate optimized model orderings for each task category (coding-smart, coding-fast, chat-smart, chat-fast, roleplay)
- Provide continuous scoring that adapts to real provider response data
- Enable intelligent selection of models based on task requirements

---

## 2. Technical Stack

**Language:** Python 3.10+

**Core Dependencies:**
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical calculations and array operations
- `pydantic` - Data validation and settings management
- `httpx` - Async HTTP client for API calls
- `python-dotenv` - Environment variable management

**Testing:**
- `pytest` - Unit and integration testing
- `pytest-asyncio` - Async test support
- `pytest-cov` - Code coverage reporting

**Optional (Free Tier):**
- `streamlit` - Interactive dashboard (free tier)
- `plotly` - Visualization (free tier)

**Storage:**
- JSON/YAML file-based storage for metrics and scores (no paid DB required)
- Local SQLite (built-in) for persistent storage if needed

---

## 3. Requirements (Numbered)

### 3.1 Data Collection & Ingestion

1. **Requirement:** Implement a metrics collection system that can ingest performance data from multiple AI providers
2. **Requirement:** Support both batch import (JSON/CSV) and real-time API polling for metrics
3. **Requirement:** Store raw metrics with timestamp, model identifier, provider, and task type
4. **Requirement:** Handle missing/null values gracefully with configurable default strategies

### 3.2 Performance Score Calculation

5. **Requirement:** Calculate aggregate performance scores using weighted multi-metric formula:
   ```
   Score = (W_speed × speed_score) + (W_quality × quality_score) + 
          (W_reliability × reliability_score) + (W_capability × capability_score)
   ```
6. **Requirement:** Provide configurable weight profiles for different task types
7. **Requirement:** Normalize all metrics to 0-100 scale before aggregation
8. **Requirement:** Implement exponential decay for older data points (recency weighting)

### 3.3 Model Ordering Algorithm

9. **Requirement:** Generate rank-ordered model lists for each task type (coding-smart, coding-fast, chat-smart, chat-fast, roleplay)
10. **Requirement:** Apply task-specific ordering rules:
    - `coding-smart`: Priority order = capability_score > reliability > speed
    - `coding-fast`: Priority order = speed_score > reliability > capability
    - `chat-smart`: Priority order = quality_score > capability > speed
    - `chat-fast`: Priority order = speed_score > quality > capability
    - `roleplay`: Priority order = creativity_score > capability > speed
11. **Requirement:** Support custom task type definitions via configuration

### 3.4 Trade-off Analysis

12. **Requirement:** Calculate and report trade-off metrics:
    - Power/Speed ratio per model
    - Cost-effectiveness score
    - Capability ceiling vs. average performance delta
13. **Requirement:** Generate comparative visualizations showing model positioning on power-speed axes

### 3.5 API & Interface

14. **Requirement:** Provide a RESTful API (using Flask/FastAPI with free tier) for:
    - Fetching current model rankings by task type
    - Submitting new performance metrics
    - Querying individual model scores
15. **Requirement:** Implement a CLI tool for local operations

### 3.6 Configuration & Extensibility

16. **Requirement:** Support configuration via YAML/JSON files
17. **Requirement:** Allow easy addition of new metrics or providers
18. **Requirement:** Provide plugin architecture for custom scoring algorithms

---

## 4. File Structure

```
model-selector/
├── README.md
├── AGENTS.md
├── QUESTIONS.md
├── requirements.txt
├── pyproject.toml
│
├── src/
│   ├── __init__.py
│   ├── main.py                 # Entry point
│   ├── config.py              # Configuration management
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── scorer.py           # Core scoring algorithm
│   │   ├── ranker.py           # Model ranking generation
│   │   ├── normalizer.py       # Metric normalization
│   │   └── tradeoffs.py        # Trade-off analysis
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── collector.py        # Metrics collection
│   │   ├── storage.py          # Data persistence
│   │   └── validator.py       # Data validation
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── metrics.py         # Metric data models
│   │   ├── score.py           # Score data models
│   │   └── task.py            # Task type definitions
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py          # API routes
│   │   └── schemas.py         # Request/response schemas
│   │
│   ├── cli/
│   │   ├── __init__.py
│   │   └── commands.py        # CLI commands
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py          # Logging utilities
│       └── math.py            # Math helpers
│
├── config/
│   ├── default.yaml          # Default configuration
│   ├── weights.yaml          # Scoring weights per task
│   └── tasks.yaml             # Task type definitions
│
├── data/
│   ├── metrics/              # Raw metrics storage
│   ├── scores/               # Calculated scores
│   └── rankings/              # Generated rankings
│
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_scorer.py
│   │   ├── test_ranker.py
│   │   └── test_normalizer.py
│   ├── integration/
│   │   ├── __init__.py
│   │   └── test_full_pipeline.py
│   └── fixtures/
│       ├── __init__.py
│       └── sample_metrics.json
│
├── scripts/
│   ├── collect_metrics.py   # Manual metrics collection
│   ├── generate_rankings.py  # Generate rankings
│   └── demo_data.py          # Generate demo data
│
└── docs/
    ├── ARCHITECTURE.md
    ├── API.md
    └── SCORING.md
```

---

## 5. Testing Requirements

### 5.1 Unit Tests (Required)

1. **Test Coverage Minimum:** 80% code coverage required
2. **Core Algorithm Tests:**
   - Test aggregate score calculation with known inputs
   - Test normalization functions (verify 0-100 bounds)
   - Test ranking generation produces correct order
   - Test trade-off calculations
3. **Data Validation Tests:**
   - Test invalid metric rejection
   - Test missing value handling
   - Test timestamp ordering

### 5.2 Integration Tests

4. **Test full pipeline:** Ingest sample metrics → Calculate scores → Generate rankings → Verify output
5. **Test configuration loading:** Ensure config files load correctly
6. **Test data persistence:** Write and read from storage

### 5.3 Test Data

7. **Provide fixture data:** Include sample_metrics.json with realistic test data
8. **Generate demo data:** Create scripts/demo_data.py that generates varied test data for all task types

### 5.4 CI/CD Testing

9. **GitHub Actions Workflow:** Run tests on every push
10. **Coverage Check:** Fail build if coverage drops below 80%

---

## 6. Git Protocol

### 6.1 Branch Strategy

- **Main Branch:** `main` - Production-ready code only
- **Development Branch:** `develop` - Integration branch
- **Feature Branches:** `feature/description` - For new features
- **Bugfix Branches:** `bugfix/description` - For bug fixes
- **Experiment Branches:** `experiment/description` - For exploratory work

### 6.2 Commit Messages

Format: