# AGENTS.md

## Auto-Discovery System for Free LLM APIs

---

## 1. Role/Mission

You are an autonomous coding agent responsible for building and maintaining an **Auto-Discovery System for Free LLM APIs**. Your mission is to create a reliable, automated system that:

1. **Discovers free LLM APIs** by periodically scanning resources like model.dev and other publicly available directories of free LLM endpoints
2. **Catalogs discovered APIs** with their metadata (endpoint URL, model name, rate limits, usage limits, authentication requirements)
3. **Integrates with the gateway** by automatically adding valid free APIs to the configuration, respecting rate limits and usage quotas
4. **Handles manual registration** for APIs requiring API keys by notifying administrators via GitHub Issues
5. **Maintains the system** with minimal human intervention while running on free GitHub Actions infrastructure

**Decision-Making Authority**: You may make independent decisions about:
- Scraping strategies and timing (within respectful bounds)
- API validity verification methods
- Classification of API tiers (truly free vs. freemium)
- Threshold values for rate limits and usage limits

**Questions to Save**: Any ambiguous decisions about API classification, legal considerations, or infrastructure changes should be documented in `QUESTIONS.md`.

---

## 2. Technical Stack

- **Language**: Python 3.11+ (excellent library ecosystem for web scraping and scheduling)
- **Scraping**: `requests` + `BeautifulSoup4` for HTML parsing, or `playwright` for JavaScript-rendered pages if needed
- **Scheduling**: GitHub Actions scheduler (`cron` syntax) - runs on free tier
- **Storage**: JSON files in repository (simple, version-controllable, no database required)
- **Configuration**: YAML for API gateway configuration
- **Notifications**: GitHub Issues API for manual registration alerts
- **Secret Management**: GitHub Secrets (for any required tokens)

**Key Libraries**:
```txt
requests>=2.31.0
beautifulsoup4>=4.12.0
pyyaml>=6.0
python-dateutil>=2.8.0
lxml>=4.9.0
```

---

## 3. Requirements (numbered)

### 3.1 Initial Discovery Scanner

1. Build a web scraper that scans **model.dev** (and at least 2 additional free API directories) for free LLM endpoints
2. Extract: endpoint URL, model name, provider name, authentication type, rate limits (requests/minute), usage limits (calls/day or tokens/day)
3. Handle pagination and dynamic content loading
4. Respect robots.txt and implement polite scraping (1 request/second maximum)
5. Cache results to avoid redundant scans

### 3.2 API Registry System

1. Create a JSON-based API registry (`registry/apis.json`) storing discovered APIs
2. Registry schema:
   ```json
   {
     "apis": [
       {
         "id": "unique-id",
         "name": "Provider/Model",
         "endpoint": "https://...",
         "auth_type": "none|api_key|oauth",
         "rate_limit": 60,
         "rate_period": "minute",
         "usage_limit": 1000,
         "usage_period": "day",
         "discovered_at": "ISO8601",
         "last_verified": "ISO8601",
         "status": "active|verified|failed|requires_key"
       }
     ]
   }
   ```
3. Implement deduplication based on endpoint URL matching
4. Support incremental updates (only add/modify changed entries)

### 3.3 Periodic Scanning Workflow

1. Create GitHub Actions workflow (`.github/workflows/scan.yml`) that runs daily
2. On each run:
   - Perform discovery scan
   - Verify previously discovered APIs are still active (ping health endpoints)
   - Update registry with new/changed/removed APIs
   - Generate summary report
3. Handle workflow failures gracefully (retry with exponential backoff)

### 3.4 Gateway Auto-Configuration

1. Generate or update gateway configuration (`config/gateway.yaml`) with discovered free APIs
2. Apply default rate limiting (conservative: 10 req/min per API)
3. Exclude APIs marked as "requires_key" or "failed"
4. Version control all configuration changes

### 3.5 Manual Key Registration Handler

1. When an API requires an API key, create a GitHub Issue titled `[API KEY NEEDED] {Provider Name}`
2. Issue内容包括:
   - Provider name and model
   - Registration URL
   - Last verification status
   - Labels: `api-discovery`, `needs-key`
3. Mark API as `requires_key` in registry until manual resolution

### 3.6 Health Verification

1. Implement a verifier that checks API accessibility without authentication
2. Perform lightweight test (e.g., simple completion request or HEAD check)
3. Update `last_verified` timestamp and `status` field
4. Mark inactive APIs as `failed` after 3 consecutive failures

---

## 4. File Structure

```
free-llm-discoverer/
├── .github/
│   └── workflows/
│       ├── scan.yml           # Daily discovery scan
│       ├── verify.yml         # Periodic health verification
│       └── update-gateway.yml # Gateway config update
├── src/
│   ├── __init__.py
│   ├── scraper/
│   │   ├── __init__.py
│   │   ├── base.py           # Abstract scraper class
│   │   ├── modeldev.py       # model.dev scraper
│   │   └── sources.py        # Source registry and loader
│   ├── registry/
│   │   ├── __init__.py
│   │   ├── api_store.py     # JSON registry management
│   │   └── validator.py    # API validity checker
│   ├── gateway/
│   │   ├── __init__.py
│   │   ├── config_gen.py    # Gateway YAML generator
│   │   └── limits.py       # Rate limit calculators
│   ├── notifier/
│   │   ├── __init__.py
│   │   └── issues.py        # GitHub Issue creator
│   └── main.py              # CLI entry point
├── config/
│   ├── sources.yaml         # List of discovery sources
│   ├── gateways.yaml        # Gateway configuration template
│   └── limits.yaml          # Default rate/usage limits
├── registry/
│   ├── apis.json            # Discovered API registry
│   └── history/             # Historical scans
├── tests/
│   ├── __init__.py
│   ├── test_scraper.py
│   ├── test_registry.py
│   ├── test_gateway.py
│   └── fixtures/
│       ├── sample_response.html
│       └── sample_registry.json
├── .gitignore
├── requirements.txt
├── README.md
├── AGENTS.md
└── QUESTIONS.md
```

---

## 5. Testing Requirements

### 5.1 Unit Tests

1. **Scraper Tests**: Mock HTTP responses and verify parsing logic
   - Test HTML parsing for each source
   - Test pagination handling
   - Test error handling for malformed pages

2. **Registry Tests**: Verify data integrity
   - Test add/update/delete operations
   - Test deduplication logic
   - Test serialization/deserialization

3. **Gateway Tests**: Verify configuration generation
   - Test YAML output structure
   - Test limit calculations
   - Test filtering logic

### 5.2 Integration Tests

1. **Full Scan Test**: Run scraper against mock/staging sources and verify output
2. **Registry Update Test**: Verify incremental updates work correctly
3. **GitHub API Tests**: (Mocked) verify issue creation logic

### 5.3 Test Coverage Requirements

- Minimum **80% code coverage** for `src/` modules
- All critical paths must have test cases
- Use `@pytest.mark.parametrize` for multiple source testing

### 5.4 Running Tests

```bash
# Run all tests with coverage
pytest --cov=src --cov-report=html

# Run specific test module
pytest tests/test_scraper.py -v
```

---

## 6. Git Protocol

### 6.1 Branch Strategy

- **Main branch**: `main` - production-ready code only
- **Working branches**: `feature/{description}` or `fix/{description}`
- **Automation branches**: `auto/scan-{date}` - for automated scan results

### 6.2 Commit Messages

Follow conventional commits:
```
