# AGENTS.md - Model Availability Monitor with g4f Integration

## 1. Role/Mission

**Mission**: Create and maintain an automated system that fetches model availability data from g4f's GitHub repository, stores it locally, and runs daily to keep the reference data current. The agent acts as a curator of community-sourced model status information, making this data discoverable and accessible while clearly communicating its reference-only nature.

**Role Definition**: You are an autonomous coding agent responsible for building a lightweight, automated pipeline that:
- Retrieves current model availability status from g4f's public GitHub data sources
- Transforms and stores this data in a usable local format
- Maintains the system with daily automated updates via GitHub Actions
- Documents usage limitations clearly so users understand this is supplementary reference data, not a definitive exclusion tool
- Provides clear indicators when models appear unavailable while emphasizing direct testing is always recommended

## 2. Technical Stack

**Core Technologies** (Free Resources Only):
- **Python 3.10+** - Primary scripting language for data fetching and processing
- **GitHub Actions** - Free CI/CD automation for daily scheduled runs
- **GitHub API** - For fetching raw data from g4f repository without authentication
- **requests library** - HTTP client for API calls (or urllib from stdlib)
- **JSON** - Primary data storage format for model status data
- **YAML** - For configuration files
- **Cron (via GitHub Actions)** - Scheduled triggers for daily updates

**Development Tools**:
- **Git** - Version control (built into GitHub Actions)
- **VS Code** or any text editor for development
- **shell** - bash scripts for automation

**External Dependencies**:
- g4f GitHub repository (public): `https://github.com/xeniaix/g4f`
- Raw GitHub content delivery via `raw.githubusercontent.com`
- No API keys or paid services required

## 3. Requirements (Numbered)

### Data Acquisition
1. **Fetch g4f Model Status Data**: Retrieve current model availability information from g4f's public GitHub repository using the GitHub raw content API
2. **Identify Data Source**: Determine the specific file(s) in g4f repository that contain model status information (typically providers or model lists)
3. **Handle Missing Data**: Gracefully handle cases where g4f data source is unavailable or returns errors, maintaining previous data when possible

### Data Processing
4. **Parse and Transform**: Parse the fetched data into a structured format suitable for local storage
5. **Extract Model Information**: Extract model names, provider associations, and availability indicators from the raw data
6. **Normalize Data**: Create a consistent data schema with fields: `model_name`, `provider`, `status`, `last_updated`, `source`

### Storage & Persistence
7. **Local Storage**: Store processed model data in JSON format at `data/model_status.json`
8. **Metadata Storage**: Maintain a metadata file with fetch timestamp, data source URL, and record count at `data/metadata.json`
9. **Data Retention**: Keep historical snapshots (daily) in `data/history/` for reference and debugging

### Automation
10. **Daily Scheduled Run**: Configure GitHub Actions workflow to run daily using cron schedule
11. **Manual Trigger**: Ensure workflow can be manually triggered via GitHub Actions UI
12. **Automated Commits**: Automatically commit updated data files when new data is fetched and differs from previous

### Documentation & User Guidance
13. **Create Reference README**: Document that this is reference-only data, not definitive, and models not in list may still work
14. **Update Notice**: Clearly note the daily update frequency and that lag may exist between actual availability and listed status
15. **Test Disclaimer**: Include guidance for direct testing to confirm model availability

### Error Handling
16. **Network Error Handling**: Implement retry logic (3 attempts) for network requests with appropriate backoff
17. **Rate Limiting**: Implement delays between requests to avoid GitHub API rate limits
18. **Logging**: Write meaningful logs to console and optionally to a log file for debugging

## 4. File Structure

```
model-availability-monitor/
├── .github/
│   └── workflows/
│       └── daily-update.yml    # GitHub Actions workflow for daily runs
├── data/
│   ├── model_status.json        # Current model status data
│   ├── metadata.json           # Fetch metadata (timestamp, source, count)
│   └── history/                 # Historical snapshots
│       └── YYYY-MM-DD.json      # Daily snapshots
├── scripts/
│   ├── fetch_data.py            # Main data fetching script
│   ├── process_data.py          # Data transformation and storage
│   └── update_workflow.py       # Helper to manage workflow files
├── docs/
│   ├── USAGE.md                # User guide for the system
│   └── REFERENCE_NOTES.md       # Important disclaimers and notes
├── tests/
│   ├── test_fetch.py            # Tests for data fetching
│   ├── test_process.py         # Tests for data processing
│   └── test_integration.py     # Integration tests
├── config/
│   └── settings.yml            # Configuration file
├── .gitignore                   # Git ignore patterns
├── README.md                   # Project overview
├── REQUIREMENTS.txt             # Python dependencies
└── AGENTS.md                   # This specification
```

## 5. Testing Requirements

### Unit Tests
- **Data Fetching Tests**: Verify the fetch script correctly handles successful responses, HTTP errors, and empty responses
- **Data Processing Tests**: Verify transformation correctly maps input fields to output schema
- **Error Handling Tests**: Verify retry logic works as expected and failures are handled gracefully

### Integration Tests
- **End-to-End Test**: Test the full pipeline (fetch → process → store) with mocked responses or when g4f source is available
- **Output Validation**: Verify output JSON matches expected schema and contains required fields

### Test Execution
- **Local Testing**: All tests must pass locally before any commit
- **Test Coverage**: At minimum, test core functionality (fetch, process, error handling)
- **Isolation**: Tests should not depend on external network availability (use mocks or record/replay)

### Verification Steps
1. Run `python -m pytest tests/` locally
2. Ensure all tests pass with green output
3. Verify `data/model_status.json` is generated with valid JSON structure
4. Verify `data/metadata.json` contains fetch timestamp and source URL

## 6. Git Protocol

### Branch Strategy
- **Main Branch**: `main` - Always deployable, contains latest stable data
- **Feature Branches**: `feature/description` - For new features or fixes
- **Workaround**: Use descriptive branch names (e.g., `feature/add-daily-scheduler`)

### Commit Messages
Follow conventional commits format:
- `feat: add model status fetch from g4f` - New features
- `fix: handle empty response from g4f` - Bug fixes
- `docs: update README with usage notes` - Documentation
- `chore: add test coverage for fetch` - Maintenance

### Workflow
1. Create feature branch from `main`
2. Make changes and commit incrementally
3. Push branch and create Pull Request
4. Require at least basic test verification (self-review acceptable for small changes)
5. Merge to `main` after verification
6. GitHub Actions will trigger workflow on `main` push

### Data Commit Rules
- **Auto-commit updated data**: The workflow can auto-commit changes to `data/` directory when run completes successfully
- **Commit message format for data**: `data: update model status - YYYY-MM-DD`
- **Require changes**: Only commit if data actually differs from previous

## 7. Completion Criteria

### Core Functionality
- [ ] Script successfully fetches model status data from g4f GitHub repository
- [ ] Data is parsed, transformed, and stored in valid JSON format at `data/model_status.json`
- [ ] Metadata file is created/updated at `data/metadata.json` with fetch timestamp and source
- [ ] GitHub Actions workflow is configured and runs on schedule (daily) and manual trigger

### Documentation
- [ ] README.md provides clear project overview
- [ ] USAGE.md documents how to use the system
- [ ] REFERENCE_NOTES.md contains explicit disclaimer that this is reference-only data

### Testing
- [ ] Basic unit tests exist and pass for core functionality
- [ ] All tests can run via `python -m pytest tests/`

### Automation
- [ ] Workflow runs daily via cron schedule
- [ ] Workflow can be triggered manually via GitHub Actions UI
- [ ] Data updates are automatically committed when significant changes are detected

### System Health
- [ ] No hardcoded credentials or secrets (all free/public resources)
- [ ] Error handling prevents workflow failures from incomplete data
- [ ] README clearly states this supplements but does not replace direct testing

---

**Questions**: Save any questions or ambiguities to `QUESTIONS.md` for human review. Do not block on unclear requirements - make reasonable assumptions and document them.

**Independence**: You are authorized to make decisions within this scope. Use free resources only. When uncertain, prefer simplest working solution and document your reasoning.