# Data Collection Setup Guide

## Prerequisites

- Python 3.8+
- pandas
- requests
- python-dotenv

## Quick Start

1. **Install Dependencies**
```bash
pip install pandas requests python-dotenv
```

2. **Configure API Keys**
Create `.env` file in project root:
```env
ARTIFICIAL_ANALYSIS_API_KEY=your_api_key_here
```

3. **Run Initial Data Fetch**
```bash
python src/rotator_library/fetch_artificial_analysis_data.py
```

4. **Verify Output**
Check for `artificial_analysis_models.csv` and `last_successful_fetch.txt` in working directory.

## Data Source Configuration

### Artificial Analysis API

**Cache Duration**: 24 hours (configurable via `CACHE_DURATION_HOURS`)

**Fields Extracted**:
- `id`, `name`, `slug`: Model identifiers
- `model_creator_name`, `model_creator_slug`: Organization info
- `eval_*`: Benchmark scores (flattened)
- `price_*`: Pricing tiers (flattened)
- `median_output_tokens_per_second`: Performance metric

**Excluded Fields**:
- `median_time_to_first_token_seconds`
- `median_time_to_first_answer_token`

### Provider Database

Edit `config/providers_database.yaml` to add/update providers:

```yaml
providers:
  - id: new_provider
    name: "Provider Name"
    signup_url: "https://..."
    free_tier: true
    free_models:
      - id: "model-id"
        context: 8192
        rpm: 10
        daily_limit: 1000
    capabilities: [chat, code, vision]
    last_verified: "2025-01-15"
```

## Automation

### Cron Setup (Linux/Mac)
```bash
# Update benchmarks every 6 hours
0 */6 * * * cd /path/to/project && python src/rotator_library/fetch_artificial_analysis_data.py >> /var/log/llm_fetch.log 2>&1
```

### Windows Task Scheduler
Schedule `fetch_artificial_analysis_data.py` to run every 6 hours.

## Troubleshooting

### "ARTIFICIAL_ANALYSIS_API_KEY not found"
- Verify `.env` file exists and is loaded
- Check `load_dotenv()` path in fetch script
- Ensure key is set: `echo $ARTIFICIAL_ANALYSIS_API_KEY`

### Cache not refreshing
- Delete `last_successful_fetch.txt` to force refresh
- Check file permissions on timestamp file

### CSV parsing errors
- Verify API response format hasn't changed
- Check `process_api_data()` for new nested structures
- Review `encoding='utf-8'` setting for special characters

## Data Schema Reference

### artificial_analysis_models.csv Columns

| Column | Type | Description |
|--------|------|-------------|
| id | string | UUID from API |
| name | string | Display name |
| slug | string | URL-friendly ID |
| model_creator_name | string | Organization |
| model_creator_slug | string | Org ID |
| eval_* | float | Benchmark scores |
| price_* | float/JSON | Pricing data |
| median_output_tokens_per_second | float | Speed metric |

### providers_database.yaml Schema

```yaml
providers:
  - id: string          # Unique identifier
    name: string        # Display name
    signup_url: string  # Registration URL
    api_base: string    # Optional: Custom endpoint
    free_tier: boolean  # Offers free tier?
    note: string        # Optional: Usage notes
    free_models:        # List of free models
      - id: string      # Model identifier
        name: string    # Optional: Display name
        context: int    # Context window
        rpm: int        # Requests per minute
        daily_limit: int # Daily request/token limit
        note: string    # Optional: Model-specific notes
    capabilities: [chat, code, vision, embeddings, image]
    last_verified: date # ISO 8601 date
```
