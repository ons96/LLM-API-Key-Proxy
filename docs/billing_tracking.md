# Billing Tracking Documentation

## Overview
The billing tracking system monitors API usage and calculates estimated costs for paid providers. It uses a SQLite database to store usage records and a YAML configuration file to define pricing models.

## Configuration

### Pricing Config (`config/pricing.yaml`)
Pricing is defined per 1,000,000 (1M) tokens. To add or update pricing for a provider, edit `config/pricing.yaml`:

```yaml
provider_name:
  model_name:
    input_cost_per_1m: 10.00
    output_cost_per_1m: 30.00
```

### Database
The system automatically creates a `billing.db` file in the root directory (or wherever configured) on startup.

## Usage

### In Proxy Logic
To track a request, import the integration helper in your proxy or router module:

```python
from proxy_app.billing_integration import track_request_cost

# After receiving a response from the provider
usage_data = {
    "prompt_tokens": response.usage.prompt_tokens,
    "completion_tokens": response.usage.completion_tokens
}

track_request_cost(
    provider="openai",
    model="gpt-4o",
    usage_data=usage_data
)
```

### Direct Access
For custom scripts or analysis, access the `BillingTracker` directly:

```python
from proxy_app.billing_tracker import BillingTracker
from datetime import datetime, timedelta

tracker = BillingTracker()

# Log usage manually
tracker.log_usage("anthropic", "claude-3-opus", 1000, 500)

# Get total spend this month
start = datetime.now().replace(day=1)
total = tracker.get_total_spend(start_date=start)
print(f"Total spend: ${total:.2f}")

# Get stats per model
stats = tracker.get_usage_stats(days=7)
for stat in stats:
    print(f"{stat['model']}: ${stat['total_cost']:.2f}")
```

## Data Schema

### `usage_records` Table
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary Key |
| timestamp | DATETIME | ISO format timestamp of the request |
| provider | TEXT | Provider name (e.g., 'openai') |
| model | TEXT | Model name (e.g., 'gpt-4o') |
| input_tokens | INTEGER | Number of input/prompt tokens |
| output_tokens | INTEGER | Number of output/completion tokens |
| total_tokens | INTEGER | Sum of input and output tokens |
| estimated_cost_usd | REAL | Calculated cost based on pricing config |

## Notes
- Costs are estimates based on public pricing. Actual billing may vary due to taxes, rounding, or specific enterprise agreements.
- If a model is not found in `pricing.yaml`, the cost is recorded as `0.0`.
- The system handles both `prompt_tokens`/`completion_tokens` and `input_tokens`/`output_tokens` naming conventions.
