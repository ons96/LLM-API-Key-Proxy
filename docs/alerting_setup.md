# Provider Outage Alerting Setup

This document describes how to configure alerting for provider outages in the LLM Proxy Router.

## Overview

The alerting system monitors provider health and sends notifications when:
- A provider goes down (consecutive failures >= threshold)
- A provider recovers from a down/degraded state
- A provider becomes degraded (optional, disabled by default)

## Configuration Methods

Alerting can be configured via:
1. **Environment Variables** (recommended for secrets)
2. **Configuration File** (`config/router_config.yaml`)

## Environment Variables

### Generic Webhook
```bash
export ALERT_WEBHOOK_URL="https://your-webhook-endpoint.com/alert"
export ALERT_WEBHOOK_HEADERS="Authorization:Bearer token123,X-Custom-Header:value"
export ALERT_COOLDOWN_MINUTES=15
export ALERT_ON_DOWN=true
export ALERT_ON_RECOVERY=true
export ALERT_ON_DEGRADED=false
```

### Discord Integration
```bash
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
export DISCORD_ALERT_COOLDOWN_MINUTES=15
export DISCORD_ALERT_ON_DOWN=true
export DISCORD_ALERT_ON_RECOVERY=true
export DISCORD_ALERT_ON_DEGRADED=false
```

### Slack Integration
```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
export SLACK_ALERT_COOLDOWN_MINUTES=15
export SLACK_ALERT_ON_DOWN=true
export SLACK_ALERT_ON_RECOVERY=true
export SLACK_ALERT_ON_DEGRADED=false
```

## Alert Cooldown

To prevent alert spam, the system implements cooldown periods:
- **Default cooldown**: 15 minutes between alerts for the same provider
- **Recovery alerts**: Always sent immediately (no cooldown)
- **Per-channel cooldown**: Each notification channel (Discord, Slack, generic) has its own cooldown

## Alert Payload Format

### Generic Webhook
```json
{
  "provider": "groq",
  "status": "down",
  "timestamp": "2024-01-15T10:30:00",
  "previous_status": "healthy",
  "response_time_ms": 5000.5,
  "consecutive_failures": 3,
  "error_message": "Connection timeout",
  "uptime_percent": 95.5,
  "alert_type": "outage"
}
```

### Discord
Rich embeds with color coding:
- 🟢 Green: Healthy/Recovery
- 🟡 Yellow: Degraded
- 🔴 Red: Down

### Slack
Formatted attachments with color coding:
- Green: Healthy/Recovery
- Yellow: Degraded
- Red: Down

## Testing Alerts

To test the alerting system:

```python
import asyncio
from src.rotator_library.alert_manager import AlertManager

async def test_alert():
    manager = AlertManager()
    await manager.send_alert(
        provider="test_provider",
        status="down",
        details={
            "error_message": "Test alert",
            "response_time_ms": 1000,
            "consecutive_failures": 3,
            "previous_status": "healthy"
        }
    )

asyncio.run(test_alert())
```

## Troubleshooting

### Alerts not sending
1. Check that webhook URLs are correctly set in environment variables
2. Verify network connectivity to webhook endpoints
3. Check logs for "Failed to send" error messages
4. Ensure cooldown period has passed since last alert

### Duplicate alerts
- Check cooldown configuration - may need to increase `ALERT_COOLDOWN_MINUTES`
- Verify that multiple instances of the tracker aren't running

### Missing recovery alerts
- Ensure `ALERT_ON_RECOVERY` is set to `true`
- Recovery alerts bypass cooldown and should always send immediately

## Integration with Status API

Alert configuration can be retrieved via the Status API:

```python
from src.rotator_library.provider_status_tracker import ProviderStatusTracker

tracker = ProviderStatusTracker()
config = tracker.get_alert_config()
print(config)
```

This returns the current alert configuration and recent activity summary.
