#!/usr/bin/env python3
"""
Alert Manager Module

Handles sending alerts for provider outages and recoveries via webhooks
and other notification channels. Includes cooldown management to prevent
alert spam.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class AlertConfig:
    """Configuration for alerting."""
    webhook_url: Optional[str] = None
    webhook_headers: Dict[str, str] = None
    cooldown_minutes: int = 15
    alert_on_degraded: bool = False
    alert_on_down: bool = True
    alert_on_recovery: bool = True
    include_details: bool = True
    
    def __post_init__(self):
        if self.webhook_headers is None:
            self.webhook_headers = {}


class AlertManager:
    """
    Manages alerting for provider status changes.
    Supports webhook notifications (Discord, Slack, generic) with cooldown
    periods to prevent spam.
    """
    
    def __init__(self):
        self.configs: Dict[str, AlertConfig] = {}
        self.last_alert_times: Dict[str, datetime] = {}
        self._load_config_from_env()
        
    def _load_config_from_env(self):
        """Load alert configurations from environment variables."""
        # Generic webhook
        generic_webhook = os.getenv("ALERT_WEBHOOK_URL")
        if generic_webhook:
            self.configs["generic"] = AlertConfig(
                webhook_url=generic_webhook,
                webhook_headers=self._parse_headers(os.getenv("ALERT_WEBHOOK_HEADERS", "")),
                cooldown_minutes=int(os.getenv("ALERT_COOLDOWN_MINUTES", "15")),
                alert_on_degraded=os.getenv("ALERT_ON_DEGRADED", "false").lower() == "true",
                alert_on_down=os.getenv("ALERT_ON_DOWN", "true").lower() == "true",
                alert_on_recovery=os.getenv("ALERT_ON_RECOVERY", "true").lower() == "true"
            )
            
        # Discord-specific
        discord_webhook = os.getenv("DISCORD_WEBHOOK_URL")
        if discord_webhook:
            self.configs["discord"] = AlertConfig(
                webhook_url=discord_webhook,
                cooldown_minutes=int(os.getenv("DISCORD_ALERT_COOLDOWN_MINUTES", "15")),
                alert_on_degraded=os.getenv("DISCORD_ALERT_ON_DEGRADED", "false").lower() == "true",
                alert_on_down=os.getenv("DISCORD_ALERT_ON_DOWN", "true").lower() == "true",
                alert_on_recovery=os.getenv("DISCORD_ALERT_ON_RECOVERY", "true").lower() == "true"
            )
            
        # Slack-specific
        slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
        if slack_webhook:
            self.configs["slack"] = AlertConfig(
                webhook_url=slack_webhook,
                cooldown_minutes=int(os.getenv("SLACK_ALERT_COOLDOWN_MINUTES", "15")),
                alert_on_degraded=os.getenv("SLACK_ALERT_ON_DEGRADED", "false").lower() == "true",
                alert_on_down=os.getenv("SLACK_ALERT_ON_DOWN", "true").lower() == "true",
                alert_on_recovery=os.getenv("SLACK_ALERT_ON_RECOVERY", "true").lower() == "true"
            )
            
        logger.info(f"AlertManager initialized with {len(self.configs)} notification channels")
        
    def _parse_headers(self, headers_str: str) -> Dict[str, str]:
        """Parse header string format 'Key1:Value1,Key2:Value2'."""
        headers = {}
        if not headers_str:
            return headers
        for pair in headers_str.split(","):
            if ":" in pair:
                key, value = pair.split(":", 1)
                headers[key.strip()] = value.strip()
        return headers
        
    def _is_cooldown_active(self, provider_name: str, config_name: str, cooldown_minutes: int) -> bool:
        """Check if alerting is in cooldown period for this provider."""
        key = f"{provider_name}:{config_name}"
        last_alert = self.last_alert_times.get(key)
        if not last_alert:
            return False
        return datetime.now() - last_alert < timedelta(minutes=cooldown_minutes)
        
    def _update_last_alert(self, provider_name: str, config_name: str):
        """Update the last alert timestamp."""
        key = f"{provider_name}:{config_name}"
        self.last_alert_times[key] = datetime.now()
        
    def _format_discord_message(self, provider: str, status: str, details: Dict[str, Any]) -> Dict:
        """Format message for Discord webhook."""
        color = 0x00FF00 if status == "healthy" else 0xFFA500 if status == "degraded" else 0xFF0000
        emoji = "✅" if status == "healthy" else "⚠️" if status == "degraded" else "❌"
        
        embed = {
            "title": f"{emoji} Provider Alert: {provider.upper()}",
            "description": f"Status changed to **{status.upper()}**",
            "color": color,
            "timestamp": datetime.now().isoformat(),
            "fields": []
        }
        
        if details.get("error_message"):
            embed["fields"].append({
                "name": "Error",
                "value": details["error_message"][:1000],  # Discord limit
                "inline": False
            })
            
        if details.get("response_time_ms"):
            embed["fields"].append({
                "name": "Response Time",
                "value": f"{details['response_time_ms']:.2f}ms",
                "inline": True
            })
            
        if details.get("consecutive_failures"):
            embed["fields"].append({
                "name": "Consecutive Failures",
                "value": str(details["consecutive_failures"]),
                "inline": True
            })
            
        if details.get("previous_status"):
            embed["fields"].append({
                "name": "Previous Status",
                "value": details["previous_status"],
                "inline": True
            })
            
        return {"embeds": [embed]}
        
    def _format_slack_message(self, provider: str, status: str, details: Dict[str, Any]) -> Dict:
        """Format message for Slack webhook."""
        color = "good" if status == "healthy" else "warning" if status == "degraded" else "danger"
        emoji = ":white_check_mark:" if status == "healthy" else ":warning:" if status == "degraded" else ":x:"
        
        fields = []
        if details.get("error_message"):
            fields.append({
                "title": "Error",
                "value": details["error_message"][:1000],
                "short": False
            })
            
        if details.get("response_time_ms"):
            fields.append({
                "title": "Response Time",
                "value": f"{details['response_time_ms']:.2f}ms",
                "short": True
            })
            
        if details.get("consecutive_failures"):
            fields.append({
                "title": "Consecutive Failures",
                "value": str(details["consecutive_failures"]),
                "short": True
            })
            
        attachment = {
            "color": color,
            "title": f"{emoji} Provider Alert: {provider.upper()}",
            "text": f"Status changed to *{status.upper()}*",
            "fields": fields,
            "footer": "Provider Status Tracker",
            "ts": int(datetime.now().timestamp())
        }
        
        return {"attachments": [attachment]}
        
    def _format_generic_message(self, provider: str, status: str, details: Dict[str, Any]) -> Dict:
        """Format generic webhook message."""
        return {
            "provider": provider,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "previous_status": details.get("previous_status"),
            "response_time_ms": details.get("response_time_ms"),
            "consecutive_failures": details.get("consecutive_failures"),
            "error_message": details.get("error_message"),
            "uptime_percent": details.get("uptime_percent"),
            "alert_type": "recovery" if status == "healthy" and details.get("previous_status") != "healthy" else "outage" if status == "down" else "degraded"
        }
        
    async def _send_webhook(self, config_name: str, config: AlertConfig, provider: str, status: str, details: Dict[str, Any]):
        """Send webhook alert."""
        if not config.webhook_url:
            return
            
        # Format message based on config type
        if config_name == "discord":
            payload = self._format_discord_message(provider, status, details)
        elif config_name == "slack":
            payload = self._format_slack_message(provider, status, details)
        else:
            payload = self._format_generic_message(provider, status, details)
            
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Content-Type": "application/json"}
                headers.update(config.webhook_headers)
                
                async with session.post(
                    config.webhook_url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status >= 400:
                        logger.error(f"Webhook alert failed for {config_name}: HTTP {response.status}")
                    else:
                        logger.info(f"Alert sent successfully via {config_name} for {provider}")
                        
        except Exception as e:
            logger.error(f"Failed to send {config_name} alert for {provider}: {e}")
            
    async def send_alert(self, provider: str, status: str, details: Dict[str, Any] = None):
        """
        Send alert for provider status change.
        
        Args:
            provider: Provider name
            status: Current status (healthy, degraded, down)
            details: Additional details like error_message, response_time_ms, etc.
        """
        if details is None:
            details = {}
            
        previous_status = details.get("previous_status", "unknown")
        
        # Determine if we should alert based on config
        for config_name, config in self.configs.items():
            should_alert = False
            
            # Check status-specific alert settings
            if status == "down" and config.alert_on_down:
                should_alert = True
            elif status == "degraded" and config.alert_on_degraded:
                should_alert = True
            elif status == "healthy" and config.alert_on_recovery and previous_status in ("down", "degraded"):
                should_alert = True
                
            if not should_alert:
                continue
                
            # Check cooldown (except for recovery alerts - always send those)
            if status != "healthy" and self._is_cooldown_active(provider, config_name, config.cooldown_minutes):
                logger.debug(f"Alert cooldown active for {provider} on {config_name}")
                continue
                
            # Send the alert
            await self._send_webhook(config_name, config, provider, status, details)
            
            # Update cooldown timestamp (except for recovery - we want immediate alerts when things break again)
            if status != "healthy":
                self._update_last_alert(provider, config_name)
                
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert configurations and recent activity."""
        return {
            "configured_channels": list(self.configs.keys()),
            "last_alerts": {k: v.isoformat() for k, v in self.last_alert_times.items()},
            "total_configs": len(self.configs)
        }
