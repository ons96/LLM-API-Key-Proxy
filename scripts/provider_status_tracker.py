#!/usr/bin/env python3
"""LLM API Provider Status Tracker.

Reads provider configs and pings their health endpoints.
Outputs status report to terminal or JSON.

Usage:
    python scripts/provider_status_tracker.py
    python scripts/provider_status_tracker.py --json
    python scripts/provider_status_tracker.py --daemon --interval 300
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"

# Providers and their health-check endpoints
# Uses /v1/models as a lightweight probe
PROVIDER_PROBES: Dict[str, str] = {
    "groq": "https://api.groq.com/openai/v1/models",
    "gemini": "https://generativelanguage.googleapis.com/v1/models",
    "together": "https://api.together.xyz/v1/models",
    "deepinfra": "https://api.deepinfra.com/v1/models",
    "cerebras": "https://api.cerebras.ai/v1/models",
    "freetheai": "https://api.freetheai.xyz/v1/models",
    "openai": "https://api.openai.com/v1/models",
    "anthropic": "https://api.anthropic.com/v1/models",
    "xai": "https://api.x.ai/v1/models",
    "mistral": "https://api.mistral.ai/v1/models",
    "openrouter": "https://openrouter.ai/api/v1/models",
    "alibaba": "https://dashscope.aliyuncs.com/compatible-mode/v1/models",
}

# Additional endpoints beyond model list probes
HEALTH_ENDPOINTS: Dict[str, str] = {
    "freetheai": "https://status.freetheai.xyz/api/v2/status",
}


@dataclass
class ProviderStatus:
    name: str
    reachable: bool
    status_code: Optional[int] = None
    response_time_ms: Optional[float] = None
    error: Optional[str] = None
    models_available: Optional[int] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


async def check_provider(
    client: httpx.AsyncClient,
    name: str,
    url: str,
    timeout: float = 10.0,
) -> ProviderStatus:
    start = time.monotonic()
    try:
        resp = await client.get(url, timeout=timeout)
        elapsed = (time.monotonic() - start) * 1000
        if resp.status_code == 200:
            data = resp.json()
            models = (
                data
                if isinstance(data, list)
                else data.get("data", data.get("models", []))
            )
            count = len(models) if isinstance(models, list) else 0
            return ProviderStatus(
                name=name,
                reachable=True,
                status_code=resp.status_code,
                response_time_ms=round(elapsed, 1),
                models_available=count,
            )
        return ProviderStatus(
            name=name,
            reachable=False,
            status_code=resp.status_code,
            response_time_ms=round(elapsed, 1),
            error=f"HTTP {resp.status_code}",
        )
    except httpx.TimeoutException:
        elapsed = (time.monotonic() - start) * 1000
        return ProviderStatus(
            name=name,
            reachable=False,
            error="timeout",
            response_time_ms=round(elapsed, 1),
        )
    except Exception as e:
        elapsed = (time.monotonic() - start) * 1000
        return ProviderStatus(
            name=name,
            reachable=False,
            error=str(e)[:80],
            response_time_ms=round(elapsed, 1),
        )


async def run_checks(
    providers: Dict[str, str], timeout: float = 10.0
) -> List[ProviderStatus]:
    async with httpx.AsyncClient(verify=False, timeout=timeout) as client:
        tasks = [
            check_provider(client, name, url, timeout)
            for name, url in providers.items()
        ]
        results = await asyncio.gather(*tasks)
        return results


def print_report(results: List[ProviderStatus]):
    up = sum(1 for r in results if r.reachable)
    total = len(results)
    print(
        f"\nLLM Provider Status Report — {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )
    print(f"{'=' * 60}")
    print(f"  {up}/{total} providers reachable\n")
    for r in sorted(results, key=lambda x: (not x.reachable, x.name)):
        if r.reachable:
            print(
                f"  ✓ {r.name:20s} {r.response_time_ms:>7.1f}ms  ({r.models_available} models)"
            )
        else:
            print(f"  ✗ {r.name:20s} {'--':>7s}  {r.error or 'unreachable'}")
    print()


def print_json(results: List[ProviderStatus]):
    data = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "total": len(results),
        "reachable": sum(1 for r in results if r.reachable),
        "providers": {
            r.name: {
                "reachable": r.reachable,
                "status_code": r.status_code,
                "response_time_ms": r.response_time_ms,
                "error": r.error,
                "models_available": r.models_available,
            }
            for r in results
        },
    }
    print(json.dumps(data, indent=2))


async def daemon_loop(interval: int, json_output: bool):
    print(f"Starting provider status daemon (interval={interval}s)")
    while True:
        results = await run_checks(PROVIDER_PROBES)
        if json_output:
            print_json(results)
        else:
            print_report(results)
        await asyncio.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="LLM API Provider Status Tracker")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--daemon", action="store_true", help="Run in continuous loop")
    parser.add_argument(
        "--interval", type=int, default=300, help="Daemon check interval (seconds)"
    )
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="Request timeout per provider"
    )
    args = parser.parse_args()

    if args.daemon:
        asyncio.run(daemon_loop(args.interval, args.json))
    else:
        results = asyncio.run(run_checks(PROVIDER_PROBES, args.timeout))
        if args.json:
            print_json(results)
        else:
            print_report(results)
        sys.exit(0 if any(r.reachable for r in results) else 1)


if __name__ == "__main__":
    main()
