#!/usr/bin/env python3
"""Lightweight benchmark runner for free LLM providers.

Sends a standardized coding prompt to each enabled provider/model,
records TTFT (time-to-first-token), TPS, latency, and success/failure
into the telemetry SQLite DB.

Designed to run:
  - As a GitHub Actions scheduled job (every 6 hours)
  - On VPS1 nightly via cron
  - Manually: python scripts/benchmark_providers.py

Outputs:
  - Telemetry written to /tmp/llm_proxy_telemetry.db (read by score_engine.py)
  - Summary CSV: config/benchmark_results.csv
  - JSON report: config/benchmark_report.json

Usage:
  python scripts/benchmark_providers.py                    # All enabled providers
  python scripts/benchmark_providers.py --provider groq   # Single provider
  python scripts/benchmark_providers.py --max-models 5    # Limit per provider
  python scripts/benchmark_providers.py --dry-run         # Show what would run
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("benchmark")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
ROUTER_CONFIG = CONFIG_DIR / "router_config.yaml"
PROVIDERS_DB = CONFIG_DIR / "providers_database.yaml"
BENCHMARK_CSV = CONFIG_DIR / "benchmark_results.csv"
BENCHMARK_JSON = CONFIG_DIR / "benchmark_report.json"
TELEMETRY_DB = Path("/tmp/llm_proxy_telemetry.db")

# Standardized benchmark prompts (short enough to complete fast, tests coding ability)
BENCHMARK_PROMPTS = [
    {
        "id": "hello_world",
        "messages": [{"role": "user", "content": "Write a Python function that returns 'hello world'. Just the function, no explanation."}],
        "max_tokens": 80,
        "expected_min_tokens": 5,
    },
    {
        "id": "fizzbuzz",
        "messages": [{"role": "user", "content": "Write a Python one-liner that prints FizzBuzz for 1-20. Just the code."}],
        "max_tokens": 120,
        "expected_min_tokens": 10,
    },
    {
        "id": "json_parse",
        "messages": [{"role": "user", "content": "Write a Python function that safely parses JSON and returns None on error. Just the function."}],
        "max_tokens": 150,
        "expected_min_tokens": 15,
    },
]


def load_providers() -> dict:
    """Load enabled providers with API keys from router_config.yaml."""
    if not ROUTER_CONFIG.exists():
        logger.error(f"Config not found: {ROUTER_CONFIG}")
        return {}
    with open(ROUTER_CONFIG) as f:
        config = yaml.safe_load(f) or {}
    return {name: cfg for name, cfg in config.get("providers", {}).items()
            if isinstance(cfg, dict) and cfg.get("enabled", False)}


def get_api_key(provider_name: str, cfg: dict) -> Optional[str]:
    """Resolve API key from environment."""
    env_var = cfg.get("env_var", "")
    if env_var:
        key = os.getenv(env_var) or os.getenv(f"{env_var}_1")
        if key:
            return key
    # Common fallbacks
    for suffix in ["", "_1", "_2"]:
        key = os.getenv(f"{provider_name.upper()}_API_KEY{suffix}")
        if key:
            return key
    return None


async def benchmark_model(
    provider: str,
    model_id: str,
    base_url: str,
    api_key: str,
    prompt: dict,
    timeout: float = 30.0,
) -> dict:
    """Run a single benchmark prompt against a provider/model."""
    result = {
        "provider": provider,
        "model": model_id,
        "prompt_id": prompt["id"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "success": False,
        "ttft_ms": None,
        "total_ms": None,
        "output_tokens": None,
        "tps": None,
        "error": None,
    }

    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model_id,
        "messages": prompt["messages"],
        "max_tokens": prompt["max_tokens"],
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    t0 = time.monotonic()
    ttft = None
    token_count = 0

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                if resp.status_code not in (200, 206):
                    body = await resp.aread()
                    result["error"] = f"HTTP {resp.status_code}: {body[:200].decode(errors='replace')}"
                    return result

                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            if ttft is None:
                                ttft = (time.monotonic() - t0) * 1000
                            token_count += len(content.split())  # rough token estimate
                    except (json.JSONDecodeError, IndexError, KeyError):
                        pass

        total_ms = (time.monotonic() - t0) * 1000
        if token_count < prompt.get("expected_min_tokens", 1):
            result["error"] = f"Too few tokens: {token_count}"
            return result

        result["success"] = True
        result["ttft_ms"] = round(ttft, 1) if ttft else None
        result["total_ms"] = round(total_ms, 1)
        result["output_tokens"] = token_count
        if token_count and total_ms > 0:
            result["tps"] = round((token_count / total_ms) * 1000, 1)

    except httpx.TimeoutException:
        result["error"] = f"Timeout after {timeout}s"
    except Exception as e:
        result["error"] = str(e)[:200]

    return result


def record_to_telemetry(result: dict) -> None:
    """Write benchmark result to telemetry SQLite."""
    try:
        import sqlite3
        conn = sqlite3.connect(str(TELEMETRY_DB))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT, model TEXT,
                success INTEGER, error_reason TEXT,
                response_time_ms INTEGER,
                time_to_first_token_ms INTEGER,
                tokens_per_second REAL,
                input_tokens INTEGER, output_tokens INTEGER,
                cost_estimate_usd REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tps_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT, model TEXT,
                tps REAL, window_minutes INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            INSERT INTO api_calls
            (provider, model, success, error_reason, response_time_ms,
             time_to_first_token_ms, tokens_per_second, output_tokens)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result["provider"], result["model"],
            1 if result["success"] else 0,
            result.get("error"),
            int(result["total_ms"]) if result["total_ms"] else None,
            int(result["ttft_ms"]) if result["ttft_ms"] else None,
            result.get("tps"),
            result.get("output_tokens"),
        ))
        if result.get("tps"):
            cursor.execute("""
                INSERT INTO tps_metrics (provider, model, tps, window_minutes)
                VALUES (?, ?, ?, 1)
            """, (result["provider"], result["model"], result["tps"]))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.debug(f"Telemetry write failed (non-fatal): {e}")


async def run_benchmarks(
    target_provider: Optional[str] = None,
    max_models: int = 3,
    dry_run: bool = False,
) -> list[dict]:
    """Run benchmarks across all (or one) provider."""
    providers = load_providers()
    if not providers:
        logger.error("No enabled providers found in router_config.yaml")
        return []

    if target_provider:
        providers = {k: v for k, v in providers.items() if k == target_provider}
        if not providers:
            logger.error(f"Provider '{target_provider}' not found or not enabled")
            return []

    all_results = []
    prompt = BENCHMARK_PROMPTS[0]  # Use hello_world for speed

    for provider_name, cfg in providers.items():
        base_url = cfg.get("base_url") or cfg.get("api_base", "")
        if not base_url:
            logger.debug(f"Skipping {provider_name}: no base_url")
            continue

        api_key = get_api_key(provider_name, cfg)
        if not api_key:
            logger.debug(f"Skipping {provider_name}: no API key in environment")
            continue

        models = cfg.get("free_tier_models", [])[:max_models]
        if not models:
            logger.debug(f"Skipping {provider_name}: no free_tier_models")
            continue

        logger.info(f"Benchmarking {provider_name} ({len(models)} models)...")

        for model_id in models:
            if dry_run:
                logger.info(f"  [DRY RUN] Would test: {provider_name}/{model_id}")
                continue

            result = await benchmark_model(provider_name, model_id, base_url, api_key, prompt)
            all_results.append(result)

            status = "OK" if result["success"] else f"FAIL: {result.get('error', '?')[:50]}"
            tps_str = f" tps={result['tps']}" if result.get('tps') else ""
            ttft_str = f" ttft={result['ttft_ms']}ms" if result.get('ttft_ms') else ""
            logger.info(f"  {model_id}: {status}{tps_str}{ttft_str}")

            if result["success"]:
                record_to_telemetry(result)

            # Small delay between calls to avoid rate limiting
            await asyncio.sleep(1.0)

    return all_results


def save_results(results: list[dict]) -> None:
    """Save results to CSV and JSON."""
    if not results:
        return

    # CSV
    fields = ["timestamp", "provider", "model", "prompt_id", "success",
              "ttft_ms", "total_ms", "output_tokens", "tps", "error"]
    with open(BENCHMARK_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"CSV written: {BENCHMARK_CSV}")

    # JSON report with summary
    successful = [r for r in results if r["success"]]
    by_provider: dict = {}
    for r in successful:
        p = r["provider"]
        by_provider.setdefault(p, [])
        if r.get("tps"):
            by_provider[p].append(r["tps"])

    summary = {}
    for p, tps_list in by_provider.items():
        if tps_list:
            summary[p] = {
                "avg_tps": round(sum(tps_list) / len(tps_list), 1),
                "max_tps": round(max(tps_list), 1),
                "min_tps": round(min(tps_list), 1),
                "models_tested": len(tps_list),
            }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_tests": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "provider_summary": summary,
        "ranked_providers": sorted(summary.keys(),
                                    key=lambda p: -summary[p]["avg_tps"]),
    }

    with open(BENCHMARK_JSON, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"JSON report written: {BENCHMARK_JSON}")
    logger.info(f"Provider TPS ranking: {report['ranked_providers']}")


async def main():
    parser = argparse.ArgumentParser(description="Benchmark free LLM providers")
    parser.add_argument("--provider", help="Only benchmark this provider")
    parser.add_argument("--max-models", type=int, default=3,
                        help="Max models per provider to test (default: 3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run, don't make API calls")
    args = parser.parse_args()

    logger.info(f"Starting benchmark run {'(DRY RUN)' if args.dry_run else ''}")
    results = await run_benchmarks(
        target_provider=args.provider,
        max_models=args.max_models,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        save_results(results)
        successful = sum(1 for r in results if r["success"])
        logger.info(f"Done: {successful}/{len(results)} tests passed")
    else:
        logger.info("Dry run complete")


if __name__ == "__main__":
    asyncio.run(main())
