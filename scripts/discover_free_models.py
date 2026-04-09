#!/usr/bin/env python3
"""
Discover new free LLM providers from models.dev

Usage:
    python scripts/discover_free_models.py              # Report only
    python scripts/discover_free_models.py --update      # Update router_config.yaml
    python scripts/discover_free_models.py --auto-enable  # Update and auto-enable new providers
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rotator_library.models_dev_discovery import run_discovery

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def print_report(discoveries: list) -> None:
    new_providers = [d for d in discoveries if d["is_new_provider"]]
    existing_with_new = [
        d for d in discoveries if not d["is_new_provider"] and d["new_models"]
    ]

    total_new = sum(len(d["new_models"]) for d in discoveries)

    print(f"\n{'=' * 70}")
    print(f"  models.dev Free LLM Discovery Report")
    print(f"{'=' * 70}")
    print(f"  New providers found:          {len(new_providers)}")
    print(f"  Existing providers w/ new:    {len(existing_with_new)}")
    print(f"  Total new free models:        {total_new}")
    print(f"{'=' * 70}\n")

    if new_providers:
        print("NEW PROVIDERS (not yet in router_config.yaml):\n")
        for d in new_providers:
            print(f"  {d['provider_name']} ({d['provider_id']})")
            print(f"    API: {d['api_url']}")
            print(f"    Env: {', '.join(d['env_vars'])}")
            print(f"    Doc: {d['doc_url']}")
            print(f"    Free models ({len(d['all_free_models'])}):")
            for m in d["all_free_models"][:8]:
                flags = []
                if m["tool_call"]:
                    flags.append("tools")
                if m["reasoning"]:
                    flags.append("reasoning")
                ctx = m["context_window"]
                ctx_str = f"{ctx // 1000}k" if ctx else "?"
                print(f"      - {m['id']} [{ctx_str}] {' '.join(flags)}")
            if len(d["all_free_models"]) > 8:
                print(f"      ... and {len(d['all_free_models']) - 8} more")
            print()

    if existing_with_new:
        print("\nEXISTING PROVIDERS WITH NEW FREE MODELS:\n")
        for d in existing_with_new:
            print(f"  {d['provider_name']} ({d['provider_id']})")
            print(f"    New models ({len(d['new_models'])}):")
            for m in d["new_models"][:5]:
                flags = []
                if m["tool_call"]:
                    flags.append("tools")
                if m["reasoning"]:
                    flags.append("reasoning")
                ctx = m["context_window"]
                ctx_str = f"{ctx // 1000}k" if ctx else "?"
                print(f"      - {m['id']} [{ctx_str}] {' '.join(flags)}")
            if len(d["new_models"]) > 5:
                print(f"      ... and {len(d['new_models']) - 5} more")
            print()


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Discover free LLM providers from models.dev"
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update router_config.yaml with discoveries",
    )
    parser.add_argument(
        "--auto-enable",
        action="store_true",
        help="Auto-enable newly discovered providers",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Skip saving discovery cache"
    )
    args = parser.parse_args()

    discoveries, changes = await run_discovery(
        update_config=args.update,
        auto_enable=args.auto_enable,
        save_cache=not args.no_cache,
    )

    print_report(discoveries)

    if changes:
        print("\nCONFIG CHANGES APPLIED:")
        for change in changes:
            print(f"  - {change}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
