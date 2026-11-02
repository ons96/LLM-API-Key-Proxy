#!/usr/bin/env python3
"""Prune free_models entries already blocked by config/dead_providers.yaml.

The catalog drifted: 160 (provider, model) pairs stayed advertised after the
policy blocked them. Chains are filtered at routing time, so this is a
metadata fix, not a routing change.

Usage:
    python3 scripts/prune_blocklisted_catalog.py            # report only
    python3 scripts/prune_blocklisted_catalog.py --apply    # write catalog
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import chain_policy  # noqa: E402

CATALOG = REPO_ROOT / "config" / "providers_database.yaml"


def offenders(database: dict, policy: dict) -> tuple[list, list]:
    """Split catalog entries into (model-gone, blocked-for-other-reasons).

    Only the first group is safe to prune: the second group is blocked by a
    quota, missing key, IP block or credit gate, and those models still exist.
    """
    gone, limited = [], []
    for provider in database["providers"]:
        pid = provider.get("id")
        for model in provider.get("free_models") or []:
            mid = model.get("id")
            reason = chain_policy.blocked_reason(pid, mid, policy)
            if not reason:
                continue
            scope = chain_policy.blocked_scope(pid, mid, policy)
            if scope == "model" and chain_policy.is_model_gone_reason(reason):
                gone.append((pid, mid, reason))
            else:
                limited.append((pid, mid, reason))
    return gone, limited
def prune_text(text: str, doomed: dict[str, set[str]], date: str) -> str:
    """Drop doomed model items and stamp last_verified on touched providers."""
    lines = text.splitlines()
    out: list[str] = []
    provider: str | None = None
    touched: set[str] = set()
    dropping = False

    def is_model_item(line: str) -> bool:
        return line.startswith("  - id: ")

    def is_provider_key(line: str) -> bool:
        return (
            line.startswith("  ")
            and not line.startswith("   ")
            and not line.startswith("  - ")
            and ":" in line
        )

    for line in lines:
        if line.startswith("- id: "):
            provider = line[len("- id: ") :].strip()
            dropping = False
            out.append(line)
            continue

        if dropping:
            if is_model_item(line) or is_provider_key(line) or line.startswith("- "):
                dropping = False
            else:
                continue

        if provider in doomed and is_model_item(line):
            model_id = line[len("  - id: ") :].strip()
            if model_id in doomed[provider]:
                dropping = True
                touched.add(provider)
                continue

        out.append(line)

    stamped: list[str] = []
    provider = None
    for line in out:
        if line.startswith("- id: "):
            provider = line[len("- id: ") :].strip()
            stamped.append(line)
            continue
        if provider in touched and line.strip() == "free_models:":
            stamped.append(f"  last_verified: '{date}'")
            touched.discard(provider)
        stamped.append(line)

    result = "\n".join(stamped)
    return result + "\n" if text.endswith("\n") else result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write the catalog")
    ap.add_argument("--date", default=None, help="last_verified stamp")
    args = ap.parse_args()

    policy = chain_policy.load_policy()
    text = CATALOG.read_text()
    database = yaml.safe_load(text)
    found, limited = offenders(database, policy)
    by_provider: dict[str, set[str]] = {}
    for pid, mid, _ in found:
        by_provider.setdefault(pid, set()).add(mid)

    print(f"model-gone offenders: {len(found)} across {len(by_provider)} providers")
    for pid in sorted(by_provider):
        print(f"  {pid}: {len(by_provider[pid])}")
    if limited:
        print(f"kept: {len(limited)} blocked for quota/key/IP/tier, not pruned")
        for pid in sorted({p for p, _, _ in limited}):
            print(f"  {pid}: {sum(1 for x in limited if x[0] == pid)}")
    if not found:
        return 0
    if not args.apply:
        print("dry run; pass --apply to write")
        return 0
    date = args.date or datetime.date.today().isoformat()
    CATALOG.write_text(prune_text(text, by_provider, date))
    after = yaml.safe_load(CATALOG.read_text())
    remaining, _ = offenders(after, policy)
    if remaining:
        print(f"ERROR: {len(remaining)} offenders remain", file=sys.stderr)
        return 1
    before_pairs = {
        (p["id"], m["id"]) for p in database["providers"] for m in (p.get("free_models") or [])
    }
    after_pairs = {
        (p["id"], m["id"]) for p in after["providers"] for m in (p.get("free_models") or [])
    }
    gone = before_pairs - after_pairs
    added = after_pairs - before_pairs
    expected = {(pid, mid) for pid, mid, _ in found}
    if added or gone != expected:
        print(f"ERROR: catalog delta added={sorted(added)[:5]}", file=sys.stderr)
        return 1
    print(f"pruned {len(gone)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
