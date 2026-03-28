#!/usr/bin/env python3
"""
Provider Database Sync Script

Reads config/providers_database.yaml (the single source of truth for providers)
and syncs the `providers:` section of config/router_config.yaml.

Also optionally exports a sortable CSV of all providers and models.

Usage:
    python scripts/sync_providers.py              # Sync router_config.yaml only
    python scripts/sync_providers.py --csv        # Sync + export providers_export.csv
    python scripts/sync_providers.py --dry-run    # Preview changes, don't write
    python scripts/sync_providers.py --enabled-only  # Only sync enabled providers
    python scripts/sync_providers.py --validate   # Validate router_models references

Workflow:
    1. Edit config/providers_database.yaml to add/remove providers or models
    2. Run this script to push changes to router_config.yaml
    3. Restart the gateway to pick up changes
"""

import argparse
import csv
import logging
import sys
import traceback
from copy import deepcopy
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("sync_providers")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DB_PATH = CONFIG_DIR / "providers_database.yaml"
ROUTER_CONFIG_PATH = CONFIG_DIR / "router_config.yaml"
CSV_EXPORT_PATH = PROJECT_ROOT / "providers_export.csv"

# Providers the router handles differently — skip their free_tier_models sync
# (they either have no model list or handle routing specially)
SKIP_FREE_MODELS_FOR = {"g4f"}  # g4f has its own dynamic model list

# Fields from router_config.yaml providers section we must preserve
# (not present in providers_database.yaml)
PRESERVE_FIELDS = {"base_url", "api_base"}


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            data, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )


def build_provider_entry(provider: dict) -> dict:
    """Convert a providers_database.yaml entry to router_config.yaml format."""
    entry = {}

    # enabled flag
    entry["enabled"] = provider.get("enabled", True)

    # env_var
    if env_var := provider.get("env_var"):
        entry["env_var"] = env_var

    # base_url / api_base
    if base_url := provider.get("base_url"):
        entry["base_url"] = base_url

    # free_tier flag
    if provider.get("free_tier"):
        entry["free_tier"] = True

    # no_api_key_required
    if provider.get("no_api_key_required"):
        entry["no_api_key_required"] = True

    # free_tier_models — extract just the IDs
    provider_id = provider["id"]
    if provider_id not in SKIP_FREE_MODELS_FOR:
        models = provider.get("free_models", [])
        if models:
            entry["free_tier_models"] = [m["id"] for m in models]

    return entry


def sync_router_config(
    db_providers: list, router_config: dict, enabled_only: bool
) -> dict:
    """Sync provider entries from database into router_config.

    Continues on per-provider errors — logs warnings but does not halt.
    """
    updated = deepcopy(router_config)
    existing_providers = updated.get("providers", {})

    changes = []
    errors = []

    for provider in db_providers:
        pid = provider.get("id", "<unknown>")

        try:
            if enabled_only and not provider.get("enabled", True):
                logger.debug(f"Skipping disabled provider: {pid}")
                continue

            new_entry = build_provider_entry(provider)

            if pid in existing_providers:
                old_entry = existing_providers[pid]
                # Preserve fields that exist in router_config but not in our DB
                for field in PRESERVE_FIELDS:
                    if field in old_entry and field not in new_entry:
                        new_entry[field] = old_entry[field]

                if old_entry != new_entry:
                    changes.append(("update", pid, old_entry, new_entry))
                    existing_providers[pid] = new_entry
            else:
                changes.append(("add", pid, None, new_entry))
                existing_providers[pid] = new_entry
        except Exception as e:
            logger.warning(
                f"Error syncing provider '{pid}': {e} — skipping, continuing with next provider"
            )
            errors.append((pid, str(e)))
            continue

    # Report providers in router_config but not in database
    db_ids = {p["id"] for p in db_providers}
    for pid in list(existing_providers.keys()):
        # Skip non-provider keys (router models, search, etc.)
        if pid.startswith("router/") or pid in (
            "router_models",
            "coding-smart",
            "coding-fast",
            "chat-smart",
            "chat-fast",
        ):
            continue
        if pid not in db_ids:
            logger.warning(
                f"Provider '{pid}' exists in router_config but not in providers_database.yaml "
                f"— leaving untouched. Add it to the database to manage it."
            )

    updated["providers"] = existing_providers
    return updated, changes, errors


def validate_router_models(router_config: dict, db_providers: list) -> list:
    """Validate that router_models candidates reference valid providers and models.

    Returns list of (model_name, candidate_index, provider, model, error_msg) tuples.
    """
    db_provider_map = {p["id"]: p for p in db_providers}
    valid_provider_ids = set(db_provider_map.keys())

    # Also include provider IDs that exist in the router_config providers section
    rc_providers = router_config.get("providers", {})
    for pid in rc_providers:
        if not pid.startswith("router/") and pid not in (
            "router_models",
            "coding-smart",
            "coding-fast",
            "chat-smart",
            "chat-fast",
        ):
            valid_provider_ids.add(pid)

    issues = []
    router_models = router_config.get("router_models", {})

    for model_name, model_def in router_models.items():
        if not isinstance(model_def, dict):
            continue
        candidates = model_def.get("candidates", [])
        for idx, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                continue
            provider = candidate.get("provider", "")
            model = candidate.get("model", "")

            if provider not in valid_provider_ids:
                issues.append(
                    (
                        model_name,
                        idx,
                        provider,
                        model,
                        f"Provider '{provider}' not found in providers config",
                    )
                )
            elif provider in db_provider_map:
                db_models = [
                    m["id"] for m in db_provider_map[provider].get("free_models", [])
                ]
                if model not in db_models:
                    issues.append(
                        (
                            model_name,
                            idx,
                            provider,
                            model,
                            f"Model '{model}' not in provider '{provider}' free_models",
                        )
                    )

    return issues


def export_csv(db_providers: list, path: Path) -> None:
    """Export all providers and models to a sortable CSV."""
    rows = []
    for provider in db_providers:
        pid = provider["id"]
        pname = provider.get("name", pid)
        enabled = provider.get("enabled", True)
        free_tier = provider.get("free_tier", True)
        no_key = provider.get("no_api_key_required", False)
        provider_caps = ",".join(provider.get("capabilities", []))
        rpm = provider.get("rate_limits", {}).get("rpm", "")
        daily = provider.get("rate_limits", {}).get("daily", "")
        signup = provider.get("signup_url", "")
        notes = provider.get("notes", "")
        last_verified = provider.get("last_verified", "")

        models = provider.get("free_models", [])
        if not models:
            rows.append(
                {
                    "provider_id": pid,
                    "provider_name": pname,
                    "enabled": enabled,
                    "free_tier": free_tier,
                    "no_api_key": no_key,
                    "provider_rpm": rpm,
                    "provider_daily_limit": daily,
                    "model_id": "",
                    "model_context": "",
                    "model_tps": "",
                    "model_capabilities": "",
                    "model_notes": "",
                    "signup_url": signup,
                    "provider_notes": notes,
                    "last_verified": last_verified,
                }
            )
        else:
            for model in models:
                model_caps = ",".join(model.get("capabilities", []))
                rows.append(
                    {
                        "provider_id": pid,
                        "provider_name": pname,
                        "enabled": enabled,
                        "free_tier": free_tier,
                        "no_api_key": no_key,
                        "provider_rpm": rpm,
                        "provider_daily_limit": daily,
                        "model_id": model["id"],
                        "model_context": model.get("context", ""),
                        "model_tps": model.get("tps", ""),
                        "model_capabilities": model_caps,
                        "model_notes": model.get("notes", ""),
                        "signup_url": signup,
                        "provider_notes": notes,
                        "last_verified": last_verified,
                    }
                )

    fieldnames = [
        "provider_id",
        "provider_name",
        "enabled",
        "free_tier",
        "no_api_key",
        "provider_rpm",
        "provider_daily_limit",
        "model_id",
        "model_context",
        "model_tps",
        "model_capabilities",
        "model_notes",
        "signup_url",
        "provider_notes",
        "last_verified",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Exported {len(rows)} rows to {path}")


def print_changes(changes: list) -> None:
    if not changes:
        logger.info("No changes detected — router_config.yaml is already up to date.")
        return

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Changes to apply ({len(changes)} providers):")
    logger.info(f"{'=' * 60}")
    for action, pid, old, new in changes:
        if action == "add":
            model_count = len(new.get("free_tier_models", []))
            logger.info(f"  + ADD    {pid} ({model_count} models)")
        elif action == "update":
            old_models = set(old.get("free_tier_models", []))
            new_models = set(new.get("free_tier_models", []))
            added_m = new_models - old_models
            removed_m = old_models - new_models
            if added_m:
                logger.info(f"  ~ UPDATE {pid}: +{len(added_m)} models added")
                for m in sorted(added_m):
                    logger.info(f"      + {m}")
            if removed_m:
                logger.info(f"  ~ UPDATE {pid}: -{len(removed_m)} models removed")
                for m in sorted(removed_m):
                    logger.info(f"      - {m}")
            if not added_m and not removed_m:
                logger.info(f"  ~ UPDATE {pid}: metadata changed")


def main():
    parser = argparse.ArgumentParser(
        description="Sync providers_database.yaml → router_config.yaml"
    )
    parser.add_argument(
        "--csv", action="store_true", help="Also export providers_export.csv"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing files",
    )
    parser.add_argument(
        "--enabled-only",
        action="store_true",
        help="Only sync enabled providers (skip disabled)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate router_models candidates reference valid providers/models",
    )
    args = parser.parse_args()

    # Load database
    logger.info(f"Loading provider database: {DB_PATH}")
    db = load_yaml(DB_PATH)
    db_providers = db.get("providers", [])
    logger.info(
        f"  Found {len(db_providers)} providers, "
        f"{sum(1 for p in db_providers if p.get('enabled', True))} enabled"
    )

    # Count total models
    total_models = sum(len(p.get("free_models", [])) for p in db_providers)
    enabled_models = sum(
        len(p.get("free_models", [])) for p in db_providers if p.get("enabled", True)
    )
    logger.info(f"  {total_models} total models, {enabled_models} in enabled providers")

    # Load router config
    logger.info(f"Loading router config: {ROUTER_CONFIG_PATH}")
    router_config = load_yaml(ROUTER_CONFIG_PATH)

    # Sync
    updated_config, changes, errors = sync_router_config(
        db_providers, router_config, args.enabled_only
    )
    print_changes(changes)

    if errors:
        logger.warning(
            f"\n{len(errors)} provider(s) had errors during sync (see above)"
        )

    if args.dry_run:
        logger.info("\nDry run — no files written.")
    else:
        if changes:
            save_yaml(ROUTER_CONFIG_PATH, updated_config)
            logger.info(
                f"\nWrote updated router_config.yaml ({len(changes)} providers changed)"
            )
        else:
            logger.info("No changes to write.")

    # Validate router_models
    if args.validate:
        logger.info(f"\n{'=' * 60}")
        logger.info("ROUTER MODELS VALIDATION")
        logger.info(f"{'=' * 60}")
        issues = validate_router_models(updated_config, db_providers)
        if issues:
            logger.warning(f"Found {len(issues)} issue(s) in router_models:")
            for model_name, idx, provider, model, error_msg in issues:
                logger.warning(
                    f"  [{model_name}] candidate #{idx}: {provider}/{model} — {error_msg}"
                )
        else:
            logger.info(
                "All router_models candidates reference valid providers and models."
            )

    # CSV export
    if args.csv:
        logger.info(f"\nExporting CSV to {CSV_EXPORT_PATH}")
        export_csv(db_providers, CSV_EXPORT_PATH)

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("PROVIDER SUMMARY")
    logger.info(f"{'=' * 60}")
    for p in db_providers:
        status = "ENABLED " if p.get("enabled", True) else "disabled"
        model_count = len(p.get("free_models", []))
        logger.info(
            f"  [{status}] {p['id']:20s} {model_count:3d} models  {p.get('name', '')}"
        )

    if errors:
        logger.warning(
            f"\nCompleted with {len(errors)} error(s). Check warnings above."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
