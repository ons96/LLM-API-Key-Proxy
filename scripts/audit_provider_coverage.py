#!/usr/bin/env python3
"""Compare free providers in the manager DB with the gateway router config.

This is a read-only audit. It never calls a provider, never reads an API key
value, and never edits router_config.yaml. It answers three separate questions:

1. How many providers does the gateway configure?
2. How many free providers exist in llm-provider-manager's DB?
3. Of the missing ones, which are complete enough to add?

"Complete enough" means: a base URL, at least one model row, and a key source
(env var, secret-file path, or no_api_key_required). A missing provider can be
free and still not be routable because its DB row has no URL or no models.

Usage:
    python scripts/audit_provider_coverage.py
    python scripts/audit_provider_coverage.py --format markdown
    python scripts/audit_provider_coverage.py --format json
    python scripts/audit_provider_coverage.py --fail-on-gap

This repo has no pyproject.toml, so uv needs an ephemeral environment:
    uv run --no-project --with pyyaml python scripts/audit_provider_coverage.py
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

LOCAL_HOSTS = {
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "::1",
    "host.docker.internal",
    "40.233.101.233",
    "100.71.95.75",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_db() -> Path:
    return Path(__file__).resolve().parents[2] / "llm-provider-manager" / "llm_providers.db"


def _normalize_name(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _normalize_url(value: str) -> str:
    raw = str(value or "").strip().rstrip("/")
    if not raw:
        return ""
    if "://" in raw:
        parsed = urlparse(raw)
        return f"{parsed.netloc.lower()}{parsed.path.rstrip('/')}"
    return raw.lower()


def _is_local_url(value: str) -> bool:
    raw = str(value or "").strip()
    if not raw:
        return False
    parsed = urlparse(raw if "://" in raw else f"http://{raw}")
    host = (parsed.hostname or "").lower()
    return host in LOCAL_HOSTS or host.startswith("127.") or host.endswith(".local")


def _key_source(env_var: str, no_api_key_required: bool) -> str:
    if no_api_key_required:
        return "no-key"
    env = str(env_var or "").strip()
    if not env:
        return "missing"
    if env.startswith("~") or env.startswith("/"):
        return "secret-file"
    return "env"


def load_db(db_path: Path) -> list[dict[str, Any]]:
    if not db_path.exists():
        raise FileNotFoundError(f"provider DB not found: {db_path}")
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        counts = {
            int(row["provider_id"]): {
                "models": int(row["models"] or 0),
                "free_models": int(row["free_models"] or 0),
            }
            for row in con.execute(
                """
                SELECT provider_id,
                       COUNT(*) AS models,
                       SUM(CASE WHEN free_tier = 1 THEN 1 ELSE 0 END) AS free_models
                FROM models
                WHERE provider_id IS NOT NULL
                GROUP BY provider_id
                """
            )
        }
        rows = []
        for row in con.execute("SELECT * FROM providers ORDER BY key_name"):
            # The DB contains orphan rows without an id; they are not providers
            # and cannot be referenced by a model row.
            if row["id"] is None or not row["key_name"]:
                continue
            provider_id = int(row["id"])
            model_counts = counts.get(provider_id, {"models": 0, "free_models": 0})
            free_tier = bool(row["free_tier"])
            no_key = bool(row["no_api_key_required"])
            free_models = model_counts["free_models"]
            rows.append(
                {
                    "provider_id": provider_id,
                    "name": row["key_name"],
                    "base_url": row["base_url"] or "",
                    "env_var": row["env_var"] or "",
                    "free_tier": free_tier,
                    "no_api_key_required": no_key,
                    "models": model_counts["models"],
                    "free_models": free_models,
                    "rpm": int(row["rate_limit_rpm"] or 0),
                    "daily_tokens": int(row["rate_limit_daily_tokens"] or 0),
                    "free_signal": free_tier or no_key or free_models > 0,
                }
            )
        return rows
    finally:
        con.close()


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"router config not found: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"router config is not a mapping: {config_path}")
    return data


def load_catalog(catalog_path: Path) -> dict[str, Any] | None:
    if not catalog_path.exists():
        return None
    import gzip

    try:
        if catalog_path.suffix == ".gz":
            with gzip.open(catalog_path, "rt", encoding="utf-8") as handle:
                data = json.load(handle)
        else:
            data = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"error": str(exc)}
    return data if isinstance(data, dict) else None


def _chain_providers(config: dict[str, Any]) -> list[str]:
    found: set[str] = set()
    for virtual in (config.get("router_models") or {}).values():
        if not isinstance(virtual, dict):
            continue
        chain = virtual.get("fallback_chain") or virtual.get("candidates") or []
        for candidate in chain:
            if isinstance(candidate, dict) and candidate.get("provider"):
                found.add(str(candidate["provider"]))
    return sorted(found)


def build_report(config_path: Path, db_path: Path, catalog_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    configured: dict[str, Any] = config.get("providers") or {}
    configured_by_name = {_normalize_name(name): name for name in configured}
    configured_by_url: dict[str, list[str]] = defaultdict(list)
    for name, details in configured.items():
        if not isinstance(details, dict):
            continue
        normalized = _normalize_url(details.get("base_url") or "")
        if normalized:
            configured_by_url[normalized].append(name)

    db_rows = load_db(db_path)
    free_rows = [row for row in db_rows if row["free_signal"]]
    free_with_limit = sum(
        1 for row in free_rows if row["rpm"] or row["daily_tokens"]
    )
    missing = [row for row in free_rows if _normalize_name(row["name"]) not in configured_by_name]

    counts = {
        "usable": 0,
        "duplicate_endpoint": 0,
        "no_base_url": 0,
        "no_models": 0,
        "no_key_source": 0,
        "local_or_self": 0,
    }
    usable: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    incomplete: dict[str, list[str]] = defaultdict(list)

    for row in missing:
        name = row["name"]
        url = row["base_url"]
        key_source = _key_source(row["env_var"], row["no_api_key_required"])
        normalized_url = _normalize_url(url)
        duplicate_of = configured_by_url.get(normalized_url, []) if normalized_url else []

        if not url:
            counts["no_base_url"] += 1
            incomplete["no_base_url"].append(name)
            continue
        if _is_local_url(url):
            counts["local_or_self"] += 1
            incomplete["local_or_self"].append(name)
            continue
        if row["models"] <= 0:
            counts["no_models"] += 1
            incomplete["no_models"].append(name)
            continue
        if key_source == "missing":
            counts["no_key_source"] += 1
            incomplete["no_key_source"].append(name)
            continue
        if duplicate_of:
            counts["duplicate_endpoint"] += 1
            duplicates.append(
                {
                    "name": name,
                    "free_models": row["free_models"],
                    "url": url,
                    "duplicate_of": sorted(duplicate_of),
                }
            )
            continue

        counts["usable"] += 1
        usable.append(
            {
                "name": name,
                "free_models": row["free_models"],
                "total_models": row["models"],
                "env_var": row["env_var"],
                "key_source": key_source,
                "rpm": row["rpm"],
                "daily_tokens": row["daily_tokens"],
                "url": url,
            }
        )

    catalog = load_catalog(catalog_path)
    catalog_providers = 0
    catalog_free = 0
    catalog_generated_at = ""
    if catalog and "error" not in catalog:
        providers = catalog.get("providers") or []
        catalog_providers = len(providers)
        catalog_free = sum(
            1 for provider in providers if int(provider.get("quota_tier", 4)) < 4
        )
        catalog_generated_at = str(catalog.get("generated_at") or "")

    return {
        "router_config": str(config_path),
        "provider_db": str(db_path),
        "configured_providers": len(configured),
        "configured_enabled": sum(
            1 for details in configured.values() if isinstance(details, dict) and details.get("enabled")
        ),
        "fallback_chain_providers": _chain_providers(config),
        "db_providers": len(db_rows),
        "db_free_providers": len(free_rows),
        "free_with_numeric_limit": free_with_limit,
        "free_without_numeric_limit": len(free_rows) - free_with_limit,
        "missing_free_providers": len(missing),
        "usable_with_numeric_limit": sum(
            1 for item in usable if item["rpm"] or item["daily_tokens"]
        ),
        "usable_without_numeric_limit": sum(
            1 for item in usable if not item["rpm"] and not item["daily_tokens"]
        ),
        "counts": counts,
        "usable": sorted(usable, key=lambda item: item["name"]),
        "duplicates": sorted(duplicates, key=lambda item: item["name"]),
        "incomplete": {key: sorted(values) for key, values in incomplete.items()},
        "catalog": {
            "providers": catalog_providers,
            "free": catalog_free,
            "generated_at": catalog_generated_at,
        },
    }


def _limit_label(rpm: int, daily_tokens: int) -> str:
    parts = []
    if rpm:
        parts.append(f"{rpm} RPM")
    if daily_tokens:
        parts.append(f"{daily_tokens} tokens/day")
    return ", ".join(parts) or "unknown"


def render_text(report: dict[str, Any]) -> str:
    counts = report["counts"]
    lines = [
        "Provider coverage audit",
        f"  router config: {report['router_config']}",
        f"  provider DB:   {report['provider_db']}",
        f"  catalog:       {report['catalog']['providers']} providers, "
        f"{report['catalog']['free']} free"
        + (f", generated {report['catalog']['generated_at']}" if report["catalog"]["generated_at"] else ""),
        "",
        f"  configured providers: {report['configured_providers']} "
        f"({report['configured_enabled']} enabled)",
        f"  providers used by fallback chains: {len(report['fallback_chain_providers'])}",
        "    " + ", ".join(report["fallback_chain_providers"]),
        f"  DB providers: {report['db_providers']}",
        f"  DB providers with a free signal: {report['db_free_providers']}",
        f"    with a numeric limit:    {report['free_with_numeric_limit']}",
        f"    without a numeric limit: {report['free_without_numeric_limit']}",
        f"  missing free providers: {report['missing_free_providers']}",
        "",
        "  Why missing:",
        f"    usable new endpoint:  {counts['usable']}",
        f"    duplicate endpoint:   {counts['duplicate_endpoint']}",
        f"    no base URL:          {counts['no_base_url']}",
        f"    no models:            {counts['no_models']}",
        f"    no key source:        {counts['no_key_source']}",
        f"    local/self endpoint:  {counts['local_or_self']}",
        "",
        "  Usable new endpoints:",
    ]
    for item in report["usable"]:
        lines.append(
            f"    {item['name']:<22} free_models={item['free_models']:<4} "
            f"limit={_limit_label(item['rpm'], item['daily_tokens']):<18} "
            f"key={item['key_source']:<11} {item['url']}"
        )
    if not report["usable"]:
        lines.append("    (none)")
    if report["duplicates"]:
        lines.extend(["", "  Duplicate endpoints already configured:"])
        for item in report["duplicates"]:
            lines.append(
                f"    {item['name']:<22} -> {', '.join(item['duplicate_of'])}"
            )
    return "\n".join(lines)


def render_markdown(report: dict[str, Any]) -> str:
    counts = report["counts"]
    chain = ", ".join(f"`{name}`" for name in report["fallback_chain_providers"])
    lines = [
        "# Provider coverage",
        "",
        "Generated by `scripts/audit_provider_coverage.py`. This report is read-only",
        "and does not read API key values. A configured provider is not necessarily",
        "used by a fallback chain, and a free provider is not necessarily routable.",
        "",
        "## Counts",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Providers in router config | {report['configured_providers']} |",
        f"| Enabled in router config | {report['configured_enabled']} |",
        f"| Providers used by fallback chains | {len(report['fallback_chain_providers'])} |",
        f"| Providers in catalog | {report['catalog']['providers']} |",
        f"| Free providers in catalog | {report['catalog']['free']} |",
        f"| Providers in manager DB | {report['db_providers']} |",
        f"| Providers in DB with a free signal | {report['db_free_providers']} |",
        f"| Free providers with a numeric limit | {report['free_with_numeric_limit']} |",
        f"| Free providers without a numeric limit | {report['free_without_numeric_limit']} |",
        f"| Free DB providers missing from router config | {report['missing_free_providers']} |",
        "",
        f"Fallback-chain providers: {chain}.",
        "",
        "## Why the missing providers are missing",
        "",
        "| Reason | Count |",
        "|---|---:|",
        f"| Usable new endpoint | {counts['usable']} |",
        f"| Duplicate of an existing endpoint | {counts['duplicate_endpoint']} |",
        f"| No base URL | {counts['no_base_url']} |",
        f"| No models | {counts['no_models']} |",
        f"| No key source | {counts['no_key_source']} |",
        f"| Local or self endpoint | {counts['local_or_self']} |",
        "",
        "## Rate-limit coverage",
        "",
        f"Of {report['db_free_providers']} free providers in the DB, "
        f"{report['free_with_numeric_limit']} have a numeric RPM or daily-token "
        f"limit and {report['free_without_numeric_limit']} do not. Missing a "
        "numeric limit is not a reason to remove a provider: the router treats "
        "unknown limits as best-effort and still reacts to upstream 429s.",
        "",
        f"Of the {report['counts']['usable']} usable new endpoints, "
        f"{report['usable_with_numeric_limit']} have a numeric limit and "
        f"{report['usable_without_numeric_limit']} do not.",
        "",
        "## Usable new endpoints",
        "",
        "| Provider | Free models | Limit | Key source | Base URL |",
        "|---|---:|---|---|---|",
    ]
    for item in report["usable"]:
        lines.append(
            f"| `{item['name']}` | {item['free_models']} | "
            f"{_limit_label(item['rpm'], item['daily_tokens'])} | "
            f"{item['key_source']} | `{item['url']}` |"
        )
    if not report["usable"]:
        lines.append("| _none_ | 0 | - | - | - |")
    lines.extend(["", "## Duplicate endpoints already covered", ""])
    if report["duplicates"]:
        lines.extend(["| Missing name | Already configured as |", "|---|---|"])
        for item in report["duplicates"]:
            names = ", ".join(f"`{name}`" for name in item["duplicate_of"])
            lines.append(f"| `{item['name']}` | {names} |")
    else:
        lines.append("_None._")
    lines.extend(
        [
            "",
            "## Limit overlay is separate",
            "",
            "`config/provider_limits.yaml` only stores providers with numeric RPM or",
            "daily-token data. It is not the provider inventory. Adding or removing a",
            "provider from the gateway happens in `config/router_config.yaml`; the",
            "limit overlay only tells the rate limiter what cap to apply.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--router-config",
        type=Path,
        default=_repo_root() / "config" / "router_config.yaml",
    )
    parser.add_argument("--db", type=Path, default=_default_db())
    parser.add_argument(
        "--catalog",
        type=Path,
        default=_repo_root() / "config" / "catalog.json.gz",
    )
    parser.add_argument("--format", choices=("text", "json", "markdown"), default="text")
    parser.add_argument("--output", type=Path, help="write the report to this path")
    parser.add_argument(
        "--fail-on-gap",
        action="store_true",
        help="exit 1 when usable free providers are missing from router config",
    )
    args = parser.parse_args(argv)

    try:
        report = build_report(args.router_config, args.db, args.catalog)
    except (FileNotFoundError, sqlite3.Error, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        rendered = json.dumps(report, indent=2, sort_keys=True)
    elif args.format == "markdown":
        rendered = render_markdown(report)
    else:
        rendered = render_text(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(rendered)

    if args.fail_on_gap and report["counts"]["usable"]:
        print(
            f"error: {report['counts']['usable']} usable free providers are missing",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
