#!/usr/bin/env python3
"""Export provider and model quota labels from the provider DB.

Read-only. It does not call providers, read API key values, edit the DB, or
change router_config.yaml. It produces a reviewable inventory:

  PROVIDER_QUOTA_INVENTORY_<date>.csv
  MODEL_QUOTA_INVENTORY_<date>.csv
  PROVIDER_DB_ORPHANS_<date>.csv
  PROVIDER_QUOTA_SUMMARY_<date>.md

The key question is not just "is this provider free?" but "what kind of free?"
The DB has separate flags for unlimited, daily, and one-time-credit usage, plus
a provider_type field that may be stale. This exporter shows both and flags
mismatches so the labels can be reviewed before routing changes.

Usage:
    python scripts/export_provider_quota_inventory.py
    uv run --no-project --with pyyaml python scripts/export_provider_quota_inventory.py
    python scripts/export_provider_quota_inventory.py --out-dir /tmp/quota-report
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from collections import Counter
from datetime import date
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

PROVIDER_CSV_FIELDS = [
    "provider_id",
    "key_name",
    "display_name",
    "provider_type_stored",
    "provider_type_derived",
    "type_mismatch",
    "free_tier",
    "free_unlimited",
    "free_daily",
    "free_one_time",
    "checkin_required",
    "checkin_unlimited",
    "no_api_key_required",
    "quota_class",
    "quota_amount",
    "quota_unit",
    "quota_reset",
    "rate_limit_rpm",
    "rate_limit_daily_tokens",
    "models_total",
    "models_free",
    "models_unlabeled",
    "key_source",
    "base_url",
    "env_var",
    "configured",
    "in_fallback_chain",
    "coverage_status",
    "duplicate_of",
    "probe_policy",
    "label_state",
]

MODEL_CSV_FIELDS = [
    "provider_id",
    "provider",
    "provider_type_stored",
    "provider_type_derived",
    "provider_quota_class",
    "model_id",
    "display_name",
    "context_window",
    "tps",
    "free_tier",
    "tier",
    "fallback_only",
    "free_label",
    "configured",
    "in_fallback_chain",
    "probe_policy",
]


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


def _expected_type(
    key_name: str,
    base_url: str,
    free_tier: bool,
    free_unlimited: bool,
    free_daily: bool,
    free_one_time: bool,
) -> str:
    """Mirror llm-provider-manager/scripts/classify_provider_types.py."""
    if "/" in key_name or not base_url or _is_local_url(base_url):
        return "internal"
    if free_unlimited:
        return "fully_free"
    if free_daily:
        return "free_daily"
    if free_one_time:
        return "free_one_time"
    if free_tier:
        return "free_tier_limited"
    return "paid"


def _quota_class(
    free_tier: bool,
    free_unlimited: bool,
    free_daily: bool,
    free_one_time: bool,
    no_api_key_required: bool,
) -> str:
    labels = []
    if free_unlimited:
        labels.append("unlimited")
    if free_daily:
        labels.append("daily")
    if free_one_time:
        labels.append("one_time")
    if not labels and free_tier:
        labels.append("limited")
    if no_api_key_required:
        labels.append("keyless")
    return "+".join(labels) or ("paid" if not free_tier else "unknown")


def _probe_policy(
    provider_type: str,
    free_signal: bool,
    free_one_time: bool,
    key_source: str,
) -> str:
    if provider_type in {"paid", "internal"} or not free_signal:
        return "not_a_probe_target"
    if free_one_time:
        return "do_not_probe_one_time"
    if key_source == "missing":
        return "needs_key"
    return "probe_one_model"


TYPED_PROVIDER_TYPES = {
    "fully_free",
    "free_daily",
    "free_rpd",
    "free_tier_limited",
    "free_one_time",
    "paid",
    "internal",
}


def _label_state(
    free_tier: bool | None,
    free_unlimited: bool,
    free_daily: bool,
    free_one_time: bool,
    provider_type: str,
) -> str:
    if free_tier is None:
        return "unknown"
    if free_unlimited or free_daily or free_one_time:
        return "labeled"
    # provider_type is the durable label. The three mode flags are optional
    # detail, so a typed row is not an unlabelled row just because the tier
    # is renewable "limited" rather than unlimited or daily.
    if provider_type in TYPED_PROVIDER_TYPES:
        return "labeled"
    return "unlabeled_free" if free_tier else "labeled_paid"


def _load_router_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"router config is not a mapping: {path}")
    return data


def _chain_providers(config: dict[str, Any]) -> set[str]:
    found: set[str] = set()
    for virtual in (config.get("router_models") or {}).values():
        if not isinstance(virtual, dict):
            continue
        for candidate in virtual.get("fallback_chain") or virtual.get("candidates") or []:
            if isinstance(candidate, dict) and candidate.get("provider"):
                found.add(str(candidate["provider"]))
    return found


def _load_db(db_path: Path) -> tuple[list[sqlite3.Row], list[sqlite3.Row], dict[int, dict[str, int]]]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        providers = con.execute(
            """
            SELECT id, key_name, display_name, base_url, env_var, provider_type,
                   free_tier, free_unlimited, free_daily, free_one_time,
                   checkin_required, checkin_unlimited, no_api_key_required,
                   quota_amount, quota_unit, quota_reset,
                   rate_limit_rpm, rate_limit_daily_tokens
            FROM providers
            ORDER BY key_name
            """
        ).fetchall()
        models = con.execute(
            """
            SELECT provider_id, model_id, display_name, context_window, tps,
                   free_tier, tier, fallback_only
            FROM models
            WHERE provider_id IS NOT NULL
            ORDER BY provider_id, model_id
            """
        ).fetchall()
        counts = {
            int(row["provider_id"]): {
                "models": int(row["models"] or 0),
                "free": int(row["free"] or 0),
                "unlabeled": int(row["unlabeled"] or 0),
            }
            for row in con.execute(
                """
                SELECT provider_id,
                       COUNT(*) AS models,
                       SUM(CASE WHEN free_tier = 1 THEN 1 ELSE 0 END) AS free,
                       SUM(CASE WHEN free_tier IS NULL THEN 1 ELSE 0 END) AS unlabeled
                FROM models
                WHERE provider_id IS NOT NULL
                GROUP BY provider_id
                """
            )
        }
        return providers, models, counts
    finally:
        con.close()


def _load_orphans(db_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        providers = [
            {
                "row_type": "provider",
                "provider_id": "",
                "key_name": row["key_name"],
                "base_url": row["base_url"],
                "provider_type": row["provider_type"],
                "free_tier": row["free_tier"],
                "model_id": "",
                "orphan_reason": "null_provider_id",
            }
            for row in con.execute(
                """
                SELECT key_name, base_url, provider_type, free_tier
                FROM providers
                WHERE id IS NULL
                ORDER BY key_name
                """
            )
        ]
        models = [
            {
                "row_type": "model",
                "provider_id": row["provider_id"] if row["provider_id"] is not None else "",
                "key_name": "",
                "base_url": "",
                "provider_type": "",
                "free_tier": row["free_tier"],
                "model_id": row["model_id"],
                "orphan_reason": row["orphan_reason"],
            }
            for row in con.execute(
                """
                SELECT model_id, free_tier, provider_id,
                       CASE WHEN provider_id IS NULL
                            THEN 'null_provider_id'
                            ELSE 'dangling_provider_id' END AS orphan_reason
                FROM models
                WHERE provider_id IS NULL
                   OR provider_id NOT IN (
                       SELECT id FROM providers WHERE id IS NOT NULL
                   )
                ORDER BY model_id
                """
            )
        ]
        return providers, models
    finally:
        con.close()


def build_inventory(db_path: Path, router_config_path: Path) -> dict[str, Any]:
    config = _load_router_config(router_config_path)
    configured: dict[str, Any] = config.get("providers") or {}
    configured_names = {_normalize_name(name) for name in configured}
    configured_by_url: dict[str, list[str]] = {}
    for name, details in configured.items():
        if not isinstance(details, dict):
            continue
        normalized = _normalize_url(details.get("base_url") or "")
        if normalized:
            configured_by_url.setdefault(normalized, []).append(name)
    chain = _chain_providers(config)

    providers, models, model_counts = _load_db(db_path)
    provider_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    provider_by_id: dict[int, dict[str, Any]] = {}

    for row in providers:
        if row["id"] is None or not row["key_name"]:
            continue
        provider_id = int(row["id"])
        name = str(row["key_name"])
        url = str(row["base_url"] or "")
        free_tier_raw = row["free_tier"]
        free_tier = bool(free_tier_raw)
        free_unlimited = bool(row["free_unlimited"])
        free_daily = bool(row["free_daily"])
        free_one_time = bool(row["free_one_time"])
        no_key = bool(row["no_api_key_required"])
        counts = model_counts.get(provider_id, {"models": 0, "free": 0, "unlabeled": 0})
        free_signal = free_tier or no_key or counts["free"] > 0
        stored_type = str(row["provider_type"] or "unknown")
        derived_type = _expected_type(
            name, url, free_tier, free_unlimited, free_daily, free_one_time
        )
        key_source = _key_source(str(row["env_var"] or ""), no_key)
        normalized_url = _normalize_url(url)
        duplicate_of = configured_by_url.get(normalized_url, [])
        is_configured = _normalize_name(name) in configured_names

        if is_configured:
            coverage = "configured"
        elif not url:
            coverage = "no_base_url"
        elif _is_local_url(url):
            coverage = "local_or_self"
        elif counts["models"] <= 0:
            coverage = "no_models"
        elif key_source == "missing":
            coverage = "no_key_source"
        elif duplicate_of:
            coverage = "duplicate_endpoint"
        else:
            coverage = "usable_missing"

        quota_class = _quota_class(
            free_tier, free_unlimited, free_daily, free_one_time, no_key
        )
        label_state = _label_state(
            free_tier_raw, free_unlimited, free_daily, free_one_time, stored_type
        )
        probe_policy = _probe_policy(
            stored_type, free_signal, free_one_time, key_source
        )

        provider_row = {
            "provider_id": provider_id,
            "key_name": name,
            "display_name": row["display_name"] or "",
            "provider_type_stored": stored_type,
            "provider_type_derived": derived_type,
            "type_mismatch": int(stored_type != derived_type),
            "free_tier": "" if free_tier_raw is None else int(free_tier),
            "free_unlimited": int(free_unlimited),
            "free_daily": int(free_daily),
            "free_one_time": int(free_one_time),
            "checkin_required": int(row["checkin_required"] or 0),
            "checkin_unlimited": int(row["checkin_unlimited"] or 0),
            "no_api_key_required": int(no_key),
            "quota_class": quota_class,
            "quota_amount": row["quota_amount"] if row["quota_amount"] is not None else "",
            "quota_unit": row["quota_unit"] or "",
            "quota_reset": row["quota_reset"] or "",
            "rate_limit_rpm": int(row["rate_limit_rpm"] or 0),
            "rate_limit_daily_tokens": int(row["rate_limit_daily_tokens"] or 0),
            "models_total": counts["models"],
            "models_free": counts["free"],
            "models_unlabeled": counts["unlabeled"],
            "key_source": key_source,
            "base_url": url,
            "env_var": row["env_var"] or "",
            "configured": int(is_configured),
            "in_fallback_chain": int(name in chain),
            "coverage_status": coverage,
            "duplicate_of": "|".join(sorted(duplicate_of)),
            "probe_policy": probe_policy,
            "label_state": label_state,
        }
        provider_rows.append(provider_row)
        provider_by_id[provider_id] = provider_row

    for model in models:
        provider_id = int(model["provider_id"])
        provider_row = provider_by_id.get(provider_id)
        if not provider_row:
            continue
        free_tier = model["free_tier"]
        if free_tier == 1:
            free_label = "free"
        elif free_tier == 0:
            free_label = "paid"
        else:
            free_label = "unlabeled"
        model_rows.append(
            {
                "provider_id": provider_id,
                "provider": provider_row["key_name"],
                "provider_type_stored": provider_row["provider_type_stored"],
                "provider_type_derived": provider_row["provider_type_derived"],
                "provider_quota_class": provider_row["quota_class"],
                "model_id": model["model_id"] or "",
                "display_name": model["display_name"] or "",
                "context_window": model["context_window"] if model["context_window"] is not None else "",
                "tps": model["tps"] if model["tps"] is not None else "",
                "free_tier": "" if free_tier is None else int(free_tier),
                "tier": model["tier"] or "",
                "fallback_only": "" if model["fallback_only"] is None else int(model["fallback_only"]),
                "free_label": free_label,
                "configured": provider_row["configured"],
                "in_fallback_chain": provider_row["in_fallback_chain"],
                "probe_policy": provider_row["probe_policy"],
            }
        )

    orphan_providers, orphan_models = _load_orphans(db_path)
    provider_classes = Counter(row["quota_class"] for row in provider_rows)
    model_labels = Counter(row["free_label"] for row in model_rows)
    mismatches = [row for row in provider_rows if row["type_mismatch"]]
    unlabeled_free = [row for row in provider_rows if row["label_state"] == "unlabeled_free"]
    unknown_free = [
        row for row in provider_rows
        if row["free_tier"] != 0 and row["quota_class"] in {"unknown", "keyless"}
    ]

    return {
        "provider_rows": provider_rows,
        "model_rows": model_rows,
        "orphan_providers": orphan_providers,
        "orphan_models": orphan_models,
        "summary": {
            "providers": len(provider_rows),
            "free_providers": sum(
                1 for row in provider_rows
                if row["free_tier"] == 1 or row["models_free"] > 0 or row["no_api_key_required"]
            ),
            "provider_classes": dict(sorted(provider_classes.items())),
            "type_mismatches": len(mismatches),
            "type_mismatch_names": [row["key_name"] for row in mismatches[:30]],
            "unlabeled_free": len(unlabeled_free),
            "unlabeled_free_names": [row["key_name"] for row in unlabeled_free[:30]],
            "unknown_free": len(unknown_free),
            "unknown_free_names": [row["key_name"] for row in unknown_free[:30]],
            "models": len(model_rows),
            "models_total": len(model_rows) + len(orphan_models),
            "model_labels": dict(sorted(model_labels.items())),
            "usable_missing_classes": dict(
                sorted(
                    Counter(
                        row["quota_class"]
                        for row in provider_rows
                        if row["coverage_status"] == "usable_missing"
                        and (
                            row["free_tier"] == 1
                            or row["models_free"] > 0
                            or row["no_api_key_required"]
                        )
                    ).items()
                )
            ),
            "orphan_providers": len(orphan_providers),
            "orphan_models": len(orphan_models),
            "orphan_models_null": sum(
                1 for row in orphan_models if row["orphan_reason"] == "null_provider_id"
            ),
            "orphan_models_dangling": sum(
                1 for row in orphan_models if row["orphan_reason"] == "dangling_provider_id"
            ),
            "one_time_providers": sum(1 for row in provider_rows if row["free_one_time"]),
            "usable_missing_one_time": sum(
                1
                for row in provider_rows
                if row["coverage_status"] == "usable_missing"
                and row["free_one_time"]
                and (
                    row["free_tier"] == 1
                    or row["models_free"] > 0
                    or row["no_api_key_required"]
                )
            ),
        },
    }


def _ascii_text(value: Any) -> Any:
    """Keep generated reports ASCII even when provider names are not."""
    if isinstance(value, str):
        return value.encode("ascii", "backslashreplace").decode("ascii")
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {key: _ascii_text(value) for key, value in row.items()}
            for row in rows
        )


def _write_summary(path: Path, report: dict[str, Any], stamp: str) -> None:
    summary = report["summary"]
    classes = summary["provider_classes"]
    labels = summary["model_labels"]
    lines = [
        "# Provider and model quota inventory",
        "",
        f"Generated: {stamp}. Read-only exporter: it did not change DB rows or router config.",
        "",
        "## Provider labels",
        "",
        f"- Valid providers: {summary['providers']}",
        f"- Providers with a free signal: {summary['free_providers']}",
        f"- Provider type mismatches: {summary['type_mismatches']}",
        f"- Free providers with no unlimited/daily/one-time flag: {summary['unlabeled_free']}",
        f"- Free providers with unknown/keyless class: {summary['unknown_free']}",
        f"- One-time-credit providers: {summary['one_time_providers']}",
        "",
        "| Quota class | Providers |",
        "|---|---:|",
    ]
    for name, count in classes.items():
        lines.append(f"| {name} | {count} |")
    lines.extend(
        [
            "",
            "## Model labels",
            "",
            f"- Total model rows: {summary['models_total']}",
            f"- Labelable model rows: {summary['models']}",
            f"- Orphan/dangling model rows excluded: {summary['orphan_models']}",
            f"- Labeled free: {labels.get('free', 0)}",
            f"- Labeled paid: {labels.get('paid', 0)}",
            f"- Unlabeled: {labels.get('unlabeled', 0)}",
            "",
            "## Usable missing providers by quota class",
            "",
            "| Quota class | Count |",
            "|---|---:|",
        ]
    )
    for name, count in summary["usable_missing_classes"].items():
        lines.append(f"| {name} | {count} |")
    lines.extend(
        [
            "",
            f"Of these, {summary['usable_missing_one_time']} are one-time-credit "
            "providers. Do not probe them.",
            "",
            "## One-time-credit warning",
            "",
            "Do not probe one-time-credit providers. The probe consumes the credit.",
            "Use metadata and provider documentation for those rows, and keep them",
            "out of automatic fallback until they are deliberately selected.",
            "",
            "## Review worklist",
            "",
            "Type mismatches (stored provider_type differs from derived):",
            "",
        ]
    )
    if summary["type_mismatch_names"]:
        lines.extend(f"- `{name}`" for name in summary["type_mismatch_names"])
    else:
        lines.append("- None.")
    lines.extend(["", "Free providers with no quota flag:", ""])
    if summary["unlabeled_free_names"]:
        lines.extend(f"- `{name}`" for name in summary["unlabeled_free_names"])
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Orphan rows",
            "",
            f"- Provider rows without an id: {summary['orphan_providers']}",
            f"- Model rows with no provider id: {summary['orphan_models_null']}",
            f"- Model rows with a dangling provider id: {summary['orphan_models_dangling']}",
            "Orphan rows are exported for review; this exporter never deletes live rows.",
            "",
            "## Files",
            "",
            f"- `PROVIDER_QUOTA_INVENTORY_{stamp}.csv`",
            f"- `MODEL_QUOTA_INVENTORY_{stamp}.csv`",
            f"- `PROVIDER_DB_ORPHANS_{stamp}.csv`",
            "",
        ]
    )
    path.write_text(_ascii_text("\n".join(lines)), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=_default_db())
    parser.add_argument(
        "--router-config",
        type=Path,
        default=_repo_root() / "config" / "router_config.yaml",
    )
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--stamp", default=date.today().isoformat())
    args = parser.parse_args(argv)

    if not args.db.exists():
        print(f"error: provider DB not found: {args.db}")
        return 2
    if not args.router_config.exists():
        print(f"error: router config not found: {args.router_config}")
        return 2

    report = build_inventory(args.db, args.router_config)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    provider_path = out_dir / f"PROVIDER_QUOTA_INVENTORY_{args.stamp}.csv"
    model_path = out_dir / f"MODEL_QUOTA_INVENTORY_{args.stamp}.csv"
    orphan_path = out_dir / f"PROVIDER_DB_ORPHANS_{args.stamp}.csv"
    summary_path = out_dir / f"PROVIDER_QUOTA_SUMMARY_{args.stamp}.md"

    _write_csv(provider_path, report["provider_rows"], PROVIDER_CSV_FIELDS)
    _write_csv(model_path, report["model_rows"], MODEL_CSV_FIELDS)
    orphan_fields = [
        "row_type",
        "provider_id",
        "key_name",
        "base_url",
        "provider_type",
        "free_tier",
        "model_id",
        "orphan_reason",
    ]
    _write_csv(
        orphan_path,
        report["orphan_providers"] + report["orphan_models"],
        orphan_fields,
    )
    _write_summary(summary_path, report, args.stamp)

    summary = report["summary"]
    print(f"wrote {provider_path} ({summary['providers']} providers)")
    print(f"wrote {model_path} ({summary['models']} models)")
    print(
        f"wrote {orphan_path} "
        f"({summary['orphan_providers']} orphan providers, "
        f"{summary['orphan_models']} orphan models)"
    )
    print(f"wrote {summary_path}")
    print(
        "label gaps: "
        f"{summary['type_mismatches']} type mismatches, "
        f"{summary['unlabeled_free']} free providers without a quota flag, "
        f"{report['summary']['model_labels'].get('unlabeled', 0)} unlabeled models"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
