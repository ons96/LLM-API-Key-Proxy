"""Read-only bridge from the gateway provider database to sidecar route classes.

`llm_providers.db` is the source of truth for provider free-tier flags. This
module does not copy provider credentials or mutate the database; it exposes a
small classification snapshot that an operator can compare with the sidecar
ledger before seeding or changing routes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import sqlite3

from quota_ledger import Route


_COLUMNS = (
    "key_name",
    "enabled",
    "free_tier",
    "no_api_key_required",
    "free_one_time",
    "free_daily",
    "free_unlimited",
    "checkin_required",
    "checkin_unlimited",
    "provider_type",
    "quota_amount",
    "quota_unit",
    "quota_reset",
    "classification_source",
    "human_confirmed",
    "rate_limit_rpm",
    "rate_limit_daily_tokens",
)
_LEGACY_NAME_COLUMN = "name"


@dataclass(frozen=True)
class ProviderClassification:
    """One provider's funding category, derived from its persisted flags."""

    provider_id: str
    quota_type: str
    free_class: str
    provider_type: str
    enabled: bool
    free_tier: bool | None
    no_api_key_required: bool
    free_one_time: bool
    free_daily: bool
    free_unlimited: bool
    checkin_required: bool
    checkin_unlimited: bool
    quota_amount: float | None
    quota_unit: str | None
    quota_reset: str | None
    classification_source: str
    human_confirmed: bool
    rate_limit_rpm: int | None
    rate_limit_daily_tokens: int | None

    def as_dict(self) -> dict:
        return asdict(self)


def _flag(value: object) -> bool:
    return bool(value) if value is not None else False


def _optional_number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number == number and number not in (float("inf"), float("-inf")) else None


def _optional_int(value: object) -> int | None:
    number = _optional_number(value)
    return None if number is None else int(number)


def classify_provider_flags(
    *,
    free_one_time: object = 0,
    free_daily: object = 0,
    free_unlimited: object = 0,
    checkin_required: object = 0,
    checkin_unlimited: object = 0,
    provider_type: object = "",
) -> tuple[str, str]:
    """Return `(quota_type, free_class)` using the gateway's flag precedence.

    One-time credits are the scarcest, then check-in-unlimited behaves like an
    unlimited credit pool, then renewable/finite tiers, then flat unlimited.
    `provider_type` is a fallback for older rows whose flags are incomplete.
    """
    kind = str(provider_type or "").strip().lower()
    if _flag(free_one_time) or kind == "free_one_time":
        return "credit_based", "free_credits"
    if _flag(checkin_unlimited):
        return "unlimited", "free_unlimited"
    if _flag(free_daily) or _flag(checkin_required) or kind in {"free_daily", "free_tier_limited"}:
        return "rate_limited", "free_limited"
    if _flag(free_unlimited) or kind == "fully_free":
        return "unlimited", "free_unlimited"
    if kind == "paid":
        return "paid", "paid"
    return "unknown", "unknown"


def classify_provider_row(row: sqlite3.Row) -> ProviderClassification:
    provider_id = str(row["provider_id"] or "").strip()
    if not provider_id or len(provider_id) > 512:
        raise ValueError("invalid provider id")
    quota_type, free_class = classify_provider_flags(
        free_one_time=row["free_one_time"],
        free_daily=row["free_daily"],
        free_unlimited=row["free_unlimited"],
        checkin_required=row["checkin_required"],
        checkin_unlimited=row["checkin_unlimited"],
        provider_type=row["provider_type"],
    )
    free_tier = None if row["free_tier"] is None else bool(row["free_tier"])
    return ProviderClassification(
        provider_id=provider_id,
        quota_type=quota_type,
        free_class=free_class,
        provider_type=str(row["provider_type"] or ""),
        enabled=bool(row["enabled"]),
        free_tier=free_tier,
        no_api_key_required=bool(row["no_api_key_required"]),
        free_one_time=bool(row["free_one_time"]),
        free_daily=bool(row["free_daily"]),
        free_unlimited=bool(row["free_unlimited"]),
        checkin_required=bool(row["checkin_required"]),
        checkin_unlimited=bool(row["checkin_unlimited"]),
        quota_amount=_optional_number(row["quota_amount"]),
        quota_unit=str(row["quota_unit"]) if row["quota_unit"] else None,
        quota_reset=str(row["quota_reset"]) if row["quota_reset"] else None,
        classification_source=str(row["classification_source"] or "unknown"),
        human_confirmed=bool(row["human_confirmed"]),
        rate_limit_rpm=_optional_int(row["rate_limit_rpm"]),
        rate_limit_daily_tokens=_optional_int(row["rate_limit_daily_tokens"]),
    )


def load_provider_classifications(path: str | Path) -> dict[str, ProviderClassification]:
    """Read provider classifications without opening or modifying credentials."""
    database = Path(path).expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(database)
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(providers)")}
        name_column = "key_name" if "key_name" in columns else (
            _LEGACY_NAME_COLUMN if _LEGACY_NAME_COLUMN in columns else None
        )
        if name_column is None:
            raise ValueError("provider database has no key_name or name column")
        selected = [f"{name_column} AS provider_id"]
        for name in _COLUMNS:
            if name == "key_name":
                continue
            selected.append(name if name in columns else f"NULL AS {name}")
        rows = connection.execute(
            "SELECT " + ", ".join(selected) + " FROM providers ORDER BY " + name_column
        ).fetchall()
    finally:
        connection.close()
    result: dict[str, ProviderClassification] = {}
    for row in rows:
        classification = classify_provider_row(row)
        result[classification.provider_id] = classification
    return result


def audit_routes(routes: list[Route],
                 classifications: dict[str, ProviderClassification]) -> list[dict]:
    """Compare ledger routes with provider metadata without changing either side.

    Explicit route metadata is preserved. A mismatch is a review signal, not an
    automatic rewrite, because credit balances and route-specific limits may be
    intentionally different from the provider default.
    """
    rows = []
    for route in routes:
        provider = classifications.get(route.provider_id)
        if provider is None:
            rows.append({
                "route_id": route.route_id, "provider_id": route.provider_id,
                "status": "missing_provider_metadata",
                "current_quota_type": route.quota_type,
                "current_free_class": route.free_class,
                "suggested_quota_type": None, "suggested_free_class": None,
                "notes": ["provider is not present in llm_providers.db"],
            })
            continue
        notes = []
        if provider.quota_type != route.quota_type:
            notes.append(f"quota_type {route.quota_type} != {provider.quota_type}")
        if provider.free_class != route.free_class:
            notes.append(f"free_class {route.free_class} != {provider.free_class}")
        if provider.quota_type == "credit_based":
            if route.credit_balance is None:
                notes.append("route has no credit_balance")
            if (route.credit_input_price is None and route.credit_output_price is None
                    and (route.input_price in (None, 0) or route.output_price in (None, 0))):
                notes.append("route has no notional credit tariff")
        status = "unclassified" if provider.quota_type == "unknown" else (
            "match" if not notes else "mismatch"
        )
        rows.append({
            "route_id": route.route_id, "provider_id": route.provider_id,
            "status": status,
            "current_quota_type": route.quota_type,
            "current_free_class": route.free_class,
            "suggested_quota_type": provider.quota_type,
            "suggested_free_class": provider.free_class,
            "provider_type": provider.provider_type,
            "classification_source": provider.classification_source,
            "human_confirmed": provider.human_confirmed,
            "notes": notes,
        })
    return rows
