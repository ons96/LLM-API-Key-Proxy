import sqlite3

from provider_classification import (audit_routes, classify_provider_flags,
                                     load_provider_classifications)
from quota_ledger import QuotaLedger, Route


def provider_database(tmp_path):
    path = tmp_path / "providers.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE providers (
            id INTEGER PRIMARY KEY, key_name TEXT, enabled INTEGER, free_tier INTEGER,
            no_api_key_required INTEGER, free_one_time INTEGER, free_daily INTEGER,
            free_unlimited INTEGER, checkin_required INTEGER, checkin_unlimited INTEGER,
            provider_type TEXT, quota_amount REAL, quota_unit TEXT, quota_reset TEXT,
            classification_source TEXT, human_confirmed INTEGER,
            rate_limit_rpm INTEGER, rate_limit_daily_tokens INTEGER
        );
        INSERT INTO providers VALUES
            (1, 'one', 1, 1, 0, 1, 0, 0, 0, 0, 'free_one_time', 5.0, 'usd', 'never', 'manual', 1, 10, NULL),
            (2, 'daily', 1, 1, 0, 0, 1, 0, 0, 0, 'free_daily', NULL, 'credits', 'day', 'legacy-audit', 1, 20, 1000),
            (3, 'unlimited', 1, 1, 0, 0, 0, 1, 0, 0, 'fully_free', NULL, NULL, NULL, 'legacy-audit', 1, 30, 14400),
            (4, 'limited', 1, 1, 0, 0, 0, 0, 0, 0, 'free_tier_limited', NULL, NULL, NULL, 'legacy-audit', 1, 15, NULL),
            (5, 'paid', 1, 0, 0, 0, 0, 0, 0, 0, 'paid', NULL, NULL, NULL, 'legacy-audit', 1, 500, NULL),
            (6, 'internal', 1, 1, 1, 0, 0, 0, 0, 0, 'internal', NULL, NULL, NULL, 'legacy-audit', 1, NULL, NULL);
        """
    )
    connection.commit()
    connection.close()
    return path


def test_flags_match_gateway_precedence():
    assert classify_provider_flags(free_one_time=1, free_daily=1) == (
        "credit_based", "free_credits")
    assert classify_provider_flags(checkin_unlimited=1, free_daily=1) == (
        "unlimited", "free_unlimited")
    assert classify_provider_flags(free_daily=1) == (
        "rate_limited", "free_limited")
    assert classify_provider_flags(free_unlimited=1) == (
        "unlimited", "free_unlimited")
    assert classify_provider_flags(provider_type="free_tier_limited") == (
        "rate_limited", "free_limited")
    assert classify_provider_flags(provider_type="paid") == ("paid", "paid")
    assert classify_provider_flags(provider_type="internal") == ("unknown", "unknown")


def test_load_provider_classifications_is_read_only(tmp_path):
    path = provider_database(tmp_path)
    before = path.read_bytes()
    providers = load_provider_classifications(path)
    assert path.read_bytes() == before
    assert set(providers) == {"one", "daily", "unlimited", "limited", "paid", "internal"}
    assert providers["one"].quota_type == "credit_based"
    assert providers["one"].quota_amount == 5.0
    assert providers["daily"].free_class == "free_limited"
    assert providers["unlimited"].free_class == "free_unlimited"
    assert providers["unlimited"].rate_limit_rpm == 30
    assert providers["limited"].free_class == "free_limited"
    assert providers["paid"].free_class == "paid"
    assert providers["internal"].free_class == "unknown"


def test_route_audit_preserves_metadata_and_reports_mismatches(tmp_path):
    classifications = load_provider_classifications(provider_database(tmp_path))
    matching = Route("one", "one", "model", "key", "sc.one", 1,
                     quota_type="credit_based", input_price=0, output_price=0,
                     credit_input_price=1, credit_output_price=1, credit_balance=5)
    mismatch = Route("daily", "daily", "model", "key", "sc.daily", 2,
                     quota_type="unlimited", input_price=0, output_price=0)
    missing = Route("unknown", "not-in-db", "model", "key", "sc.unknown", 3,
                    quota_type="unlimited", input_price=0, output_price=0)
    rows = {row["route_id"]: row for row in audit_routes(
        [matching, mismatch, missing], classifications)}
    assert rows["one"]["status"] == "match"
    assert rows["daily"]["status"] == "mismatch"
    assert rows["daily"]["suggested_quota_type"] == "rate_limited"
    assert rows["unknown"]["status"] == "missing_provider_metadata"
    assert rows["daily"]["current_quota_type"] == "unlimited"


def test_operator_classify_uses_ledger_without_writing(tmp_path):
    from newapi_operator import classify_routes

    ledger = QuotaLedger()
    try:
        ledger.configure(Route("daily", "daily", "model", "key", "sc.daily", 1,
                               quota_type="rate_limited", input_price=0, output_price=0,
                               limits={"rpd": 10}))
        result = classify_routes(ledger, provider_database(tmp_path))
    finally:
        ledger.close()
    assert result["apply"] is False
    assert result["providers_in_metadata"] == 6
    assert result["routes"][0]["status"] == "match"
