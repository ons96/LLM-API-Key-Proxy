import math

import pytest

from router_state import RouterState


def test_state_persists_hashes_and_route_metadata(tmp_path):
    path = tmp_path / "router.sqlite"
    state = RouterState(path)
    state.record("session-1", "private prompt", "fast_general", "general")
    state.set_preferred("session-1", "implement", "general")
    state.record_outcome("session-1", "general", "success")
    state.record_cache("session-1", "general", "private prompt", cached_tokens=10)
    state.close()

    reopened = RouterState(path)
    assert reopened.count() == 1
    assert reopened.preferred("session-1", "implement") == "general"
    assert reopened.outcome_count("success") == 1
    assert reopened.cache_count("general") == 1
    assert reopened.total_cache_hits() == 1
    hints = reopened.cache_hints("session-1", "private prompt")
    assert hints["general"].cached_tokens == 10
    assert hints["general"].ttl_seconds == 600
    columns = reopened.connection.execute("PRAGMA table_info(decisions)").fetchall()
    assert {column[1] for column in columns} == {
        "id", "session_hash", "prompt_hash", "capability", "deployment", "provider", "config_fingerprint", "created_at"
    }
    assert reopened.connection.execute("SELECT prompt_hash FROM decisions").fetchone()[0] != "private prompt"
    reopened.close()


def test_state_rejects_unbounded_outcome_categories():
    state = RouterState()
    try:
        state.record_outcome("session", "general", "raw-prompt-data")
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported outcome should be rejected")
    finally:
        state.close()


def test_state_keeps_failure_classes_distinct():
    state = RouterState()
    state.record_outcome("session", "general", "operational_failure", "timeout")
    state.record_outcome("session", "general", "semantic_failure", "bad_result")
    assert state.outcome_count("operational_failure") == 1
    assert state.outcome_count("semantic_failure") == 1
    state.close()


def test_state_rejects_negative_cache_counts():
    state = RouterState()
    try:
        state.record_cache("session", "general", "prefix", cached_tokens=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative cache count should be rejected")
    finally:
        state.close()


def test_state_rejects_negative_cache_ttl():
    state = RouterState()
    try:
        state.record_cache("session", "general", "prefix", ttl_seconds=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative cache TTL should be rejected")
    finally:
        state.close()


@pytest.mark.parametrize("field", ["cached_tokens", "cache_write_tokens", "ttl_seconds"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_state_rejects_nonfinite_cache_values(field, value):
    state = RouterState()
    values = {"cached_tokens": 0, "cache_write_tokens": 0, "ttl_seconds": 600}
    values[field] = value
    try:
        with pytest.raises(ValueError):
            state.record_cache("session", "general", "prefix", **values)
    finally:
        state.close()


def test_state_ignores_corrupt_nonfinite_cache_rows():
    state = RouterState()
    state.connection.execute(
        "INSERT INTO cache_observations(session_hash, deployment, prefix_hash, cached_tokens, cache_write_tokens, ttl_seconds, observed_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (state._hash("session"), "general", state._hash("prefix"), 10, 0, math.inf, 100),
    )
    state.connection.commit()
    assert state.cache_hints("session", "prefix") == {}
    state.close()


def test_state_migrates_legacy_decisions_table(tmp_path):
    import sqlite3

    path = tmp_path / "legacy.sqlite"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE decisions (id INTEGER PRIMARY KEY, session_hash TEXT, prompt_hash TEXT, capability TEXT, deployment TEXT, created_at REAL)"
    )
    connection.commit()
    connection.close()
    state = RouterState(path)
    columns = {row[1] for row in state.connection.execute("PRAGMA table_info(decisions)")}
    assert {"provider", "config_fingerprint"} <= columns
    state.close()
