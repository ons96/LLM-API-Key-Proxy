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
