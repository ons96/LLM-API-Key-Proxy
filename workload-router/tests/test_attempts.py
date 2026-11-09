"""Attempt grading: dispatch is not correctness, only a client verdict is."""

import pytest

from router_state import RouterState, VERDICTS


def test_open_attempt_is_pending_and_gradeable():
    state = RouterState()
    try:
        attempt = state.open_attempt("session", "general")
        assert len(attempt) == 32
        assert state.pending_attempts() == 1
        assert state.graded_attempts() == 0

        state.grade_attempt(attempt, True, "pass")

        assert state.pending_attempts() == 0
        assert state.graded_attempts(True) == 1
        assert state.graded_attempts(False) == 0
        row = state.attempt(attempt)
        assert row is not None
        assert row["deployment"] == "general"
        assert row["grade"] == 1
        assert row["verdict"] == "pass"
        assert row["graded_at"] is not None
    finally:
        state.close()


def test_attempt_ids_are_unique():
    state = RouterState()
    try:
        ids = {state.open_attempt("session", "general") for _ in range(50)}
        assert len(ids) == 50
        assert state.pending_attempts() == 50
    finally:
        state.close()


def test_regrade_is_rejected():
    state = RouterState()
    try:
        attempt = state.open_attempt("session", "general")
        state.grade_attempt(attempt, False, "fail_test")
        with pytest.raises(ValueError, match="already graded"):
            state.grade_attempt(attempt, True, "pass")
    finally:
        state.close()


def test_unknown_attempt_is_rejected():
    state = RouterState()
    try:
        with pytest.raises(ValueError, match="unknown attempt_id"):
            state.grade_attempt("0" * 32, True, "pass")
        assert state.attempt("0" * 32) is None
    finally:
        state.close()


def test_verdict_enum_is_closed():
    state = RouterState()
    try:
        attempt = state.open_attempt("session", "general")
        for bad in ("ok", "PASS", "", "fail", "raw stderr here"):
            with pytest.raises(ValueError, match="unsupported verdict"):
                state.grade_attempt(attempt, True, bad)
        assert state.pending_attempts() == 1
        for verdict in sorted(VERDICTS):
            recovered = state.open_attempt("session", "general")
            state.grade_attempt(recovered, verdict == "pass", verdict)
    finally:
        state.close()


def test_grade_must_be_a_boolean():
    state = RouterState()
    try:
        attempt = state.open_attempt("session", "general")
        for bad in (1, 0, "true", None):
            with pytest.raises(ValueError, match="grade must be a boolean"):
                state.grade_attempt(attempt, bad, "pass")
    finally:
        state.close()


def test_prompt_text_never_enters_attempt_rows(tmp_path):
    path = tmp_path / "router.db"
    state = RouterState(path)
    try:
        attempt = state.open_attempt("session-with-secret", "general")
        state.grade_attempt(attempt, True, "pass")
        state.close()
        reopened = RouterState(path)
        try:
            rows = reopened.connection.execute(
                "SELECT session_hash, deployment, verdict FROM attempts"
            ).fetchall()
            assert len(rows) == 1
            assert rows[0][0] != "session-with-secret"
            assert rows[0][2] == "pass"
            assert reopened.graded_attempts(True) == 1
        finally:
            reopened.close()
    finally:
        pass


def test_legacy_outcome_path_still_records():
    """Old clients keep working; they are just not trusted as grades."""
    state = RouterState()
    try:
        state.record_outcome("session", "general", "success")
        assert state.outcome_count("success") == 1
        assert state.graded_attempts() == 0
    finally:
        state.close()
