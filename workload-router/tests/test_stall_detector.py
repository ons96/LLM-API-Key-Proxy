from stall_detector import StallState


def test_repeated_actions_trigger_bounded_escalation():
    state = StallState()
    for _ in range(3):
        state.observe("pytest")
    assert state.should_escalate()
    assert state.escalate()
    assert state.escalation_level == 1
    assert state.escalate()
    assert not state.escalate()


def test_progress_resets_failure_signals():
    state = StallState(phase="debug")
    state.observe("test", verification_failed=True)
    state.observe("test", verification_failed=True)
    assert state.should_escalate()
    state.mark_progress("implement")
    assert not state.should_escalate()
    assert state.phase == "implement"
