import pytest

from lazarus.appliance.state import StateMachine


def test_transitions_and_listener():
    machine = StateMachine()
    seen = []
    machine.on_change(seen.append)
    machine.transition("loading")
    machine.transition("healthy")
    assert machine.state == "healthy"
    assert seen == ["initializing", "loading", "healthy"]


def test_unknown_state_rejected():
    with pytest.raises(ValueError):
        StateMachine().transition("exploded")


def test_error_dedupe_keeps_first_seen():
    machine = StateMachine()
    machine.record_error("MODEL_LOAD_FAILED", "first", recoverable=True, role="embedding")
    first_seen = machine.errors[0].first_seen
    machine.record_error("MODEL_LOAD_FAILED", "second", recoverable=True, role="embedding")
    assert len(machine.errors) == 1
    assert machine.errors[0].message == "second"
    assert machine.errors[0].first_seen == first_seen


def test_errors_payload_shape():
    machine = StateMachine()
    machine.record_error("CONFIG_INVALID", "bad config", recoverable=True)
    payload = machine.errors_payload()
    assert payload["errors"][0]["code"] == "CONFIG_INVALID"
    assert payload["errors"][0]["recoverable"] is True
    assert "first_seen" in payload["errors"][0]
