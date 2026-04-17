"""Standard agent event model."""

from mini_devin.orchestrator.standard_events import (
    AgentEventKind,
    AgentStreamEvent,
    from_legacy_session_event,
)


def test_to_log_dict_roundtrip_fields():
    ev = AgentStreamEvent(
        kind=AgentEventKind.TOOL_CALL,
        tool_name="terminal",
        tool_args={"command": "ls"},
        tool_call_id="c1",
    )
    d = ev.to_log_dict()
    assert d["kind"] == "tool_call"
    assert d["tool_name"] == "terminal"
    assert d["tool"] == "terminal"
    assert d["tool_args"]["command"] == "ls"
    assert d["command"] == "ls"


def test_from_legacy_think():
    ev = from_legacy_session_event({"type": "think", "text": "planning"})
    assert ev.kind == AgentEventKind.STATUS
    assert ev.legacy_type == "think"


def test_from_legacy_observe():
    ev = from_legacy_session_event({"type": "observe", "tool": "editor", "exit_code": 0})
    assert ev.kind == AgentEventKind.OBSERVATION
    assert ev.tool_name == "editor"


def test_observe_to_log_dict_flattens_filesystem_delta():
    ev = AgentStreamEvent(
        kind=AgentEventKind.OBSERVATION,
        tool_name="terminal",
        exit_code=0,
        legacy_type="observe",
        meta={"plan_step": 1, "filesystem_delta": {"added_sorted": ["a.py"]}},
    )
    d = ev.to_log_dict()
    assert d["filesystem_delta"]["added_sorted"] == ["a.py"]
    assert d["plan_step"] == 1
