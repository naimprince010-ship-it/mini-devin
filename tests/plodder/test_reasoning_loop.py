"""Tests for OpenHands-style reasoning / monologue helpers."""

import json

from plodder.orchestration.reasoning_loop import (
    build_agent_thought_text,
    goal_suggests_frontend_stack,
    monologue_validation_error,
    parse_driver_turn,
    shell_argv_suggests_dev_server,
    terminal_failure_followup_hints,
)


def test_goal_suggests_frontend_stack() -> None:
    assert goal_suggests_frontend_stack("Build a React dashboard with Vite")
    assert not goal_suggests_frontend_stack("Fix typo in README")


def test_monologue_required_for_mutation_tools() -> None:
    turn = {
        "status": "continue",
        "rationale": "x",
        "tool_calls": [{"name": "fs_write", "args": {"path": "a", "content": "b"}}],
    }
    err = monologue_validation_error(turn)
    assert err is not None
    assert "sub_goal" in err


def test_monologue_ok_when_complete() -> None:
    turn = {
        "status": "continue",
        "rationale": "Write file",
        "sub_goal": "Create package manifest for the app",
        "risk_assessment": "Wrong cwd could write outside workspace; verify with pwd first",
        "expected_outcome": "package.json exists and npm install succeeds",
        "tool_calls": [{"name": "fs_write", "args": {"path": "package.json", "content": "{}"}}],
    }
    assert monologue_validation_error(turn) is None


def test_monologue_not_required_for_fs_list_only() -> None:
    turn = {
        "status": "continue",
        "rationale": "list",
        "tool_calls": [{"name": "fs_list", "args": {"path": "."}}],
    }
    assert monologue_validation_error(turn) is None


def test_parse_driver_turn_monologue_keys() -> None:
    raw = json.dumps(
        {
            "status": "continue",
            "rationale": "r",
            "sub_goal": "sg " * 5,
            "risk_assessment": "ra " * 5,
            "expected_outcome": "eo " * 5,
            "tool_calls": [],
        }
    )
    t = parse_driver_turn(raw)
    assert "sub_goal" in t
    assert len(t["sub_goal"]) >= 10


def test_build_agent_thought_text_merges_fields() -> None:
    txt = build_agent_thought_text(
        {
            "rationale": "Main",
            "sub_goal": "Sub",
            "observe": "Observed ok",
        }
    )
    assert "Main" in txt
    assert "sub_goal: Sub" in txt


def test_shell_argv_suggests_dev_server() -> None:
    assert shell_argv_suggests_dev_server(["sh", "-c", "npm run dev"])
    assert not shell_argv_suggests_dev_server(["sh", "-c", "npm run build"])


def test_terminal_failure_followup_hints() -> None:
    hint = terminal_failure_followup_hints(
        [{"tool": "sandbox_shell", "ok": False, "stderr": "nope", "argv": ["sh", "-c", "npm run dev"]}]
    )
    assert "playwright_observe" in hint
    assert "capture_console" in hint
