"""Tests for OpenHands-style reasoning / monologue helpers."""

import json

from plodder.orchestration.reasoning_loop import (
    build_agent_thought_text,
    goal_suggests_frontend_stack,
    merge_prefix_summary_and_tail,
    monologue_validation_error,
    parse_driver_turn,
    partition_messages_sliding_window,
    path_suggests_ui_surface,
    shell_argv_suggests_dev_server,
    terminal_failure_followup_hints,
    visual_review_done_gate,
    worklog_has_ui_mutation,
)


def test_goal_suggests_frontend_stack() -> None:
    assert goal_suggests_frontend_stack("Build a React dashboard with Vite")
    assert goal_suggests_frontend_stack("Premium Dashboard with charts")
    assert not goal_suggests_frontend_stack("Fix typo in README")


def test_path_suggests_ui_surface() -> None:
    assert path_suggests_ui_surface("src/components/Foo.tsx")
    assert path_suggests_ui_surface("tailwind.config.js")
    assert path_suggests_ui_surface("app/routes/index.tsx")
    assert not path_suggests_ui_surface("README.md")
    assert not path_suggests_ui_surface("api/server.py")


class _FakeWorklog:
    __slots__ = ("events",)

    def __init__(self, events: list) -> None:
        self.events = events


def test_worklog_has_ui_mutation() -> None:
    ev = [
        {
            "event_type": "action_observation",
            "action": {"tool": "atomic_edit", "arguments": {"path": "src/App.tsx"}},
        }
    ]
    assert worklog_has_ui_mutation(ev)
    assert not worklog_has_ui_mutation(
        [{"event_type": "action_observation", "action": {"tool": "fs_write", "arguments": {"path": "README.md"}}}]
    )


def test_visual_review_done_gate_skips_non_frontend() -> None:
    wl = _FakeWorklog(
        [
            {
                "event_type": "action_observation",
                "action": {"tool": "atomic_edit", "arguments": {"path": "src/App.tsx"}},
            }
        ]
    )
    assert visual_review_done_gate("Fix typo in README", wl) is None


def test_visual_review_done_gate_blocks_without_observe() -> None:
    wl = _FakeWorklog(
        [
            {
                "event_type": "action_observation",
                "action": {"tool": "atomic_edit", "arguments": {"path": "src/App.tsx"}},
            }
        ]
    )
    msg = visual_review_done_gate("Build a React dashboard with Vite", wl)
    assert msg is not None
    assert "playwright_observe" in msg


def test_partition_messages_sliding_window_keeps_goal_prefix() -> None:
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "## Workspace root\nx\n\n## Goal\nBuild app"},
    ]
    for i in range(21):
        messages.append({"role": "assistant", "content": json.dumps({"status": "continue", "rationale": str(i)})})
        messages.append({"role": "user", "content": "## Tool results\n```json\n{}\n```"})
    prefix, stale, tail = partition_messages_sliding_window(messages, max_tool_rounds=20)
    assert len(prefix) == 2
    assert "## Goal" in prefix[1]["content"]
    assert len(stale) == 2
    assert stale[0]["role"] == "assistant"
    assert "## Tool results" in stale[1]["content"]
    assert len(tail) == len(messages) - 4
    assert "## Tool results" in tail[-1]["content"]


def test_merge_prefix_summary_and_tail_inserts_summary_after_goal() -> None:
    prefix = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "## Goal\nG"},
    ]
    tail = [{"role": "assistant", "content": "a"}, {"role": "user", "content": "## Tool results\n{}"}]
    out = merge_prefix_summary_and_tail(prefix, tail, "- fixed imports\n- ran tests")
    assert len(out) == 5
    assert out[0]["role"] == "system"
    assert "## Goal" in out[1]["content"]
    assert "Running summary" in out[2]["content"]
    assert "fixed imports" in out[2]["content"]
    assert out[3]["role"] == "assistant"


def test_visual_review_done_gate_passes_with_two_viewports() -> None:
    wl = _FakeWorklog(
        [
            {
                "event_type": "action_observation",
                "action": {"tool": "atomic_edit", "arguments": {"path": "src/App.tsx"}},
            },
            {
                "event_type": "action_observation",
                "action": {
                    "tool": "playwright_observe",
                    "arguments": {"url": "http://127.0.0.1:5173/", "viewport_width": 375},
                },
                "observation": {"raw": {"ok": True, "viewport_width": 375}},
            },
            {
                "event_type": "action_observation",
                "action": {
                    "tool": "playwright_observe",
                    "arguments": {"viewport_width": 1440},
                },
                "observation": {"raw": {"ok": True, "viewport_width": 1440}},
            },
        ]
    )
    assert visual_review_done_gate("Build a React dashboard with Vite", wl) is None


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
