"""Tests for simple file task completion helpers."""

from unittest.mock import MagicMock

from mini_devin.orchestrator.agent import Agent
from mini_devin.schemas.state import TaskGoal, TaskState


def test_simple_file_task_satisfied_uses_task_goal_description():
    agent = Agent(llm_client=MagicMock(), auto_verify=False)
    task = TaskState(
        task_id="task-simple-file",
        goal=TaskGoal(
            description=(
                "Create a file named mini-devin-smoke.txt with exactly this text: "
                "Mini Devin smoke test passed."
            )
        ),
    )

    assert agent._simple_file_task_satisfied(
        task,
        "editor",
        {
            "action": "write_file",
            "path": "mini-devin-smoke.txt",
            "content": "Mini Devin smoke test passed.",
        },
        "",
    )


def test_simple_file_task_satisfied_does_not_match_wrong_content():
    agent = Agent(llm_client=MagicMock(), auto_verify=False)
    task = TaskState(
        task_id="task-simple-file-wrong",
        goal=TaskGoal(
            description=(
                "Create a file named mini-devin-smoke.txt with exactly this text: "
                "Mini Devin smoke test passed."
            )
        ),
    )

    assert not agent._simple_file_task_satisfied(
        task,
        "editor",
        {
            "action": "write_file",
            "path": "mini-devin-smoke.txt",
            "content": "Different content.",
        },
        "",
    )


def test_progress_guard_nudges_after_repeated_inspection():
    agent = Agent(llm_client=MagicMock(), auto_verify=False)
    task = TaskState(
        task_id="task-progress-guard",
        goal=TaskGoal(description="Create index.html for FocusFlow."),
    )

    messages = [
        agent._progress_guard_message(
            task,
            "terminal",
            {"command": "git status"},
            tool_success=True,
        )
        for _ in range(4)
    ]

    assert messages[:3] == [None, None, None]
    assert messages[3] is not None
    assert "Stop listing directories" in messages[3]
    assert "Create index.html" in messages[3]


def test_progress_guard_resets_after_write_action():
    agent = Agent(llm_client=MagicMock(), auto_verify=False)
    task = TaskState(
        task_id="task-progress-guard-reset",
        goal=TaskGoal(description="Create index.html for FocusFlow."),
    )

    assert agent._progress_guard_message(
        task,
        "terminal",
        {"command": "git status"},
        tool_success=True,
    ) is None
    assert agent._progress_guard_message(
        task,
        "editor",
        {"action": "write_file", "path": "index.html"},
        tool_success=True,
    ) is None
    assert agent._no_progress_inspection_streak == 0
