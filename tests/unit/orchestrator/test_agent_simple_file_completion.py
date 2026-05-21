"""Tests for simple file task completion helpers."""

import asyncio
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
        for _ in range(2)
    ]

    assert messages[0] is None
    assert messages[1] is not None
    assert "Stop listing directories" in messages[1]
    assert "Create index.html" in messages[1]


def test_progress_guard_respects_no_edit_verification_task():
    agent = Agent(llm_client=MagicMock(), auto_verify=False)
    task = TaskState(
        task_id="task-no-edit-progress-guard",
        goal=TaskGoal(
            description=(
                "Inspect repo, run git status, then run python -m pytest tests/test_simple.py. "
                "Do not edit files."
            )
        ),
    )

    assert agent._progress_guard_message(
        task,
        "terminal",
        {"command": "git status"},
        tool_success=True,
    ) is None
    message = agent._progress_guard_message(
        task,
        "terminal",
        {"command": "git status"},
        tool_success=True,
    )

    assert message is not None
    assert "Do not edit files" in message
    assert "create or edit" not in message.lower()


def test_no_edit_verification_satisfied_after_pytest_passes():
    agent = Agent(llm_client=MagicMock(), auto_verify=False)
    task = TaskState(
        task_id="task-no-edit-verification",
        goal=TaskGoal(
            description=(
                "Reliability smoke test. Inspect this repository, run git status, "
                "then run python -m pytest tests/test_simple.py. Do not edit files."
            )
        ),
    )

    assert agent._no_edit_verification_satisfied(
        task,
        "terminal",
        {"command": "python -m pytest tests/test_simple.py"},
        "1 passed\n\nExit code: 0",
    )


def test_terminal_task_complete_echo_satisfies_completion():
    agent = Agent(llm_client=MagicMock(), auto_verify=False)

    assert agent._terminal_task_complete_satisfied(
        "terminal",
        "STDOUT:\nTASK COMPLETE: tests passed\n\nExit code: 0",
    )


def test_clone_status_verification_satisfied_after_clean_git_status():
    agent = Agent(llm_client=MagicMock(), auto_verify=False)
    task = TaskState(
        task_id="task-clone-status",
        goal=TaskGoal(
            description=(
                "Clone this public GitHub repository into the current workspace: "
                "https://github.com/pypa/sampleproject. After cloning, run git status "
                "inside the cloned repository and finish with TASK COMPLETE."
            )
        ),
    )

    assert agent._clone_status_verification_satisfied(
        task,
        "terminal",
        {"command": "git status", "working_directory": "./sampleproject"},
        "On branch main\nYour branch is up to date with 'origin/main'.\n\n"
        "nothing to commit, working tree clean\n\nExit code: 0",
    )


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


def test_malformed_editor_call_gets_actionable_error():
    agent = Agent(llm_client=MagicMock(), auto_verify=False)

    result = asyncio.run(agent._execute_tool("editor", {}))

    assert "malformed editor tool call" in result
    assert "write_file" in result
    assert "path" in result


def test_browser_open_emits_visible_browser_event():
    seen = []
    agent = Agent(
        llm_client=MagicMock(),
        auto_verify=False,
        callbacks={"on_browser_event": lambda payload: seen.append(payload)},
    )

    result = asyncio.run(
        agent._execute_tool(
            "browser_open",
            {"url": "https://halalzi.com", "note": "Open halalzi"},
        )
    )

    assert "Opened URL" in result
    assert seen == [
        {
            "event_type": "navigate",
            "url": "https://halalzi.com",
            "query": "Open halalzi",
        }
    ]
