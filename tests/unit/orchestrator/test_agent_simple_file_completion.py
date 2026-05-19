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
