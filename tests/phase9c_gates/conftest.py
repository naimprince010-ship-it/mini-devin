"""Shared fixtures for Phase 9C tests."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from mini_devin.config.settings import AgentGatesSettings
from mini_devin.schemas.state import (
    TaskState,
    TaskGoal,
    PlanState,
    PlanStep,
    StepType,
    StepStatus,
)
from mini_devin.agents.reviewer import ReviewFeedback, ReviewComment, ReviewSeverity


@pytest.fixture
def gates_enabled():
    """Gates settings with all gates enabled (default)."""
    return AgentGatesSettings(
        planning_required=True,
        max_plan_steps=5,
        review_required=True,
        block_on_high_severity=True,
        use_llm_planning=False,
    )


@pytest.fixture
def gates_disabled():
    """Gates settings with all gates disabled."""
    return AgentGatesSettings(
        planning_required=False,
        max_plan_steps=5,
        review_required=False,
        block_on_high_severity=False,
        use_llm_planning=False,
    )


@pytest.fixture
def sample_task():
    """Sample task for testing."""
    return TaskState(
        task_id="test-task-001",
        goal=TaskGoal(
            description="Fix the failing test in test_utils.py",
            acceptance_criteria=["All tests pass", "No lint errors"],
        ),
    )


@pytest.fixture
def sample_plan():
    """Sample plan with steps."""
    return PlanState(
        plan_id="test-plan-001",
        task_description="Fix the failing test",
        steps=[
            PlanStep(
                step_id="step-1",
                order=1,
                description="Read the failing test file",
                step_type=StepType.EXPLORE,
                status=StepStatus.PENDING,
            ),
            PlanStep(
                step_id="step-2",
                order=2,
                description="Identify the bug causing the failure",
                step_type=StepType.ANALYZE,
                status=StepStatus.PENDING,
            ),
            PlanStep(
                step_id="step-3",
                order=3,
                description="Fix the bug in the code",
                step_type=StepType.IMPLEMENT,
                status=StepStatus.PENDING,
            ),
            PlanStep(
                step_id="step-4",
                order=4,
                description="Run tests to verify the fix",
                step_type=StepType.VERIFY,
                status=StepStatus.PENDING,
            ),
        ],
    )


@pytest.fixture
def review_with_high_severity():
    """Review feedback with high severity findings."""
    return ReviewFeedback(
        overall_assessment="Changes have critical issues",
        should_commit=False,
        confidence=0.9,
        comments=[
            ReviewComment(
                file_path="src/utils.py",
                line_number=42,
                severity=ReviewSeverity.HIGH,
                category="security",
                message="Potential SQL injection vulnerability",
                suggestion="Use parameterized queries",
            ),
            ReviewComment(
                file_path="src/utils.py",
                line_number=50,
                severity=ReviewSeverity.CRITICAL,
                category="security",
                message="Hardcoded credentials detected",
                suggestion="Use environment variables",
            ),
        ],
        patch_improvements=[],
    )


@pytest.fixture
def review_with_low_severity():
    """Review feedback with only low severity findings."""
    return ReviewFeedback(
        overall_assessment="Changes look good with minor suggestions",
        should_commit=True,
        confidence=0.85,
        comments=[
            ReviewComment(
                file_path="src/utils.py",
                line_number=42,
                severity=ReviewSeverity.LOW,
                category="style",
                message="Consider using more descriptive variable name",
                suggestion="Rename 'x' to 'user_count'",
            ),
            ReviewComment(
                file_path="src/utils.py",
                line_number=50,
                severity=ReviewSeverity.INFO,
                category="documentation",
                message="Missing docstring",
                suggestion="Add docstring explaining the function",
            ),
        ],
        patch_improvements=[],
    )


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    client = MagicMock()
    client.complete = AsyncMock()
    client.add_user_message = MagicMock()
    client.add_assistant_message = MagicMock()
    client.add_tool_result = MagicMock()
    client.set_system_prompt = MagicMock()
    client.get_usage_stats = MagicMock(return_value={"total_tokens": 1000})
    client.conversation = []
    return client
