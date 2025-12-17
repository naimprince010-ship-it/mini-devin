"""Unit tests for the planner agent."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mini_devin.agents.planner import (
    PlannerAgent,
    PlanningStrategy,
    PlanQuality,
    PlanValidationResult,
    TaskAnalysis,
    PlanningResult,
)
from mini_devin.schemas.state import PlanStep, PlanState, TaskState, TaskGoal


class TestPlanningStrategy:
    """Tests for PlanningStrategy enum."""

    def test_strategy_values(self):
        """Test that all strategy values exist."""
        assert PlanningStrategy.WATERFALL is not None
        assert PlanningStrategy.ITERATIVE is not None
        assert PlanningStrategy.EXPLORATORY is not None
        assert PlanningStrategy.TEST_DRIVEN is not None
        assert PlanningStrategy.MINIMAL is not None


class TestPlanQuality:
    """Tests for PlanQuality enum."""

    def test_quality_values(self):
        """Test that all quality values exist."""
        assert PlanQuality.EXCELLENT is not None
        assert PlanQuality.GOOD is not None
        assert PlanQuality.ACCEPTABLE is not None
        assert PlanQuality.NEEDS_WORK is not None
        assert PlanQuality.POOR is not None


class TestPlanValidationResult:
    """Tests for PlanValidationResult dataclass."""

    def test_validation_result_creation(self):
        """Test creating a PlanValidationResult."""
        result = PlanValidationResult(
            is_valid=True,
            quality=PlanQuality.GOOD,
            issues=[],
            suggestions=["Consider adding tests"],
            estimated_steps=5,
            estimated_complexity="medium",
        )
        assert result.is_valid is True
        assert result.quality == PlanQuality.GOOD
        assert len(result.issues) == 0
        assert len(result.suggestions) == 1

    def test_validation_result_to_dict(self):
        """Test converting validation result to dict."""
        result = PlanValidationResult(
            is_valid=False,
            quality=PlanQuality.NEEDS_WORK,
            issues=["Missing steps"],
        )
        d = result.to_dict()
        assert d["is_valid"] is False
        assert d["quality"] == "needs_work"
        assert "Missing steps" in d["issues"]


class TestPlannerAgent:
    """Tests for PlannerAgent class."""

    def test_planner_initialization(self):
        """Test PlannerAgent initialization."""
        planner = PlannerAgent()
        assert planner is not None
        assert planner.max_steps == 50
        assert planner.include_verification_steps is True

    def test_planner_with_custom_settings(self):
        """Test PlannerAgent with custom settings."""
        planner = PlannerAgent(
            max_steps=20,
            include_verification_steps=False,
            default_strategy=PlanningStrategy.MINIMAL,
        )
        assert planner.max_steps == 20
        assert planner.include_verification_steps is False
        assert planner.default_strategy == PlanningStrategy.MINIMAL

    def test_planner_with_llm_client(self):
        """Test PlannerAgent with LLM client."""
        mock_client = MagicMock()
        planner = PlannerAgent(llm_client=mock_client)
        assert planner.llm == mock_client

    def test_create_minimal_plan(self):
        """Test creating a minimal plan without LLM."""
        planner = PlannerAgent()
        task = TaskState(
            task_id="test-task-1",
            goal=TaskGoal(description="Fix the bug in main.py"),
        )
        plan = planner.create_minimal_plan(task)
        
        assert plan is not None
        assert plan.task_id == "test-task-1"
        assert len(plan.steps) >= 1
        assert len(plan.milestones) >= 1

    def test_create_minimal_plan_has_exploration_step(self):
        """Test that minimal plan includes exploration step."""
        planner = PlannerAgent()
        task = TaskState(
            task_id="test-task-2",
            goal=TaskGoal(description="Add a new feature"),
        )
        plan = planner.create_minimal_plan(task)
        
        # Should have an exploration/understanding step
        descriptions = [s.description.lower() for s in plan.steps]
        has_exploration = any("understand" in d or "explore" in d for d in descriptions)
        assert has_exploration

    def test_get_next_step(self):
        """Test getting next step from plan."""
        planner = PlannerAgent()
        task = TaskState(
            task_id="test-task-3",
            goal=TaskGoal(description="Test task"),
        )
        plan = planner.create_minimal_plan(task)
        
        next_step = planner.get_next_step(plan)
        assert next_step is not None
        assert next_step.step_id is not None

    def test_mark_step_complete(self):
        """Test marking a step as complete."""
        from mini_devin.schemas.state import StepStatus
        
        planner = PlannerAgent()
        task = TaskState(
            task_id="test-task-4",
            goal=TaskGoal(description="Test task"),
        )
        plan = planner.create_minimal_plan(task)
        
        first_step = plan.steps[0]
        updated_plan = planner.mark_step_complete(plan, first_step.step_id, "Done")
        
        completed_step = next(s for s in updated_plan.steps if s.step_id == first_step.step_id)
        assert completed_step.status == StepStatus.COMPLETED

    def test_mark_step_failed(self):
        """Test marking a step as failed."""
        from mini_devin.schemas.state import StepStatus
        
        planner = PlannerAgent()
        task = TaskState(
            task_id="test-task-5",
            goal=TaskGoal(description="Test task"),
        )
        plan = planner.create_minimal_plan(task)
        
        first_step = plan.steps[0]
        updated_plan = planner.mark_step_failed(plan, first_step.step_id, "Error occurred")
        
        failed_step = next(s for s in updated_plan.steps if s.step_id == first_step.step_id)
        assert failed_step.status == StepStatus.FAILED
        assert failed_step.error == "Error occurred"

    def test_get_plan_progress(self):
        """Test getting plan progress statistics."""
        planner = PlannerAgent()
        task = TaskState(
            task_id="test-task-6",
            goal=TaskGoal(description="Test task"),
        )
        plan = planner.create_minimal_plan(task)
        
        progress = planner.get_plan_progress(plan)
        
        assert "total_steps" in progress
        assert "completed_steps" in progress
        assert "progress_percent" in progress
        assert progress["total_steps"] == len(plan.steps)

    def test_validate_plan(self):
        """Test plan validation."""
        planner = PlannerAgent()
        task = TaskState(
            task_id="test-task-7",
            goal=TaskGoal(description="Test task"),
        )
        plan = planner.create_minimal_plan(task)
        
        validation = planner.validate_plan(plan, task)
        
        assert validation is not None
        assert isinstance(validation.is_valid, bool)
        assert isinstance(validation.quality, PlanQuality)


class TestPlannerAgentValidation:
    """Validation tests for PlannerAgent."""

    def test_validate_empty_plan(self):
        """Test validating an empty plan."""
        planner = PlannerAgent()
        task = TaskState(
            task_id="test-task-8",
            goal=TaskGoal(description="Test task"),
        )
        
        empty_plan = PlanState(
            plan_id="empty-plan",
            task_id="test-task-8",
            milestones=[],
            steps=[],
        )
        
        validation = planner.validate_plan(empty_plan, task)
        
        assert validation.is_valid is False
        assert "no steps" in validation.issues[0].lower()

    def test_validate_plan_with_too_many_steps(self):
        """Test validating a plan with too many steps."""
        planner = PlannerAgent(max_steps=3)
        task = TaskState(
            task_id="test-task-9",
            goal=TaskGoal(description="Test task"),
        )
        
        # Create a plan with more steps than allowed
        steps = [
            PlanStep(
                step_id=f"step-{i}",
                description=f"Step {i}",
                expected_outcome=f"Outcome {i}",
            )
            for i in range(5)
        ]
        
        large_plan = PlanState(
            plan_id="large-plan",
            task_id="test-task-9",
            milestones=[],
            steps=steps,
        )
        
        validation = planner.validate_plan(large_plan, task)
        
        assert validation.is_valid is False
        assert any("too many steps" in issue.lower() for issue in validation.issues)
