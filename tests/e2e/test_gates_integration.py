"""
End-to-End Tests for Planner + Reviewer Gates Integration (Phase 10).

Tests that the planner and reviewer gates work correctly in the
full agent execution flow.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mini_devin.orchestrator.agent import Agent
from mini_devin.schemas.state import (
    TaskState,
    TaskGoal,
    TaskStatus,
    PlanState,
    PlanStep,
    StepType,
    StepStatus,
)
from mini_devin.config.settings import AgentGatesSettings


class TestPlannerGateIntegration:
    """Integration tests for planner gate in full execution flow."""
    
    @pytest.mark.asyncio
    async def test_planning_gate_creates_plan_before_execution(
        self, mock_llm_client, temp_workspace
    ):
        """Test that planning gate creates a plan before task execution."""
        agent = Agent(
            llm_client=mock_llm_client,
            working_directory=temp_workspace,
            gates_settings=AgentGatesSettings(
                planning_required=True,
                use_llm_planning=False,
            ),
        )
        
        with patch.object(agent, 'create_minimal_plan') as mock_create_plan:
            with patch.object(agent, 'get_planner_agent') as mock_get_planner:
                mock_planner = MagicMock()
                mock_planner.current_plan = PlanState(
                    plan_id="test-plan",
                    task_description="Test task",
                    steps=[
                        PlanStep(
                            step_id="step-1",
                            order=1,
                            description="Test step",
                            step_type=StepType.EXPLORE,
                            status=StepStatus.PENDING,
                        )
                    ],
                )
                mock_get_planner.return_value = mock_planner
                
                with patch.object(agent, '_run_with_plan', new_callable=AsyncMock) as mock_run:
                    task = TaskState(
                        task_id="test-task",
                        goal=TaskGoal(description="Test task"),
                    )
                    mock_run.return_value = task
                    
                    await agent.run(task)
                    
                    mock_create_plan.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_planning_gate_enforces_max_steps(
        self, mock_llm_client, temp_workspace
    ):
        """Test that planning gate enforces max plan steps."""
        agent = Agent(
            llm_client=mock_llm_client,
            working_directory=temp_workspace,
            gates_settings=AgentGatesSettings(
                planning_required=True,
                max_plan_steps=3,
                use_llm_planning=False,
            ),
        )
        
        assert agent.gates.max_plan_steps == 3
    
    @pytest.mark.asyncio
    async def test_planning_disabled_uses_legacy_mode(
        self, mock_llm_client, temp_workspace
    ):
        """Test that disabling planning gate uses legacy execution mode."""
        agent = Agent(
            llm_client=mock_llm_client,
            working_directory=temp_workspace,
            gates_settings=AgentGatesSettings(
                planning_required=False,
                review_required=False,
            ),
        )
        
        with patch.object(agent, '_run_legacy', new_callable=AsyncMock) as mock_legacy:
            task = TaskState(
                task_id="test-task",
                goal=TaskGoal(description="Test task"),
            )
            mock_legacy.return_value = task
            
            await agent.run(task)
            
            mock_legacy.assert_called_once()


class TestReviewerGateIntegration:
    """Integration tests for reviewer gate in full execution flow."""
    
    @pytest.mark.asyncio
    async def test_reviewer_gate_called_on_completion(
        self, mock_llm_client, temp_workspace
    ):
        """Test that reviewer gate is called when task completes."""
        agent = Agent(
            llm_client=mock_llm_client,
            working_directory=temp_workspace,
            gates_settings=AgentGatesSettings(
                planning_required=True,
                review_required=True,
                use_llm_planning=False,
            ),
        )
        
        with patch.object(agent, 'create_minimal_plan'):
            with patch.object(agent, 'get_planner_agent') as mock_get_planner:
                mock_planner = MagicMock()
                mock_planner.current_plan = PlanState(
                    plan_id="test-plan",
                    task_description="Test task",
                    steps=[],
                )
                mock_get_planner.return_value = mock_planner
                
                with patch.object(agent, 'get_next_plan_step') as mock_get_step:
                    mock_get_step.return_value = None
                    
                    with patch.object(agent, '_check_reviewer_gate', new_callable=AsyncMock) as mock_gate:
                        mock_gate.return_value = (True, "Review passed")
                        
                        with patch.object(agent, '_finalize_task', new_callable=AsyncMock) as mock_finalize:
                            task = TaskState(
                                task_id="test-task",
                                goal=TaskGoal(description="Test task"),
                            )
                            task.status = TaskStatus.COMPLETED
                            mock_finalize.return_value = task
                            
                            await agent.run(task)
                            
                            mock_gate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_reviewer_gate_blocks_on_high_severity(
        self, mock_llm_client, temp_workspace
    ):
        """Test that reviewer gate blocks commit on high severity findings."""
        agent = Agent(
            llm_client=mock_llm_client,
            working_directory=temp_workspace,
            gates_settings=AgentGatesSettings(
                planning_required=True,
                review_required=True,
                block_on_high_severity=True,
                use_llm_planning=False,
            ),
        )
        
        with patch.object(agent, 'create_minimal_plan'):
            with patch.object(agent, 'get_planner_agent') as mock_get_planner:
                mock_planner = MagicMock()
                mock_planner.current_plan = PlanState(
                    plan_id="test-plan",
                    task_description="Test task",
                    steps=[],
                )
                mock_get_planner.return_value = mock_planner
                
                with patch.object(agent, 'get_next_plan_step') as mock_get_step:
                    mock_get_step.return_value = None
                    
                    with patch.object(agent, '_check_reviewer_gate', new_callable=AsyncMock) as mock_gate:
                        mock_gate.return_value = (False, "Blocked: 2 high/critical findings")
                        
                        with patch.object(agent, '_finalize_task', new_callable=AsyncMock) as mock_finalize:
                            task = TaskState(
                                task_id="test-task",
                                goal=TaskGoal(description="Test task"),
                            )
                            task.status = TaskStatus.COMPLETED
                            mock_finalize.return_value = task
                            
                            result = await agent.run(task)
                            
                            assert result.status == TaskStatus.FAILED
                            assert "Reviewer gate blocked" in result.last_error


class TestGatesConfigurationIntegration:
    """Integration tests for gates configuration."""
    
    def test_gates_default_configuration(self, mock_llm_client, temp_workspace):
        """Test that gates have correct default configuration."""
        agent = Agent(
            llm_client=mock_llm_client,
            working_directory=temp_workspace,
        )
        
        assert agent.gates.planning_required is True
        assert agent.gates.max_plan_steps == 5
        assert agent.gates.review_required is True
        assert agent.gates.block_on_high_severity is True
        assert agent.gates.use_llm_planning is True
    
    def test_gates_custom_configuration(self, mock_llm_client, temp_workspace):
        """Test that gates accept custom configuration."""
        custom_gates = AgentGatesSettings(
            planning_required=False,
            max_plan_steps=10,
            review_required=False,
            block_on_high_severity=False,
            use_llm_planning=False,
        )
        
        agent = Agent(
            llm_client=mock_llm_client,
            working_directory=temp_workspace,
            gates_settings=custom_gates,
        )
        
        assert agent.gates.planning_required is False
        assert agent.gates.max_plan_steps == 10
        assert agent.gates.review_required is False
        assert agent.gates.block_on_high_severity is False
        assert agent.gates.use_llm_planning is False
    
    def test_gates_from_environment(self, mock_llm_client, temp_workspace):
        """Test that gates can be configured from environment variables."""
        import os
        
        with patch.dict(os.environ, {
            "PLANNING_REQUIRED": "false",
            "MAX_PLAN_STEPS": "3",
            "REVIEW_REQUIRED": "false",
        }):
            gates = AgentGatesSettings.from_env()
            
            assert gates.planning_required is False
            assert gates.max_plan_steps == 3
            assert gates.review_required is False


class TestStepByStepExecution:
    """Integration tests for step-by-step execution with gates."""
    
    @pytest.mark.asyncio
    async def test_step_execution_marks_steps_complete(
        self, mock_llm_client, temp_workspace
    ):
        """Test that step execution marks steps as complete."""
        agent = Agent(
            llm_client=mock_llm_client,
            working_directory=temp_workspace,
            gates_settings=AgentGatesSettings(
                planning_required=True,
                review_required=False,
                use_llm_planning=False,
            ),
        )
        
        mock_response = MagicMock()
        mock_response.tool_calls = None
        mock_response.content = "Step completed successfully"
        mock_response.finish_reason = "stop"
        mock_llm_client.complete.return_value = mock_response
        
        success, result = await agent._execute_plan_step("Test step description")
        
        assert success is True
        assert "completed" in result.lower() or "successfully" in result.lower()
    
    @pytest.mark.asyncio
    async def test_verification_runs_after_implementation_steps(
        self, mock_llm_client, temp_workspace
    ):
        """Test that verification runs after implementation steps."""
        agent = Agent(
            llm_client=mock_llm_client,
            working_directory=temp_workspace,
            gates_settings=AgentGatesSettings(
                planning_required=True,
                review_required=False,
                use_llm_planning=False,
            ),
            auto_verify=False,
        )
        
        passed, summary = await agent._run_verification_step()
        
        assert passed is True
        assert "skipped" in summary.lower()
