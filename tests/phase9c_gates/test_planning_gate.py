"""Tests for planning gate enforcement (Phase 9C)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mini_devin.config.settings import AgentGatesSettings
from mini_devin.schemas.state import (
    TaskStatus,
    PlanState,
    PlanStep,
    StepType,
    StepStatus,
)
from mini_devin.orchestrator.agent import Agent


class TestPlanningGateSettings:
    """Test AgentGatesSettings configuration."""
    
    def test_default_settings(self):
        """Test default gates settings have planning required."""
        settings = AgentGatesSettings()
        assert settings.planning_required is True
        assert settings.max_plan_steps == 5
        assert settings.review_required is True
        assert settings.block_on_high_severity is True
        assert settings.use_llm_planning is True
    
    def test_custom_settings(self):
        """Test custom gates settings."""
        settings = AgentGatesSettings(
            planning_required=False,
            max_plan_steps=10,
            review_required=False,
            block_on_high_severity=False,
            use_llm_planning=False,
        )
        assert settings.planning_required is False
        assert settings.max_plan_steps == 10
        assert settings.review_required is False
        assert settings.block_on_high_severity is False
        assert settings.use_llm_planning is False
    
    def test_from_env(self):
        """Test loading settings from environment variables."""
        import os
        
        with patch.dict(os.environ, {
            "PLANNING_REQUIRED": "false",
            "MAX_PLAN_STEPS": "3",
            "REVIEW_REQUIRED": "false",
            "BLOCK_ON_HIGH_SEVERITY": "false",
            "USE_LLM_PLANNING": "false",
        }):
            settings = AgentGatesSettings.from_env()
            assert settings.planning_required is False
            assert settings.max_plan_steps == 3
            assert settings.review_required is False
            assert settings.block_on_high_severity is False
            assert settings.use_llm_planning is False


class TestPlanningGateEnforcement:
    """Test planning gate enforcement in Agent.run()."""
    
    def test_agent_accepts_gates_settings(self, mock_llm_client, gates_enabled):
        """Test Agent accepts gates_settings parameter."""
        agent = Agent(
            llm_client=mock_llm_client,
            gates_settings=gates_enabled,
        )
        assert agent.gates.planning_required is True
        assert agent.gates.max_plan_steps == 5
    
    def test_agent_default_gates_settings(self, mock_llm_client):
        """Test Agent uses default gates settings when not provided."""
        agent = Agent(llm_client=mock_llm_client)
        assert agent.gates.planning_required is True
        assert agent.gates.review_required is True
    
    @pytest.mark.asyncio
    async def test_planning_required_creates_plan(
        self, mock_llm_client, gates_enabled, sample_task
    ):
        """Test that planning gate creates a plan before execution."""
        agent = Agent(
            llm_client=mock_llm_client,
            gates_settings=gates_enabled,
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
                    mock_run.return_value = sample_task
                    
                    await agent.run(sample_task)
                    
                    mock_create_plan.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_planning_disabled_skips_plan(
        self, mock_llm_client, gates_disabled, sample_task
    ):
        """Test that planning gate is skipped when disabled."""
        agent = Agent(
            llm_client=mock_llm_client,
            gates_settings=gates_disabled,
        )
        
        with patch.object(agent, 'create_minimal_plan') as mock_create_plan:
            with patch.object(agent, '_run_legacy', new_callable=AsyncMock) as mock_run:
                mock_run.return_value = sample_task
                
                await agent.run(sample_task)
                
                mock_create_plan.assert_not_called()
                mock_run.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_planning_failure_blocks_execution(
        self, mock_llm_client, gates_enabled, sample_task
    ):
        """Test that execution is blocked when planning fails."""
        agent = Agent(
            llm_client=mock_llm_client,
            gates_settings=gates_enabled,
        )
        
        with patch.object(agent, 'create_minimal_plan') as mock_create_plan:
            mock_create_plan.return_value = None
            
            with patch.object(agent, 'get_planner_agent') as mock_get_planner:
                mock_get_planner.return_value = None
                
                result = await agent.run(sample_task)
                
                assert result.status == TaskStatus.FAILED
                assert "Planning gate failed" in result.last_error


class TestMaxPlanStepsEnforcement:
    """Test max plan steps enforcement."""
    
    def test_max_plan_steps_default(self):
        """Test default max plan steps is 5."""
        settings = AgentGatesSettings()
        assert settings.max_plan_steps == 5
    
    def test_max_plan_steps_custom(self):
        """Test custom max plan steps."""
        settings = AgentGatesSettings(max_plan_steps=10)
        assert settings.max_plan_steps == 10


class TestStepByStepExecution:
    """Test step-by-step execution with plan."""
    
    @pytest.mark.asyncio
    async def test_execute_plan_step(self, mock_llm_client, gates_enabled):
        """Test executing a single plan step."""
        agent = Agent(
            llm_client=mock_llm_client,
            gates_settings=gates_enabled,
        )
        
        mock_response = MagicMock()
        mock_response.tool_calls = None
        mock_response.content = "Step completed successfully"
        mock_response.finish_reason = "stop"
        mock_llm_client.complete.return_value = mock_response
        
        success, result = await agent._execute_plan_step("Read the test file")
        
        assert success is True
        assert "Step completed" in result or "successfully" in result.lower()
    
    @pytest.mark.asyncio
    async def test_run_verification_step(self, mock_llm_client, gates_enabled):
        """Test running verification after implementation step."""
        agent = Agent(
            llm_client=mock_llm_client,
            gates_settings=gates_enabled,
            auto_verify=False,
        )
        
        passed, summary = await agent._run_verification_step()
        
        assert passed is True
        assert "skipped" in summary.lower()
