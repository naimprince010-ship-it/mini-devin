"""Tests for reviewer gate enforcement (Phase 9C)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mini_devin.config.settings import AgentGatesSettings
from mini_devin.schemas.state import TaskStatus
from mini_devin.orchestrator.agent import Agent


class TestReviewerGateSettings:
    """Test reviewer gate settings."""
    
    def test_review_required_default(self):
        """Test review is required by default."""
        settings = AgentGatesSettings()
        assert settings.review_required is True
    
    def test_block_on_high_severity_default(self):
        """Test blocking on high severity is enabled by default."""
        settings = AgentGatesSettings()
        assert settings.block_on_high_severity is True
    
    def test_review_disabled(self):
        """Test review can be disabled."""
        settings = AgentGatesSettings(review_required=False)
        assert settings.review_required is False


class TestReviewerGateEnforcement:
    """Test reviewer gate enforcement in Agent."""
    
    @pytest.mark.asyncio
    async def test_reviewer_gate_skipped_when_disabled(
        self, mock_llm_client, gates_disabled
    ):
        """Test reviewer gate is skipped when review_required=False."""
        agent = Agent(
            llm_client=mock_llm_client,
            gates_settings=gates_disabled,
        )
        
        can_commit, reason = await agent._check_reviewer_gate()
        
        assert can_commit is True
        assert "not required" in reason.lower()
    
    @pytest.mark.asyncio
    async def test_reviewer_gate_passes_with_no_diff(
        self, mock_llm_client, gates_enabled
    ):
        """Test reviewer gate passes when there's no diff to review."""
        agent = Agent(
            llm_client=mock_llm_client,
            gates_settings=gates_enabled,
        )
        
        with patch.object(agent, 'get_reviewer_agent') as mock_get_reviewer:
            mock_get_reviewer.return_value = MagicMock()
            
            with patch.object(agent, '_get_git_manager') as mock_get_git:
                mock_git = MagicMock()
                mock_git.get_diff = AsyncMock(return_value=MagicMock(
                    success=True,
                    data={"diff": ""},
                ))
                mock_get_git.return_value = mock_git
                
                can_commit, reason = await agent._check_reviewer_gate()
                
                assert can_commit is True
                assert "no changes" in reason.lower()
    
    @pytest.mark.asyncio
    async def test_reviewer_gate_blocks_high_severity(
        self, mock_llm_client, gates_enabled, review_with_high_severity
    ):
        """Test reviewer gate blocks commit on high/critical findings."""
        agent = Agent(
            llm_client=mock_llm_client,
            gates_settings=gates_enabled,
        )
        
        with patch.object(agent, 'get_reviewer_agent') as mock_get_reviewer:
            mock_reviewer = MagicMock()
            mock_get_reviewer.return_value = mock_reviewer
            
            with patch.object(agent, '_get_git_manager') as mock_get_git:
                mock_git = MagicMock()
                mock_git.get_diff = AsyncMock(return_value=MagicMock(
                    success=True,
                    data={"diff": "some diff content"},
                ))
                mock_get_git.return_value = mock_git
                
                with patch.object(agent, 'review_before_commit', new_callable=AsyncMock) as mock_review:
                    mock_review.return_value = (False, review_with_high_severity)
                    
                    can_commit, reason = await agent._check_reviewer_gate()
                    
                    assert can_commit is False
                    assert "blocked" in reason.lower()
    
    @pytest.mark.asyncio
    async def test_reviewer_gate_allows_low_severity(
        self, mock_llm_client, gates_enabled, review_with_low_severity
    ):
        """Test reviewer gate allows commit with only low severity findings."""
        agent = Agent(
            llm_client=mock_llm_client,
            gates_settings=gates_enabled,
        )
        
        with patch.object(agent, 'get_reviewer_agent') as mock_get_reviewer:
            mock_reviewer = MagicMock()
            mock_get_reviewer.return_value = mock_reviewer
            
            with patch.object(agent, '_get_git_manager') as mock_get_git:
                mock_git = MagicMock()
                mock_git.get_diff = AsyncMock(return_value=MagicMock(
                    success=True,
                    data={"diff": "some diff content"},
                ))
                mock_get_git.return_value = mock_git
                
                with patch.object(agent, 'review_before_commit', new_callable=AsyncMock) as mock_review:
                    mock_review.return_value = (True, review_with_low_severity)
                    
                    can_commit, reason = await agent._check_reviewer_gate()
                    
                    assert can_commit is True
                    assert "passed" in reason.lower()
    
    @pytest.mark.asyncio
    async def test_reviewer_gate_non_blocking_when_disabled(
        self, mock_llm_client, review_with_high_severity
    ):
        """Test high severity findings are non-blocking when block_on_high_severity=False."""
        gates = AgentGatesSettings(
            planning_required=False,
            review_required=True,
            block_on_high_severity=False,
        )
        
        agent = Agent(
            llm_client=mock_llm_client,
            gates_settings=gates,
        )
        
        with patch.object(agent, 'get_reviewer_agent') as mock_get_reviewer:
            mock_reviewer = MagicMock()
            mock_get_reviewer.return_value = mock_reviewer
            
            with patch.object(agent, '_get_git_manager') as mock_get_git:
                mock_git = MagicMock()
                mock_git.get_diff = AsyncMock(return_value=MagicMock(
                    success=True,
                    data={"diff": "some diff content"},
                ))
                mock_get_git.return_value = mock_git
                
                with patch.object(agent, 'review_before_commit', new_callable=AsyncMock) as mock_review:
                    mock_review.return_value = (False, review_with_high_severity)
                    
                    can_commit, reason = await agent._check_reviewer_gate()
                    
                    assert can_commit is True
                    assert "non-blocking" in reason.lower()


class TestReviewerGateIntegration:
    """Integration tests for reviewer gate with task execution."""
    
    @pytest.mark.asyncio
    async def test_task_fails_on_reviewer_block(
        self, mock_llm_client, sample_task, sample_plan, review_with_high_severity
    ):
        """Test task fails when reviewer gate blocks commit."""
        gates = AgentGatesSettings(
            planning_required=True,
            review_required=True,
            block_on_high_severity=True,
            use_llm_planning=False,
        )
        
        agent = Agent(
            llm_client=mock_llm_client,
            gates_settings=gates,
        )
        
        with patch.object(agent, 'create_minimal_plan'):
            with patch.object(agent, 'get_planner_agent') as mock_get_planner:
                mock_planner = MagicMock()
                mock_planner.current_plan = sample_plan
                mock_get_planner.return_value = mock_planner
                
                with patch.object(agent, 'get_next_plan_step') as mock_get_step:
                    mock_get_step.return_value = None
                    
                    with patch.object(agent, '_check_reviewer_gate', new_callable=AsyncMock) as mock_gate:
                        mock_gate.return_value = (False, "Blocked: 2 high/critical findings")
                        
                        with patch.object(agent, '_finalize_task', new_callable=AsyncMock) as mock_finalize:
                            sample_task.status = TaskStatus.COMPLETED
                            mock_finalize.return_value = sample_task
                            
                            result = await agent.run(sample_task)
                            
                            assert result.status == TaskStatus.FAILED
                            assert "Reviewer gate blocked" in result.last_error
