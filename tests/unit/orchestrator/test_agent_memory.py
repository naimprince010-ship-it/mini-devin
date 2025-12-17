"""Unit tests for agent memory integration (Phase 18)."""

import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone

from mini_devin.orchestrator.agent import Agent
from mini_devin.schemas.state import TaskState, TaskGoal, TaskStatus
from mini_devin.memory.conversation_memory import (
    ConversationMemory,
    ConversationEntryType,
    Importance,
)


class TestAgentMemoryIntegration:
    """Tests for agent memory integration."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM client."""
        llm = MagicMock()
        llm.conversation = []
        llm.add_user_message = MagicMock()
        llm.add_assistant_message = MagicMock()
        llm.add_tool_result = MagicMock()
        llm.get_usage_stats = MagicMock(return_value={"total_tokens": 100})
        return llm
    
    @pytest.fixture
    def agent(self, mock_llm, tmp_path):
        """Create an agent with mock LLM and fresh memory."""
        agent = Agent(llm_client=mock_llm, verbose=False)
        agent._use_conversation_memory = True
        agent._conversation_memory = ConversationMemory(max_entries=100)
        return agent
    
    def test_conversation_memory_initialization(self, agent):
        """Test that conversation memory is initialized."""
        memory = agent.get_conversation_memory()
        
        assert memory is not None
        assert isinstance(memory, ConversationMemory)
        assert agent._conversation_memory is memory
    
    def test_enable_conversation_memory(self, agent):
        """Test enabling/disabling conversation memory."""
        assert agent._use_conversation_memory is True
        
        agent.enable_conversation_memory(False)
        assert agent._use_conversation_memory is False
        
        agent.enable_conversation_memory(True)
        assert agent._use_conversation_memory is True
    
    def test_get_context_from_memory_disabled(self, agent):
        """Test that context retrieval returns empty when disabled."""
        agent.enable_conversation_memory(False)
        
        context = agent.get_context_from_memory("Fix a bug")
        
        assert context == ""
    
    def test_get_context_from_memory_enabled(self, agent):
        """Test context retrieval when enabled."""
        memory = agent.get_conversation_memory()
        memory.add_lesson(
            lesson="Always check error logs first",
            context="Debugging",
            tags=["debugging"],
        )
        
        context = agent.get_context_from_memory("Debug an error")
        
        assert isinstance(context, str)
    
    def test_add_lesson_to_memory(self, agent):
        """Test adding a lesson to memory."""
        entry_id = agent.add_lesson_to_memory(
            lesson="Always run tests before committing",
            context="Before pushing code",
            importance="high",
            tags=["testing", "best-practice"],
        )
        
        assert entry_id != ""
        
        memory = agent.get_conversation_memory()
        entry = memory.get(entry_id)
        assert entry is not None
        assert entry.entry_type == ConversationEntryType.LESSON_LEARNED
        assert entry.importance == Importance.HIGH
    
    def test_add_lesson_disabled(self, agent):
        """Test that adding lesson returns empty when disabled."""
        agent.enable_conversation_memory(False)
        
        entry_id = agent.add_lesson_to_memory("Test lesson")
        
        assert entry_id == ""
    
    def test_add_error_pattern_to_memory(self, agent):
        """Test adding an error pattern to memory."""
        entry_id = agent.add_error_pattern_to_memory(
            error="ImportError: No module named 'numpy'",
            cause="Missing package",
            solution="Run pip install numpy",
            tags=["python", "dependencies"],
        )
        
        assert entry_id != ""
        
        memory = agent.get_conversation_memory()
        entry = memory.get(entry_id)
        assert entry is not None
        assert entry.entry_type == ConversationEntryType.ERROR_PATTERN
    
    def test_add_error_pattern_disabled(self, agent):
        """Test that adding error pattern returns empty when disabled."""
        agent.enable_conversation_memory(False)
        
        entry_id = agent.add_error_pattern_to_memory(
            error="Error",
            cause="cause",
            solution="fix",
        )
        
        assert entry_id == ""
    
    def test_add_solution_pattern_to_memory(self, agent):
        """Test adding a solution pattern to memory."""
        entry_id = agent.add_solution_pattern_to_memory(
            problem="Slow database queries",
            solution="Add an index to the column",
            code_example="CREATE INDEX idx_name ON table(column);",
            tags=["database", "performance"],
        )
        
        assert entry_id != ""
        
        memory = agent.get_conversation_memory()
        entry = memory.get(entry_id)
        assert entry is not None
        assert entry.entry_type == ConversationEntryType.SOLUTION_PATTERN
    
    def test_add_solution_pattern_disabled(self, agent):
        """Test that adding solution pattern returns empty when disabled."""
        agent.enable_conversation_memory(False)
        
        entry_id = agent.add_solution_pattern_to_memory(
            problem="problem",
            solution="solution",
        )
        
        assert entry_id == ""
    
    def test_get_error_solutions(self, agent):
        """Test getting error solutions from memory."""
        memory = agent.get_conversation_memory()
        memory.add_error_pattern(
            error="ModuleNotFoundError: numpy",
            cause="Missing package",
            solution="pip install numpy",
        )
        
        solutions = agent.get_error_solutions("ModuleNotFoundError: numpy")
        
        assert isinstance(solutions, list)
    
    def test_get_error_solutions_disabled(self, agent):
        """Test that getting error solutions returns empty when disabled."""
        agent.enable_conversation_memory(False)
        
        solutions = agent.get_error_solutions("Error")
        
        assert solutions == []
    
    def test_get_recent_lessons(self, agent):
        """Test getting recent lessons from memory."""
        memory = agent.get_conversation_memory()
        memory.add_lesson("Lesson 1", "context1")
        memory.add_lesson("Lesson 2", "context2")
        memory.add_lesson("Lesson 3", "context3")
        
        lessons = agent.get_recent_lessons(limit=2)
        
        assert isinstance(lessons, list)
        assert len(lessons) <= 2
    
    def test_get_recent_lessons_disabled(self, agent):
        """Test that getting recent lessons returns empty when disabled."""
        agent.enable_conversation_memory(False)
        
        lessons = agent.get_recent_lessons()
        
        assert lessons == []
    
    def test_search_memory(self, agent):
        """Test searching memory."""
        memory = agent.get_conversation_memory()
        memory.add_lesson("Python testing best practices", "context")
        
        results = agent.search_memory("python testing")
        
        assert isinstance(results, list)
    
    def test_search_memory_with_type_filter(self, agent):
        """Test searching memory with type filter."""
        memory = agent.get_conversation_memory()
        memory.add_lesson("Lesson content", "context")
        memory.add_error_pattern("Error", "Cause", "Solution")
        
        results = agent.search_memory("content", entry_type="lesson")
        
        assert isinstance(results, list)
        for result in results:
            assert result.get("entry_type") == "lesson_learned"
    
    def test_search_memory_disabled(self, agent):
        """Test that searching memory returns empty when disabled."""
        agent.enable_conversation_memory(False)
        
        results = agent.search_memory("query")
        
        assert results == []
    
    def test_save_task_summary(self, agent):
        """Test saving a task summary to memory."""
        task = TaskState(
            task_id="test-task-1",
            goal=TaskGoal(
                description="Fix a bug in the login system",
                acceptance_criteria=["Bug is fixed", "Tests pass"],
            ),
            status=TaskStatus.COMPLETED,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            commands_executed=["pytest", "git status"],
        )
        
        entry_id = agent.save_task_summary(
            task=task,
            summary="Fixed the login bug by updating the auth logic",
            lessons_learned=["Always check session expiry"],
        )
        
        assert entry_id != ""
    
    def test_save_task_summary_disabled(self, agent):
        """Test that saving task summary returns empty when disabled."""
        agent.enable_conversation_memory(False)
        
        task = TaskState(
            task_id="test-task-1",
            goal=TaskGoal(description="Test task"),
            status=TaskStatus.COMPLETED,
        )
        
        entry_id = agent.save_task_summary(task=task, summary="summary")
        
        assert entry_id == ""
    
    def test_memory_statistics_includes_conversation_memory(self, agent):
        """Test that memory statistics include conversation memory."""
        memory = agent.get_conversation_memory()
        memory.add_lesson("Test lesson", "context")
        
        stats = agent.get_memory_statistics()
        
        assert "conversation_memory" in stats
        assert stats["conversation_memory"]["total_entries"] == 1


class TestAgentMemoryImportance:
    """Tests for importance level mapping in agent memory."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM client."""
        llm = MagicMock()
        llm.conversation = []
        return llm
    
    @pytest.fixture
    def agent(self, mock_llm):
        """Create an agent with mock LLM and fresh memory."""
        agent = Agent(llm_client=mock_llm, verbose=False)
        agent._conversation_memory = ConversationMemory(max_entries=100)
        return agent
    
    def test_importance_low(self, agent):
        """Test low importance mapping."""
        entry_id = agent.add_lesson_to_memory("lesson", importance="low")
        entry = agent.get_conversation_memory().get(entry_id)
        assert entry.importance == Importance.LOW
    
    def test_importance_medium(self, agent):
        """Test medium importance mapping."""
        entry_id = agent.add_lesson_to_memory("lesson", importance="medium")
        entry = agent.get_conversation_memory().get(entry_id)
        assert entry.importance == Importance.MEDIUM
    
    def test_importance_high(self, agent):
        """Test high importance mapping."""
        entry_id = agent.add_lesson_to_memory("lesson", importance="high")
        entry = agent.get_conversation_memory().get(entry_id)
        assert entry.importance == Importance.HIGH
    
    def test_importance_critical(self, agent):
        """Test critical importance mapping."""
        entry_id = agent.add_lesson_to_memory("lesson", importance="critical")
        entry = agent.get_conversation_memory().get(entry_id)
        assert entry.importance == Importance.CRITICAL
    
    def test_importance_default(self, agent):
        """Test default importance mapping for unknown values."""
        entry_id = agent.add_lesson_to_memory("lesson", importance="unknown")
        entry = agent.get_conversation_memory().get(entry_id)
        assert entry.importance == Importance.MEDIUM
