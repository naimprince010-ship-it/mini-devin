"""
Integration tests for Mini-Devin Agent.

These tests validate that all components work together correctly
in real-world scenarios without mocking the core functionality.
"""

import asyncio
import tempfile
from pathlib import Path

import pytest

from mini_devin.schemas.tools import TerminalInput, ReadFileInput, WriteFileInput


class TestAgentInitialization:
    """Test that the agent initializes correctly with all components."""

    def test_agent_imports(self):
        """Verify all agent imports work correctly."""
        from mini_devin.orchestrator.agent import Agent, create_agent
        from mini_devin.schemas.state import AgentPhase, TaskStatus
        from mini_devin.core.llm_client import LLMClient  # noqa: F401
        from mini_devin.core.tool_interface import ToolRegistry  # noqa: F401
        
        assert Agent is not None
        assert create_agent is not None
        assert AgentPhase is not None
        assert TaskStatus is not None

    @pytest.mark.asyncio
    async def test_create_agent_with_defaults(self):
        """Test creating an agent with default settings."""
        from mini_devin.orchestrator.agent import create_agent
        
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = await create_agent(
                working_directory=tmpdir,
                api_key="test-key",
            )
            
            assert agent is not None
            assert agent.working_directory == tmpdir
            assert agent._enable_parallel_execution is True
            assert agent._use_conversation_memory is True

    @pytest.mark.asyncio
    async def test_agent_tool_registration(self):
        """Test that default tools are registered."""
        from mini_devin.orchestrator.agent import create_agent
        
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = await create_agent(
                working_directory=tmpdir,
                api_key="test-key",
            )
            
            assert agent.registry.get("terminal") is not None
            assert agent.registry.get("editor") is not None


class TestToolsIntegration:
    """Test that tools work correctly in isolation."""

    def test_terminal_tool_creation(self):
        """Test terminal tool can be created."""
        from mini_devin.tools.terminal import create_terminal_tool
        
        with tempfile.TemporaryDirectory() as tmpdir:
            terminal = create_terminal_tool(working_directory=tmpdir)
            assert terminal is not None
            assert terminal.name == "terminal"

    def test_editor_tool_creation(self):
        """Test editor tool can be created."""
        from mini_devin.tools.editor import create_editor_tool
        
        with tempfile.TemporaryDirectory() as tmpdir:
            editor = create_editor_tool(working_directory=tmpdir)
            assert editor is not None
            assert editor.name == "editor"

    @pytest.mark.asyncio
    async def test_terminal_execute_command(self):
        """Test terminal can execute a simple command."""
        from mini_devin.tools.terminal import create_terminal_tool
        
        with tempfile.TemporaryDirectory() as tmpdir:
            terminal = create_terminal_tool(working_directory=tmpdir)
            input_data = TerminalInput(command="echo 'hello world'")
            result = await terminal.execute(input_data)
            
            assert "hello world" in result.stdout

    @pytest.mark.asyncio
    async def test_editor_read_file(self):
        """Test editor can read a file."""
        from mini_devin.tools.editor import create_editor_tool
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("Hello, Mini-Devin!")
            
            editor = create_editor_tool(working_directory=tmpdir)
            input_data = ReadFileInput(path=str(test_file))
            result = await editor.execute(input_data)
            
            assert "Hello, Mini-Devin!" in result.content

    @pytest.mark.asyncio
    async def test_editor_write_file(self):
        """Test editor can write a file."""
        from mini_devin.tools.editor import create_editor_tool
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "new_file.txt"
            
            editor = create_editor_tool(working_directory=tmpdir)
            input_data = WriteFileInput(
                path=str(test_file),
                content="New content from Mini-Devin",
            )
            await editor._execute(input_data)
            
            assert test_file.exists()
            assert test_file.read_text() == "New content from Mini-Devin"


class TestMemoryIntegration:
    """Test memory components work correctly."""

    def test_conversation_memory_creation(self):
        """Test conversation memory can be created."""
        from mini_devin.memory import create_conversation_memory
        
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_file = Path(tmpdir) / "memory.json"
            memory = create_conversation_memory(storage_path=str(storage_file))
            assert memory is not None

    def test_conversation_memory_add_lesson(self):
        """Test adding a lesson to conversation memory."""
        from mini_devin.memory import create_conversation_memory, Importance
        
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_file = Path(tmpdir) / "memory.json"
            memory = create_conversation_memory(storage_path=str(storage_file))
            
            entry_id = memory.add_lesson(
                lesson="Always run tests before committing",
                context="Code review feedback",
                importance=Importance.HIGH,
            )
            
            assert entry_id is not None
            
            lessons = memory.get_recent_lessons(limit=10)
            assert len(lessons) >= 1

    def test_working_memory_creation(self):
        """Test working memory can be created."""
        from mini_devin.memory import create_working_memory
        
        memory = create_working_memory()
        assert memory is not None

    def test_symbol_index_creation(self):
        """Test symbol index can be created."""
        from mini_devin.memory import create_symbol_index
        
        with tempfile.TemporaryDirectory() as tmpdir:
            index = create_symbol_index(workspace_path=tmpdir)
            assert index is not None


class TestParallelExecutionIntegration:
    """Test parallel execution components."""

    def test_parallel_executor_creation(self):
        """Test parallel executor can be created."""
        from mini_devin.core.parallel_executor import create_parallel_executor
        
        async def mock_execute(tool_name: str, arguments: dict) -> str:
            return f"Executed {tool_name}"
        
        executor = create_parallel_executor(
            execute_fn=mock_execute,
            max_concurrent=5,
        )
        
        assert executor is not None

    def test_batch_caller_creation(self):
        """Test batch caller can be created."""
        from mini_devin.core.parallel_executor import create_batch_caller
        
        async def mock_execute(tool_name: str, arguments: dict) -> str:
            return f"Executed {tool_name}"
        
        caller = create_batch_caller(
            execute_fn=mock_execute,
            max_concurrent=5,
        )
        
        assert caller is not None

    @pytest.mark.asyncio
    async def test_parallel_executor_execute(self):
        """Test parallel executor can execute tool calls."""
        from mini_devin.core.parallel_executor import (
            create_parallel_executor,
            ToolCall,
        )
        
        async def mock_execute(tool_name: str, arguments: dict) -> str:
            await asyncio.sleep(0.01)
            return f"Executed {tool_name} with {arguments}"
        
        executor = create_parallel_executor(
            execute_fn=mock_execute,
            max_concurrent=5,
        )
        
        calls = [
            ToolCall.create(tool_name="tool1", arguments={"arg": "1"}),
            ToolCall.create(tool_name="tool2", arguments={"arg": "2"}),
            ToolCall.create(tool_name="tool3", arguments={"arg": "3"}),
        ]
        
        result = await executor.execute(calls)
        
        assert len(result.results) == 3
        assert all(r.success for r in result.results)


class TestAgentsIntegration:
    """Test specialized agents (Planner, Reviewer)."""

    def test_planner_agent_creation(self):
        """Test planner agent can be created."""
        from mini_devin.agents import create_planner_agent, PlannerAgent
        
        planner = create_planner_agent()
        assert planner is not None
        assert isinstance(planner, PlannerAgent)

    def test_reviewer_agent_creation(self):
        """Test reviewer agent can be created."""
        from mini_devin.agents import create_reviewer_agent, ReviewerAgent
        
        reviewer = create_reviewer_agent()
        assert reviewer is not None
        assert isinstance(reviewer, ReviewerAgent)


class TestSkillsIntegration:
    """Test skills framework."""

    def test_skill_registry_creation(self):
        """Test skill registry can be created."""
        from mini_devin.skills import SkillRegistry
        
        registry = SkillRegistry()
        assert registry is not None

    def test_builtin_skill_registration(self):
        """Test builtin skills can be registered."""
        from mini_devin.skills import SkillRegistry, AddEndpointSkill
        
        registry = SkillRegistry()
        
        registry.register(AddEndpointSkill)
        
        retrieved = registry.get("add_endpoint")
        assert retrieved is not None
        assert retrieved.name == "add_endpoint"


class TestSafetyIntegration:
    """Test safety guards."""

    def test_safety_guard_creation(self):
        """Test safety guard can be created."""
        from mini_devin.safety.guards import SafetyGuard, SafetyPolicy
        
        policy = SafetyPolicy()
        guard = SafetyGuard(policy=policy)
        
        assert guard is not None

    def test_safety_guard_blocks_dangerous_command(self):
        """Test safety guard blocks dangerous commands."""
        from mini_devin.safety.guards import SafetyGuard, SafetyPolicy
        
        policy = SafetyPolicy()
        guard = SafetyGuard(policy=policy)
        
        result = guard.check_command("rm -rf /")
        assert result is not None
        assert result.blocked is True

    def test_safety_guard_allows_safe_command(self):
        """Test safety guard allows safe commands."""
        from mini_devin.safety.guards import SafetyGuard, SafetyPolicy
        
        policy = SafetyPolicy()
        guard = SafetyGuard(policy=policy)
        
        result = guard.check_command("echo 'hello'")
        assert result is None or result.blocked is False


class TestVerificationIntegration:
    """Test verification components."""

    def test_verification_runner_creation(self):
        """Test verification runner can be created."""
        from mini_devin.verification.runner import VerificationRunner
        
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = VerificationRunner(working_directory=tmpdir)
            assert runner is not None


class TestAPIIntegration:
    """Test API components."""

    def test_api_app_creation(self):
        """Test FastAPI app can be created."""
        from mini_devin.api.app import app
        
        assert app is not None

    def test_api_routes_exist(self):
        """Test API routes are registered."""
        from mini_devin.api.app import app
        
        routes = [route.path for route in app.routes]
        
        assert "/api/health" in routes or "/health" in routes


class TestEndToEndScenarios:
    """End-to-end test scenarios."""

    @pytest.mark.asyncio
    async def test_agent_can_create_file(self):
        """Test agent can create a file using tools."""
        from mini_devin.orchestrator.agent import create_agent
        
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = await create_agent(
                working_directory=tmpdir,
                api_key="test-key",
            )
            
            editor = agent.registry.get("editor")
            test_file = Path(tmpdir) / "created_by_agent.txt"
            
            input_data = WriteFileInput(
                path=str(test_file),
                content="Created by Mini-Devin agent!",
            )
            await editor._execute(input_data)
            
            assert test_file.exists()
            assert "Created by Mini-Devin agent!" in test_file.read_text()

    @pytest.mark.asyncio
    async def test_agent_can_run_command(self):
        """Test agent can run a command using terminal tool directly."""
        from mini_devin.tools.terminal import create_terminal_tool
        
        with tempfile.TemporaryDirectory() as tmpdir:
            terminal = create_terminal_tool(working_directory=tmpdir)
            input_data = TerminalInput(command="echo 'Hello from Mini-Devin'")
            result = await terminal.execute(input_data)
            
            assert result.status.value == "success"
            assert "Hello from Mini-Devin" in result.stdout

    @pytest.mark.asyncio
    async def test_agent_memory_persistence(self):
        """Test agent memory persists across operations."""
        from mini_devin.orchestrator.agent import create_agent
        from mini_devin.memory import Importance
        
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = await create_agent(
                working_directory=tmpdir,
                api_key="test-key",
            )
            
            agent.add_lesson_to_memory(
                lesson="Test lesson for persistence",
                context="Integration test",
                importance=Importance.MEDIUM,
            )
            
            lessons = agent.get_recent_lessons(limit=10)
            assert any("Test lesson" in str(lesson) for lesson in lessons)

    @pytest.mark.asyncio
    async def test_agent_parallel_execution(self):
        """Test parallel executor can execute multiple tool calls."""
        from mini_devin.core.parallel_executor import (
            create_parallel_executor,
            ToolCall,
        )
        
        results_store = {}
        
        async def mock_execute(tool_name: str, arguments: dict) -> str:
            await asyncio.sleep(0.01)
            result = f"Executed {tool_name} with {arguments}"
            results_store[tool_name] = result
            return result
        
        executor = create_parallel_executor(
            execute_fn=mock_execute,
            max_concurrent=5,
        )
        
        calls = [
            ToolCall.create(tool_name="read_file_1", arguments={"path": "/file1.txt"}),
            ToolCall.create(tool_name="read_file_2", arguments={"path": "/file2.txt"}),
            ToolCall.create(tool_name="read_file_3", arguments={"path": "/file3.txt"}),
        ]
        
        result = await executor.execute(calls)
        
        assert len(result.results) == 3
        assert all(r.success for r in result.results)
        assert len(results_store) == 3


class TestConfigurationIntegration:
    """Test configuration components."""

    def test_settings_loading(self):
        """Test settings can be loaded."""
        from mini_devin.config.settings import Settings
        
        settings = Settings()
        assert settings is not None

    def test_safety_settings_loading(self):
        """Test safety settings can be loaded."""
        from mini_devin.config.settings import SafetySettings
        
        settings = SafetySettings()
        assert hasattr(settings, "max_iterations")
        assert hasattr(settings, "max_repair_iterations")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
