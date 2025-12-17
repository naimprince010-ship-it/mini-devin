"""Unit tests for agent parallel execution integration (Phase 19)."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from mini_devin.orchestrator.agent import Agent
from mini_devin.core.parallel_executor import (
    ParallelExecutor,
    BatchToolCaller,
    ParallelExecutionResult,
    ToolCallResult,
)


class TestAgentParallelExecution:
    """Tests for agent parallel execution integration."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM client."""
        llm = MagicMock()
        llm.conversation = []
        return llm
    
    @pytest.fixture
    def agent(self, mock_llm):
        """Create an agent with mock LLM."""
        return Agent(llm_client=mock_llm, verbose=False)
    
    def test_parallel_execution_enabled_by_default(self, agent):
        """Test that parallel execution is enabled by default."""
        assert agent._enable_parallel_execution is True
        assert agent._max_parallel_tools == 5
    
    def test_parallel_execution_can_be_disabled(self, agent):
        """Test that parallel execution can be disabled."""
        agent.enable_parallel_execution(False)
        assert agent._enable_parallel_execution is False
        
        agent.enable_parallel_execution(True)
        assert agent._enable_parallel_execution is True
    
    def test_set_max_parallel_tools(self, agent):
        """Test setting max parallel tools."""
        agent.set_max_parallel_tools(10)
        assert agent._max_parallel_tools == 10
        assert agent._parallel_executor is None
        assert agent._batch_caller is None
    
    def test_get_parallel_executor(self, agent):
        """Test getting parallel executor."""
        executor = agent.get_parallel_executor()
        
        assert executor is not None
        assert isinstance(executor, ParallelExecutor)
        assert agent._parallel_executor is executor
        
        executor2 = agent.get_parallel_executor()
        assert executor2 is executor
    
    def test_get_batch_caller(self, agent):
        """Test getting batch caller."""
        caller = agent.get_batch_caller()
        
        assert caller is not None
        assert isinstance(caller, BatchToolCaller)
        assert agent._batch_caller is caller
        
        caller2 = agent.get_batch_caller()
        assert caller2 is caller
    
    def test_get_parallel_execution_stats(self, agent):
        """Test getting parallel execution stats."""
        stats = agent.get_parallel_execution_stats()
        
        assert stats["enabled"] is True
        assert stats["max_parallel_tools"] == 5
        assert stats["executor_initialized"] is False
        assert stats["batch_caller_initialized"] is False
        
        agent.get_parallel_executor()
        stats = agent.get_parallel_execution_stats()
        assert stats["executor_initialized"] is True
    
    def test_parallel_execution_disabled_in_constructor(self, mock_llm):
        """Test creating agent with parallel execution disabled."""
        agent = Agent(
            llm_client=mock_llm,
            verbose=False,
            enable_parallel_execution=False,
            max_parallel_tools=3,
        )
        
        assert agent._enable_parallel_execution is False
        assert agent._max_parallel_tools == 3


class TestAgentParallelExecutionAsync:
    """Async tests for agent parallel execution."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM client."""
        llm = MagicMock()
        llm.conversation = []
        return llm
    
    @pytest.fixture
    def agent(self, mock_llm):
        """Create an agent with mock LLM."""
        return Agent(llm_client=mock_llm, verbose=False)
    
    @pytest.mark.asyncio
    async def test_execute_tools_parallel_single_call(self, agent):
        """Test executing a single tool call (no parallelization)."""
        agent._execute_tool = AsyncMock(return_value="result1")
        
        tool_calls = [
            {"name": "editor", "arguments": {"action": "read_file", "path": "/test.py"}},
        ]
        
        result = await agent.execute_tools_parallel(tool_calls)
        
        assert isinstance(result, ParallelExecutionResult)
        assert len(result.results) == 1
        assert result.results[0].success is True
        assert result.results[0].result == "result1"
    
    @pytest.mark.asyncio
    async def test_execute_tools_parallel_disabled(self, agent):
        """Test executing tools with parallelization disabled."""
        agent.enable_parallel_execution(False)
        agent._execute_tool = AsyncMock(side_effect=["result1", "result2"])
        
        tool_calls = [
            {"name": "editor", "arguments": {"action": "read_file", "path": "/test1.py"}},
            {"name": "editor", "arguments": {"action": "read_file", "path": "/test2.py"}},
        ]
        
        result = await agent.execute_tools_parallel(tool_calls)
        
        assert isinstance(result, ParallelExecutionResult)
        assert len(result.results) == 2
        assert result.parallel_speedup == 1.0
    
    @pytest.mark.asyncio
    async def test_execute_tools_parallel_with_error(self, agent):
        """Test executing tools with an error."""
        agent.enable_parallel_execution(False)
        agent._execute_tool = AsyncMock(side_effect=Exception("Test error"))
        
        tool_calls = [
            {"name": "editor", "arguments": {"action": "read_file", "path": "/test.py"}},
        ]
        
        result = await agent.execute_tools_parallel(tool_calls)
        
        assert len(result.results) == 1
        assert result.results[0].success is False
        assert "Test error" in result.results[0].error
    
    @pytest.mark.asyncio
    async def test_batch_read_files(self, agent):
        """Test batch reading files."""
        agent._execute_tool = AsyncMock(side_effect=[
            "content1",
            "content2",
            "content3",
        ])
        agent.enable_parallel_execution(False)
        
        paths = ["/file1.py", "/file2.py", "/file3.py"]
        contents = await agent.batch_read_files(paths)
        
        assert len(contents) == 3
        assert contents["/file1.py"] == "content1"
        assert contents["/file2.py"] == "content2"
        assert contents["/file3.py"] == "content3"
    
    @pytest.mark.asyncio
    async def test_batch_read_files_with_error(self, agent):
        """Test batch reading files with an error."""
        agent._execute_tool = AsyncMock(side_effect=[
            "content1",
            "Error: File not found",
        ])
        agent.enable_parallel_execution(False)
        
        paths = ["/file1.py", "/file2.py"]
        contents = await agent.batch_read_files(paths)
        
        assert contents["/file1.py"] == "content1"
        assert "Error" in contents["/file2.py"]
    
    @pytest.mark.asyncio
    async def test_batch_search(self, agent):
        """Test batch searching."""
        agent._execute_tool = AsyncMock(side_effect=[
            "match1: line 10",
            "match2: line 20",
        ])
        agent.enable_parallel_execution(False)
        
        patterns = ["pattern1", "pattern2"]
        results = await agent.batch_search(patterns, "/src")
        
        assert len(results) == 2
        assert "match1" in results["pattern1"]
        assert "match2" in results["pattern2"]
    
    @pytest.mark.asyncio
    async def test_batch_terminal_commands(self, agent):
        """Test batch terminal commands."""
        agent._execute_tool = AsyncMock(side_effect=[
            "output1",
            "output2",
        ])
        agent.enable_parallel_execution(False)
        
        commands = ["ls -la", "pwd"]
        results = await agent.batch_terminal_commands(commands)
        
        assert len(results) == 2
        assert isinstance(results[0], ToolCallResult)
        assert results[0].result == "output1"
        assert results[1].result == "output2"
    
    @pytest.mark.asyncio
    async def test_batch_terminal_commands_with_working_dir(self, agent):
        """Test batch terminal commands with working directory."""
        agent._execute_tool = AsyncMock(return_value="output")
        agent.enable_parallel_execution(False)
        
        commands = ["ls"]
        await agent.batch_terminal_commands(commands, working_directory="/home")
        
        agent._execute_tool.assert_called_once()
        call_args = agent._execute_tool.call_args
        assert call_args[0][1]["working_directory"] == "/home"
