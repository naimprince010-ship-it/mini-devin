"""Unit tests for the parallel executor module."""

import pytest
import asyncio
from datetime import datetime

from mini_devin.core.parallel_executor import (
    DependencyType,
    ToolCall,
    ToolCallResult,
    ParallelExecutionResult,
    DependencyAnalyzer,
    ParallelExecutor,
    BatchToolCaller,
    create_parallel_executor,
    create_batch_caller,
)


class TestToolCall:
    """Tests for ToolCall dataclass."""
    
    def test_create_tool_call(self):
        """Test creating a tool call."""
        call = ToolCall.create(
            tool_name="terminal",
            arguments={"command": "ls"},
        )
        assert call.tool_name == "terminal"
        assert call.arguments == {"command": "ls"}
        assert call.id is not None
        assert len(call.id) == 8
        assert call.depends_on == []
        assert call.priority == 0
    
    def test_create_with_dependencies(self):
        """Test creating a tool call with dependencies."""
        call = ToolCall.create(
            tool_name="editor",
            arguments={"action": "read_file", "path": "test.py"},
            depends_on=["abc123"],
            priority=1,
        )
        assert call.depends_on == ["abc123"]
        assert call.priority == 1


class TestToolCallResult:
    """Tests for ToolCallResult dataclass."""
    
    def test_create_result(self):
        """Test creating a tool call result."""
        result = ToolCallResult(
            call_id="abc123",
            tool_name="terminal",
            success=True,
            result="output",
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            duration_ms=100,
        )
        assert result.success
        assert result.result == "output"
        assert result.error is None
    
    def test_failed_result(self):
        """Test creating a failed result."""
        result = ToolCallResult(
            call_id="abc123",
            tool_name="terminal",
            success=False,
            result=None,
            error="Command failed",
        )
        assert not result.success
        assert result.error == "Command failed"


class TestParallelExecutionResult:
    """Tests for ParallelExecutionResult dataclass."""
    
    def test_all_successful(self):
        """Test all_successful property."""
        results = [
            ToolCallResult(call_id="1", tool_name="t", success=True, result="ok"),
            ToolCallResult(call_id="2", tool_name="t", success=True, result="ok"),
        ]
        exec_result = ParallelExecutionResult(
            results=results,
            total_duration_ms=100,
            parallel_speedup=1.5,
            execution_order=[["1", "2"]],
        )
        assert exec_result.all_successful
    
    def test_not_all_successful(self):
        """Test all_successful when one fails."""
        results = [
            ToolCallResult(call_id="1", tool_name="t", success=True, result="ok"),
            ToolCallResult(call_id="2", tool_name="t", success=False, result=None, error="fail"),
        ]
        exec_result = ParallelExecutionResult(
            results=results,
            total_duration_ms=100,
            parallel_speedup=1.0,
            execution_order=[["1", "2"]],
        )
        assert not exec_result.all_successful
    
    def test_failed_calls(self):
        """Test failed_calls property."""
        results = [
            ToolCallResult(call_id="1", tool_name="t", success=True, result="ok"),
            ToolCallResult(call_id="2", tool_name="t", success=False, result=None, error="fail"),
        ]
        exec_result = ParallelExecutionResult(
            results=results,
            total_duration_ms=100,
            parallel_speedup=1.0,
            execution_order=[["1", "2"]],
        )
        failed = exec_result.failed_calls
        assert len(failed) == 1
        assert failed[0].call_id == "2"
    
    def test_get_result(self):
        """Test get_result method."""
        results = [
            ToolCallResult(call_id="1", tool_name="t", success=True, result="ok1"),
            ToolCallResult(call_id="2", tool_name="t", success=True, result="ok2"),
        ]
        exec_result = ParallelExecutionResult(
            results=results,
            total_duration_ms=100,
            parallel_speedup=1.0,
            execution_order=[["1", "2"]],
        )
        assert exec_result.get_result("1").result == "ok1"
        assert exec_result.get_result("2").result == "ok2"
        assert exec_result.get_result("3") is None


class TestDependencyAnalyzer:
    """Tests for DependencyAnalyzer."""
    
    def test_no_dependencies(self):
        """Test analyzing calls with no dependencies."""
        analyzer = DependencyAnalyzer()
        calls = [
            ToolCall.create("terminal", {"command": "ls"}),
            ToolCall.create("terminal", {"command": "pwd"}),
        ]
        deps = analyzer.analyze(calls)
        assert deps[calls[0].id] == []
        assert deps[calls[1].id] == []
    
    def test_explicit_dependencies(self):
        """Test explicit dependencies are preserved."""
        analyzer = DependencyAnalyzer()
        call1 = ToolCall.create("terminal", {"command": "ls"})
        call2 = ToolCall.create("terminal", {"command": "pwd"}, depends_on=[call1.id])
        deps = analyzer.analyze([call1, call2])
        assert call1.id in deps[call2.id]
    
    def test_file_read_after_write_dependency(self):
        """Test detecting read after write dependency."""
        analyzer = DependencyAnalyzer()
        call1 = ToolCall.create("editor", {"action": "write_file", "path": "test.py", "content": "x"})
        call2 = ToolCall.create("editor", {"action": "read_file", "path": "test.py"})
        deps = analyzer.analyze([call1, call2])
        assert call1.id in deps[call2.id]
    
    def test_no_dependency_different_files(self):
        """Test no dependency for different files."""
        analyzer = DependencyAnalyzer()
        call1 = ToolCall.create("editor", {"action": "write_file", "path": "a.py", "content": "x"})
        call2 = ToolCall.create("editor", {"action": "read_file", "path": "b.py"})
        deps = analyzer.analyze([call1, call2])
        assert call1.id not in deps[call2.id]
    
    def test_git_command_sequence(self):
        """Test detecting git command sequence dependency."""
        analyzer = DependencyAnalyzer()
        call1 = ToolCall.create("terminal", {"command": "git add ."})
        call2 = ToolCall.create("terminal", {"command": "git commit -m 'test'"})
        deps = analyzer.analyze([call1, call2])
        assert call1.id in deps[call2.id]


class TestParallelExecutor:
    """Tests for ParallelExecutor."""
    
    @pytest.fixture
    def mock_execute_fn(self):
        """Create a mock execute function."""
        async def execute(tool_name: str, arguments: dict) -> str:
            await asyncio.sleep(0.01)
            return f"Result for {tool_name}"
        return execute
    
    @pytest.mark.asyncio
    async def test_execute_empty_calls(self, mock_execute_fn):
        """Test executing empty call list."""
        executor = ParallelExecutor(mock_execute_fn)
        result = await executor.execute([])
        assert result.results == []
        assert result.all_successful
    
    @pytest.mark.asyncio
    async def test_execute_single_call(self, mock_execute_fn):
        """Test executing a single call."""
        executor = ParallelExecutor(mock_execute_fn)
        calls = [ToolCall.create("terminal", {"command": "ls"})]
        result = await executor.execute(calls)
        assert len(result.results) == 1
        assert result.results[0].success
        assert "terminal" in result.results[0].result
    
    @pytest.mark.asyncio
    async def test_execute_parallel_calls(self, mock_execute_fn):
        """Test executing independent calls in parallel."""
        executor = ParallelExecutor(mock_execute_fn)
        calls = [
            ToolCall.create("terminal", {"command": "ls"}),
            ToolCall.create("terminal", {"command": "pwd"}),
            ToolCall.create("terminal", {"command": "whoami"}),
        ]
        result = await executor.execute(calls)
        assert len(result.results) == 3
        assert result.all_successful
        assert len(result.execution_order) == 1
        assert len(result.execution_order[0]) == 3
    
    @pytest.mark.asyncio
    async def test_execute_sequential_calls(self, mock_execute_fn):
        """Test executing dependent calls sequentially."""
        executor = ParallelExecutor(mock_execute_fn)
        call1 = ToolCall.create("terminal", {"command": "git add ."})
        call2 = ToolCall.create("terminal", {"command": "git commit -m 'test'"}, depends_on=[call1.id])
        result = await executor.execute([call1, call2])
        assert len(result.results) == 2
        assert result.all_successful
        assert len(result.execution_order) == 2
    
    @pytest.mark.asyncio
    async def test_fail_fast(self):
        """Test fail fast behavior."""
        async def failing_execute(tool_name: str, arguments: dict) -> str:
            if arguments.get("fail"):
                return "Error: failed"
            return "ok"
        
        executor = ParallelExecutor(failing_execute, fail_fast=True)
        call1 = ToolCall.create("terminal", {"command": "ls", "fail": True})
        call2 = ToolCall.create("terminal", {"command": "pwd"}, depends_on=[call1.id])
        result = await executor.execute([call1, call2])
        assert len(result.results) == 1
        assert not result.results[0].success
    
    @pytest.mark.asyncio
    async def test_max_concurrent(self, mock_execute_fn):
        """Test max concurrent limit."""
        executor = ParallelExecutor(mock_execute_fn, max_concurrent=2)
        calls = [ToolCall.create("terminal", {"command": f"cmd{i}"}) for i in range(5)]
        result = await executor.execute(calls)
        assert len(result.results) == 5
        assert result.all_successful


class TestBatchToolCaller:
    """Tests for BatchToolCaller."""
    
    @pytest.fixture
    def mock_execute_fn(self):
        """Create a mock execute function."""
        async def execute(tool_name: str, arguments: dict) -> str:
            return f"Result for {tool_name}"
        return execute
    
    def test_add_call(self, mock_execute_fn):
        """Test adding a call to the batch."""
        caller = BatchToolCaller(mock_execute_fn)
        caller.add("terminal", {"command": "ls"})
        assert caller.pending_count == 1
    
    def test_method_chaining(self, mock_execute_fn):
        """Test method chaining."""
        caller = BatchToolCaller(mock_execute_fn)
        result = caller.terminal("ls").terminal("pwd").read_file("test.py")
        assert result is caller
        assert caller.pending_count == 3
    
    def test_depends_on_previous(self, mock_execute_fn):
        """Test depends_on_previous flag."""
        caller = BatchToolCaller(mock_execute_fn)
        caller.terminal("git add .").terminal("git commit", depends_on_previous=True)
        assert caller.pending_count == 2
    
    def test_clear(self, mock_execute_fn):
        """Test clearing pending calls."""
        caller = BatchToolCaller(mock_execute_fn)
        caller.terminal("ls").terminal("pwd")
        caller.clear()
        assert caller.pending_count == 0
    
    @pytest.mark.asyncio
    async def test_execute(self, mock_execute_fn):
        """Test executing the batch."""
        caller = BatchToolCaller(mock_execute_fn)
        caller.terminal("ls").terminal("pwd")
        result = await caller.execute()
        assert len(result.results) == 2
        assert result.all_successful
        assert caller.pending_count == 0
    
    def test_convenience_methods(self, mock_execute_fn):
        """Test convenience methods."""
        caller = BatchToolCaller(mock_execute_fn)
        caller.terminal("ls")
        caller.read_file("test.py")
        caller.write_file("out.py", "content")
        caller.search("pattern", ".")
        caller.list_directory(".")
        caller.browser_search("query")
        caller.browser_fetch("http://example.com")
        assert caller.pending_count == 7


class TestFactoryFunctions:
    """Tests for factory functions."""
    
    def test_create_parallel_executor(self):
        """Test create_parallel_executor factory."""
        async def execute(tool_name: str, arguments: dict) -> str:
            return "ok"
        
        executor = create_parallel_executor(execute, max_concurrent=3, fail_fast=True)
        assert isinstance(executor, ParallelExecutor)
        assert executor.max_concurrent == 3
        assert executor.fail_fast
    
    def test_create_batch_caller(self):
        """Test create_batch_caller factory."""
        async def execute(tool_name: str, arguments: dict) -> str:
            return "ok"
        
        caller = create_batch_caller(execute, max_concurrent=3)
        assert isinstance(caller, BatchToolCaller)
        assert caller.executor.max_concurrent == 3
