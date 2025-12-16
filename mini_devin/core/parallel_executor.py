"""
Parallel Tool Executor for Mini-Devin

This module provides parallel execution of independent tool calls,
analyzing dependencies and executing non-dependent calls concurrently.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine
import uuid


class DependencyType(str, Enum):
    """Types of dependencies between tool calls."""
    NONE = "none"
    FILE_READ_AFTER_WRITE = "file_read_after_write"
    DIRECTORY_DEPENDENCY = "directory_dependency"
    COMMAND_SEQUENCE = "command_sequence"
    EXPLICIT = "explicit"


@dataclass
class ToolCall:
    """Represents a tool call to be executed."""
    id: str
    tool_name: str
    arguments: dict[str, Any]
    depends_on: list[str] = field(default_factory=list)
    priority: int = 0
    
    @classmethod
    def create(
        cls,
        tool_name: str,
        arguments: dict[str, Any],
        depends_on: list[str] | None = None,
        priority: int = 0,
    ) -> "ToolCall":
        """Create a new tool call with auto-generated ID."""
        return cls(
            id=str(uuid.uuid4())[:8],
            tool_name=tool_name,
            arguments=arguments,
            depends_on=depends_on or [],
            priority=priority,
        )


@dataclass
class ToolCallResult:
    """Result of a tool call execution."""
    call_id: str
    tool_name: str
    success: bool
    result: Any
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: int = 0


@dataclass
class ParallelExecutionResult:
    """Result of parallel execution of multiple tool calls."""
    results: list[ToolCallResult]
    total_duration_ms: int
    parallel_speedup: float
    execution_order: list[list[str]]
    
    @property
    def all_successful(self) -> bool:
        """Check if all tool calls succeeded."""
        return all(r.success for r in self.results)
    
    @property
    def failed_calls(self) -> list[ToolCallResult]:
        """Get list of failed tool calls."""
        return [r for r in self.results if not r.success]
    
    def get_result(self, call_id: str) -> ToolCallResult | None:
        """Get result for a specific call ID."""
        for r in self.results:
            if r.call_id == call_id:
                return r
        return None


class DependencyAnalyzer:
    """Analyzes dependencies between tool calls."""
    
    def __init__(self):
        self._file_operations: dict[str, list[str]] = {}
    
    def analyze(self, calls: list[ToolCall]) -> dict[str, list[str]]:
        """
        Analyze dependencies between tool calls.
        
        Returns a dict mapping call IDs to their dependencies.
        """
        dependencies: dict[str, list[str]] = {call.id: list(call.depends_on) for call in calls}
        
        for i, earlier_call in enumerate(calls):
            for j, later_call in enumerate(calls):
                if j <= i:
                    continue
                
                dep = self._check_dependency(later_call, earlier_call)
                if dep != DependencyType.NONE:
                    if earlier_call.id not in dependencies[later_call.id]:
                        dependencies[later_call.id].append(earlier_call.id)
        
        return dependencies
    
    def _check_dependency(self, call: ToolCall, earlier_call: ToolCall) -> DependencyType:
        """Check if call depends on earlier_call."""
        if call.tool_name == "editor" and earlier_call.tool_name == "editor":
            return self._check_editor_dependency(call, earlier_call)
        
        if call.tool_name == "terminal" and earlier_call.tool_name == "terminal":
            return self._check_terminal_dependency(call, earlier_call)
        
        if call.tool_name == "editor" and earlier_call.tool_name == "terminal":
            return self._check_editor_after_terminal(call, earlier_call)
        
        return DependencyType.NONE
    
    def _check_editor_dependency(self, call: ToolCall, earlier_call: ToolCall) -> DependencyType:
        """Check dependency between two editor calls."""
        call_path = call.arguments.get("path", "")
        earlier_path = earlier_call.arguments.get("path", "")
        call_action = call.arguments.get("action", "")
        earlier_action = earlier_call.arguments.get("action", "")
        
        if not call_path or not earlier_path:
            return DependencyType.NONE
        
        if earlier_action in ("write_file", "apply_patch"):
            if call_action == "read_file" and self._paths_overlap(call_path, earlier_path):
                return DependencyType.FILE_READ_AFTER_WRITE
        
        if earlier_action == "write_file" and call_action == "write_file":
            if call_path == earlier_path:
                return DependencyType.FILE_READ_AFTER_WRITE
        
        return DependencyType.NONE
    
    def _check_terminal_dependency(self, call: ToolCall, earlier_call: ToolCall) -> DependencyType:
        """Check dependency between two terminal calls."""
        command = call.arguments.get("command", "")
        earlier_command = earlier_call.arguments.get("command", "")
        
        sequential_patterns = [
            ("git add", "git commit"),
            ("git commit", "git push"),
            ("npm install", "npm run"),
            ("pip install", "python"),
            ("poetry install", "poetry run"),
            ("mkdir", "cd"),
            ("cd", ""),
        ]
        
        for pattern1, pattern2 in sequential_patterns:
            if pattern1 in earlier_command and (not pattern2 or pattern2 in command):
                return DependencyType.COMMAND_SEQUENCE
        
        return DependencyType.NONE
    
    def _check_editor_after_terminal(self, call: ToolCall, earlier_call: ToolCall) -> DependencyType:
        """Check if editor call depends on terminal call."""
        command = earlier_call.arguments.get("command", "")
        call_path = call.arguments.get("path", "")
        
        file_creating_commands = ["touch", "mkdir", "cp", "mv", "git checkout"]
        for cmd in file_creating_commands:
            if cmd in command and call_path:
                return DependencyType.COMMAND_SEQUENCE
        
        return DependencyType.NONE
    
    def _paths_overlap(self, path1: str, path2: str) -> bool:
        """Check if two paths overlap (same file or parent/child)."""
        if path1 == path2:
            return True
        if path1.startswith(path2 + "/") or path2.startswith(path1 + "/"):
            return True
        return False


class ParallelExecutor:
    """
    Executes tool calls in parallel when possible.
    
    Analyzes dependencies between calls and groups independent
    calls for concurrent execution.
    """
    
    def __init__(
        self,
        execute_fn: Callable[[str, dict[str, Any]], Coroutine[Any, Any, str]],
        max_concurrent: int = 5,
        fail_fast: bool = False,
    ):
        """
        Initialize the parallel executor.
        
        Args:
            execute_fn: Async function to execute a single tool call.
                       Signature: (tool_name, arguments) -> result_string
            max_concurrent: Maximum number of concurrent executions
            fail_fast: If True, stop execution on first failure
        """
        self.execute_fn = execute_fn
        self.max_concurrent = max_concurrent
        self.fail_fast = fail_fast
        self.analyzer = DependencyAnalyzer()
    
    async def execute(self, calls: list[ToolCall]) -> ParallelExecutionResult:
        """
        Execute tool calls with automatic parallelization.
        
        Analyzes dependencies and executes independent calls in parallel.
        """
        if not calls:
            return ParallelExecutionResult(
                results=[],
                total_duration_ms=0,
                parallel_speedup=1.0,
                execution_order=[],
            )
        
        start_time = datetime.utcnow()
        
        dependencies = self.analyzer.analyze(calls)
        
        execution_groups = self._create_execution_groups(calls, dependencies)
        
        results: list[ToolCallResult] = []
        execution_order: list[list[str]] = []
        sequential_duration = 0
        
        for group in execution_groups:
            group_results = await self._execute_group(group)
            results.extend(group_results)
            execution_order.append([call.id for call in group])
            
            sequential_duration += sum(r.duration_ms for r in group_results)
            
            if self.fail_fast and any(not r.success for r in group_results):
                break
        
        end_time = datetime.utcnow()
        total_duration = int((end_time - start_time).total_seconds() * 1000)
        
        speedup = sequential_duration / total_duration if total_duration > 0 else 1.0
        
        return ParallelExecutionResult(
            results=results,
            total_duration_ms=total_duration,
            parallel_speedup=speedup,
            execution_order=execution_order,
        )
    
    def _create_execution_groups(
        self,
        calls: list[ToolCall],
        dependencies: dict[str, list[str]],
    ) -> list[list[ToolCall]]:
        """
        Create groups of calls that can be executed in parallel.
        
        Uses topological sorting to respect dependencies.
        """
        call_map = {call.id: call for call in calls}
        remaining = set(call.id for call in calls)
        completed: set[str] = set()
        groups: list[list[ToolCall]] = []
        
        while remaining:
            ready = []
            for call_id in remaining:
                deps = dependencies.get(call_id, [])
                if all(dep in completed or dep not in remaining for dep in deps):
                    ready.append(call_id)
            
            if not ready:
                ready = [min(remaining, key=lambda x: call_map[x].priority)]
            
            ready.sort(key=lambda x: call_map[x].priority)
            
            group = [call_map[call_id] for call_id in ready]
            groups.append(group)
            
            for call_id in ready:
                remaining.remove(call_id)
                completed.add(call_id)
        
        return groups
    
    async def _execute_group(self, group: list[ToolCall]) -> list[ToolCallResult]:
        """Execute a group of independent calls in parallel."""
        if len(group) == 1:
            return [await self._execute_single(group[0])]
        
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def execute_with_semaphore(call: ToolCall) -> ToolCallResult:
            async with semaphore:
                return await self._execute_single(call)
        
        tasks = [execute_with_semaphore(call) for call in group]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ToolCallResult(
                    call_id=group[i].id,
                    tool_name=group[i].tool_name,
                    success=False,
                    result=None,
                    error=str(result),
                    started_at=datetime.utcnow(),
                    completed_at=datetime.utcnow(),
                    duration_ms=0,
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _execute_single(self, call: ToolCall) -> ToolCallResult:
        """Execute a single tool call."""
        started_at = datetime.utcnow()
        
        try:
            result = await self.execute_fn(call.tool_name, call.arguments)
            completed_at = datetime.utcnow()
            duration_ms = int((completed_at - started_at).total_seconds() * 1000)
            
            success = "Error" not in str(result) and "BLOCKED" not in str(result)
            
            return ToolCallResult(
                call_id=call.id,
                tool_name=call.tool_name,
                success=success,
                result=result,
                error=None if success else result,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
            )
        except Exception as e:
            completed_at = datetime.utcnow()
            duration_ms = int((completed_at - started_at).total_seconds() * 1000)
            
            return ToolCallResult(
                call_id=call.id,
                tool_name=call.tool_name,
                success=False,
                result=None,
                error=str(e),
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
            )


class BatchToolCaller:
    """
    High-level interface for batching and executing tool calls.
    
    Provides a fluent API for building batches of tool calls
    with automatic dependency detection.
    """
    
    def __init__(
        self,
        execute_fn: Callable[[str, dict[str, Any]], Coroutine[Any, Any, str]],
        max_concurrent: int = 5,
    ):
        self.executor = ParallelExecutor(execute_fn, max_concurrent)
        self._calls: list[ToolCall] = []
        self._last_call_id: str | None = None
    
    def add(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        depends_on_previous: bool = False,
        depends_on: list[str] | None = None,
        priority: int = 0,
    ) -> "BatchToolCaller":
        """
        Add a tool call to the batch.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments for the tool
            depends_on_previous: If True, depends on the previous call
            depends_on: List of call IDs this call depends on
            priority: Execution priority (lower = earlier)
        
        Returns:
            Self for method chaining
        """
        deps = list(depends_on or [])
        if depends_on_previous and self._last_call_id:
            deps.append(self._last_call_id)
        
        call = ToolCall.create(
            tool_name=tool_name,
            arguments=arguments,
            depends_on=deps,
            priority=priority,
        )
        self._calls.append(call)
        self._last_call_id = call.id
        
        return self
    
    def terminal(
        self,
        command: str,
        working_directory: str | None = None,
        depends_on_previous: bool = False,
    ) -> "BatchToolCaller":
        """Add a terminal command to the batch."""
        args: dict[str, Any] = {"command": command}
        if working_directory:
            args["working_directory"] = working_directory
        return self.add("terminal", args, depends_on_previous=depends_on_previous)
    
    def read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> "BatchToolCaller":
        """Add a file read to the batch."""
        args: dict[str, Any] = {"action": "read_file", "path": path}
        if start_line:
            args["start_line"] = start_line
        if end_line:
            args["end_line"] = end_line
        return self.add("editor", args)
    
    def write_file(
        self,
        path: str,
        content: str,
        depends_on_previous: bool = False,
    ) -> "BatchToolCaller":
        """Add a file write to the batch."""
        return self.add(
            "editor",
            {"action": "write_file", "path": path, "content": content},
            depends_on_previous=depends_on_previous,
        )
    
    def search(
        self,
        pattern: str,
        path: str,
        file_pattern: str | None = None,
    ) -> "BatchToolCaller":
        """Add a search to the batch."""
        args: dict[str, Any] = {"action": "search", "pattern": pattern, "path": path}
        if file_pattern:
            args["file_pattern"] = file_pattern
        return self.add("editor", args)
    
    def list_directory(
        self,
        path: str,
        recursive: bool = False,
    ) -> "BatchToolCaller":
        """Add a directory listing to the batch."""
        return self.add(
            "editor",
            {"action": "list_directory", "path": path, "recursive": recursive},
        )
    
    def browser_search(
        self,
        query: str,
        max_results: int = 10,
    ) -> "BatchToolCaller":
        """Add a web search to the batch."""
        return self.add(
            "browser_search",
            {"query": query, "max_results": max_results},
        )
    
    def browser_fetch(
        self,
        url: str,
        extract_content: bool = True,
    ) -> "BatchToolCaller":
        """Add a page fetch to the batch."""
        return self.add(
            "browser_fetch",
            {"url": url, "extract_content": extract_content},
        )
    
    async def execute(self) -> ParallelExecutionResult:
        """Execute all batched calls with automatic parallelization."""
        result = await self.executor.execute(self._calls)
        self._calls = []
        self._last_call_id = None
        return result
    
    def clear(self) -> "BatchToolCaller":
        """Clear all pending calls."""
        self._calls = []
        self._last_call_id = None
        return self
    
    @property
    def pending_count(self) -> int:
        """Get the number of pending calls."""
        return len(self._calls)


def create_parallel_executor(
    execute_fn: Callable[[str, dict[str, Any]], Coroutine[Any, Any, str]],
    max_concurrent: int = 5,
    fail_fast: bool = False,
) -> ParallelExecutor:
    """Create a parallel executor instance."""
    return ParallelExecutor(execute_fn, max_concurrent, fail_fast)


def create_batch_caller(
    execute_fn: Callable[[str, dict[str, Any]], Coroutine[Any, Any, str]],
    max_concurrent: int = 5,
) -> BatchToolCaller:
    """Create a batch tool caller instance."""
    return BatchToolCaller(execute_fn, max_concurrent)
