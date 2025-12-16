"""
Tool Interface for Mini-Devin

This module defines the base interface for all tools and the tool registry.
Tools are the primary way the agent interacts with the environment.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Generic, TypeVar
import asyncio
import uuid

from pydantic import BaseModel

from ..schemas.tools import (
    BaseToolInput,
    BaseToolOutput,
    ToolStatus,
)
from ..schemas.state import ToolCallTrace, AgentPhase


# Type variables for generic tool typing
TInput = TypeVar("TInput", bound=BaseToolInput)
TOutput = TypeVar("TOutput", bound=BaseToolOutput)


class ToolExecutionError(Exception):
    """Raised when a tool execution fails."""
    
    def __init__(self, message: str, tool_name: str, original_error: Exception | None = None):
        self.message = message
        self.tool_name = tool_name
        self.original_error = original_error
        super().__init__(f"[{tool_name}] {message}")


class ToolPolicy(BaseModel):
    """Policy configuration for tool execution."""
    
    # Timeouts
    default_timeout_seconds: int = 30
    max_timeout_seconds: int = 300
    
    # Retries
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    
    # Rate limiting
    max_calls_per_minute: int = 60
    
    # Safety
    dry_run: bool = False
    require_confirmation: bool = False
    
    # Logging
    log_inputs: bool = True
    log_outputs: bool = True
    truncate_output_length: int = 10000


class BaseTool(ABC, Generic[TInput, TOutput]):
    """
    Abstract base class for all tools.
    
    Tools must implement:
    - name: The unique name of the tool
    - description: A description for the LLM
    - input_schema: The Pydantic model for input validation
    - output_schema: The Pydantic model for output
    - _execute: The actual execution logic
    """
    
    def __init__(self, policy: ToolPolicy | None = None):
        self.policy = policy or ToolPolicy()
        self._call_count = 0
        self._last_call_time: datetime | None = None
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name of the tool."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the tool does (for LLM)."""
        pass
    
    @property
    @abstractmethod
    def input_schema(self) -> type[TInput]:
        """Pydantic model class for input validation."""
        pass
    
    @property
    @abstractmethod
    def output_schema(self) -> type[TOutput]:
        """Pydantic model class for output."""
        pass
    
    @abstractmethod
    async def _execute(self, input_data: TInput) -> TOutput:
        """
        Execute the tool with validated input.
        
        This method should be implemented by subclasses.
        It should NOT handle retries or timeouts - those are handled by execute().
        """
        pass
    
    def validate_input(self, input_data: dict[str, Any] | TInput) -> TInput:
        """Validate and parse input data."""
        if isinstance(input_data, self.input_schema):
            return input_data
        return self.input_schema.model_validate(input_data)
    
    async def execute(
        self,
        input_data: dict[str, Any] | TInput,
        timeout: float | None = None,
    ) -> TOutput:
        """
        Execute the tool with validation, timeout, and retry handling.
        
        Args:
            input_data: The input data (dict or Pydantic model)
            timeout: Optional timeout override in seconds
            
        Returns:
            The tool output
            
        Raises:
            ToolExecutionError: If execution fails after retries
        """
        # Validate input
        validated_input = self.validate_input(input_data)
        
        # Determine timeout
        effective_timeout = min(
            timeout or self.policy.default_timeout_seconds,
            self.policy.max_timeout_seconds,
        )
        
        # Execute with retries
        last_error: Exception | None = None
        for attempt in range(self.policy.max_retries + 1):
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    self._execute(validated_input),
                    timeout=effective_timeout,
                )
                self._call_count += 1
                self._last_call_time = datetime.utcnow()
                return result
                
            except asyncio.TimeoutError:
                last_error = TimeoutError(f"Tool execution timed out after {effective_timeout}s")
                
            except Exception as e:
                last_error = e
            
            # Wait before retry (if not last attempt)
            if attempt < self.policy.max_retries:
                await asyncio.sleep(self.policy.retry_delay_seconds)
        
        # All retries exhausted
        raise ToolExecutionError(
            message=f"Execution failed after {self.policy.max_retries + 1} attempts: {last_error}",
            tool_name=self.name,
            original_error=last_error,
        )
    
    def get_schema_for_llm(self) -> dict[str, Any]:
        """
        Get the tool schema in a format suitable for LLM function calling.
        
        Returns a dict compatible with OpenAI/Anthropic function calling format.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema.model_json_schema(),
        }
    
    def __repr__(self) -> str:
        return f"<Tool: {self.name}>"


class ToolRegistry:
    """
    Registry for managing available tools.
    
    The registry:
    - Stores tool instances by name
    - Provides tool lookup and enumeration
    - Generates schemas for LLM function calling
    - Tracks tool usage statistics
    """
    
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._call_traces: list[ToolCallTrace] = []
    
    def register(self, tool: BaseTool) -> None:
        """Register a tool in the registry."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
    
    def unregister(self, name: str) -> None:
        """Remove a tool from the registry."""
        if name in self._tools:
            del self._tools[name]
    
    def get(self, name: str) -> BaseTool | None:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def get_required(self, name: str) -> BaseTool:
        """Get a tool by name, raising if not found."""
        tool = self.get(name)
        if tool is None:
            raise KeyError(f"Tool '{name}' not found in registry")
        return tool
    
    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
    
    def get_all_tools(self) -> list[BaseTool]:
        """Get all registered tools."""
        return list(self._tools.values())
    
    def get_schemas_for_llm(self) -> list[dict[str, Any]]:
        """Get all tool schemas for LLM function calling."""
        return [tool.get_schema_for_llm() for tool in self._tools.values()]
    
    async def execute(
        self,
        tool_name: str,
        input_data: dict[str, Any],
        session_id: str,
        task_id: str,
        agent_phase: AgentPhase,
        plan_step_id: str | None = None,
        iteration: int = 0,
    ) -> tuple[BaseToolOutput, ToolCallTrace]:
        """
        Execute a tool and record the trace.
        
        Args:
            tool_name: Name of the tool to execute
            input_data: Input data for the tool
            session_id: Current session ID
            task_id: Current task ID
            agent_phase: Current agent phase
            plan_step_id: Current plan step ID (if any)
            iteration: Current iteration number
            
        Returns:
            Tuple of (tool output, execution trace)
        """
        tool = self.get_required(tool_name)
        
        # Create trace
        trace = ToolCallTrace(
            trace_id=str(uuid.uuid4()),
            session_id=session_id,
            task_id=task_id,
            tool_name=tool_name,
            tool_input=input_data,
            agent_phase=agent_phase,
            plan_step_id=plan_step_id,
            iteration=iteration,
            started_at=datetime.utcnow(),
        )
        
        try:
            # Execute tool
            output = await tool.execute(input_data)
            
            # Update trace
            trace.completed_at = datetime.utcnow()
            trace.duration_ms = int(
                (trace.completed_at - trace.started_at).total_seconds() * 1000
            )
            trace.tool_output = output.model_dump()
            trace.success = output.status == ToolStatus.SUCCESS
            
        except ToolExecutionError as e:
            trace.completed_at = datetime.utcnow()
            trace.duration_ms = int(
                (trace.completed_at - trace.started_at).total_seconds() * 1000
            )
            trace.success = False
            trace.error_message = str(e)
            raise
        
        finally:
            self._call_traces.append(trace)
        
        return output, trace
    
    def get_traces(
        self,
        session_id: str | None = None,
        task_id: str | None = None,
        tool_name: str | None = None,
        limit: int | None = None,
    ) -> list[ToolCallTrace]:
        """Get tool call traces with optional filtering."""
        traces = self._call_traces
        
        if session_id:
            traces = [t for t in traces if t.session_id == session_id]
        if task_id:
            traces = [t for t in traces if t.task_id == task_id]
        if tool_name:
            traces = [t for t in traces if t.tool_name == tool_name]
        
        if limit:
            traces = traces[-limit:]
        
        return traces
    
    def clear_traces(self) -> None:
        """Clear all stored traces."""
        self._call_traces.clear()
    
    def get_usage_stats(self) -> dict[str, Any]:
        """Get usage statistics for all tools."""
        stats = {}
        for name, tool in self._tools.items():
            tool_traces = [t for t in self._call_traces if t.tool_name == name]
            success_traces = [t for t in tool_traces if t.success]
            
            stats[name] = {
                "total_calls": len(tool_traces),
                "successful_calls": len(success_traces),
                "failed_calls": len(tool_traces) - len(success_traces),
                "success_rate": len(success_traces) / len(tool_traces) if tool_traces else 0,
                "avg_duration_ms": (
                    sum(t.duration_ms or 0 for t in tool_traces) / len(tool_traces)
                    if tool_traces else 0
                ),
            }
        
        return stats


# Global registry instance
_global_registry: ToolRegistry | None = None


def get_global_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def register_tool(tool: BaseTool) -> BaseTool:
    """Register a tool in the global registry."""
    get_global_registry().register(tool)
    return tool
