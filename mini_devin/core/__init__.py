"""
Mini-Devin Core

This package contains core infrastructure for the Mini-Devin agent:
- tool_interface: Base tool interface and registry
- llm_client: LLM client wrapper using LiteLLM
"""

from .tool_interface import (
    ToolExecutionError,
    ToolPolicy,
    BaseTool,
    ToolRegistry,
    get_global_registry,
    register_tool,
)

from .llm_client import (
    LLMClient,
    LLMConfig,
    LLMMessage,
    LLMResponse,
    ToolCall,
    create_llm_client,
)

from .parallel_executor import (
    DependencyType,
    ToolCall as ParallelToolCall,
    ToolCallResult,
    ParallelExecutionResult,
    DependencyAnalyzer,
    ParallelExecutor,
    BatchToolCaller,
    create_parallel_executor,
    create_batch_caller,
)

__all__ = [
    "ToolExecutionError",
    "ToolPolicy",
    "BaseTool",
    "ToolRegistry",
    "get_global_registry",
    "register_tool",
    "LLMClient",
    "LLMConfig",
    "LLMMessage",
    "LLMResponse",
    "ToolCall",
    "create_llm_client",
    "DependencyType",
    "ParallelToolCall",
    "ToolCallResult",
    "ParallelExecutionResult",
    "DependencyAnalyzer",
    "ParallelExecutor",
    "BatchToolCaller",
    "create_parallel_executor",
    "create_batch_caller",
]
