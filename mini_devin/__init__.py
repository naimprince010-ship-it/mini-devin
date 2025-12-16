"""
Mini-Devin: An Autonomous AI Software Engineer Agent

Mini-Devin is an AI agent that can autonomously solve software engineering tasks
by using tools to interact with a terminal, code editor, and web browser.

Core Components:
- schemas: Pydantic schemas for tools, state, and verification
- core: Base tool interface and registry
- tools: Tool implementations (terminal, editor, browser)
- memory: Memory and context management
- orchestrator: Agent orchestration and planning
- sandbox: Execution sandbox and isolation

Usage:
    from mini_devin import AgentState, TaskState, ToolRegistry
    from mini_devin.core import BaseTool, register_tool
"""

__version__ = "0.1.0"

from .schemas import (
    # Tool schemas
    ToolStatus,
    TerminalInput,
    TerminalOutput,
    EditorAction,
    ReadFileInput,
    ReadFileOutput,
    WriteFileInput,
    WriteFileOutput,
    SearchInput,
    SearchOutput,
    BrowserAction,
    SearchWebInput,
    SearchWebOutput,
    FetchPageInput,
    FetchPageOutput,
    # State schemas
    TaskStatus,
    TaskType,
    TaskState,
    TaskGoal,
    TaskConstraints,
    AgentState,
    AgentPhase,
    PlanState,
    PlanStep,
    WorkingMemory,
    # Verification schemas
    VerificationCheck,
    VerificationResult,
    DoneSignal,
    CompletionCriteria,
)

from .core import (
    BaseTool,
    ToolRegistry,
    ToolPolicy,
    ToolExecutionError,
    get_global_registry,
    register_tool,
)

__all__ = [
    # Version
    "__version__",
    # Tool schemas
    "ToolStatus",
    "TerminalInput",
    "TerminalOutput",
    "EditorAction",
    "ReadFileInput",
    "ReadFileOutput",
    "WriteFileInput",
    "WriteFileOutput",
    "SearchInput",
    "SearchOutput",
    "BrowserAction",
    "SearchWebInput",
    "SearchWebOutput",
    "FetchPageInput",
    "FetchPageOutput",
    # State schemas
    "TaskStatus",
    "TaskType",
    "TaskState",
    "TaskGoal",
    "TaskConstraints",
    "AgentState",
    "AgentPhase",
    "PlanState",
    "PlanStep",
    "WorkingMemory",
    # Verification schemas
    "VerificationCheck",
    "VerificationResult",
    "DoneSignal",
    "CompletionCriteria",
    # Core
    "BaseTool",
    "ToolRegistry",
    "ToolPolicy",
    "ToolExecutionError",
    "get_global_registry",
    "register_tool",
]
