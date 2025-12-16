"""
Mini-Devin Schemas

This package contains all Pydantic schemas for the Mini-Devin agent:
- tools: Tool input/output schemas (Terminal, Editor, Browser)
- state: Agent and task state schemas
- verification: Verification and done signal schemas
"""

from .tools import (
    # Base
    ToolStatus,
    BaseToolInput,
    BaseToolOutput,
    # Terminal
    TerminalInput,
    TerminalOutput,
    # Editor
    EditorAction,
    FileRange,
    ReadFileInput,
    ReadFileOutput,
    WriteFileInput,
    WriteFileOutput,
    ApplyPatchInput,
    ApplyPatchOutput,
    SearchInput,
    SearchMatch,
    SearchOutput,
    ListDirectoryInput,
    FileInfo,
    ListDirectoryOutput,
    SymbolInfo,
    GetSymbolsInput,
    GetSymbolsOutput,
    GotoDefinitionInput,
    GotoDefinitionOutput,
    FindReferencesInput,
    ReferenceInfo,
    FindReferencesOutput,
    DiagnosticSeverity,
    Diagnostic,
    GetDiagnosticsInput,
    GetDiagnosticsOutput,
    # Browser
    BrowserAction,
    SearchWebInput,
    SearchResult,
    SearchWebOutput,
    FetchPageInput,
    LinkInfo,
    FetchPageOutput,
    NavigateInput,
    NavigateOutput,
    ClickInput,
    ClickOutput,
    TypeInput,
    TypeOutput,
    ScreenshotInput,
    ScreenshotOutput,
    # Union types
    EditorInput,
    EditorOutput,
    BrowserInput,
    BrowserOutput,
    ToolInput,
    ToolOutput,
)

from .state import (
    # Task
    TaskStatus,
    TaskType,
    TaskConstraints,
    TaskGoal,
    FileChange,
    TaskState,
    # Plan
    StepStatus,
    PlanStep,
    Milestone,
    PlanState,
    # Agent
    ConversationMessage,
    WorkingMemory,
    EnvironmentInfo,
    LongTermMemory,
    AgentPhase,
    AgentState,
    # Trace
    ToolCallTrace,
)

from .verification import (
    # Check types
    CheckType,
    CheckStatus,
    CheckSeverity,
    # Verification
    VerificationCheck,
    CheckResult,
    VerificationSuite,
    VerificationResult,
    # Done signals
    DoneSignalType,
    DoneSignal,
    CompletionCriteria,
    # Pre-built suites
    create_python_verification_suite,
    create_javascript_verification_suite,
    create_git_verification_suite,
)

__all__ = [
    # Tools - Base
    "ToolStatus",
    "BaseToolInput",
    "BaseToolOutput",
    # Tools - Terminal
    "TerminalInput",
    "TerminalOutput",
    # Tools - Editor
    "EditorAction",
    "FileRange",
    "ReadFileInput",
    "ReadFileOutput",
    "WriteFileInput",
    "WriteFileOutput",
    "ApplyPatchInput",
    "ApplyPatchOutput",
    "SearchInput",
    "SearchMatch",
    "SearchOutput",
    "ListDirectoryInput",
    "FileInfo",
    "ListDirectoryOutput",
    "SymbolInfo",
    "GetSymbolsInput",
    "GetSymbolsOutput",
    "GotoDefinitionInput",
    "GotoDefinitionOutput",
    "FindReferencesInput",
    "ReferenceInfo",
    "FindReferencesOutput",
    "DiagnosticSeverity",
    "Diagnostic",
    "GetDiagnosticsInput",
    "GetDiagnosticsOutput",
    # Tools - Browser
    "BrowserAction",
    "SearchWebInput",
    "SearchResult",
    "SearchWebOutput",
    "FetchPageInput",
    "LinkInfo",
    "FetchPageOutput",
    "NavigateInput",
    "NavigateOutput",
    "ClickInput",
    "ClickOutput",
    "TypeInput",
    "TypeOutput",
    "ScreenshotInput",
    "ScreenshotOutput",
    # Tools - Union types
    "EditorInput",
    "EditorOutput",
    "BrowserInput",
    "BrowserOutput",
    "ToolInput",
    "ToolOutput",
    # State - Task
    "TaskStatus",
    "TaskType",
    "TaskConstraints",
    "TaskGoal",
    "FileChange",
    "TaskState",
    # State - Plan
    "StepStatus",
    "PlanStep",
    "Milestone",
    "PlanState",
    # State - Agent
    "ConversationMessage",
    "WorkingMemory",
    "EnvironmentInfo",
    "LongTermMemory",
    "AgentPhase",
    "AgentState",
    # State - Trace
    "ToolCallTrace",
    # Verification
    "CheckType",
    "CheckStatus",
    "CheckSeverity",
    "VerificationCheck",
    "CheckResult",
    "VerificationSuite",
    "VerificationResult",
    # Done signals
    "DoneSignalType",
    "DoneSignal",
    "CompletionCriteria",
    # Pre-built suites
    "create_python_verification_suite",
    "create_javascript_verification_suite",
    "create_git_verification_suite",
]
