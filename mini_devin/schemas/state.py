"""
State Schemas for Mini-Devin

This module defines the Pydantic schemas for managing agent and task state:
- TaskState: The current state of a task being worked on
- AgentState: The overall state of the agent including memory and context
- PlanState: The current plan and its execution status
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field



# =============================================================================
# Task State Schemas
# =============================================================================


class TaskStatus(str, Enum):
    """Status of a task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    """Type of task the agent is working on."""
    BUG_FIX = "bug_fix"
    FEATURE = "feature"
    REFACTOR = "refactor"
    CODE_REVIEW = "code_review"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    EXPLORATION = "exploration"
    GENERAL = "general"


class TaskConstraints(BaseModel):
    """Constraints and boundaries for task execution."""
    max_file_changes: int = Field(
        default=50,
        description="Maximum number of files that can be modified"
    )
    max_iterations: int = Field(
        default=100,
        description="Maximum number of agent iterations"
    )
    allowed_directories: list[str] = Field(
        default_factory=list,
        description="Directories the agent is allowed to modify (empty = all)"
    )
    forbidden_directories: list[str] = Field(
        default_factory=lambda: [".git", "node_modules", "__pycache__", ".env"],
        description="Directories the agent must not modify"
    )
    forbidden_commands: list[str] = Field(
        default_factory=lambda: [
            "rm -rf /",
            "sudo rm",
            "git push --force",
            "git reset --hard",
            "DROP DATABASE",
            "DELETE FROM",
        ],
        description="Commands that are blocked"
    )
    require_tests: bool = Field(
        default=True,
        description="Whether tests must pass before completion"
    )
    require_lint: bool = Field(
        default=True,
        description="Whether lint must pass before completion"
    )
    auto_commit: bool = Field(
        default=True,
        description="Whether to automatically commit changes"
    )
    max_cost_usd: float = Field(
        default=10.0,
        description="Maximum cost in USD for LLM calls"
    )


class TaskGoal(BaseModel):
    """The goal of a task."""
    description: str = Field(description="Natural language description of the goal")
    acceptance_criteria: list[str] = Field(
        default_factory=list,
        description="Specific criteria that must be met"
    )
    success_signals: list[str] = Field(
        default_factory=list,
        description="Signals that indicate success (e.g., 'tests pass', 'lint clean')"
    )


class FileChange(BaseModel):
    """Record of a file change."""
    path: str
    action: str = Field(description="created, modified, deleted")
    diff: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TaskState(BaseModel):
    """Complete state of a task being worked on."""
    task_id: str = Field(description="Unique identifier for the task")
    task_type: TaskType = TaskType.GENERAL
    status: TaskStatus = TaskStatus.PENDING
    goal: TaskGoal
    constraints: TaskConstraints = Field(default_factory=TaskConstraints)
    
    # Context
    repository_path: str | None = Field(
        default=None,
        description="Path to the repository being worked on"
    )
    branch_name: str | None = None
    base_branch: str | None = Field(default="main")
    
    # Progress tracking
    files_changed: list[FileChange] = Field(default_factory=list)
    commands_executed: list[str] = Field(default_factory=list)
    urls_visited: list[str] = Field(default_factory=list)
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    
    # Metrics
    iteration_count: int = 0
    total_tokens_used: int = 0
    estimated_cost_usd: float = 0.0
    
    # Error tracking
    last_error: str | None = None
    error_count: int = 0
    
    # User interaction
    awaiting_user_input: bool = False
    user_messages: list[str] = Field(default_factory=list)


# =============================================================================
# Plan State Schemas
# =============================================================================


class StepStatus(str, Enum):
    """Status of a plan step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PlanStep(BaseModel):
    """A single step in the execution plan."""
    step_id: str
    description: str = Field(description="What this step accomplishes")
    expected_outcome: str = Field(description="How we know this step succeeded")
    status: StepStatus = StepStatus.PENDING
    
    # Execution details
    tool_calls: list[str] = Field(
        default_factory=list,
        description="Tool calls made during this step"
    )
    result: str | None = None
    error: str | None = None
    
    # Dependencies
    depends_on: list[str] = Field(
        default_factory=list,
        description="Step IDs this step depends on"
    )
    
    # Timing
    started_at: datetime | None = None
    completed_at: datetime | None = None


class Milestone(BaseModel):
    """A major milestone in the plan."""
    milestone_id: str
    name: str = Field(description="Name of the milestone (e.g., 'Bug Reproduced')")
    description: str
    steps: list[str] = Field(description="Step IDs that belong to this milestone")
    verification: str = Field(description="How to verify this milestone is complete")
    completed: bool = False


class PlanState(BaseModel):
    """The current execution plan."""
    plan_id: str
    task_id: str = Field(description="ID of the task this plan is for")
    
    # High-level structure
    milestones: list[Milestone] = Field(default_factory=list)
    steps: list[PlanStep] = Field(default_factory=list)
    
    # Current position
    current_milestone_id: str | None = None
    current_step_id: str | None = None
    
    # Plan metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_replanned_at: datetime | None = None
    replan_count: int = 0
    
    # Reasoning
    initial_analysis: str | None = Field(
        default=None,
        description="Initial analysis of the task"
    )
    current_reasoning: str | None = Field(
        default=None,
        description="Current reasoning about what to do next"
    )


# =============================================================================
# Agent State Schemas
# =============================================================================


class ConversationMessage(BaseModel):
    """A message in the conversation history."""
    role: str = Field(description="user, assistant, system, or tool")
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tool_call_id: str | None = None
    tool_name: str | None = None


class WorkingMemory(BaseModel):
    """Short-term working memory for the agent."""
    active_files: dict[str, str] = Field(
        default_factory=dict,
        description="Currently relevant file paths and their summaries"
    )
    active_symbols: list[str] = Field(
        default_factory=list,
        description="Currently relevant code symbols"
    )
    recent_tool_outputs: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Recent tool outputs (last N)"
    )
    key_findings: list[str] = Field(
        default_factory=list,
        description="Important findings during exploration"
    )
    current_hypothesis: str | None = Field(
        default=None,
        description="Current hypothesis about the problem/solution"
    )


class EnvironmentInfo(BaseModel):
    """Information about the execution environment."""
    os: str = "linux"
    python_version: str | None = None
    node_version: str | None = None
    git_available: bool = True
    docker_available: bool = False
    
    # Repository info
    repo_root: str | None = None
    repo_language: str | None = None
    package_manager: str | None = Field(
        default=None,
        description="npm, yarn, pip, poetry, cargo, etc."
    )
    test_command: str | None = None
    lint_command: str | None = None
    build_command: str | None = None


class LongTermMemory(BaseModel):
    """Long-term memory persisted across sessions."""
    session_id: str
    
    # Learned facts
    environment_facts: dict[str, str] = Field(
        default_factory=dict,
        description="Facts about the environment (e.g., 'test_command': 'npm test')"
    )
    codebase_facts: dict[str, str] = Field(
        default_factory=dict,
        description="Facts about the codebase"
    )
    
    # History
    successful_patterns: list[str] = Field(
        default_factory=list,
        description="Patterns that worked well"
    )
    failed_attempts: list[str] = Field(
        default_factory=list,
        description="Approaches that didn't work"
    )
    
    # Decisions
    decisions_made: list[str] = Field(
        default_factory=list,
        description="Key decisions and their rationale"
    )


class AgentPhase(str, Enum):
    """Current phase of the agent's workflow."""
    INTAKE = "intake"
    CLARIFY = "clarify"
    EXPLORE = "explore"
    PLAN = "plan"
    EXECUTE = "execute"
    VERIFY = "verify"
    REPAIR = "repair"
    FINALIZE = "finalize"
    BLOCKED = "blocked"
    COMPLETE = "complete"


class AgentState(BaseModel):
    """Complete state of the agent."""
    agent_id: str
    session_id: str
    
    # Current phase
    phase: AgentPhase = AgentPhase.INTAKE
    
    # Task and plan
    current_task: TaskState | None = None
    current_plan: PlanState | None = None
    
    # Memory layers
    working_memory: WorkingMemory = Field(default_factory=WorkingMemory)
    long_term_memory: LongTermMemory | None = None
    
    # Environment
    environment: EnvironmentInfo = Field(default_factory=EnvironmentInfo)
    
    # Conversation
    conversation_history: list[ConversationMessage] = Field(default_factory=list)
    
    # Execution trace
    tool_call_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Complete history of tool calls for replay"
    )
    
    # Control
    iteration: int = 0
    max_iterations: int = 100
    paused: bool = False
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Tool Call Trace (for replay and debugging)
# =============================================================================


class ToolCallTrace(BaseModel):
    """A trace of a single tool call for replay and debugging."""
    trace_id: str
    session_id: str
    task_id: str
    
    # Tool info
    tool_name: str
    tool_input: dict[str, Any]
    tool_output: dict[str, Any] | None = None
    
    # Context
    agent_phase: AgentPhase
    plan_step_id: str | None = None
    iteration: int
    
    # Timing
    started_at: datetime
    completed_at: datetime | None = None
    duration_ms: int | None = None
    
    # Status
    success: bool = False
    error_message: str | None = None
    
    # LLM context (for debugging)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    model_used: str | None = None
