"""
Skill Base Classes

This module defines the base classes for skills in Mini-Devin.
Skills are reusable, composable procedures that combine multiple
tool calls into coherent workflows.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SkillStatus(str, Enum):
    """Status of a skill execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SkillContext:
    """
    Context for skill execution.
    
    Contains all the information and tools a skill needs to execute.
    """
    workspace_path: str
    session_id: str | None = None
    task_id: str | None = None
    
    tools: dict[str, Any] = field(default_factory=dict)
    
    env_vars: dict[str, str] = field(default_factory=dict)
    
    metadata: dict[str, Any] = field(default_factory=dict)
    
    dry_run: bool = False
    
    max_iterations: int = 10
    timeout_seconds: int = 300
    
    def get_tool(self, name: str) -> Any | None:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def has_tool(self, name: str) -> bool:
        """Check if a tool is available."""
        return name in self.tools


@dataclass
class SkillStep:
    """A single step in a skill execution."""
    name: str
    description: str
    status: SkillStatus = SkillStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: Any = None
    error: str | None = None
    
    def start(self) -> None:
        """Mark the step as started."""
        self.status = SkillStatus.RUNNING
        self.started_at = datetime.utcnow()
    
    def complete(self, result: Any = None) -> None:
        """Mark the step as completed."""
        self.status = SkillStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.result = result
    
    def fail(self, error: str) -> None:
        """Mark the step as failed."""
        self.status = SkillStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error = error
    
    @property
    def duration_ms(self) -> int | None:
        """Get the duration in milliseconds."""
        if self.started_at and self.completed_at:
            return int((self.completed_at - self.started_at).total_seconds() * 1000)
        return None


@dataclass
class SkillResult:
    """Result of a skill execution."""
    success: bool
    message: str
    status: SkillStatus = SkillStatus.COMPLETED
    
    steps: list[SkillStep] = field(default_factory=list)
    
    outputs: dict[str, Any] = field(default_factory=dict)
    
    files_created: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    files_deleted: list[str] = field(default_factory=list)
    
    started_at: datetime | None = None
    completed_at: datetime | None = None
    
    error: str | None = None
    error_details: dict[str, Any] | None = None
    
    @property
    def duration_ms(self) -> int | None:
        """Get the total duration in milliseconds."""
        if self.started_at and self.completed_at:
            return int((self.completed_at - self.started_at).total_seconds() * 1000)
        return None
    
    @classmethod
    def success_result(
        cls,
        message: str,
        outputs: dict[str, Any] | None = None,
        files_created: list[str] | None = None,
        files_modified: list[str] | None = None,
    ) -> "SkillResult":
        """Create a successful result."""
        return cls(
            success=True,
            message=message,
            status=SkillStatus.COMPLETED,
            outputs=outputs or {},
            files_created=files_created or [],
            files_modified=files_modified or [],
        )
    
    @classmethod
    def failure_result(
        cls,
        message: str,
        error: str | None = None,
        error_details: dict[str, Any] | None = None,
    ) -> "SkillResult":
        """Create a failed result."""
        return cls(
            success=False,
            message=message,
            status=SkillStatus.FAILED,
            error=error,
            error_details=error_details,
        )


@dataclass
class SkillParameter:
    """Definition of a skill parameter."""
    name: str
    description: str
    type: str  # "string", "integer", "boolean", "array", "object"
    required: bool = True
    default: Any = None
    enum: list[Any] | None = None
    
    def to_schema(self) -> dict[str, Any]:
        """Convert to JSON schema format."""
        schema: dict[str, Any] = {
            "type": self.type,
            "description": self.description,
        }
        if self.default is not None:
            schema["default"] = self.default
        if self.enum is not None:
            schema["enum"] = self.enum
        return schema


class Skill(ABC):
    """
    Base class for all skills.
    
    Skills are reusable procedures that combine multiple tool calls
    into coherent workflows. They provide:
    - Clear input/output contracts
    - Step-by-step execution with progress tracking
    - Error handling and recovery
    - Dry-run support for previewing changes
    """
    
    name: str = "base_skill"
    description: str = "Base skill class"
    version: str = "1.0.0"
    
    parameters: list[SkillParameter] = []
    
    required_tools: list[str] = []
    
    tags: list[str] = []
    
    def __init__(self):
        self._steps: list[SkillStep] = []
        self._current_step: SkillStep | None = None
    
    @abstractmethod
    async def execute(
        self,
        context: SkillContext,
        **kwargs: Any,
    ) -> SkillResult:
        """
        Execute the skill.
        
        Args:
            context: Execution context with tools and configuration
            **kwargs: Skill-specific parameters
            
        Returns:
            SkillResult with success/failure status and outputs
        """
        pass
    
    def validate_parameters(self, **kwargs: Any) -> tuple[bool, str | None]:
        """
        Validate the provided parameters.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        for param in self.parameters:
            if param.required and param.name not in kwargs:
                if param.default is None:
                    return False, f"Missing required parameter: {param.name}"
            
            if param.name in kwargs and param.enum is not None:
                if kwargs[param.name] not in param.enum:
                    return False, f"Invalid value for {param.name}: must be one of {param.enum}"
        
        return True, None
    
    def validate_context(self, context: SkillContext) -> tuple[bool, str | None]:
        """
        Validate the execution context.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        for tool_name in self.required_tools:
            if not context.has_tool(tool_name):
                return False, f"Missing required tool: {tool_name}"
        
        return True, None
    
    def add_step(self, name: str, description: str) -> SkillStep:
        """Add a new step to the execution."""
        step = SkillStep(name=name, description=description)
        self._steps.append(step)
        return step
    
    def start_step(self, name: str, description: str) -> SkillStep:
        """Start a new step."""
        step = self.add_step(name, description)
        step.start()
        self._current_step = step
        return step
    
    def complete_step(self, result: Any = None) -> None:
        """Complete the current step."""
        if self._current_step:
            self._current_step.complete(result)
            self._current_step = None
    
    def fail_step(self, error: str) -> None:
        """Fail the current step."""
        if self._current_step:
            self._current_step.fail(error)
            self._current_step = None
    
    def get_steps(self) -> list[SkillStep]:
        """Get all execution steps."""
        return self._steps.copy()
    
    def reset(self) -> None:
        """Reset the skill state for re-execution."""
        self._steps = []
        self._current_step = None
    
    def to_schema(self) -> dict[str, Any]:
        """Convert the skill to a JSON schema for tool calling."""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = param.to_schema()
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }
    
    def __repr__(self) -> str:
        return f"<Skill {self.name} v{self.version}>"
