"""
Stop Conditions for Mini-Devin

This module defines strong STOP and BLOCKED conditions that halt agent execution.
These conditions ensure safe operation and prevent runaway behavior.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class StopReason(str, Enum):
    """Reasons for stopping agent execution."""
    
    # Normal completion
    TASK_COMPLETED = "task_completed"
    USER_REQUESTED = "user_requested"
    
    # Safety stops
    MAX_ITERATIONS_REACHED = "max_iterations_reached"
    MAX_REPAIR_ITERATIONS_REACHED = "max_repair_iterations_reached"
    SAFETY_VIOLATION = "safety_violation"
    DANGEROUS_COMMAND_BLOCKED = "dangerous_command_blocked"
    
    # Resource limits
    TIMEOUT_EXCEEDED = "timeout_exceeded"
    MEMORY_LIMIT_EXCEEDED = "memory_limit_exceeded"
    DISK_SPACE_LOW = "disk_space_low"
    
    # Error conditions
    LLM_API_ERROR = "llm_api_error"
    TOOL_EXECUTION_ERROR = "tool_execution_error"
    UNRECOVERABLE_ERROR = "unrecoverable_error"
    
    # Blocked conditions (require user intervention)
    MISSING_API_KEY = "missing_api_key"
    PERMISSION_DENIED = "permission_denied"
    DEPENDENCY_BUMP_REQUIRED = "dependency_bump_required"
    MANUAL_INTERVENTION_REQUIRED = "manual_intervention_required"
    AMBIGUOUS_TASK = "ambiguous_task"
    EXTERNAL_SERVICE_UNAVAILABLE = "external_service_unavailable"


class StopSeverity(str, Enum):
    """Severity level of stop condition."""
    
    INFO = "info"  # Normal completion
    WARNING = "warning"  # Soft stop, can be overridden
    ERROR = "error"  # Hard stop, requires attention
    CRITICAL = "critical"  # Critical stop, immediate halt


@dataclass
class StopCondition:
    """
    Represents a stop condition that halts agent execution.
    
    Attributes:
        reason: The reason for stopping
        severity: Severity level
        message: Human-readable description
        details: Additional context
        recoverable: Whether the condition can be recovered from
        requires_user_action: Whether user intervention is needed
        timestamp: When the condition was triggered
    """
    
    reason: StopReason
    severity: StopSeverity
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    recoverable: bool = False
    requires_user_action: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "reason": self.reason.value,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "recoverable": self.recoverable,
            "requires_user_action": self.requires_user_action,
            "timestamp": self.timestamp.isoformat(),
        }


class StopConditionChecker:
    """
    Checks for stop conditions during agent execution.
    
    This class evaluates various conditions that should halt the agent.
    """
    
    def __init__(
        self,
        max_iterations: int = 50,
        max_repair_iterations: int = 3,
        timeout_seconds: int = 300,
        max_lines_edit: int = 300,
        max_files_delete: int = 1,
        allow_dependency_bump: bool = False,
    ):
        self.max_iterations = max_iterations
        self.max_repair_iterations = max_repair_iterations
        self.timeout_seconds = timeout_seconds
        self.max_lines_edit = max_lines_edit
        self.max_files_delete = max_files_delete
        self.allow_dependency_bump = allow_dependency_bump
        
        self._start_time: datetime | None = None
        self._iteration_count = 0
        self._repair_iteration_count = 0
        self._consecutive_errors = 0
        self._max_consecutive_errors = 5
    
    def start(self) -> None:
        """Start tracking execution time."""
        self._start_time = datetime.utcnow()
        self._iteration_count = 0
        self._repair_iteration_count = 0
        self._consecutive_errors = 0
    
    def increment_iteration(self) -> StopCondition | None:
        """
        Increment iteration count and check for max iterations.
        
        Returns:
            StopCondition if max iterations reached, None otherwise
        """
        self._iteration_count += 1
        
        if self._iteration_count >= self.max_iterations:
            return StopCondition(
                reason=StopReason.MAX_ITERATIONS_REACHED,
                severity=StopSeverity.ERROR,
                message=f"Maximum iterations ({self.max_iterations}) reached. Agent stopped to prevent runaway execution.",
                details={
                    "iteration_count": self._iteration_count,
                    "max_iterations": self.max_iterations,
                },
                recoverable=False,
                requires_user_action=True,
            )
        
        return None
    
    def increment_repair_iteration(self) -> StopCondition | None:
        """
        Increment repair iteration count and check for max repair iterations.
        
        Returns:
            StopCondition if max repair iterations reached, None otherwise
        """
        self._repair_iteration_count += 1
        
        if self._repair_iteration_count >= self.max_repair_iterations:
            return StopCondition(
                reason=StopReason.MAX_REPAIR_ITERATIONS_REACHED,
                severity=StopSeverity.ERROR,
                message=f"Maximum repair iterations ({self.max_repair_iterations}) reached. Unable to automatically fix issues.",
                details={
                    "repair_iteration_count": self._repair_iteration_count,
                    "max_repair_iterations": self.max_repair_iterations,
                },
                recoverable=False,
                requires_user_action=True,
            )
        
        return None
    
    def reset_repair_iterations(self) -> None:
        """Reset repair iteration count (called after successful repair)."""
        self._repair_iteration_count = 0
    
    def check_timeout(self) -> StopCondition | None:
        """
        Check if execution has exceeded timeout.
        
        Returns:
            StopCondition if timeout exceeded, None otherwise
        """
        if self._start_time is None:
            return None
        
        elapsed = (datetime.utcnow() - self._start_time).total_seconds()
        
        if elapsed >= self.timeout_seconds:
            return StopCondition(
                reason=StopReason.TIMEOUT_EXCEEDED,
                severity=StopSeverity.ERROR,
                message=f"Execution timeout ({self.timeout_seconds}s) exceeded.",
                details={
                    "elapsed_seconds": elapsed,
                    "timeout_seconds": self.timeout_seconds,
                },
                recoverable=False,
                requires_user_action=True,
            )
        
        return None
    
    def check_edit_size(self, lines_changed: int) -> StopCondition | None:
        """
        Check if edit size exceeds maximum allowed.
        
        Args:
            lines_changed: Number of lines being changed
            
        Returns:
            StopCondition if edit too large, None otherwise
        """
        if lines_changed > self.max_lines_edit:
            return StopCondition(
                reason=StopReason.SAFETY_VIOLATION,
                severity=StopSeverity.ERROR,
                message=f"Edit size ({lines_changed} lines) exceeds maximum allowed ({self.max_lines_edit} lines).",
                details={
                    "lines_changed": lines_changed,
                    "max_lines_edit": self.max_lines_edit,
                },
                recoverable=False,
                requires_user_action=True,
            )
        
        return None
    
    def check_delete_count(self, files_to_delete: int) -> StopCondition | None:
        """
        Check if delete count exceeds maximum allowed.
        
        Args:
            files_to_delete: Number of files being deleted
            
        Returns:
            StopCondition if too many deletions, None otherwise
        """
        if files_to_delete > self.max_files_delete:
            return StopCondition(
                reason=StopReason.SAFETY_VIOLATION,
                severity=StopSeverity.ERROR,
                message=f"Delete count ({files_to_delete} files) exceeds maximum allowed ({self.max_files_delete} files).",
                details={
                    "files_to_delete": files_to_delete,
                    "max_files_delete": self.max_files_delete,
                },
                recoverable=False,
                requires_user_action=True,
            )
        
        return None
    
    def check_dependency_bump(self, file_path: str) -> StopCondition | None:
        """
        Check if dependency file modification is allowed.
        
        Args:
            file_path: Path to file being modified
            
        Returns:
            StopCondition if dependency bump not allowed, None otherwise
        """
        dependency_files = [
            "package.json",
            "package-lock.json",
            "yarn.lock",
            "pnpm-lock.yaml",
            "requirements.txt",
            "requirements-dev.txt",
            "Pipfile",
            "Pipfile.lock",
            "pyproject.toml",
            "poetry.lock",
            "Cargo.toml",
            "Cargo.lock",
            "go.mod",
            "go.sum",
            "Gemfile",
            "Gemfile.lock",
        ]
        
        file_name = file_path.split("/")[-1]
        
        if file_name in dependency_files and not self.allow_dependency_bump:
            return StopCondition(
                reason=StopReason.DEPENDENCY_BUMP_REQUIRED,
                severity=StopSeverity.WARNING,
                message=f"Modification of dependency file '{file_name}' requires explicit permission.",
                details={
                    "file_path": file_path,
                    "file_name": file_name,
                    "allow_dependency_bump": self.allow_dependency_bump,
                },
                recoverable=True,
                requires_user_action=True,
            )
        
        return None
    
    def check_dangerous_command(self, command: str) -> StopCondition | None:
        """
        Check if command is dangerous and should be blocked.
        
        Args:
            command: Command to check
            
        Returns:
            StopCondition if command is dangerous, None otherwise
        """
        dangerous_patterns = [
            "rm -rf /",
            "rm -rf /*",
            "rm -rf ~",
            "rm -rf ~/*",
            ":(){ :|:& };:",  # Fork bomb
            "mkfs.",
            "dd if=/dev/zero",
            "dd if=/dev/random",
            "chmod -R 777 /",
            "> /dev/sda",
            "wget | sh",
            "curl | sh",
            "wget | bash",
            "curl | bash",
            "git push --force",
            "git push -f",
            "git reset --hard HEAD~",
            "sudo rm -rf",
            "sudo dd",
            "sudo mkfs",
        ]
        
        command_lower = command.lower().strip()
        
        for pattern in dangerous_patterns:
            if pattern.lower() in command_lower:
                return StopCondition(
                    reason=StopReason.DANGEROUS_COMMAND_BLOCKED,
                    severity=StopSeverity.CRITICAL,
                    message=f"Dangerous command blocked: '{command[:100]}...'",
                    details={
                        "command": command,
                        "matched_pattern": pattern,
                    },
                    recoverable=False,
                    requires_user_action=True,
                )
        
        return None
    
    def record_error(self) -> StopCondition | None:
        """
        Record an error and check for consecutive error limit.
        
        Returns:
            StopCondition if too many consecutive errors, None otherwise
        """
        self._consecutive_errors += 1
        
        if self._consecutive_errors >= self._max_consecutive_errors:
            return StopCondition(
                reason=StopReason.UNRECOVERABLE_ERROR,
                severity=StopSeverity.ERROR,
                message=f"Too many consecutive errors ({self._consecutive_errors}). Agent stopped.",
                details={
                    "consecutive_errors": self._consecutive_errors,
                    "max_consecutive_errors": self._max_consecutive_errors,
                },
                recoverable=False,
                requires_user_action=True,
            )
        
        return None
    
    def record_success(self) -> None:
        """Record a successful operation (resets consecutive error count)."""
        self._consecutive_errors = 0
    
    def create_blocked_condition(
        self,
        reason: StopReason,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> StopCondition:
        """
        Create a BLOCKED condition that requires user intervention.
        
        Args:
            reason: The reason for blocking
            message: Human-readable description
            details: Additional context
            
        Returns:
            StopCondition with requires_user_action=True
        """
        return StopCondition(
            reason=reason,
            severity=StopSeverity.ERROR,
            message=message,
            details=details or {},
            recoverable=True,
            requires_user_action=True,
        )
    
    def get_status(self) -> dict[str, Any]:
        """Get current status of the checker."""
        elapsed = 0.0
        if self._start_time:
            elapsed = (datetime.utcnow() - self._start_time).total_seconds()
        
        return {
            "iteration_count": self._iteration_count,
            "max_iterations": self.max_iterations,
            "repair_iteration_count": self._repair_iteration_count,
            "max_repair_iterations": self.max_repair_iterations,
            "elapsed_seconds": elapsed,
            "timeout_seconds": self.timeout_seconds,
            "consecutive_errors": self._consecutive_errors,
            "max_consecutive_errors": self._max_consecutive_errors,
        }


# Predefined stop conditions for common scenarios
def create_missing_api_key_condition(key_name: str) -> StopCondition:
    """Create a stop condition for missing API key."""
    return StopCondition(
        reason=StopReason.MISSING_API_KEY,
        severity=StopSeverity.ERROR,
        message=f"Required API key '{key_name}' is not configured.",
        details={"key_name": key_name},
        recoverable=True,
        requires_user_action=True,
    )


def create_permission_denied_condition(resource: str, action: str) -> StopCondition:
    """Create a stop condition for permission denied."""
    return StopCondition(
        reason=StopReason.PERMISSION_DENIED,
        severity=StopSeverity.ERROR,
        message=f"Permission denied: Cannot {action} '{resource}'.",
        details={"resource": resource, "action": action},
        recoverable=True,
        requires_user_action=True,
    )


def create_ambiguous_task_condition(task: str, clarification_needed: str) -> StopCondition:
    """Create a stop condition for ambiguous task."""
    return StopCondition(
        reason=StopReason.AMBIGUOUS_TASK,
        severity=StopSeverity.WARNING,
        message=f"Task is ambiguous and requires clarification: {clarification_needed}",
        details={"task": task, "clarification_needed": clarification_needed},
        recoverable=True,
        requires_user_action=True,
    )


def create_task_completed_condition(summary: str) -> StopCondition:
    """Create a stop condition for successful task completion."""
    return StopCondition(
        reason=StopReason.TASK_COMPLETED,
        severity=StopSeverity.INFO,
        message=f"Task completed successfully: {summary}",
        details={"summary": summary},
        recoverable=False,
        requires_user_action=False,
    )
