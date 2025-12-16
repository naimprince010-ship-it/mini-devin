"""
Safety Guards for Mini-Devin

This module implements safety guards that prevent dangerous operations:
- Deleting more than 1 file in one operation
- Editing more than 300 lines in one iteration
- Dependency bumps unless explicitly allowed
- Other configurable safety policies
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ViolationType(str, Enum):
    """Types of safety violations."""
    DELETE_MULTIPLE_FILES = "delete_multiple_files"
    EDIT_TOO_MANY_LINES = "edit_too_many_lines"
    DEPENDENCY_BUMP = "dependency_bump"
    DANGEROUS_COMMAND = "dangerous_command"
    FORCE_PUSH = "force_push"
    DELETE_BRANCH = "delete_branch"
    MODIFY_PROTECTED_FILE = "modify_protected_file"
    EXCEED_ITERATION_LIMIT = "exceed_iteration_limit"


@dataclass
class SafetyViolation:
    """Record of a safety violation."""
    violation_type: ViolationType
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    blocked: bool = True  # Whether this violation blocks the operation


@dataclass
class SafetyPolicy:
    """Configurable safety policy."""
    # File deletion limits
    max_files_delete_per_operation: int = 1
    
    # Edit limits
    max_lines_edit_per_iteration: int = 300
    
    # Dependency management
    allow_dependency_bump: bool = False
    allowed_dependency_files: list[str] = field(default_factory=lambda: [
        "requirements.txt",
        "pyproject.toml",
        "package.json",
        "Cargo.toml",
        "go.mod",
    ])
    
    # Dangerous commands
    blocked_commands: list[str] = field(default_factory=lambda: [
        "rm -rf /",
        "rm -rf /*",
        "rm -rf ~",
        "rm -rf ~/",
        "> /dev/sda",
        "mkfs",
        "dd if=",
        ":(){:|:&};:",  # Fork bomb
    ])
    
    # Git safety
    block_force_push: bool = True
    block_delete_branch: bool = True
    protected_branches: list[str] = field(default_factory=lambda: [
        "main",
        "master",
        "production",
        "release",
    ])
    
    # Protected files (cannot be modified)
    protected_files: list[str] = field(default_factory=lambda: [
        ".env",
        ".env.local",
        ".env.production",
        "credentials.json",
        "secrets.yaml",
        "id_rsa",
        "id_ed25519",
    ])
    
    # Iteration limits
    max_iterations_per_task: int = 50
    max_repair_iterations: int = 3


class SafetyGuard:
    """
    Safety guard that validates operations before execution.
    
    This guard enforces safety policies to prevent dangerous operations
    and transitions the agent to BLOCKED state on violations.
    """
    
    def __init__(self, policy: SafetyPolicy | None = None):
        self.policy = policy or SafetyPolicy()
        self.violations: list[SafetyViolation] = []
        self.lines_edited_this_iteration: int = 0
        self.files_deleted_this_operation: int = 0
        self.current_iteration: int = 0
    
    def reset_iteration(self) -> None:
        """Reset per-iteration counters."""
        self.lines_edited_this_iteration = 0
        self.files_deleted_this_operation = 0
        self.current_iteration += 1
    
    def reset_all(self) -> None:
        """Reset all counters and violations."""
        self.violations = []
        self.lines_edited_this_iteration = 0
        self.files_deleted_this_operation = 0
        self.current_iteration = 0
    
    def check_command(self, command: str) -> SafetyViolation | None:
        """
        Check if a shell command is safe to execute.
        
        Returns a SafetyViolation if the command is blocked, None otherwise.
        """
        command_lower = command.lower().strip()
        
        # Check for blocked commands
        for blocked in self.policy.blocked_commands:
            if blocked.lower() in command_lower:
                violation = SafetyViolation(
                    violation_type=ViolationType.DANGEROUS_COMMAND,
                    message=f"Blocked dangerous command: {command[:100]}",
                    details={"command": command, "matched_pattern": blocked},
                )
                self.violations.append(violation)
                return violation
        
        # Check for force push
        if self.policy.block_force_push:
            if re.search(r"git\s+push\s+.*(-f|--force)", command_lower):
                violation = SafetyViolation(
                    violation_type=ViolationType.FORCE_PUSH,
                    message="Force push is blocked by safety policy",
                    details={"command": command},
                )
                self.violations.append(violation)
                return violation
        
        # Check for branch deletion
        if self.policy.block_delete_branch:
            branch_delete_match = re.search(
                r"git\s+(branch\s+-[dD]|push\s+.*--delete)",
                command_lower,
            )
            if branch_delete_match:
                # Check if it's a protected branch
                for branch in self.policy.protected_branches:
                    if branch.lower() in command_lower:
                        violation = SafetyViolation(
                            violation_type=ViolationType.DELETE_BRANCH,
                            message=f"Deleting protected branch '{branch}' is blocked",
                            details={"command": command, "branch": branch},
                        )
                        self.violations.append(violation)
                        return violation
        
        # Check for multiple file deletion
        rm_match = re.search(r"rm\s+(-[rf]+\s+)?(.+)", command)
        if rm_match:
            files_part = rm_match.group(2)
            # Count files (rough estimate based on spaces)
            files = [f for f in files_part.split() if not f.startswith("-")]
            if len(files) > self.policy.max_files_delete_per_operation:
                violation = SafetyViolation(
                    violation_type=ViolationType.DELETE_MULTIPLE_FILES,
                    message=f"Deleting {len(files)} files exceeds limit of {self.policy.max_files_delete_per_operation}",
                    details={"command": command, "files_count": len(files)},
                )
                self.violations.append(violation)
                return violation
        
        return None
    
    def check_file_edit(
        self,
        file_path: str,
        lines_changed: int,
        is_delete: bool = False,
    ) -> SafetyViolation | None:
        """
        Check if a file edit is safe.
        
        Args:
            file_path: Path to the file being edited
            lines_changed: Number of lines being changed
            is_delete: Whether this is a file deletion
            
        Returns a SafetyViolation if blocked, None otherwise.
        """
        # Check protected files
        file_name = file_path.split("/")[-1]
        for protected in self.policy.protected_files:
            if protected in file_path or file_name == protected:
                violation = SafetyViolation(
                    violation_type=ViolationType.MODIFY_PROTECTED_FILE,
                    message=f"Cannot modify protected file: {file_path}",
                    details={"file_path": file_path, "protected_pattern": protected},
                )
                self.violations.append(violation)
                return violation
        
        # Check file deletion limit
        if is_delete:
            self.files_deleted_this_operation += 1
            if self.files_deleted_this_operation > self.policy.max_files_delete_per_operation:
                violation = SafetyViolation(
                    violation_type=ViolationType.DELETE_MULTIPLE_FILES,
                    message=f"Deleting {self.files_deleted_this_operation} files exceeds limit of {self.policy.max_files_delete_per_operation}",
                    details={
                        "file_path": file_path,
                        "files_deleted": self.files_deleted_this_operation,
                        "limit": self.policy.max_files_delete_per_operation,
                    },
                )
                self.violations.append(violation)
                return violation
        
        # Check lines edited limit
        self.lines_edited_this_iteration += lines_changed
        if self.lines_edited_this_iteration > self.policy.max_lines_edit_per_iteration:
            violation = SafetyViolation(
                violation_type=ViolationType.EDIT_TOO_MANY_LINES,
                message=f"Editing {self.lines_edited_this_iteration} lines exceeds limit of {self.policy.max_lines_edit_per_iteration}",
                details={
                    "file_path": file_path,
                    "lines_this_edit": lines_changed,
                    "total_lines_this_iteration": self.lines_edited_this_iteration,
                    "limit": self.policy.max_lines_edit_per_iteration,
                },
            )
            self.violations.append(violation)
            return violation
        
        return None
    
    def check_dependency_change(
        self,
        file_path: str,
        change_type: str = "bump",
    ) -> SafetyViolation | None:
        """
        Check if a dependency change is allowed.
        
        Args:
            file_path: Path to the dependency file
            change_type: Type of change (bump, add, remove)
            
        Returns a SafetyViolation if blocked, None otherwise.
        """
        file_name = file_path.split("/")[-1]
        
        # Check if this is a dependency file
        is_dependency_file = any(
            dep_file in file_path or file_name == dep_file
            for dep_file in self.policy.allowed_dependency_files
        )
        
        if is_dependency_file and not self.policy.allow_dependency_bump:
            violation = SafetyViolation(
                violation_type=ViolationType.DEPENDENCY_BUMP,
                message=f"Dependency {change_type} not allowed without explicit permission",
                details={
                    "file_path": file_path,
                    "change_type": change_type,
                },
            )
            self.violations.append(violation)
            return violation
        
        return None
    
    def check_iteration_limit(self, current_iteration: int) -> SafetyViolation | None:
        """Check if iteration limit has been exceeded."""
        if current_iteration > self.policy.max_iterations_per_task:
            violation = SafetyViolation(
                violation_type=ViolationType.EXCEED_ITERATION_LIMIT,
                message=f"Exceeded maximum iterations ({self.policy.max_iterations_per_task})",
                details={
                    "current_iteration": current_iteration,
                    "limit": self.policy.max_iterations_per_task,
                },
            )
            self.violations.append(violation)
            return violation
        return None
    
    def get_violations(self) -> list[SafetyViolation]:
        """Get all recorded violations."""
        return self.violations.copy()
    
    def has_blocking_violations(self) -> bool:
        """Check if there are any blocking violations."""
        return any(v.blocked for v in self.violations)
    
    def get_violation_summary(self) -> str:
        """Get a summary of all violations."""
        if not self.violations:
            return "No safety violations"
        
        summary_parts = [f"Safety Violations ({len(self.violations)}):"]
        for v in self.violations:
            status = "[BLOCKED]" if v.blocked else "[WARNING]"
            summary_parts.append(f"  {status} {v.violation_type.value}: {v.message}")
        
        return "\n".join(summary_parts)
    
    def allow_dependency_changes(self, allow: bool = True) -> None:
        """Temporarily allow or disallow dependency changes."""
        self.policy.allow_dependency_bump = allow
    
    def set_max_lines_per_iteration(self, max_lines: int) -> None:
        """Set the maximum lines that can be edited per iteration."""
        self.policy.max_lines_edit_per_iteration = max_lines
    
    def set_max_files_delete(self, max_files: int) -> None:
        """Set the maximum files that can be deleted per operation."""
        self.policy.max_files_delete_per_operation = max_files


def create_safety_guard(
    max_files_delete: int = 1,
    max_lines_edit: int = 300,
    allow_dependency_bump: bool = False,
) -> SafetyGuard:
    """Create a safety guard with custom settings."""
    policy = SafetyPolicy(
        max_files_delete_per_operation=max_files_delete,
        max_lines_edit_per_iteration=max_lines_edit,
        allow_dependency_bump=allow_dependency_bump,
    )
    return SafetyGuard(policy)
