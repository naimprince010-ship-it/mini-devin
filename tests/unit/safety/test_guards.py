"""Unit tests for safety guards module."""

import pytest
from unittest.mock import MagicMock, patch

from mini_devin.safety.guards import (
    SafetyGuard,
    SafetyViolation,
    SafetyPolicy,
    ViolationType,
    create_safety_guard,
)


class TestViolationType:
    """Tests for ViolationType enum."""

    def test_violation_type_values(self):
        """Test that all violation types exist."""
        assert ViolationType.DELETE_MULTIPLE_FILES is not None
        assert ViolationType.EDIT_TOO_MANY_LINES is not None
        assert ViolationType.DEPENDENCY_BUMP is not None
        assert ViolationType.DANGEROUS_COMMAND is not None
        assert ViolationType.FORCE_PUSH is not None
        assert ViolationType.DELETE_BRANCH is not None
        assert ViolationType.MODIFY_PROTECTED_FILE is not None
        assert ViolationType.EXCEED_ITERATION_LIMIT is not None


class TestSafetyViolation:
    """Tests for SafetyViolation dataclass."""

    def test_violation_creation(self):
        """Test creating a SafetyViolation."""
        violation = SafetyViolation(
            violation_type=ViolationType.DANGEROUS_COMMAND,
            message="Attempted to delete root filesystem",
            details={"command": "rm -rf /"},
            blocked=True,
        )
        assert violation.violation_type == ViolationType.DANGEROUS_COMMAND
        assert violation.message == "Attempted to delete root filesystem"
        assert violation.blocked is True
        assert violation.details["command"] == "rm -rf /"

    def test_violation_with_timestamp(self):
        """Test SafetyViolation has timestamp."""
        violation = SafetyViolation(
            violation_type=ViolationType.EDIT_TOO_MANY_LINES,
            message="Too many lines edited",
        )
        assert violation.timestamp is not None


class TestSafetyPolicy:
    """Tests for SafetyPolicy dataclass."""

    def test_default_policy(self):
        """Test default SafetyPolicy values."""
        policy = SafetyPolicy()
        assert policy.max_files_delete_per_operation == 1
        assert policy.max_lines_edit_per_iteration == 300
        assert policy.allow_dependency_bump is False
        assert policy.block_force_push is True
        assert policy.block_delete_branch is True
        assert policy.max_iterations_per_task == 50

    def test_custom_policy(self):
        """Test custom SafetyPolicy values."""
        policy = SafetyPolicy(
            max_files_delete_per_operation=5,
            max_lines_edit_per_iteration=500,
            allow_dependency_bump=True,
        )
        assert policy.max_files_delete_per_operation == 5
        assert policy.max_lines_edit_per_iteration == 500
        assert policy.allow_dependency_bump is True

    def test_policy_blocked_commands(self):
        """Test that policy has blocked commands."""
        policy = SafetyPolicy()
        assert len(policy.blocked_commands) > 0
        assert "rm -rf /" in policy.blocked_commands

    def test_policy_protected_branches(self):
        """Test that policy has protected branches."""
        policy = SafetyPolicy()
        assert "main" in policy.protected_branches
        assert "master" in policy.protected_branches

    def test_policy_protected_files(self):
        """Test that policy has protected files."""
        policy = SafetyPolicy()
        assert ".env" in policy.protected_files


class TestSafetyGuard:
    """Tests for SafetyGuard class."""

    def test_guard_initialization(self):
        """Test SafetyGuard initialization."""
        guard = SafetyGuard()
        assert guard is not None
        assert guard.policy is not None
        assert len(guard.violations) == 0

    def test_guard_with_custom_policy(self):
        """Test SafetyGuard with custom policy."""
        policy = SafetyPolicy(max_files_delete_per_operation=10)
        guard = SafetyGuard(policy=policy)
        assert guard.policy.max_files_delete_per_operation == 10

    def test_reset_iteration(self):
        """Test resetting iteration counters."""
        guard = SafetyGuard()
        guard.lines_edited_this_iteration = 100
        guard.files_deleted_this_operation = 5
        guard.reset_iteration()
        assert guard.lines_edited_this_iteration == 0
        assert guard.files_deleted_this_operation == 0

    def test_reset_all(self):
        """Test resetting all counters and violations."""
        guard = SafetyGuard()
        guard.violations.append(SafetyViolation(
            violation_type=ViolationType.DANGEROUS_COMMAND,
            message="Test",
        ))
        guard.current_iteration = 10
        guard.reset_all()
        assert len(guard.violations) == 0
        assert guard.current_iteration == 0


class TestSafetyGuardCommandChecks:
    """Command safety check tests for SafetyGuard."""

    def test_safe_command(self):
        """Test that safe commands pass."""
        guard = SafetyGuard()
        result = guard.check_command("ls -la")
        assert result is None

    def test_safe_echo_command(self):
        """Test that echo commands pass."""
        guard = SafetyGuard()
        result = guard.check_command("echo 'hello world'")
        assert result is None

    def test_dangerous_rm_rf_root(self):
        """Test that rm -rf / is blocked."""
        guard = SafetyGuard()
        result = guard.check_command("rm -rf /")
        assert result is not None
        assert result.violation_type == ViolationType.DANGEROUS_COMMAND
        assert result.blocked is True

    def test_dangerous_rm_rf_all(self):
        """Test that rm -rf /* is blocked."""
        guard = SafetyGuard()
        result = guard.check_command("rm -rf /*")
        assert result is not None
        assert result.violation_type == ViolationType.DANGEROUS_COMMAND

    def test_force_push_blocked(self):
        """Test that force push is blocked."""
        guard = SafetyGuard()
        result = guard.check_command("git push -f origin main")
        assert result is not None
        assert result.violation_type == ViolationType.FORCE_PUSH

    def test_force_push_long_flag(self):
        """Test that --force push is blocked."""
        guard = SafetyGuard()
        result = guard.check_command("git push --force origin main")
        assert result is not None
        assert result.violation_type == ViolationType.FORCE_PUSH

    def test_delete_protected_branch(self):
        """Test that deleting protected branch is blocked."""
        guard = SafetyGuard()
        result = guard.check_command("git branch -D main")
        assert result is not None
        assert result.violation_type == ViolationType.DELETE_BRANCH

    def test_multiple_file_deletion(self):
        """Test that deleting multiple files is blocked."""
        guard = SafetyGuard()
        result = guard.check_command("rm file1.txt file2.txt file3.txt")
        assert result is not None
        assert result.violation_type == ViolationType.DELETE_MULTIPLE_FILES


class TestSafetyGuardFileChecks:
    """File operation safety check tests for SafetyGuard."""

    def test_safe_file_edit(self):
        """Test that safe file edits pass."""
        guard = SafetyGuard()
        result = guard.check_file_edit(
            file_path="/home/user/project/src/main.py",
            lines_changed=50,
        )
        assert result is None

    def test_protected_file_blocked(self):
        """Test that editing protected files is blocked."""
        guard = SafetyGuard()
        result = guard.check_file_edit(
            file_path="/home/user/project/.env",
            lines_changed=1,
        )
        assert result is not None
        assert result.violation_type == ViolationType.MODIFY_PROTECTED_FILE

    def test_too_many_lines_edited(self):
        """Test that editing too many lines is blocked."""
        guard = SafetyGuard()
        result = guard.check_file_edit(
            file_path="/home/user/project/src/main.py",
            lines_changed=500,
        )
        assert result is not None
        assert result.violation_type == ViolationType.EDIT_TOO_MANY_LINES

    def test_multiple_file_deletion(self):
        """Test that deleting multiple files is blocked."""
        guard = SafetyGuard()
        
        # First deletion should pass
        result1 = guard.check_file_edit(
            file_path="/home/user/project/file1.txt",
            lines_changed=0,
            is_delete=True,
        )
        assert result1 is None
        
        # Second deletion should be blocked
        result2 = guard.check_file_edit(
            file_path="/home/user/project/file2.txt",
            lines_changed=0,
            is_delete=True,
        )
        assert result2 is not None
        assert result2.violation_type == ViolationType.DELETE_MULTIPLE_FILES


class TestSafetyGuardDependencyChecks:
    """Dependency change safety check tests for SafetyGuard."""

    def test_dependency_bump_blocked_by_default(self):
        """Test that dependency bumps are blocked by default."""
        guard = SafetyGuard()
        result = guard.check_dependency_change(
            file_path="requirements.txt",
            change_type="bump",
        )
        assert result is not None
        assert result.violation_type == ViolationType.DEPENDENCY_BUMP

    def test_dependency_bump_allowed_when_enabled(self):
        """Test that dependency bumps pass when allowed."""
        policy = SafetyPolicy(allow_dependency_bump=True)
        guard = SafetyGuard(policy=policy)
        result = guard.check_dependency_change(
            file_path="requirements.txt",
            change_type="bump",
        )
        assert result is None

    def test_non_dependency_file_passes(self):
        """Test that non-dependency files pass."""
        guard = SafetyGuard()
        result = guard.check_dependency_change(
            file_path="src/main.py",
            change_type="bump",
        )
        assert result is None


class TestSafetyGuardIterationChecks:
    """Iteration limit safety check tests for SafetyGuard."""

    def test_iteration_within_limit(self):
        """Test that iterations within limit pass."""
        guard = SafetyGuard()
        result = guard.check_iteration_limit(10)
        assert result is None

    def test_iteration_exceeds_limit(self):
        """Test that exceeding iteration limit is blocked."""
        guard = SafetyGuard()
        result = guard.check_iteration_limit(100)
        assert result is not None
        assert result.violation_type == ViolationType.EXCEED_ITERATION_LIMIT


class TestSafetyGuardViolationTracking:
    """Violation tracking tests for SafetyGuard."""

    def test_get_violations(self):
        """Test getting recorded violations."""
        guard = SafetyGuard()
        guard.check_command("rm -rf /")
        violations = guard.get_violations()
        assert len(violations) == 1

    def test_has_blocking_violations(self):
        """Test checking for blocking violations."""
        guard = SafetyGuard()
        assert guard.has_blocking_violations() is False
        
        guard.check_command("rm -rf /")
        assert guard.has_blocking_violations() is True

    def test_get_violation_summary(self):
        """Test getting violation summary."""
        guard = SafetyGuard()
        
        # No violations
        summary = guard.get_violation_summary()
        assert "No safety violations" in summary
        
        # With violations
        guard.check_command("rm -rf /")
        summary = guard.get_violation_summary()
        assert "BLOCKED" in summary


class TestSafetyGuardConfiguration:
    """Configuration tests for SafetyGuard."""

    def test_allow_dependency_changes(self):
        """Test allowing dependency changes."""
        guard = SafetyGuard()
        assert guard.policy.allow_dependency_bump is False
        
        guard.allow_dependency_changes(True)
        assert guard.policy.allow_dependency_bump is True

    def test_set_max_lines_per_iteration(self):
        """Test setting max lines per iteration."""
        guard = SafetyGuard()
        guard.set_max_lines_per_iteration(500)
        assert guard.policy.max_lines_edit_per_iteration == 500

    def test_set_max_files_delete(self):
        """Test setting max files to delete."""
        guard = SafetyGuard()
        guard.set_max_files_delete(5)
        assert guard.policy.max_files_delete_per_operation == 5


class TestCreateSafetyGuard:
    """Tests for create_safety_guard function."""

    def test_create_with_defaults(self):
        """Test creating guard with defaults."""
        guard = create_safety_guard()
        assert guard.policy.max_files_delete_per_operation == 1
        assert guard.policy.max_lines_edit_per_iteration == 300
        assert guard.policy.allow_dependency_bump is False

    def test_create_with_custom_settings(self):
        """Test creating guard with custom settings."""
        guard = create_safety_guard(
            max_files_delete=5,
            max_lines_edit=500,
            allow_dependency_bump=True,
        )
        assert guard.policy.max_files_delete_per_operation == 5
        assert guard.policy.max_lines_edit_per_iteration == 500
        assert guard.policy.allow_dependency_bump is True
