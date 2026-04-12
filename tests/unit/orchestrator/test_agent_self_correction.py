"""Unit tests for the Agent Self-Correction Engine."""

import pytest
from unittest.mock import MagicMock

from mini_devin.reliability.self_correction import SelfCorrectionEngine, ErrorType

class TestSelfCorrectionEngine:
    
    @pytest.fixture
    def engine(self):
        return SelfCorrectionEngine(max_immediate_retries=3, max_replan_attempts=2)

    def test_classify_error_terminal_success(self, engine):
        assert engine.classify_error("terminal", "Completed successfully", 0) == ErrorType.SUCCESS
        
    def test_classify_error_terminal_syntax_error(self, engine):
        assert engine.classify_error("terminal", "SyntaxError: invalid syntax", 1) == ErrorType.SYNTAX_ERROR
        assert engine.classify_error("terminal", "IndentationError: unexpected indent", 1) == ErrorType.SYNTAX_ERROR
        
    def test_classify_error_terminal_missing_dependency(self, engine):
        assert engine.classify_error("terminal", "ModuleNotFoundError: No module named 'pytest'", 1) == ErrorType.DEPENDENCY_MISSING
        assert engine.classify_error("terminal", "bash: some_command: command not found", 127) == ErrorType.DEPENDENCY_MISSING

    def test_classify_error_terminal_file_not_found(self, engine):
        assert engine.classify_error("terminal", "cat: nonexistent.txt: No such file or directory", 1) == ErrorType.FILE_NOT_FOUND
        
    def test_classify_error_terminal_permission_denied(self, engine):
        assert engine.classify_error("terminal", "bash: /etc/shadow: Permission denied", 1) == ErrorType.PERMISSION_DENIED
        
    def test_classify_error_terminal_timeout(self, engine):
        assert engine.classify_error("terminal", "Command timed out after 30 seconds", 124) == ErrorType.TIMEOUT
        
    def test_classify_error_terminal_general_failure(self, engine):
        assert engine.classify_error("terminal", "Something went wrong", 1) == ErrorType.COMMAND_FAILED

    def test_classify_error_terminal_windows_path_on_linux(self, engine):
        out = (
            "STDERR:\nThis shell runs on Linux (container/cloud). Windows paths like G:\\ do not exist here.\n"
            "Exit code: -1"
        )
        assert engine.classify_error("terminal", out, -1) == ErrorType.ENVIRONMENT_MISMATCH

    def test_should_retry_environment_mismatch_never(self, engine):
        assert not engine.should_retry(ErrorType.ENVIRONMENT_MISMATCH, 0)
        assert not engine.should_retry(ErrorType.ENVIRONMENT_MISMATCH, 2)

    def test_get_retry_hint_environment_mismatch(self, engine):
        hint = engine.get_retry_hint(ErrorType.ENVIRONMENT_MISMATCH, "terminal", {}, "")
        assert "drive" in hint.lower() or "workspace" in hint.lower()

    def test_classify_error_editor_not_found(self, engine):
        assert engine.classify_error("editor", "Error: File not found", None) == ErrorType.FILE_NOT_FOUND
        
    def test_should_retry(self, engine):
        # Should not retry successes
        assert not engine.should_retry(ErrorType.SUCCESS, 0)
        
        # Should retry normal errors until max
        assert engine.should_retry(ErrorType.SYNTAX_ERROR, 0)
        assert engine.should_retry(ErrorType.SYNTAX_ERROR, 2)
        assert not engine.should_retry(ErrorType.SYNTAX_ERROR, 3) # Max reached
        
        # Special logic for permissions: retry only once
        assert engine.should_retry(ErrorType.PERMISSION_DENIED, 0)
        assert not engine.should_retry(ErrorType.PERMISSION_DENIED, 1)

    def test_should_replan(self, engine):
        # Replanning triggered after retries exhausted + 1 buffer
        assert not engine.should_replan(0)
        assert not engine.should_replan(2)
        assert engine.should_replan(4) # 3 retries + 1 failure threshold

    def test_record_correction(self, engine):
        failed_call = {"name": "terminal", "arguments": {"command": "python script.py"}}
        success_call = {"name": "terminal", "arguments": {"command": "python3 script.py"}}
        
        lesson = engine.record_correction(failed_call, "SyntaxError", success_call)
        
        assert len(engine.successful_corrections) == 1
        assert engine.successful_corrections[0]["solution"] == success_call
        assert engine.successful_corrections[0]["error_pattern"] == "SyntaxError"

    def test_get_retry_hint_syntax(self, engine):
        hint = engine.get_retry_hint(ErrorType.SYNTAX_ERROR, "terminal", {}, "SyntaxError")
        assert "syntax error" in hint.lower()
        
    def test_get_retry_hint_dependency(self, engine):
        hint = engine.get_retry_hint(ErrorType.DEPENDENCY_MISSING, "terminal", {}, "NotFound")
        assert "install" in hint.lower()
