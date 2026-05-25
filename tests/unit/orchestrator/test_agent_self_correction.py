"""Unit tests for the Agent Self-Correction Engine."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from mini_devin.reliability.self_correction import (
    SelfCorrectionEngine,
    ErrorType,
    error_fingerprint,
    terminal_sanity_check,
    incremental_recovery_hint,
)
from mini_devin.schemas.state import TaskState, TaskGoal, PlanState, PlanStep, StepStatus, TaskStatus
from mini_devin.orchestrator.agent import Agent

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

    def test_get_retry_hint_dependency_node_package(self, engine):
        hint = engine.get_retry_hint(
            ErrorType.DEPENDENCY_MISSING,
            "terminal",
            {"command": "npm install express"},
            "package.json\nnode_modules missing",
        )
        low = hint.lower()
        assert "npm" in low or "pnpm" in low or "yarn" in low
        assert "not `pip`" in hint or "not pip" in low

    def test_get_retry_hint_timeout_dev_server(self, engine):
        hint = engine.get_retry_hint(
            ErrorType.TIMEOUT,
            "terminal",
            {"command": "node server.js"},
            "Command timed out after 30 seconds",
        )
        low = hint.lower()
        assert "live_preview" in low or "preview" in low
        assert "sandbox" in low or "detached" in low or "host/process" in low
        assert "run it in the background using `&`" not in hint

    def test_get_retry_hint_browser(self, engine):
        hint = engine.get_retry_hint(
            ErrorType.UNKNOWN,
            "browser_click",
            {"selector": ".submit"},
            "Error: browser_click failed: timeout",
        )
        low = hint.lower()
        assert "browser_screenshot" in hint or "browser_playwright" in hint
        assert "selector" in low
        assert "coordinates" in low or "raw coordinates" in low
        assert "submit" in low or "overlay" in low or "modal" in low

    def test_get_retry_hint_browser_navigate(self, engine):
        hint = engine.get_retry_hint(
            ErrorType.UNKNOWN,
            "browser_navigate",
            {"url": "example.com"},
            "Error: browser_navigate failed: net::ERR_CONNECTION_REFUSED",
        )
        low = hint.lower()
        assert "browser_screenshot" in hint or "browser_playwright" in hint
        assert "url" in low
        assert "https://" in hint or "dev server" in low or "redirect" in low
        assert "submit: true" not in hint


def test_error_fingerprint_stable_for_same_failure():
    fp1 = error_fingerprint("terminal", "No such file: foo\n", 1)
    fp2 = error_fingerprint("terminal", "No such file: foo\n", 1)
    assert fp1 == fp2
    assert fp1 != error_fingerprint("terminal", "other", 1)


def test_terminal_sanity_check_rejects_windows_paths_on_linux():
    ok, msg = terminal_sanity_check("cat G:\\\\repo\\\\x.py", is_windows=False)
    assert not ok
    assert "windows" in msg.lower() or "linux" in msg.lower()


def test_terminal_sanity_check_accepts_simple_posix():
    ok, msg = terminal_sanity_check("ls -la ./src", is_windows=False)
    assert ok
    assert msg == ""


def test_terminal_sanity_check_rejects_bash_listing_flags_on_windows():
    ok, msg = terminal_sanity_check("ls -la", is_windows=True)
    assert not ok
    assert "powershell" in msg.lower()
    assert "get-childitem" in msg.lower()

    ok, msg = terminal_sanity_check("dir /a", is_windows=True)
    assert not ok
    assert "plain `dir`" in msg.lower()


def test_terminal_sanity_check_rejects_pip_install_stdlib():
    ok, msg = terminal_sanity_check("pip install unittest", is_windows=False)
    assert not ok
    assert "standard library" in msg.lower()
    assert "python -m unittest" in msg.lower()


def test_incremental_recovery_hint_pytest():
    h = incremental_recovery_hint(
        "terminal",
        {"command": "pytest tests/"},
        ErrorType.COMMAND_FAILED,
        "collected 0 items",
        last_failed_command="pytest tests/",
    )
    low = h.lower()
    assert "pytest" in low
    assert "pythonpath" in low or "sys.path" in low
    assert "-v" in low or "tb=long" in low or "full-trace" in low


def test_incremental_recovery_hint_unittest():
    h = incremental_recovery_hint(
        "terminal",
        {},
        ErrorType.COMMAND_FAILED,
        "ERROR",
        last_failed_command="python -m unittest discover",
    )
    assert "unittest" in h.lower()


def test_incremental_recovery_hint_unittest_is_not_pip_package():
    h = incremental_recovery_hint(
        "terminal",
        {},
        ErrorType.COMMAND_FAILED,
        "ERROR: No matching distribution found for unittest",
        last_failed_command="pip install unittest",
    )
    low = h.lower()
    assert "built into python" in low
    assert "pip install unittest" in low
    assert "python -m unittest discover" in low


def test_incremental_recovery_hint_npm():
    h = incremental_recovery_hint(
        "terminal",
        {},
        ErrorType.COMMAND_FAILED,
        "npm ERR!",
        last_failed_command="npm test",
    )
    low = h.lower()
    assert "package.json" in low or "node_modules" in low


def test_incremental_recovery_hint_npm_missing_package_json():
    h = incremental_recovery_hint(
        "terminal",
        {},
        ErrorType.COMMAND_FAILED,
        "npm ERR! enoent Could not read package.json: Error: ENOENT: no such file or directory",
        last_failed_command="npm install",
    )
    low = h.lower()
    assert "package.json" in low
    assert "pwd" in low
    assert "wrong folder" in low or "wrong directory" in low or "create it first" in low


def test_incremental_recovery_hint_dev_server_port_conflict():
    h = incremental_recovery_hint(
        "terminal",
        {},
        ErrorType.COMMAND_FAILED,
        "listen EADDRINUSE: address already in use :::3000",
        last_failed_command="npm run dev",
    )
    low = h.lower()
    assert "3001" in h or "5173" in h or "4173" in h
    assert "lsof" in low or "netstat" in low
    assert "live preview" in low or "preview" in low


def test_incremental_recovery_hint_missing_port_tools():
    h = incremental_recovery_hint(
        "terminal",
        {},
        ErrorType.COMMAND_FAILED,
        "/bin/bash: line 1: lsof: command not found",
        last_failed_command="lsof -i :3000",
    )
    low = h.lower()
    assert "lsof" in low
    assert "different port" in low or "another port" in low


class TestAgentAutoReplan:
    @pytest.fixture
    def agent(self):
        llm = MagicMock()
        llm.conversation = []
        llm.add_user_message = MagicMock()
        llm.add_assistant_message = MagicMock()
        llm.add_tool_result = MagicMock()
        llm.get_usage_stats = MagicMock(return_value={"total_tokens": 0})
        agent = Agent(llm_client=llm, verbose=False)
        agent._use_conversation_memory = True
        agent._conversation_memory = None
        return agent

    def test_auto_replan_from_repeated_failure_marks_step_and_replans(self, agent, monkeypatch):
        task = TaskState(
            task_id="task-1",
            goal=TaskGoal(description="Fix login", acceptance_criteria=["works"]),
            status=TaskStatus.IN_PROGRESS,
        )
        plan = PlanState(
            plan_id="plan-1",
            task_id="task-1",
            steps=[
                PlanStep(step_id="step-1", description="Inspect", expected_outcome="understand", status=StepStatus.IN_PROGRESS),
                PlanStep(step_id="step-2", description="Fix", expected_outcome="code changed"),
            ],
        )
        agent.state.current_task = task
        agent.state.current_plan = plan

        agent.mark_plan_step_failed = MagicMock(side_effect=agent.mark_plan_step_failed)
        agent.add_to_memory = MagicMock(return_value="mem-1")
        agent.replan_from_failure = AsyncMock(return_value=MagicMock(success=True, plan=plan))

        ok = asyncio.run(agent.auto_replan_from_repeated_failure("same failure repeated"))

        assert ok is True
        agent.mark_plan_step_failed.assert_called_once_with("step-1", "same failure repeated")
        agent.replan_from_failure.assert_called_once_with("step-1", "same failure repeated")
        agent.add_to_memory.assert_called()


def test_incremental_recovery_hint_cwd_mismatch():
    h = incremental_recovery_hint(
        "terminal",
        {},
        ErrorType.FILE_NOT_FOUND,
        "/bin/bash: line 8: cd: my-web-app: No such file or directory",
        last_failed_command="cd my-web-app && npm run dev",
    )
    low = h.lower()
    assert "working directory" in low or "cwd" in low
    assert "pwd" in low
    assert "ls" in low
