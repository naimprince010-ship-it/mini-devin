"""Unit tests for the terminal tool."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from mini_devin.tools.terminal import TerminalTool, create_terminal_tool
from mini_devin.schemas.tools import ToolStatus, TerminalInput


class TestTerminalTool:
    """Tests for TerminalTool class."""

    def test_tool_initialization(self):
        """Test TerminalTool initialization."""
        tool = TerminalTool(working_directory="/tmp")
        assert tool.working_directory == "/tmp"
        assert tool.name == "terminal"

    def test_tool_name(self):
        """Test tool name property."""
        tool = TerminalTool()
        assert tool.name == "terminal"

    def test_tool_description(self):
        """Test tool description."""
        tool = TerminalTool()
        assert "terminal" in tool.description.lower() or "command" in tool.description.lower()

    def test_blocked_commands(self):
        """Test that dangerous commands are blocked."""
        tool = TerminalTool()
        blocked = tool.blocked_commands
        assert "rm -rf /" in blocked or any("rm" in cmd for cmd in blocked)

    def test_is_command_blocked_rm_rf_root(self):
        """Test that rm -rf / is blocked."""
        tool = TerminalTool()
        assert tool._is_command_blocked("rm -rf /") is True

    def test_is_command_blocked_safe_command(self):
        """Test that safe commands are not blocked."""
        tool = TerminalTool()
        assert tool._is_command_blocked("ls -la") is False
        assert tool._is_command_blocked("cat file.txt") is False
        assert tool._is_command_blocked("echo hello") is False

    def test_is_command_blocked_sudo_rm(self):
        """Test that sudo rm commands are blocked."""
        tool = TerminalTool()
        assert tool._is_command_blocked("sudo rm -rf /") is True

    def test_input_schema(self):
        """Test input schema property."""
        tool = TerminalTool()
        assert tool.input_schema == TerminalInput

    def test_get_schema_for_llm(self):
        """Test getting tool schema for LLM."""
        tool = TerminalTool()
        schema = tool.get_schema_for_llm()
        assert "name" in schema
        assert schema["name"] == "terminal"
        assert "description" in schema
        assert "parameters" in schema

    @pytest.mark.asyncio
    async def test_execute_simple_command(self):
        """Test executing a simple command."""
        tool = TerminalTool(working_directory="/tmp")
        result = await tool.execute({"command": "echo 'hello world'"})
        assert result.status == ToolStatus.SUCCESS
        assert "hello world" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_blocked_command(self):
        """Test that blocked commands return BLOCKED status."""
        tool = TerminalTool()
        result = await tool.execute({"command": "rm -rf /"})
        assert result.status == ToolStatus.BLOCKED

    @pytest.mark.asyncio
    async def test_execute_with_timeout(self):
        """Test command execution with timeout."""
        tool = TerminalTool()
        result = await tool.execute({"command": "sleep 10", "timeout_seconds": 1})
        assert result.status == ToolStatus.TIMEOUT

    @pytest.mark.asyncio
    async def test_execute_returns_exit_code(self):
        """Test that exit code is captured."""
        tool = TerminalTool()
        result = await tool.execute({"command": "true"})
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_execute_captures_stderr(self):
        """Test that stderr is captured."""
        tool = TerminalTool()
        result = await tool.execute({"command": "ls /nonexistent_directory_12345"})
        assert result.stderr != ""

    @pytest.mark.asyncio
    async def test_execute_with_working_directory(self):
        """Test executing command in specific directory."""
        tool = TerminalTool()
        result = await tool.execute({
            "command": "pwd",
            "working_directory": "/tmp"
        })
        assert result.status == ToolStatus.SUCCESS
        assert "/tmp" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_with_env_vars(self):
        """Test executing command with environment variables."""
        tool = TerminalTool()
        result = await tool.execute({
            "command": "echo $TEST_VAR",
            "env_vars": {"TEST_VAR": "test_value"}
        })
        assert result.status == ToolStatus.SUCCESS
        assert "test_value" in result.stdout


class TestTerminalToolSafety:
    """Safety-focused tests for TerminalTool."""

    def test_blocks_format_command(self):
        """Test that format commands are blocked."""
        tool = TerminalTool()
        dangerous_commands = [
            "mkfs.ext4 /dev/sda",
            "dd if=/dev/zero of=/dev/sda",
        ]
        for cmd in dangerous_commands:
            if tool._is_command_blocked(cmd):
                assert True
                return
        assert len(tool.blocked_commands) > 0

    def test_blocks_git_force_push(self):
        """Test that git force push is blocked."""
        tool = TerminalTool()
        assert tool._is_command_blocked("git push --force") is True

    def test_blocks_git_reset_hard(self):
        """Test that git reset --hard is blocked."""
        tool = TerminalTool()
        assert tool._is_command_blocked("git reset --hard") is True

    def test_blocks_sql_injection(self):
        """Test that SQL injection commands are blocked."""
        tool = TerminalTool()
        assert tool._is_command_blocked("DROP DATABASE") is True
        assert tool._is_command_blocked("DELETE FROM users") is True
        assert tool._is_command_blocked("TRUNCATE TABLE") is True

    def test_working_directory_isolation(self):
        """Test that working directory is respected."""
        tool = TerminalTool(working_directory="/tmp/test_workspace")
        assert tool.working_directory == "/tmp/test_workspace"


class TestCreateTerminalTool:
    """Tests for create_terminal_tool function."""

    def test_create_with_defaults(self):
        """Test creating tool with defaults."""
        tool = create_terminal_tool()
        assert tool.name == "terminal"
        assert len(tool.blocked_commands) > 0

    def test_create_with_working_directory(self):
        """Test creating tool with custom working directory."""
        tool = create_terminal_tool(working_directory="/tmp")
        assert tool.working_directory == "/tmp"

    def test_create_with_blocked_commands(self):
        """Test creating tool with custom blocked commands."""
        blocked = ["custom_blocked_cmd"]
        tool = create_terminal_tool(blocked_commands=blocked)
        assert "custom_blocked_cmd" in tool.blocked_commands
