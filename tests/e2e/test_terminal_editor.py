"""
End-to-End Tests for Terminal and Editor Tools (Phase 10).

Tests that the terminal and editor tools work correctly together
for common software engineering tasks.
"""

import pytest
from pathlib import Path

from mini_devin.core.tool_interface import ToolRegistry
from mini_devin.orchestrator.agent import Agent
from mini_devin.tools.editor import create_editor_tool
from mini_devin.tools.terminal import create_terminal_tool


def _create_test_agent(mock_llm_client, working_directory: str) -> Agent:
    """Create an isolated agent with workspace-scoped tools."""
    registry = ToolRegistry()
    registry.register(create_terminal_tool(working_directory=working_directory))
    registry.register(create_editor_tool(working_directory=working_directory))
    return Agent(
        llm_client=mock_llm_client,
        working_directory=working_directory,
        tool_registry=registry,
    )



class TestTerminalTool:
    """Tests for terminal tool functionality."""
    
    @pytest.mark.asyncio
    async def test_terminal_executes_command(self, mock_llm_client, temp_workspace):
        """Test that terminal tool can execute basic commands."""
        agent = _create_test_agent(mock_llm_client, temp_workspace)
        
        result = await agent._execute_tool("terminal", {"command": "echo 'hello world'"})
        
        assert "hello world" in result
    
    @pytest.mark.asyncio
    async def test_terminal_captures_output(self, mock_llm_client, temp_workspace):
        """Test that terminal captures command output correctly."""
        agent = _create_test_agent(mock_llm_client, temp_workspace)
        
        result = await agent._execute_tool(
            "terminal",
            {"command": "python -c \"import os; print(os.getcwd())\""},
        )
        
        assert temp_workspace in result or "/" in result
    
    @pytest.mark.asyncio
    async def test_terminal_handles_errors(self, mock_llm_client, temp_workspace):
        """Test that terminal handles command errors gracefully."""
        agent = _create_test_agent(mock_llm_client, temp_workspace)
        
        result = await agent._execute_tool(
            "terminal",
            {"command": "python -c \"import os; os.chdir(r'nonexistent_directory_12345')\""},
        )
        
        assert "exit code: 1" in result.lower() or "no such file" in result.lower() or "cannot find" in result.lower()


class TestEditorTool:
    """Tests for editor tool functionality."""
    
    @pytest.mark.asyncio
    async def test_editor_creates_file(self, mock_llm_client, temp_workspace):
        """Test that editor can create new files."""
        agent = _create_test_agent(mock_llm_client, temp_workspace)
        
        test_file = Path(temp_workspace) / "test_file.txt"
        await agent._execute_tool("editor", {
            "action": "write_file",
            "path": "test_file.txt",
            "content": "Hello, World!",
        })
        
        assert test_file.exists()
        assert test_file.read_text() == "Hello, World!"
    
    @pytest.mark.asyncio
    async def test_editor_reads_file(self, mock_llm_client, temp_workspace):
        """Test that editor can read existing files."""
        test_file = Path(temp_workspace) / "existing_file.txt"
        test_file.write_text("This is existing content.")
        
        agent = _create_test_agent(mock_llm_client, temp_workspace)
        
        result = await agent._execute_tool("editor", {
            "action": "read_file",
            "path": str(test_file),
        })
        
        assert "This is existing content." in result
    
    @pytest.mark.asyncio
    async def test_editor_updates_file(self, mock_llm_client, temp_workspace):
        """Test that editor can update existing files."""
        test_file = Path(temp_workspace) / "update_file.txt"
        test_file.write_text("Original content")
        
        agent = _create_test_agent(mock_llm_client, temp_workspace)
        
        await agent._execute_tool("editor", {
            "action": "write_file",
            "path": "update_file.txt",
            "content": "Updated content",
        })
        
        assert test_file.read_text() == "Updated content"
    



class TestTerminalEditorIntegration:
    """Integration tests for terminal and editor working together."""
    
    @pytest.mark.asyncio
    async def test_create_and_run_script(self, mock_llm_client, temp_workspace):
        """Test creating a script with editor and running it with terminal."""
        agent = _create_test_agent(mock_llm_client, temp_workspace)
        
        script_path = Path(temp_workspace) / "hello.py"
        await agent._execute_tool("editor", {
            "action": "write_file",
            "path": "hello.py",
            "content": "print('Hello from script!')",
        })
        
        result = await agent._execute_tool("terminal", {
            "command": "python hello.py",
        })
        
        assert "Hello from script!" in result
    
    @pytest.mark.asyncio
    async def test_create_and_test_module(self, mock_llm_client, python_project):
        """Test creating a module and running tests on it."""
        agent = _create_test_agent(mock_llm_client, python_project)
        
        result = await agent._execute_tool("terminal", {
            "command": "python -m pytest test_main.py -v",
        })
        
        assert "passed" in result.lower() or "PASSED" in result
