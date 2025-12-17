"""
End-to-End Tests for Terminal and Editor Tools (Phase 10).

Tests that the terminal and editor tools work correctly together
for common software engineering tasks.
"""

import pytest
from pathlib import Path

from mini_devin.orchestrator.agent import Agent
from mini_devin.config.settings import AgentGatesSettings


class TestTerminalTool:
    """Tests for terminal tool functionality."""
    
    @pytest.mark.asyncio
    async def test_terminal_executes_command(self, mock_llm_client, temp_workspace):
        """Test that terminal tool can execute basic commands."""
        agent = Agent(
            llm_client=mock_llm_client,
            working_directory=temp_workspace,
            gates_settings=AgentGatesSettings(
                planning_required=False,
                review_required=False,
            ),
        )
        
        result = await agent._execute_tool("terminal", {"command": "echo 'hello world'"})
        
        assert "hello world" in result
    
    @pytest.mark.asyncio
    async def test_terminal_captures_output(self, mock_llm_client, temp_workspace):
        """Test that terminal captures command output correctly."""
        agent = Agent(
            llm_client=mock_llm_client,
            working_directory=temp_workspace,
            gates_settings=AgentGatesSettings(
                planning_required=False,
                review_required=False,
            ),
        )
        
        result = await agent._execute_tool("terminal", {"command": "pwd"})
        
        assert temp_workspace in result or "/" in result
    
    @pytest.mark.asyncio
    async def test_terminal_handles_errors(self, mock_llm_client, temp_workspace):
        """Test that terminal handles command errors gracefully."""
        agent = Agent(
            llm_client=mock_llm_client,
            working_directory=temp_workspace,
            gates_settings=AgentGatesSettings(
                planning_required=False,
                review_required=False,
            ),
        )
        
        result = await agent._execute_tool("terminal", {"command": "ls /nonexistent_directory_12345"})
        
        assert "error" in result.lower() or "no such file" in result.lower()


class TestEditorTool:
    """Tests for editor tool functionality."""
    
    @pytest.mark.asyncio
    async def test_editor_creates_file(self, mock_llm_client, temp_workspace):
        """Test that editor can create new files."""
        agent = Agent(
            llm_client=mock_llm_client,
            working_directory=temp_workspace,
            gates_settings=AgentGatesSettings(
                planning_required=False,
                review_required=False,
            ),
        )
        
        test_file = str(Path(temp_workspace) / "test_file.txt")
        await agent._execute_tool("editor", {
            "action": "create",
            "path": test_file,
            "content": "Hello, World!",
        })
        
        assert Path(test_file).exists()
        assert Path(test_file).read_text() == "Hello, World!"
    
    @pytest.mark.asyncio
    async def test_editor_reads_file(self, mock_llm_client, temp_workspace):
        """Test that editor can read existing files."""
        test_file = Path(temp_workspace) / "existing_file.txt"
        test_file.write_text("This is existing content.")
        
        agent = Agent(
            llm_client=mock_llm_client,
            working_directory=temp_workspace,
            gates_settings=AgentGatesSettings(
                planning_required=False,
                review_required=False,
            ),
        )
        
        result = await agent._execute_tool("editor", {
            "action": "read",
            "path": str(test_file),
        })
        
        assert "This is existing content." in result
    
    @pytest.mark.asyncio
    async def test_editor_updates_file(self, mock_llm_client, temp_workspace):
        """Test that editor can update existing files."""
        test_file = Path(temp_workspace) / "update_file.txt"
        test_file.write_text("Original content")
        
        agent = Agent(
            llm_client=mock_llm_client,
            working_directory=temp_workspace,
            gates_settings=AgentGatesSettings(
                planning_required=False,
                review_required=False,
            ),
        )
        
        await agent._execute_tool("editor", {
            "action": "update",
            "path": str(test_file),
            "old_content": "Original content",
            "new_content": "Updated content",
        })
        
        assert test_file.read_text() == "Updated content"
    
    @pytest.mark.asyncio
    async def test_editor_deletes_file(self, mock_llm_client, temp_workspace):
        """Test that editor can delete files."""
        test_file = Path(temp_workspace) / "delete_file.txt"
        test_file.write_text("To be deleted")
        
        agent = Agent(
            llm_client=mock_llm_client,
            working_directory=temp_workspace,
            gates_settings=AgentGatesSettings(
                planning_required=False,
                review_required=False,
            ),
        )
        
        await agent._execute_tool("editor", {
            "action": "delete",
            "path": str(test_file),
        })
        
        assert not test_file.exists()


class TestTerminalEditorIntegration:
    """Integration tests for terminal and editor working together."""
    
    @pytest.mark.asyncio
    async def test_create_and_run_script(self, mock_llm_client, temp_workspace):
        """Test creating a script with editor and running it with terminal."""
        agent = Agent(
            llm_client=mock_llm_client,
            working_directory=temp_workspace,
            gates_settings=AgentGatesSettings(
                planning_required=False,
                review_required=False,
            ),
        )
        
        script_path = str(Path(temp_workspace) / "hello.py")
        await agent._execute_tool("editor", {
            "action": "create",
            "path": script_path,
            "content": "print('Hello from script!')",
        })
        
        result = await agent._execute_tool("terminal", {
            "command": f"python {script_path}",
        })
        
        assert "Hello from script!" in result
    
    @pytest.mark.asyncio
    async def test_create_and_test_module(self, mock_llm_client, python_project):
        """Test creating a module and running tests on it."""
        agent = Agent(
            llm_client=mock_llm_client,
            working_directory=python_project,
            gates_settings=AgentGatesSettings(
                planning_required=False,
                review_required=False,
            ),
        )
        
        result = await agent._execute_tool("terminal", {
            "command": "python -m pytest test_main.py -v",
        })
        
        assert "passed" in result.lower() or "PASSED" in result
