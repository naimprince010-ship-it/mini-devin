"""
Terminal Tool for Mini-Devin

This module implements the terminal tool that allows the agent to execute
shell commands in a controlled environment with proper output capture,
timeouts, and safety policies.
"""

import asyncio
import os
import time

from ..core.tool_interface import BaseTool, ToolPolicy
from ..schemas.tools import TerminalInput, TerminalOutput, ToolStatus


class TerminalTool(BaseTool[TerminalInput, TerminalOutput]):
    """
    Terminal tool for executing shell commands.
    
    Features:
    - Executes commands in a subprocess with proper output capture
    - Supports timeouts and resource limits
    - Tracks files modified by commands
    - Enforces command blocklist for safety
    - Truncates large outputs to prevent context overflow
    """
    
    def __init__(
        self,
        policy: ToolPolicy | None = None,
        working_directory: str | None = None,
        blocked_commands: list[str] | None = None,
        max_output_length: int = 50000,
    ):
        super().__init__(policy)
        self.working_directory = working_directory or os.getcwd()
        self.max_output_length = max_output_length
        
        # Default blocked commands for safety
        self.blocked_commands = blocked_commands or [
            "rm -rf /",
            "rm -rf /*",
            "sudo rm -rf",
            "mkfs",
            "dd if=/dev/zero",
            ":(){ :|:& };:",  # Fork bomb
            "> /dev/sda",
            "chmod -R 777 /",
            "git push --force",
            "git reset --hard",
            "DROP DATABASE",
            "DELETE FROM",
            "TRUNCATE TABLE",
        ]
    
    @property
    def name(self) -> str:
        return "terminal"
    
    @property
    def description(self) -> str:
        return """Execute shell commands in the terminal.
        
Use this tool to:
- Run build commands (npm install, pip install, cargo build, etc.)
- Execute tests (pytest, npm test, etc.)
- Run linters and formatters
- Check git status and make commits
- Navigate the filesystem
- Run any shell command needed for development

The command runs in a subprocess with the specified working directory.
Output is captured and returned. Long outputs are truncated."""
    
    @property
    def input_schema(self) -> type[TerminalInput]:
        return TerminalInput
    
    @property
    def output_schema(self) -> type[TerminalOutput]:
        return TerminalOutput
    
    def _is_command_blocked(self, command: str) -> bool:
        """Check if a command is in the blocklist."""
        command_lower = command.lower().strip()
        for blocked in self.blocked_commands:
            if blocked.lower() in command_lower:
                return True
        return False
    
    def _get_modified_files(
        self,
        working_dir: str,
        before_time: float,
    ) -> list[str]:
        """Get list of files modified after a given time."""
        modified = []
        try:
            for root, _, files in os.walk(working_dir):
                # Skip common directories that shouldn't be tracked
                if any(skip in root for skip in [".git", "node_modules", "__pycache__", ".venv"]):
                    continue
                for file in files:
                    filepath = os.path.join(root, file)
                    try:
                        if os.path.getmtime(filepath) > before_time:
                            modified.append(filepath)
                    except OSError:
                        continue
        except OSError:
            pass
        return modified[:100]  # Limit to 100 files
    
    def _truncate_output(self, output: str) -> tuple[str, bool]:
        """Truncate output if it exceeds max length."""
        if len(output) <= self.max_output_length:
            return output, False
        
        # Keep first and last parts
        half = self.max_output_length // 2
        truncated = (
            output[:half]
            + f"\n\n... [TRUNCATED {len(output) - self.max_output_length} characters] ...\n\n"
            + output[-half:]
        )
        return truncated, True
    
    async def _execute(self, input_data: TerminalInput) -> TerminalOutput:
        """Execute a shell command."""
        start_time = time.time()
        
        # Check if command is blocked
        if self._is_command_blocked(input_data.command):
            return TerminalOutput(
                status=ToolStatus.BLOCKED,
                error_message=f"Command blocked by safety policy: {input_data.command}",
                stdout="",
                stderr="Command blocked by safety policy",
                exit_code=-1,
                execution_time_ms=0,
            )
        
        # Resolve working directory
        working_dir = input_data.working_directory
        if not os.path.isabs(working_dir):
            working_dir = os.path.join(self.working_directory, working_dir)
        working_dir = os.path.abspath(working_dir)
        
        # Ensure working directory exists
        if not os.path.isdir(working_dir):
            return TerminalOutput(
                status=ToolStatus.FAILURE,
                error_message=f"Working directory does not exist: {working_dir}",
                stdout="",
                stderr=f"Directory not found: {working_dir}",
                exit_code=-1,
                execution_time_ms=0,
            )
        
        # Prepare environment
        env = os.environ.copy()
        env.update(input_data.env_vars)
        
        # Track file modifications
        before_time = time.time()
        
        try:
            # Run command with timeout
            process = await asyncio.create_subprocess_shell(
                input_data.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
                env=env,
            )
            
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=input_data.timeout_seconds,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                execution_time = int((time.time() - start_time) * 1000)
                return TerminalOutput(
                    status=ToolStatus.TIMEOUT,
                    error_message=f"Command timed out after {input_data.timeout_seconds} seconds",
                    stdout="",
                    stderr=f"Command timed out after {input_data.timeout_seconds}s",
                    exit_code=-1,
                    execution_time_ms=execution_time,
                )
            
            # Decode output
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            
            # Truncate if needed
            stdout, stdout_truncated = self._truncate_output(stdout)
            stderr, stderr_truncated = self._truncate_output(stderr)
            truncated = stdout_truncated or stderr_truncated
            
            # Get modified files
            modified_files = self._get_modified_files(working_dir, before_time)
            
            # Determine status
            exit_code = process.returncode or 0
            status = ToolStatus.SUCCESS if exit_code == 0 else ToolStatus.FAILURE
            
            execution_time = int((time.time() - start_time) * 1000)
            
            return TerminalOutput(
                status=status,
                error_message=None if exit_code == 0 else f"Command exited with code {exit_code}",
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                files_modified=modified_files,
                truncated=truncated,
                execution_time_ms=execution_time,
            )
            
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            return TerminalOutput(
                status=ToolStatus.FAILURE,
                error_message=str(e),
                stdout="",
                stderr=str(e),
                exit_code=-1,
                execution_time_ms=execution_time,
            )


# Convenience function to create a terminal tool
def create_terminal_tool(
    working_directory: str | None = None,
    blocked_commands: list[str] | None = None,
) -> TerminalTool:
    """Create a terminal tool with default settings."""
    return TerminalTool(
        working_directory=working_directory,
        blocked_commands=blocked_commands,
    )
