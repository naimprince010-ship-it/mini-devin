"""
Terminal Tool for Plodder

This module implements the terminal tool that allows the agent to execute
shell commands in a controlled environment with proper output capture,
timeouts, and safety policies.

On Windows the tool automatically:
  1. Translates common Linux/Unix commands to PowerShell equivalents
  2. Runs through PowerShell so Unix-like aliases (ls, cat, pwd…) work
"""

import asyncio
import os
import re
import sys
import time
from collections.abc import Callable

from ..core.tool_interface import BaseTool, ToolPolicy
from .host_paths import command_uses_windows_drive_paths, linux_workspace_hint
from ..schemas.tools import TerminalInput, TerminalOutput, ToolStatus

# ── Windows command translation ────────────────────────────────────────────────

_IS_WINDOWS = sys.platform == "win32"


def _split_compound_command(cmd: str) -> list[str]:
    """
    Split on ';' and '&&' so each clause can be translated separately.
    Naive split (does not handle semicolons inside quoted strings); good enough
    for typical agent one-liners like: rm -rf a; rm -rf b; npx create-react-app c
    """
    parts: list[str] = []
    for chunk in re.split(r"\s*;\s*|\s*&&\s*", cmd):
        c = chunk.strip()
        if c:
            parts.append(c)
    return parts if parts else [cmd.strip()]


def _translate_windows_statement(cmd: str) -> str:
    """
    Translate a single shell statement (no ';' / '&&' at this level).
    Called per segment after compound split.
    """
    # ── python3 → python ──────────────────────────────────────────────────────
    cmd = re.sub(r'\bpython3(\b|\s)', r'python\1', cmd)
    cmd = re.sub(r'\bpython3\.\d+\b', 'python', cmd)

    # ── pip3 → pip ────────────────────────────────────────────────────────────
    cmd = re.sub(r'\bpip3\b', 'pip', cmd)

    # ── chmod / chown → noop (no-op on Windows) ───────────────────────────────
    if re.match(r'^\s*ch(mod|own)\b', cmd):
        return 'echo "(chmod/chown skipped on Windows)"'

    # ── which → where ─────────────────────────────────────────────────────────
    cmd = re.sub(r'\bwhich\b', 'where', cmd)

    # ── touch file → New-Item -Force ──────────────────────────────────────────
    m = re.match(r'^\s*touch\s+(.+)$', cmd)
    if m:
        files = m.group(1).strip()
        return f'foreach ($f in @({", ".join(repr(f) for f in files.split())})) {{ New-Item -Force -ItemType File -Path $f | Out-Null }}'

    # ── mkdir -p dir → New-Item -Force ────────────────────────────────────────
    m = re.match(r'^\s*mkdir\s+-p\s+(.+)$', cmd)
    if m:
        d = m.group(1).strip()
        return f'New-Item -Force -ItemType Directory -Path "{d}" | Out-Null'

    # ── rm -rf dir → Remove-Item (skip if missing so chains do not abort) ─────
    m = re.match(r'^\s*rm\s+-rf?\s+(.+)$', cmd)
    if m:
        target = m.group(1).strip().strip('"').strip("'")
        return (
            f'if (Test-Path "{target}") '
            f'{{ Remove-Item -Recurse -Force "{target}" -ErrorAction Stop }}'
        )

    # ── rm file ────────────────────────────────────────────────────────────────
    m = re.match(r'^\s*rm\s+(.+)$', cmd)
    if m:
        target = m.group(1).strip()
        return f'Remove-Item "{target}"'

    # ── cp src dst → Copy-Item ────────────────────────────────────────────────
    m = re.match(r'^\s*cp\s+(-r\s+)?(.+?)\s+(\S+)$', cmd)
    if m:
        recurse = '-Recurse ' if m.group(1) else ''
        src, dst = m.group(2).strip(), m.group(3).strip()
        return f'Copy-Item {recurse}"{src}" "{dst}"'

    # ── mv src dst → Move-Item ────────────────────────────────────────────────
    m = re.match(r'^\s*mv\s+(.+?)\s+(\S+)$', cmd)
    if m:
        src, dst = m.group(1).strip(), m.group(2).strip()
        return f'Move-Item "{src}" "{dst}"'

    # ── cat file → Get-Content ────────────────────────────────────────────────
    m = re.match(r'^\s*cat\s+(.+)$', cmd)
    if m:
        f = m.group(1).strip()
        return f'Get-Content "{f}"'

    # ── head -n N file ────────────────────────────────────────────────────────
    m = re.match(r'^\s*head\s+(?:-n\s+)?(\d+)\s+(.+)$', cmd)
    if m:
        n, f = m.group(1), m.group(2).strip()
        return f'Get-Content "{f}" | Select-Object -First {n}'

    # ── tail -n N file ────────────────────────────────────────────────────────
    m = re.match(r'^\s*tail\s+(?:-n\s+)?(\d+)\s+(.+)$', cmd)
    if m:
        n, f = m.group(1), m.group(2).strip()
        return f'Get-Content "{f}" | Select-Object -Last {n}'

    # ── grep pattern file → Select-String ────────────────────────────────────
    m = re.match(r'^\s*grep\s+(?:-[^ ]+\s+)*"?([^"]+)"?\s+(.+)$', cmd)
    if m:
        pat, f = m.group(1), m.group(2).strip()
        return f'Select-String -Pattern "{pat}" -Path "{f}"'

    # ── wc -l file ────────────────────────────────────────────────────────────
    m = re.match(r'^\s*wc\s+-l\s+(.+)$', cmd)
    if m:
        f = m.group(1).strip()
        return f'(Get-Content "{f}").Count'

    # ── export VAR=val → $env:VAR = val ──────────────────────────────────────
    m = re.match(r'^\s*export\s+(\w+)=(.*)$', cmd)
    if m:
        var, val = m.group(1), m.group(2).strip()
        return f'$env:{var} = "{val}"'

    # ── source / . file → . file (PowerShell dot-sourcing) ───────────────────
    m = re.match(r'^\s*(source|\\.)\s+(.+)$', cmd)
    if m:
        f = m.group(2).strip()
        return f'. "{f}"'

    # ── clear → Clear-Host ───────────────────────────────────────────────────
    if re.match(r'^\s*clear\s*$', cmd):
        return 'Clear-Host'

    # ── /dev/null → $null ─────────────────────────────────────────────────────
    cmd = cmd.replace('/dev/null', '$null')

    return cmd


def _translate_for_windows(cmd: str) -> str:
    """
    Translate common Linux/Unix shell commands to PowerShell-compatible
    equivalents so the agent's Linux-style commands work on Windows.

    Compound commands (separated by ';' or '&&') are split so each clause is
    translated; otherwise a line like ``rm -rf a; rm -rf b; npx ...`` was
    incorrectly turned into a single Remove-Item with an invalid path.
    """
    if not _IS_WINDOWS:
        return cmd

    segments = _split_compound_command(cmd)
    if len(segments) == 1:
        return _translate_windows_statement(segments[0])
    return "; ".join(_translate_windows_statement(s) for s in segments)


async def _run_subprocess_with_streaming(
    process: asyncio.subprocess.Process,
    timeout: float,
    on_line: Callable[[str], None] | None,
) -> tuple[bytes, bytes, int]:
    out_chunks: list[bytes] = []
    err_chunks: list[bytes] = []

    async def drain(
        reader: asyncio.StreamReader | None,
        chunks: list[bytes],
        prefix: str,
    ) -> None:
        if reader is None:
            return
        while True:
            line = await reader.readline()
            if not line:
                break
            chunks.append(line)
            if on_line:
                text = line.decode("utf-8", errors="replace").rstrip("\r\n")
                if text:
                    on_line(f"{prefix}{text}" if prefix else text)

    await asyncio.wait_for(
        asyncio.gather(
            drain(process.stdout, out_chunks, ""),
            drain(process.stderr, err_chunks, "[stderr] "),
        ),
        timeout=timeout,
    )
    code = await process.wait()
    return b"".join(out_chunks), b"".join(err_chunks), int(code if code is not None else 0)


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
        bridge_session_id: str | None = None,
        on_output_line: Callable[[str], None] | None = None,
    ):
        super().__init__(policy)
        self.working_directory = working_directory or os.getcwd()
        self.max_output_length = max_output_length
        self.bridge_session_id = bridge_session_id
        self.on_output_line = on_output_line
        
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

        # Fail fast on Windows paths when the server is POSIX (avoids useless retry loops).
        if os.name != "nt" and command_uses_windows_drive_paths(input_data.command):
            wd = os.path.abspath(self.working_directory)
            msg = linux_workspace_hint(wd)
            return TerminalOutput(
                status=ToolStatus.FAILURE,
                error_message="Windows-style paths are invalid on this Linux environment.",
                stdout="",
                stderr=msg,
                exit_code=-1,
                execution_time_ms=0,
            )
        
        # Resolve working directory
        working_dir = input_data.working_directory
        if not os.path.isabs(working_dir):
            working_dir = os.path.join(self.working_directory, working_dir)
        working_dir = os.path.abspath(working_dir)

        cs = input_data.command.strip()
        pkg_json = os.path.join(working_dir, "package.json")

        # Clearer than npm's error when agents blindly re-run init/install in the wrong JS directory
        if re.match(r"^\s*npm\s+install(\s|$)", cs) and not os.path.isfile(pkg_json):
            return TerminalOutput(
                status=ToolStatus.FAILURE,
                error_message="package.json not found; create it or switch to the app directory before npm install.",
                stdout="",
                stderr=(
                    f"`npm install` requires `package.json` in the working directory. None found at {pkg_json}. "
                    "Create `package.json` first or run the command from the folder that contains it."
                ),
                exit_code=1,
                execution_time_ms=0,
            )

        if not _IS_WINDOWS:
            if cs.startswith("npm init") and "-y" in cs:
                if os.path.isfile(pkg_json):
                    return TerminalOutput(
                        status=ToolStatus.FAILURE,
                        error_message="package.json already exists; skip npm init -y in this directory.",
                        stdout="",
                        stderr=(
                            f"`package.json` already exists at {pkg_json}. "
                            "Edit it or use a new subfolder before running `npm init -y`."
                        ),
                        exit_code=1,
                        execution_time_ms=0,
                    )
        
        # Ensure working directory exists (server-side); local bridge uses its own cwd
        if not self.bridge_session_id and not os.path.isdir(working_dir):
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

        # Translate Linux commands to Windows equivalents when on Windows
        command = _translate_for_windows(input_data.command)

        # Local bridge: run on the user's machine (Cursor-like) when connected
        if self.bridge_session_id:
            try:
                from ..bridge.manager import get_bridge_manager

                bm = get_bridge_manager()
                if bm.is_connected(self.bridge_session_id):
                    raw = await bm.forward_exec(
                        self.bridge_session_id,
                        input_data.command,
                        working_dir,
                        float(input_data.timeout_seconds),
                        input_data.env_vars,
                    )
                    if raw is not None:
                        execution_time = int((time.time() - start_time) * 1000)
                        exit_code = int(raw.get("exit_code", -1))
                        stdout = str(raw.get("stdout") or "")
                        stderr = str(raw.get("stderr") or "")
                        stdout, st1 = self._truncate_output(stdout)
                        stderr, st2 = self._truncate_output(stderr)
                        status = ToolStatus.SUCCESS if exit_code == 0 else ToolStatus.FAILURE
                        prefix = "[local bridge] "
                        return TerminalOutput(
                            status=status,
                            error_message=None if exit_code == 0 else f"Command exited with code {exit_code}",
                            stdout=prefix + stdout,
                            stderr=stderr,
                            exit_code=exit_code,
                            files_modified=[],
                            truncated=st1 or st2,
                            execution_time_ms=execution_time,
                        )
            except Exception as e:
                print(f"[terminal] local bridge error (falling back to server shell): {e}")

        if not os.path.isdir(working_dir):
            return TerminalOutput(
                status=ToolStatus.FAILURE,
                error_message=f"Working directory does not exist: {working_dir}",
                stdout="",
                stderr=f"Directory not found: {working_dir}",
                exit_code=-1,
                execution_time_ms=0,
            )

        # Track file modifications
        before_time = time.time()

        try:
            # On Windows: run via PowerShell so Unix-like aliases (ls, cat, pwd…) work
            if _IS_WINDOWS:
                process = await asyncio.create_subprocess_exec(
                    "powershell", "-NoProfile", "-NonInteractive",
                    "-ExecutionPolicy", "Bypass",
                    "-Command", command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=working_dir,
                    env=env,
                )
            else:
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=working_dir,
                    env=env,
                )
            
            try:
                if self.on_output_line:
                    stdout_bytes, stderr_bytes, exit_code = await _run_subprocess_with_streaming(
                        process,
                        float(input_data.timeout_seconds),
                        self.on_output_line,
                    )
                else:
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(
                        process.communicate(),
                        timeout=input_data.timeout_seconds,
                    )
                    exit_code = process.returncode or 0
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
            status = ToolStatus.SUCCESS if exit_code == 0 else ToolStatus.FAILURE
            
            execution_time = int((time.time() - start_time) * 1000)
            
            # Prefix translated command note if it changed
            translated_note = ""
            if _IS_WINDOWS and command != input_data.command:
                translated_note = f"[Windows: translated '{input_data.command}' → '{command}']\n"

            return TerminalOutput(
                status=status,
                error_message=None if exit_code == 0 else f"Command exited with code {exit_code}",
                stdout=translated_note + stdout,
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
    bridge_session_id: str | None = None,
    on_output_line: Callable[[str], None] | None = None,
) -> TerminalTool:
    """Create a terminal tool with default settings."""
    return TerminalTool(
        working_directory=working_directory,
        blocked_commands=blocked_commands,
        bridge_session_id=bridge_session_id,
        on_output_line=on_output_line,
    )
