"""
Self-Correction Engine for Plodder

This module provides the core logic for detecting tool failures, classifying
errors, and generating retry strategies (self-correction) before falling back
to full replanning.

Also includes **tool-loop resilience** helpers: error fingerprints, terminal
sanity checks, forced workspace diagnostics, and incremental recovery hints
(OpenHands-style guardrails — no extra LLM calls).
"""

from __future__ import annotations

from enum import Enum
import os
import re
import subprocess
from typing import Any, Dict, Optional, Tuple

def error_fingerprint(tool_name: str, tool_output: str, exit_code: Optional[int]) -> str:
    """
    Stable key for “same failure” detection (tool + exit + normalized error text).
    """
    snippet = re.sub(r"\s+", " ", (tool_output or "")[:500]).strip().lower()
    return f"{tool_name}|{exit_code}|{snippet}"


def terminal_sanity_check(command: str, *, is_windows: bool) -> Tuple[bool, str]:
    """
    Lightweight syntax / environment checks before running ``terminal``.

    Does not shell out — avoids bogus commands that always fail the same way.
    """
    cmd = (command or "").strip()
    if not cmd:
        return False, "Empty command."
    if "\x00" in cmd:
        return False, "Command contains null bytes."

    if not is_windows:
        if re.search(r"(?i)[A-Za-z]:\\", cmd) or re.search(r"(?i)G:\\\\|C:\\\\|D:\\\\", cmd.replace("/", "\\")):
            return (
                False,
                "Windows-style paths are invalid in this Linux/bash environment. "
                "Use POSIX paths under the workspace (e.g. `./src` or paths from `pwd`).",
            )
        if "&&" in cmd and "bash" not in os.environ.get("SHELL", "").lower():
            # Many deployments still run bash -lc; && is fine. Flag only obvious cmd.exe mixups.
            if re.match(r"(?i)^\s*cmd\s", cmd):
                return False, "Mixed cmd.exe syntax with bash; use `;` to chain in bash or a single `bash -lc` string."

    else:
        # Windows agent shell is often PowerShell — `&&` is invalid in older PS.
        if "&&" in cmd and not re.search(r"(?i)cmd\s+/c", cmd):
            return (
                False,
                "Use `;` to chain commands in PowerShell, or wrap in `cmd /c \"...\"` if you need `&&`.",
            )

    # Unbalanced quotes (simple heuristic)
    if cmd.count('"') % 2 != 0 or cmd.count("'") % 2 != 0:
        return False, "Unbalanced single or double quotes in command."

    return True, ""


def gather_workspace_diagnostics_sync(cwd: str, *, max_chars: int = 10_000) -> str:
    """
    Run ``pwd`` + directory listing without the LLM (bounded output).

    Uses POSIX commands on non-Windows and ``cd`` / ``dir`` on Windows.
    """
    root = os.path.abspath(cwd or ".")
    if not os.path.isdir(root):
        return f"(not a directory: {root})"
    chunks: list[str] = []
    try:
        if os.name == "nt":
            r = subprocess.run(
                ["cmd", "/c", "cd"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=20,
                shell=False,
            )
            chunks.append(f"cd:\n{(r.stdout or r.stderr or '').strip()}")
            r2 = subprocess.run(
                ["cmd", "/c", "dir"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=30,
                shell=False,
            )
            chunks.append(f"dir:\n{(r2.stdout or r2.stderr or '').strip()}")
        else:
            r = subprocess.run(
                ["pwd"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=15,
                shell=False,
            )
            chunks.append(f"pwd:\n{(r.stdout or r.stderr or '').strip()}")
            r2 = subprocess.run(
                ["ls", "-la"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=25,
                shell=False,
            )
            chunks.append(f"ls -la:\n{(r2.stdout or r2.stderr or '').strip()}")
            r3 = subprocess.run(
                ["sh", "-c", "find . -maxdepth 3 -type f 2>/dev/null | head -n 80"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=35,
                shell=False,
            )
            chunks.append(f"find (depth≤3, cap 80 files):\n{(r3.stdout or r3.stderr or '').strip()}")
    except Exception as e:
        chunks.append(f"(diagnostic subprocess error: {e})")
    out = "\n\n".join(chunks).strip()
    if len(out) > max_chars:
        out = out[:max_chars] + "\n…(truncated)…"
    return out or "(no diagnostic output)"


def format_system_correction_block(*, workspace_display: str, is_windows: bool) -> str:
    """Hard 'System Correction' text injected as a user message on repeated failures."""
    if is_windows:
        env = "You are on **Windows** (typically PowerShell). Use `;` to chain commands, `python` (not `python3` unless available), and drive paths only when they exist on this machine."
        path_hint = "Verify paths with `Test-Path` in PowerShell or `dir` before editing."
    else:
        env = "You are on **Linux/bash**. Do **not** use Windows paths (`C:\\\\`, `G:\\\\`, etc.)."
        path_hint = "The shell cwd should stay under the task workspace; use `pwd` and relative paths like `./src`."
    ws = workspace_display or "."
    return (
        "## System Correction (mandatory read)\n\n"
        f"{env}\n\n"
        f"**Current workspace (authoritative):** `{ws}`\n\n"
        f"{path_hint} For Python, prefer `os.path.exists(\"relative/path\")` (or `Path(...).exists()`) "
        "before `editor` read/write on files you have not listed yet.\n\n"
        "**Backtrack:** do not repeat the exact same failing command. Take one simpler diagnostic step "
        "(e.g. `ls` / `ls tests` / `python -m pytest --collect-only`) before retrying the heavy command."
    )


class ErrorType(str, Enum):
    """Types of errors that can occur during tool execution."""
    SUCCESS = "success"
    COMMAND_FAILED = "command_failed"
    # Host mismatch (e.g. Windows drive paths on Linux container) — retrying the same command never helps.
    ENVIRONMENT_MISMATCH = "environment_mismatch"
    FILE_NOT_FOUND = "file_not_found"
    PERMISSION_DENIED = "permission_denied"
    SYNTAX_ERROR = "syntax_error"
    TIMEOUT = "timeout"
    LLM_PARSE_ERROR = "llm_parse_error"
    DEPENDENCY_MISSING = "dependency_missing"
    UNKNOWN = "unknown"


def incremental_recovery_hint(
    tool_name: str,
    args: Dict[str, Any],
    error_type: ErrorType,
    output: str,
    *,
    last_failed_command: Optional[str] = None,
) -> str:
    """
    Action-aware incremental strategy (no LLM).

    ``last_failed_command`` overrides the terminal command string parsed from ``args``
    when you have a normalized view of the exact failing shell line.
    """
    ol = (output or "").lower()
    cmd = ""
    if tool_name == "terminal":
        cmd = (last_failed_command or "").strip() or str(args.get("command", ""))
    cmd_l = cmd.lower()

    if tool_name == "terminal" and (
        "lsof" in ol and "not found" in ol
        or "netstat" in ol and "not found" in ol
        or re.search(r"\bss:\s+command not found\b", ol)
        or "fuser" in ol and "not found" in ol
    ):
        return (
            "**Incremental recovery (missing port tools):** The environment lacks `lsof`/`netstat`/`ss`/`fuser`. "
            "Do not loop on shell diagnostics; retry the dev server on a different port and then inspect the "
            "app or attach live preview once it is running."
        )

    if tool_name == "terminal" and "no such file" in ol and "cd " in cmd_l:
        return (
            "**Incremental recovery (cwd/path):** The working directory may be wrong. "
            "Run **`pwd`** and **`ls`** first, confirm the target directory exists, then retry the command from the "
            "correct repo/app folder instead of assuming the previous `cd` still applies."
        )

    # --- Python tests (pytest / unittest) ---------------------------------
    if tool_name == "terminal" and ("pytest" in cmd_l or "py.test" in cmd_l):
        return (
            "**Incremental recovery (pytest):** (1) Confirm test files exist: `ls tests` / `find . -name 'test_*.py' | head`. "
            "(2) Check **PYTHONPATH** / package layout: `echo \"$PYTHONPATH\"`, `python -c \"import sys; print(sys.path)\"`. "
            "(3) Re-run with visibility: `python -m pytest -v --tb=long` (or `--full-trace`) so hidden tracebacks surface."
        )
    if tool_name == "terminal" and (
        "unittest" in cmd_l
        or "python -m unittest" in cmd_l
        or re.search(r"\bunittest\b", cmd_l)
    ):
        return (
            "**Incremental recovery (unittest):** Verify discovery paths: `ls -la`, "
            "`python -m unittest discover -v` from the package root, and ensure the module path matches the repo layout."
        )

    # --- Node / npm ecosystem --------------------------------------------
    if tool_name == "terminal" and any(
        k in cmd_l for k in (" npm ", "npm ", "npx ", "yarn ", "pnpm ", "node ", "pnpm", "yarn")
    ):
        if (
            "eaddrinuse" in ol
            or "address already in use" in ol
            or "port 3000 is in use" in ol
            or "port is already in use" in ol
        ) and any(k in cmd_l for k in ("dev", "start", "next", "vite")):
            return (
                "**Incremental recovery (dev server port conflict):** The default port is busy. "
                "Do not get stuck on `lsof`, `netstat`, `ss`, or `fuser` if they are missing. "
                "Retry the dev server on another port such as `3001`, `3002`, `4173`, or `5173`, "
                "then probe/attach live preview to the new port."
            )
        return (
            "**Incremental recovery (Node/npm):** Inspect `package.json` (`cat package.json` / `head package.json`), "
            "confirm `ls node_modules` (or `ls -la node_modules/<pkg>`), and verify you are in the directory that contains `package.json`."
        )

    # --- Editor / path ----------------------------------------------------
    if tool_name == "editor" and error_type == ErrorType.FILE_NOT_FOUND:
        p = args.get("path", "")
        bn = os.path.basename(str(p)).strip()
        find_hint = (
            f"`find . -maxdepth 4 -name '*{bn}*' 2>/dev/null | head`"
            if bn
            else "`find . -maxdepth 4 -type f 2>/dev/null | head -n 40`"
        )
        return (
            f"**Incremental recovery (editor):** Target `{p}` may be wrong. Run **`pwd`**, then {find_hint}, "
            "or `ls` on the parent directory before read/write."
        )

    if "file not found" in ol or "no such file" in ol:
        if "cd " in cmd_l or "cwd" in ol:
            return (
                "**Incremental recovery (cwd/path):** The working directory may be wrong. "
                "Run **`pwd`** and **`ls`** first, confirm the target directory exists, then retry the command from the "
                "correct repo/app folder instead of assuming the previous `cd` still applies."
            )
        return (
            "**Incremental recovery:** From workspace root, run **`pwd`**, **`ls`**, or **`find . -maxdepth 3`** "
            "to confirm the path before opening or executing files."
        )

    if error_type == ErrorType.DEPENDENCY_MISSING:
        return "**Incremental recovery:** Install the missing dependency or use `python -m <module>` with the interpreter from Runtime context."

    return "**Incremental recovery:** Change one variable at a time (path, cwd, or command) and verify with a minimal check."


class SelfCorrectionEngine:
    """
    Engine that provides self-correction capabilities for the agent.
    Analyzes tool outputs, classifies errors, and provides retry hints.
    """
    
    def __init__(self, max_immediate_retries: int = 3, max_replan_attempts: int = 2):
        self.max_immediate_retries = max_immediate_retries
        self.max_replan_attempts = max_replan_attempts
        
        # Track successful corrections for learning
        self.successful_corrections = []
        
    def classify_error(self, tool_name: str, tool_output: str, exit_code: Optional[int] = None) -> ErrorType:
        """
        Analyze tool output and exit code to determine the error type.
        
        Args:
            tool_name: The tool that was executed
            tool_output: The string output/result from the tool
            exit_code: Process exit code if applicable (e.g. for terminal)
            
        Returns:
            The classified ErrorType
        """
        if tool_name == "terminal":
            if exit_code == 0 and "Error:" not in tool_output and "Exception" not in tool_output:
                return ErrorType.SUCCESS
                
            out_lower = tool_output.lower()

            if (
                "windows-style paths" in out_lower
                or "invalid on this linux environment" in out_lower
                or "this shell runs on linux" in out_lower
            ):
                return ErrorType.ENVIRONMENT_MISMATCH
            
            if "syntaxerror" in out_lower or "indentationerror" in out_lower:
                return ErrorType.SYNTAX_ERROR

            if (
                "modulenotfounderror" in out_lower
                or "importerror" in out_lower
                or "command not found" in out_lower
                or "not recognized as an internal or external command" in out_lower
                or "'python3' is not recognized" in out_lower
                or '"python3" is not recognized' in out_lower
                or "no module named pytest" in out_lower
            ):
                return ErrorType.DEPENDENCY_MISSING

            if (
                "no such file or directory" in out_lower
                or "cannot find the path specified" in out_lower
                or "the system cannot find the path specified" in out_lower
            ):
                return ErrorType.FILE_NOT_FOUND

            if "permission denied" in out_lower or "access is denied" in out_lower:
                return ErrorType.PERMISSION_DENIED
                
            if "timeout" in out_lower or "timed out" in out_lower:
                return ErrorType.TIMEOUT
                
            return ErrorType.COMMAND_FAILED
            
        elif tool_name == "editor":
            if tool_output.startswith("Error:"):
                out_lower = tool_output.lower()
                if "no such file" in out_lower or "not found" in out_lower:
                    return ErrorType.FILE_NOT_FOUND
                if "permission denied" in out_lower:
                    return ErrorType.PERMISSION_DENIED
                return ErrorType.UNKNOWN
            return ErrorType.SUCCESS
            
        # For browser tools
        elif tool_name.startswith("browser_"):
            if tool_output.startswith("Error:") or tool_output.endswith("failed") or "failed:" in tool_output.lower():
                if "timeout" in tool_output.lower():
                    return ErrorType.TIMEOUT
                return ErrorType.UNKNOWN
            return ErrorType.SUCCESS
            
        # Default fallback
        if "Error:" in tool_output or "BLOCKED:" in tool_output:
            return ErrorType.UNKNOWN
            
        return ErrorType.SUCCESS

    def get_retry_hint(self, error_type: ErrorType, tool_name: str, args: Dict[str, Any], output: str) -> str:
        """
        Generate a helpful hint for the LLM to fix the error.
        
        Args:
            error_type: The classified ErrorType
            tool_name: The tool that failed
            args: The arguments passed to the tool
            output: The error output
            
        Returns:
            A string hint to guide the LLM's retry attempt
        """
        if error_type == ErrorType.SYNTAX_ERROR:
            return "There is a syntax error in the code you executed or wrote. Please carefully check indentation, brackets, and syntax rules for the language, fix the error, and try again."
            
        elif error_type == ErrorType.DEPENDENCY_MISSING:
            return (
                "A dependency or command is missing. Use the **Python executable from Runtime context**: "
                "`<that-python> -m pip install <package>` or `<that-python> -m pytest` / `-m unittest`. "
                "On Windows use `python`, not `python3`. On Linux servers use `python3` if `python` is missing."
            )
            
        elif error_type == ErrorType.FILE_NOT_FOUND:
            return (
                "The file or directory was not found. Use `editor` list_directory or terminal `pwd` / `ls` "
                "under the task workspace. On Linux, do NOT use Windows paths (C:\\\\). "
                "Use relative paths like `./src` from the workspace root shown in Runtime context."
            )
            
        elif error_type == ErrorType.PERMISSION_DENIED:
            return "Permission denied. You may need to change file permissions, use a different directory, or avoid modifying system-protected files."
            
        elif error_type == ErrorType.TIMEOUT:
            return "The operation timed out. If it's a server, run it in the background using `&` or as a daemon. Otherwise, try a smaller or optimized command."

        elif error_type == ErrorType.ENVIRONMENT_MISMATCH:
            return (
                "This shell does not support Windows drive paths (E:\\\\, C:\\\\, etc.). "
                "Use the workspace path from Runtime context: `pwd`, `ls`, or relative paths like `./mini-devin`. "
                "Do not repeat the same `cd` to a drive letter."
            )
            
        elif error_type == ErrorType.COMMAND_FAILED:
            if "Windows-style paths" in output or "runs on Linux" in output:
                return (
                    "The command used a Windows path on a Linux host. Remove drive letters (G:\\\\, C:\\\\); "
                    "use `mkdir -p ./name`, POSIX paths under the workspace, or `editor` write_file instead."
                )
            return (
                "The command failed with a non-zero exit code. Read STDERR, fix the root cause "
                "(wrong path, missing package, syntax), then retry—do not repeat the same failing command."
            )

        elif tool_name.startswith("browser_"):
            action = str(args.get("action") or tool_name).strip().lower()
            base = (
                "First capture a fresh `browser_screenshot` or `browser_playwright` snapshot and inspect "
                "the screenshot, accessibility tree, console, and interactive elements before retrying. "
            )
            if action in ("browser_navigate", "navigate"):
                return (
                    "The browser navigation failed. "
                    + base
                    + "Verify that the URL is valid and complete, including `https://` when needed. "
                    + "If a local app is still booting, confirm the dev server is running before navigating again. "
                    + "If the page may still be loading or redirecting, wait for it to settle and inspect any "
                    + "console or network errors before repeating `browser_navigate`."
                )
            if action in ("browser_click", "click"):
                return (
                    "The browser click failed. "
                    + base
                    + "Prefer a more specific CSS selector on retry; only use raw coordinates when the target is "
                    + "visually unambiguous. If a modal, cookie banner, or overlay may be intercepting input, "
                    + "dismiss that first before repeating the click."
                )
            if action in ("browser_type", "type"):
                return (
                    "The browser typing action failed. "
                    + base
                    + "Retry with a clearer input selector, use `clear_first` if the field already has text, "
                    + "and use `submit: true` only when Enter should submit the form."
                )
            if action in ("browser_scroll", "scroll"):
                return (
                    "The browser scroll failed. "
                    + base
                    + "Check whether the page is still loading, whether a modal locked scrolling, or whether "
                    + "the target content is inside a nested scroll container that needs a different interaction."
                )
            return (
                "The browser action failed. "
                + base
                + "Retry with a more precise target and only repeat the action after checking what changed on the page."
            )
            
        elif "BLOCKED" in output:
             return "This action was blocked by the safety guard. You must find an alternative approach that does not violate safety policies."
             
        # Fallback for UNKNOWN
        return "The tool execution failed. Please analyze the error message, correct your approach, and try again."

    def should_retry(self, error_type: ErrorType, current_retry_count: int) -> bool:
        """
        Determine if we should attempt an immediate retry for this error.
        
        Args:
            error_type: The classified error
            current_retry_count: How many retries we've already attempted for this step
            
        Returns:
            True if we should retry, False if we should escalate/fail
        """
        if error_type == ErrorType.SUCCESS:
            return False
            
        if current_retry_count >= self.max_immediate_retries:
            return False
            
        # Certain errors might never be worth retrying immediately without a new plan
        if error_type == ErrorType.PERMISSION_DENIED:
            return current_retry_count < 1  # Only retry once for perms

        if error_type == ErrorType.ENVIRONMENT_MISMATCH:
            return False
            
        return True

    def should_replan(self, current_consecutive_failures: int) -> bool:
        """
        Determine if failures have escalated enough to require replanning.
        
        Args:
            current_consecutive_failures: How many times tool execution has failed consecutively
            
        Returns:
            True if we should trigger full replanning
        """
        # If we've exhausted immediate retries (usually 3) plus an extra buffer
        return current_consecutive_failures >= (self.max_immediate_retries + 1)

    def record_correction(self, original_call: Dict[str, Any], original_error: str, successful_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Record a successful correction for future learning/memory.
        
        Args:
            original_call: The tool call that failed
            original_error: The error output
            successful_call: The tool call that eventually succeeded
            
        Returns:
            A dictionary representing the lesson learned
        """
        lesson = {
            "error_pattern": original_error[-500:],  # Store last 500 chars 
            "original_attempt": original_call,
            "solution": successful_call
        }
        self.successful_corrections.append(lesson)
        return lesson
