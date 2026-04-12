"""
Self-Correction Engine for Plodder

This module provides the core logic for detecting tool failures, classifying
errors, and generating retry strategies (self-correction) before falling back
to full replanning.
"""

from enum import Enum
from typing import Any, Dict, Optional, Tuple
import re

class ErrorType(str, Enum):
    """Types of errors that can occur during tool execution."""
    SUCCESS = "success"
    COMMAND_FAILED = "command_failed"
    FILE_NOT_FOUND = "file_not_found"
    PERMISSION_DENIED = "permission_denied"
    SYNTAX_ERROR = "syntax_error"
    TIMEOUT = "timeout"
    LLM_PARSE_ERROR = "llm_parse_error"
    DEPENDENCY_MISSING = "dependency_missing"
    UNKNOWN = "unknown"

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
