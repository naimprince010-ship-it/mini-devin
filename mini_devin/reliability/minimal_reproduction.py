"""
Minimal Reproduction Module for Mini-Devin

This module extracts exact failing test/error information to focus fixes:
- Parse test output to find failing test names and line numbers
- Extract relevant error messages and stack traces
- Identify the minimal code context needed for the fix
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class FailureType(str, Enum):
    """Types of failures that can be extracted."""
    
    TEST_FAILURE = "test_failure"
    TEST_ERROR = "test_error"
    ASSERTION_ERROR = "assertion_error"
    IMPORT_ERROR = "import_error"
    SYNTAX_ERROR = "syntax_error"
    TYPE_ERROR = "type_error"
    RUNTIME_ERROR = "runtime_error"
    LINT_ERROR = "lint_error"
    UNKNOWN = "unknown"


@dataclass
class FailureLocation:
    """Location of a failure in code."""
    
    file_path: str
    line_number: int
    column: int | None = None
    function_name: str | None = None
    class_name: str | None = None
    
    def __str__(self) -> str:
        loc = f"{self.file_path}:{self.line_number}"
        if self.column:
            loc += f":{self.column}"
        return loc


@dataclass
class FailureInfo:
    """
    Information about a specific failure.
    
    Contains all the context needed to understand and fix the failure.
    """
    
    failure_type: FailureType
    message: str
    location: FailureLocation | None = None
    test_name: str | None = None
    expected: str | None = None
    actual: str | None = None
    stack_trace: list[str] = field(default_factory=list)
    relevant_code: str | None = None
    suggestion: str | None = None
    raw_output: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "failure_type": self.failure_type.value,
            "message": self.message,
            "location": str(self.location) if self.location else None,
            "test_name": self.test_name,
            "expected": self.expected,
            "actual": self.actual,
            "stack_trace": self.stack_trace,
            "suggestion": self.suggestion,
        }
    
    def get_focus_prompt(self) -> str:
        """Generate a focused prompt for fixing this failure."""
        parts = [f"Fix the following {self.failure_type.value}:"]
        
        if self.test_name:
            parts.append(f"Test: {self.test_name}")
        
        if self.location:
            parts.append(f"Location: {self.location}")
        
        parts.append(f"Error: {self.message}")
        
        if self.expected and self.actual:
            parts.append(f"Expected: {self.expected}")
            parts.append(f"Actual: {self.actual}")
        
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        
        return "\n".join(parts)


class FailureExtractor:
    """
    Extracts failure information from test/lint/build output.
    
    Supports multiple output formats:
    - pytest (Python)
    - jest (JavaScript)
    - ruff/flake8 (Python linting)
    - eslint (JavaScript linting)
    - mypy (Python type checking)
    - tsc (TypeScript)
    """
    
    def __init__(self):
        self._patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> dict[str, list[re.Pattern]]:
        """Compile regex patterns for different output formats."""
        return {
            "pytest": [
                # FAILED test_file.py::test_name - AssertionError
                re.compile(r"FAILED\s+(\S+)::(\S+)\s*[-–]\s*(.+)"),
                # E       AssertionError: assert 1 == 2
                re.compile(r"E\s+(\w+Error):\s*(.+)"),
                # >       assert result == expected
                re.compile(r">\s+assert\s+(.+)"),
                # test_file.py:10: AssertionError
                re.compile(r"(\S+\.py):(\d+):\s*(\w+Error)"),
            ],
            "jest": [
                # FAIL src/test.js
                re.compile(r"FAIL\s+(\S+)"),
                # ● Test Suite › test name
                re.compile(r"●\s+(.+?)\s*›\s*(.+)"),
                # expect(received).toBe(expected)
                re.compile(r"expect\((.+?)\)\.(\w+)\((.+?)\)"),
                # Expected: "foo"
                # Received: "bar"
                re.compile(r"Expected:\s*(.+)"),
                re.compile(r"Received:\s*(.+)"),
                # at Object.<anonymous> (src/test.js:10:5)
                re.compile(r"at\s+.+\s+\((.+):(\d+):(\d+)\)"),
            ],
            "ruff": [
                # file.py:10:5: E501 Line too long
                re.compile(r"(\S+\.py):(\d+):(\d+):\s*(\w+)\s+(.+)"),
            ],
            "eslint": [
                # /path/file.js:10:5: error message (rule-name)
                re.compile(r"(\S+\.js):(\d+):(\d+):\s*(error|warning)\s+(.+?)\s+\((.+)\)"),
            ],
            "mypy": [
                # file.py:10: error: Type mismatch
                re.compile(r"(\S+\.py):(\d+):\s*(error|warning):\s*(.+)"),
            ],
            "tsc": [
                # file.ts(10,5): error TS2322: Type mismatch
                re.compile(r"(\S+\.ts)\((\d+),(\d+)\):\s*(error)\s+(TS\d+):\s*(.+)"),
            ],
            "generic_error": [
                # Traceback (most recent call last):
                re.compile(r"Traceback \(most recent call last\):"),
                # File "file.py", line 10, in function
                re.compile(r'File "(.+)", line (\d+), in (\w+)'),
                # ErrorType: message
                re.compile(r"(\w+Error):\s*(.+)"),
                # SyntaxError: invalid syntax
                re.compile(r"(SyntaxError):\s*(.+)"),
            ],
        }
    
    def extract(self, output: str, tool_type: str | None = None) -> list[FailureInfo]:
        """
        Extract failure information from output.
        
        Args:
            output: The raw output from test/lint/build
            tool_type: Optional hint about the tool type (pytest, jest, etc.)
            
        Returns:
            List of FailureInfo objects
        """
        failures = []
        
        # Auto-detect tool type if not provided
        if tool_type is None:
            tool_type = self._detect_tool_type(output)
        
        # Extract based on tool type
        if tool_type == "pytest":
            failures.extend(self._extract_pytest(output))
        elif tool_type == "jest":
            failures.extend(self._extract_jest(output))
        elif tool_type == "ruff":
            failures.extend(self._extract_ruff(output))
        elif tool_type == "eslint":
            failures.extend(self._extract_eslint(output))
        elif tool_type == "mypy":
            failures.extend(self._extract_mypy(output))
        elif tool_type == "tsc":
            failures.extend(self._extract_tsc(output))
        else:
            failures.extend(self._extract_generic(output))
        
        # If no specific failures found, try generic extraction
        if not failures:
            failures.extend(self._extract_generic(output))
        
        return failures
    
    def _detect_tool_type(self, output: str) -> str:
        """Auto-detect the tool type from output."""
        output_lower = output.lower()
        
        if "pytest" in output_lower or "====" in output and "FAILED" in output:
            return "pytest"
        elif "jest" in output_lower or "PASS" in output and "FAIL" in output:
            return "jest"
        elif "ruff" in output_lower:
            return "ruff"
        elif "eslint" in output_lower:
            return "eslint"
        elif "mypy" in output_lower or "error: " in output and ".py:" in output:
            return "mypy"
        elif "tsc" in output_lower or "error TS" in output:
            return "tsc"
        else:
            return "generic"
    
    def _extract_pytest(self, output: str) -> list[FailureInfo]:
        """Extract failures from pytest output."""
        failures = []
        lines = output.split("\n")
        
        current_test = None
        current_file = None
        current_line = None
        error_message = None
        expected = None
        actual = None
        stack_trace = []
        
        for i, line in enumerate(lines):
            # Match FAILED line
            match = self._patterns["pytest"][0].search(line)
            if match:
                # Save previous failure if exists
                if current_test and error_message:
                    failures.append(FailureInfo(
                        failure_type=FailureType.TEST_FAILURE,
                        message=error_message,
                        location=FailureLocation(
                            file_path=current_file or "",
                            line_number=current_line or 0,
                        ) if current_file else None,
                        test_name=current_test,
                        expected=expected,
                        actual=actual,
                        stack_trace=stack_trace,
                        raw_output=output,
                    ))
                
                current_file = match.group(1)
                current_test = match.group(2)
                error_message = match.group(3)
                current_line = None
                expected = None
                actual = None
                stack_trace = []
                continue
            
            # Match error line
            match = self._patterns["pytest"][1].search(line)
            if match:
                error_message = f"{match.group(1)}: {match.group(2)}"
                continue
            
            # Match file:line
            match = self._patterns["pytest"][3].search(line)
            if match:
                current_file = match.group(1)
                current_line = int(match.group(2))
                continue
            
            # Collect stack trace
            if line.strip().startswith(">") or line.strip().startswith("E "):
                stack_trace.append(line.strip())
        
        # Save last failure
        if current_test and error_message:
            failures.append(FailureInfo(
                failure_type=FailureType.TEST_FAILURE,
                message=error_message,
                location=FailureLocation(
                    file_path=current_file or "",
                    line_number=current_line or 0,
                ) if current_file else None,
                test_name=current_test,
                expected=expected,
                actual=actual,
                stack_trace=stack_trace,
                raw_output=output,
            ))
        
        return failures
    
    def _extract_jest(self, output: str) -> list[FailureInfo]:
        """Extract failures from jest output."""
        failures = []
        lines = output.split("\n")
        
        current_test = None
        current_file = None
        current_line = None
        error_message = None
        expected = None
        actual = None
        
        for i, line in enumerate(lines):
            # Match FAIL line
            match = self._patterns["jest"][0].search(line)
            if match:
                current_file = match.group(1)
                continue
            
            # Match test name
            match = self._patterns["jest"][1].search(line)
            if match:
                current_test = f"{match.group(1)} › {match.group(2)}"
                continue
            
            # Match expected
            match = self._patterns["jest"][3].search(line)
            if match:
                expected = match.group(1)
                continue
            
            # Match received (actual)
            match = self._patterns["jest"][4].search(line)
            if match:
                actual = match.group(1)
                continue
            
            # Match location
            match = self._patterns["jest"][5].search(line)
            if match:
                current_file = match.group(1)
                current_line = int(match.group(2))
                
                # Create failure info
                if current_test:
                    error_message = f"Expected {expected}, received {actual}" if expected and actual else "Test failed"
                    failures.append(FailureInfo(
                        failure_type=FailureType.TEST_FAILURE,
                        message=error_message,
                        location=FailureLocation(
                            file_path=current_file,
                            line_number=current_line,
                            column=int(match.group(3)),
                        ),
                        test_name=current_test,
                        expected=expected,
                        actual=actual,
                        raw_output=output,
                    ))
                    current_test = None
                    expected = None
                    actual = None
        
        return failures
    
    def _extract_ruff(self, output: str) -> list[FailureInfo]:
        """Extract failures from ruff output."""
        failures = []
        
        for match in self._patterns["ruff"][0].finditer(output):
            failures.append(FailureInfo(
                failure_type=FailureType.LINT_ERROR,
                message=f"{match.group(4)}: {match.group(5)}",
                location=FailureLocation(
                    file_path=match.group(1),
                    line_number=int(match.group(2)),
                    column=int(match.group(3)),
                ),
                raw_output=output,
            ))
        
        return failures
    
    def _extract_eslint(self, output: str) -> list[FailureInfo]:
        """Extract failures from eslint output."""
        failures = []
        
        for match in self._patterns["eslint"][0].finditer(output):
            failures.append(FailureInfo(
                failure_type=FailureType.LINT_ERROR,
                message=f"{match.group(5)} ({match.group(6)})",
                location=FailureLocation(
                    file_path=match.group(1),
                    line_number=int(match.group(2)),
                    column=int(match.group(3)),
                ),
                raw_output=output,
            ))
        
        return failures
    
    def _extract_mypy(self, output: str) -> list[FailureInfo]:
        """Extract failures from mypy output."""
        failures = []
        
        for match in self._patterns["mypy"][0].finditer(output):
            failures.append(FailureInfo(
                failure_type=FailureType.TYPE_ERROR,
                message=match.group(4),
                location=FailureLocation(
                    file_path=match.group(1),
                    line_number=int(match.group(2)),
                ),
                raw_output=output,
            ))
        
        return failures
    
    def _extract_tsc(self, output: str) -> list[FailureInfo]:
        """Extract failures from TypeScript compiler output."""
        failures = []
        
        for match in self._patterns["tsc"][0].finditer(output):
            failures.append(FailureInfo(
                failure_type=FailureType.TYPE_ERROR,
                message=f"{match.group(5)}: {match.group(6)}",
                location=FailureLocation(
                    file_path=match.group(1),
                    line_number=int(match.group(2)),
                    column=int(match.group(3)),
                ),
                raw_output=output,
            ))
        
        return failures
    
    def _extract_generic(self, output: str) -> list[FailureInfo]:
        """Extract failures using generic patterns."""
        failures = []
        lines = output.split("\n")
        
        current_file = None
        current_line = None
        current_function = None
        error_type = None
        error_message = None
        stack_trace = []
        in_traceback = False
        
        for line in lines:
            # Check for traceback start
            if self._patterns["generic_error"][0].search(line):
                in_traceback = True
                stack_trace = []
                continue
            
            # Match file location in traceback
            match = self._patterns["generic_error"][1].search(line)
            if match and in_traceback:
                current_file = match.group(1)
                current_line = int(match.group(2))
                current_function = match.group(3)
                stack_trace.append(line.strip())
                continue
            
            # Match error type and message
            match = self._patterns["generic_error"][2].search(line)
            if match:
                error_type = match.group(1)
                error_message = match.group(2)
                
                # Determine failure type
                failure_type = FailureType.RUNTIME_ERROR
                if "AssertionError" in error_type:
                    failure_type = FailureType.ASSERTION_ERROR
                elif "ImportError" in error_type or "ModuleNotFoundError" in error_type:
                    failure_type = FailureType.IMPORT_ERROR
                elif "SyntaxError" in error_type:
                    failure_type = FailureType.SYNTAX_ERROR
                elif "TypeError" in error_type:
                    failure_type = FailureType.TYPE_ERROR
                
                failures.append(FailureInfo(
                    failure_type=failure_type,
                    message=f"{error_type}: {error_message}",
                    location=FailureLocation(
                        file_path=current_file or "",
                        line_number=current_line or 0,
                        function_name=current_function,
                    ) if current_file else None,
                    stack_trace=stack_trace.copy(),
                    raw_output=output,
                ))
                
                in_traceback = False
                stack_trace = []
                continue
            
            # Collect stack trace lines
            if in_traceback and line.strip():
                stack_trace.append(line.strip())
        
        return failures
    
    def get_minimal_context(
        self,
        failure: FailureInfo,
        workspace_path: str,
        context_lines: int = 10,
    ) -> str | None:
        """
        Get the minimal code context around a failure.
        
        Args:
            failure: The failure info
            workspace_path: Path to the workspace
            context_lines: Number of lines of context to include
            
        Returns:
            The relevant code snippet or None if not found
        """
        if not failure.location:
            return None
        
        file_path = Path(workspace_path) / failure.location.file_path
        if not file_path.exists():
            # Try without leading path components
            file_path = Path(workspace_path) / Path(failure.location.file_path).name
            if not file_path.exists():
                return None
        
        try:
            with open(file_path) as f:
                lines = f.readlines()
            
            line_num = failure.location.line_number
            start = max(0, line_num - context_lines - 1)
            end = min(len(lines), line_num + context_lines)
            
            context_lines_list = []
            for i in range(start, end):
                prefix = ">>> " if i == line_num - 1 else "    "
                context_lines_list.append(f"{prefix}{i + 1}: {lines[i].rstrip()}")
            
            return "\n".join(context_lines_list)
        except Exception:
            return None


def extract_failure_info(output: str, tool_type: str | None = None) -> list[FailureInfo]:
    """
    Convenience function to extract failure info from output.
    
    Args:
        output: The raw output from test/lint/build
        tool_type: Optional hint about the tool type
        
    Returns:
        List of FailureInfo objects
    """
    extractor = FailureExtractor()
    return extractor.extract(output, tool_type)
