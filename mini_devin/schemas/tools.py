"""
Tool Schemas for Mini-Devin

This module defines the Pydantic schemas for all tools available to the agent:
- Terminal Tool: Execute shell commands
- Editor Tool: Read, write, search, and modify code files
- Browser Tool: Search the web, fetch pages, and interact with websites
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# =============================================================================
# Base Tool Schemas
# =============================================================================


class ToolStatus(str, Enum):
    """Status of a tool execution."""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"  # Blocked by policy/guardrails


class BaseToolInput(BaseModel):
    """Base class for all tool inputs."""
    pass


class BaseToolOutput(BaseModel):
    """Base class for all tool outputs."""
    status: ToolStatus
    error_message: str | None = None
    execution_time_ms: int = Field(description="Execution time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Terminal Tool Schemas
# =============================================================================


class TerminalInput(BaseToolInput):
    """Input schema for terminal command execution."""
    command: str = Field(description="The shell command to execute")
    working_directory: str = Field(
        default=".",
        description="Working directory for command execution"
    )
    timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Maximum execution time in seconds"
    )
    env_vars: dict[str, str] = Field(
        default_factory=dict,
        description="Additional environment variables"
    )


class TerminalOutput(BaseToolOutput):
    """Output schema for terminal command execution."""
    stdout: str = Field(description="Standard output from the command")
    stderr: str = Field(description="Standard error from the command")
    exit_code: int = Field(description="Exit code of the command")
    files_modified: list[str] = Field(
        default_factory=list,
        description="List of files modified by the command"
    )
    truncated: bool = Field(
        default=False,
        description="Whether output was truncated due to size limits"
    )


# =============================================================================
# Editor Tool Schemas
# =============================================================================


class EditorAction(str, Enum):
    """Available editor actions."""
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    APPLY_PATCH = "apply_patch"
    SEARCH = "search"
    LIST_DIRECTORY = "list_directory"
    GET_SYMBOLS = "get_symbols"
    GOTO_DEFINITION = "goto_definition"
    FIND_REFERENCES = "find_references"
    GET_DIAGNOSTICS = "get_diagnostics"


class FileRange(BaseModel):
    """Represents a range within a file."""
    start_line: int = Field(ge=1, description="Starting line number (1-indexed)")
    end_line: int | None = Field(
        default=None,
        ge=1,
        description="Ending line number (1-indexed, inclusive)"
    )


class ReadFileInput(BaseToolInput):
    """Input for reading a file."""
    action: EditorAction = EditorAction.READ_FILE
    path: str = Field(description="Path to the file to read")
    line_range: FileRange | None = Field(
        default=None,
        description="Optional line range to read"
    )


class ReadFileOutput(BaseToolOutput):
    """Output from reading a file."""
    content: str = Field(description="File content")
    total_lines: int = Field(description="Total number of lines in the file")
    language: str | None = Field(
        default=None,
        description="Detected programming language"
    )


class WriteFileInput(BaseToolInput):
    """Input for writing a file."""
    action: EditorAction = EditorAction.WRITE_FILE
    path: str = Field(description="Path to the file to write")
    content: str = Field(description="Content to write to the file")
    create_directories: bool = Field(
        default=True,
        description="Create parent directories if they don't exist"
    )


class WriteFileOutput(BaseToolOutput):
    """Output from writing a file."""
    bytes_written: int = Field(description="Number of bytes written")
    path: str = Field(description="Absolute path to the written file")


class ApplyPatchInput(BaseToolInput):
    """Input for applying a unified diff patch."""
    action: EditorAction = EditorAction.APPLY_PATCH
    path: str = Field(description="Path to the file to patch")
    patch: str = Field(description="Unified diff patch to apply")
    dry_run: bool = Field(
        default=False,
        description="If true, validate patch without applying"
    )


class ApplyPatchOutput(BaseToolOutput):
    """Output from applying a patch."""
    hunks_applied: int = Field(description="Number of hunks successfully applied")
    hunks_failed: int = Field(description="Number of hunks that failed to apply")
    resulting_content: str | None = Field(
        default=None,
        description="Resulting file content (only in dry_run mode)"
    )


class SearchInput(BaseToolInput):
    """Input for searching files."""
    action: EditorAction = EditorAction.SEARCH
    pattern: str = Field(description="Search pattern (regex supported)")
    path: str = Field(
        default=".",
        description="Directory or file to search in"
    )
    file_pattern: str | None = Field(
        default=None,
        description="Glob pattern to filter files (e.g., '*.py')"
    )
    case_sensitive: bool = Field(default=True)
    max_results: int = Field(default=100, ge=1, le=1000)


class SearchMatch(BaseModel):
    """A single search match."""
    file_path: str
    line_number: int
    line_content: str
    match_start: int = Field(description="Character offset of match start")
    match_end: int = Field(description="Character offset of match end")


class SearchOutput(BaseToolOutput):
    """Output from searching files."""
    matches: list[SearchMatch] = Field(default_factory=list)
    total_matches: int
    files_searched: int
    truncated: bool = Field(
        default=False,
        description="Whether results were truncated"
    )


class ListDirectoryInput(BaseToolInput):
    """Input for listing directory contents."""
    action: EditorAction = EditorAction.LIST_DIRECTORY
    path: str = Field(default=".", description="Directory path to list")
    recursive: bool = Field(default=False)
    max_depth: int = Field(default=3, ge=1, le=10)
    include_hidden: bool = Field(default=False)


class FileInfo(BaseModel):
    """Information about a file or directory."""
    name: str
    path: str
    is_directory: bool
    size_bytes: int | None = None
    modified_time: datetime | None = None


class ListDirectoryOutput(BaseToolOutput):
    """Output from listing a directory."""
    entries: list[FileInfo] = Field(default_factory=list)
    total_files: int
    total_directories: int


class SymbolInfo(BaseModel):
    """Information about a code symbol."""
    name: str
    kind: str = Field(description="Symbol kind: function, class, variable, etc.")
    file_path: str
    line_number: int
    column: int
    signature: str | None = None
    docstring: str | None = None


class GetSymbolsInput(BaseToolInput):
    """Input for getting symbols from a file."""
    action: EditorAction = EditorAction.GET_SYMBOLS
    path: str = Field(description="File path to extract symbols from")


class GetSymbolsOutput(BaseToolOutput):
    """Output from getting symbols."""
    symbols: list[SymbolInfo] = Field(default_factory=list)


class GotoDefinitionInput(BaseToolInput):
    """Input for going to a symbol's definition."""
    action: EditorAction = EditorAction.GOTO_DEFINITION
    path: str = Field(description="Current file path")
    line: int = Field(ge=1, description="Line number (1-indexed)")
    column: int = Field(ge=0, description="Column number (0-indexed)")


class GotoDefinitionOutput(BaseToolOutput):
    """Output from goto definition."""
    definitions: list[SymbolInfo] = Field(default_factory=list)


class FindReferencesInput(BaseToolInput):
    """Input for finding references to a symbol."""
    action: EditorAction = EditorAction.FIND_REFERENCES
    path: str = Field(description="Current file path")
    line: int = Field(ge=1)
    column: int = Field(ge=0)
    include_declaration: bool = Field(default=True)


class ReferenceInfo(BaseModel):
    """Information about a reference."""
    file_path: str
    line_number: int
    column: int
    line_content: str


class FindReferencesOutput(BaseToolOutput):
    """Output from finding references."""
    references: list[ReferenceInfo] = Field(default_factory=list)


class DiagnosticSeverity(str, Enum):
    """Severity of a diagnostic."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"


class Diagnostic(BaseModel):
    """A diagnostic message (error, warning, etc.)."""
    severity: DiagnosticSeverity
    message: str
    file_path: str
    line_number: int
    column: int
    end_line: int | None = None
    end_column: int | None = None
    source: str | None = Field(
        default=None,
        description="Source of the diagnostic (e.g., 'typescript', 'pylint')"
    )
    code: str | None = Field(
        default=None,
        description="Diagnostic code (e.g., 'E501')"
    )


class GetDiagnosticsInput(BaseToolInput):
    """Input for getting diagnostics."""
    action: EditorAction = EditorAction.GET_DIAGNOSTICS
    path: str = Field(description="File path to get diagnostics for")


class GetDiagnosticsOutput(BaseToolOutput):
    """Output from getting diagnostics."""
    diagnostics: list[Diagnostic] = Field(default_factory=list)


# =============================================================================
# Browser Tool Schemas
# =============================================================================


class BrowserAction(str, Enum):
    """Available browser actions."""
    SEARCH = "search"
    FETCH = "fetch"
    NAVIGATE = "navigate"
    CLICK = "click"
    TYPE = "type"
    SCREENSHOT = "screenshot"
    EXTRACT_TEXT = "extract_text"


class SearchWebInput(BaseToolInput):
    """Input for web search."""
    action: BrowserAction = BrowserAction.SEARCH
    query: str = Field(description="Search query")
    num_results: int = Field(default=10, ge=1, le=50)
    search_engine: str = Field(default="google")


class SearchResult(BaseModel):
    """A single search result."""
    title: str
    url: str
    snippet: str
    position: int


class SearchWebOutput(BaseToolOutput):
    """Output from web search."""
    results: list[SearchResult] = Field(default_factory=list)
    query: str


class FetchPageInput(BaseToolInput):
    """Input for fetching a web page."""
    action: BrowserAction = BrowserAction.FETCH
    url: str = Field(description="URL to fetch")
    extract_text: bool = Field(
        default=True,
        description="Extract clean text content"
    )
    include_links: bool = Field(
        default=False,
        description="Include links in the output"
    )
    timeout_seconds: int = Field(default=30, ge=1, le=120)


class LinkInfo(BaseModel):
    """Information about a link."""
    text: str
    url: str


class FetchPageOutput(BaseToolOutput):
    """Output from fetching a page."""
    url: str
    title: str | None = None
    content: str = Field(description="Extracted text content")
    links: list[LinkInfo] = Field(default_factory=list)
    content_type: str | None = None
    http_status: int


class NavigateInput(BaseToolInput):
    """Input for navigating to a URL (interactive browser)."""
    action: BrowserAction = BrowserAction.NAVIGATE
    url: str = Field(description="URL to navigate to")
    wait_for_load: bool = Field(default=True)
    timeout_seconds: int = Field(default=30)


class NavigateOutput(BaseToolOutput):
    """Output from navigation."""
    url: str = Field(description="Final URL after navigation")
    title: str | None = None
    page_text: str | None = Field(
        default=None,
        description="Visible text on the page"
    )


class ClickInput(BaseToolInput):
    """Input for clicking an element."""
    action: BrowserAction = BrowserAction.CLICK
    selector: str = Field(description="CSS selector or XPath for the element")
    wait_for_navigation: bool = Field(default=False)


class ClickOutput(BaseToolOutput):
    """Output from clicking."""
    clicked: bool
    new_url: str | None = None


class TypeInput(BaseToolInput):
    """Input for typing text."""
    action: BrowserAction = BrowserAction.TYPE
    selector: str = Field(description="CSS selector for the input element")
    text: str = Field(description="Text to type")
    clear_first: bool = Field(default=True)
    press_enter: bool = Field(default=False)


class TypeOutput(BaseToolOutput):
    """Output from typing."""
    typed: bool


class ScreenshotInput(BaseToolInput):
    """Input for taking a screenshot."""
    action: BrowserAction = BrowserAction.SCREENSHOT
    full_page: bool = Field(default=False)
    selector: str | None = Field(
        default=None,
        description="Optional selector to screenshot specific element"
    )


class ScreenshotOutput(BaseToolOutput):
    """Output from taking a screenshot."""
    image_path: str = Field(description="Path to the saved screenshot")
    width: int
    height: int


# =============================================================================
# Union Types for Tool Dispatch
# =============================================================================


EditorInput = (
    ReadFileInput
    | WriteFileInput
    | ApplyPatchInput
    | SearchInput
    | ListDirectoryInput
    | GetSymbolsInput
    | GotoDefinitionInput
    | FindReferencesInput
    | GetDiagnosticsInput
)

EditorOutput = (
    ReadFileOutput
    | WriteFileOutput
    | ApplyPatchOutput
    | SearchOutput
    | ListDirectoryOutput
    | GetSymbolsOutput
    | GotoDefinitionOutput
    | FindReferencesOutput
    | GetDiagnosticsOutput
)

BrowserInput = (
    SearchWebInput
    | FetchPageInput
    | NavigateInput
    | ClickInput
    | TypeInput
    | ScreenshotInput
)

BrowserOutput = (
    SearchWebOutput
    | FetchPageOutput
    | NavigateOutput
    | ClickOutput
    | TypeOutput
    | ScreenshotOutput
)

ToolInput = TerminalInput | EditorInput | BrowserInput
ToolOutput = TerminalOutput | EditorOutput | BrowserOutput
