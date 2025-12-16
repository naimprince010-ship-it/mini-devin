"""
Agent Orchestrator for Mini-Devin

This module implements the main agent that orchestrates task execution
using a state machine approach with planning, execution, and verification phases.
"""

import json
import uuid
from datetime import datetime
from typing import Any

from ..core.llm_client import LLMClient, create_llm_client
from ..core.tool_interface import ToolRegistry, get_global_registry
from ..schemas.state import (
    AgentPhase,
    AgentState,
    TaskGoal,
    TaskState,
    TaskStatus,
)
from ..schemas.verification import (
    VerificationSuite,
    create_python_verification_suite,
)
from ..tools.terminal import create_terminal_tool
from ..tools.editor import create_editor_tool
from ..tools.browser import (
    create_search_tool,
    create_fetch_tool,
    create_interactive_tool,
    create_citation_store,
)
from ..safety.guards import SafetyGuard, SafetyPolicy, SafetyViolation
from ..artifacts.logger import ArtifactLogger, create_artifact_logger
from ..memory import (
    SymbolIndex,
    VectorStore,
    RetrievalManager,
    WorkingMemory,
    create_symbol_index,
    create_vector_store,
    create_retrieval_manager,
    create_working_memory,
    ConversationMemory,
    ConversationEntryType,
    Importance,
    TaskSummary,
    create_conversation_memory,
)
from ..agents import (
    ReviewerAgent,
    ReviewFeedback,
    create_reviewer_agent,
    PlannerAgent,
    PlanningResult,
    PlanningStrategy,
    create_planner_agent,
)


# System prompt for the agent
SYSTEM_PROMPT = """You are Mini-Devin, an autonomous AI software engineer agent. You can solve software engineering tasks by using tools to interact with the terminal, code editor, file system, and web browser.

## Your Capabilities
- Execute shell commands (terminal tool)
- Read, write, and search files (editor tool)
- Apply patches to modify code
- Navigate and explore codebases
- Search the web for documentation and solutions (browser_search tool)
- Fetch and read web pages (browser_fetch tool)
- Interact with web pages for complex scenarios (browser_interactive tool)

## Your Workflow
1. UNDERSTAND: First understand the task and explore the codebase if needed
2. RESEARCH: Search the web for relevant documentation or solutions if needed
3. PLAN: Create a clear plan with specific steps
4. EXECUTE: Execute each step using the appropriate tools
5. VERIFY: Verify your changes work (run tests, lint, etc.)
6. ITERATE: If something fails, analyze the error and try again

## Guidelines
- Always explore the codebase before making changes
- Search the web when you need documentation or are stuck on an error
- Make small, incremental changes
- Test your changes after each modification
- If you encounter errors, read them carefully and fix them
- Be thorough and complete the entire task

## Tool Usage
When you need to perform an action, use the appropriate tool:
- Use `terminal` to run shell commands
- Use `editor` with action="read_file" to read files
- Use `editor` with action="write_file" to write files
- Use `editor` with action="search" to search for patterns
- Use `editor` with action="list_directory" to explore directories
- Use `browser_search` to search the web for information
- Use `browser_fetch` to fetch and read web page content
- Use `browser_interactive` for complex web interactions (forms, JS-heavy pages)

Always provide your reasoning before using tools. After each tool use, analyze the result and decide on the next step.

When the task is complete, provide a summary of what you did."""


class Agent:
    """
    The main agent that orchestrates task execution.
    
    The agent follows a state machine with these phases:
    - INTAKE: Receive and understand the task
    - EXPLORE: Explore the codebase to understand context
    - PLAN: Create an execution plan
    - EXECUTE: Execute plan steps using tools
    - VERIFY: Verify changes work correctly
    - REPAIR: Fix issues if verification fails
    - COMPLETE: Task completed successfully
    - BLOCKED: Waiting for user input
    """
    
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        tool_registry: ToolRegistry | None = None,
        working_directory: str | None = None,
        max_iterations: int = 50,
        verbose: bool = True,
        auto_verify: bool = True,
        max_repair_iterations: int = 3,
        safety_policy: SafetyPolicy | None = None,
        artifact_dir: str = "runs",
    ):
        self.llm = llm_client or create_llm_client()
        self.registry = tool_registry or get_global_registry()
        self.working_directory = working_directory
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.auto_verify = auto_verify
        self.max_repair_iterations = max_repair_iterations
        
        # Initialize state
        self.state = AgentState(
            agent_id=str(uuid.uuid4()),
            session_id=str(uuid.uuid4()),
            phase=AgentPhase.INTAKE,
        )
        
        # Verification components (lazy initialized)
        self._verification_runner = None
        self._git_manager = None
        self._repair_loop = None
        
        # Safety guard
        self.safety_guard = SafetyGuard(safety_policy or SafetyPolicy())
        
        # Artifact logger (lazy initialized per task)
        self.artifact_dir = artifact_dir
        self._artifact_logger: ArtifactLogger | None = None
        
        # Memory system (Phase 4)
        self._symbol_index: SymbolIndex | None = None
        self._vector_store: VectorStore | None = None
        self._retrieval_manager: RetrievalManager | None = None
        self._working_memory: WorkingMemory | None = None
        self._memory_indexed = False
        
        # Reviewer agent (Phase 9A)
        self._reviewer_agent: ReviewerAgent | None = None
        
        # Planner agent (Phase 9B)
        self._planner_agent: PlannerAgent | None = None
        
        # Conversation memory for cross-session learning (Phase 18)
        self._conversation_memory: ConversationMemory | None = None
        self._use_conversation_memory: bool = True
        
        # Register default tools
        self._register_default_tools()
        
        # Set system prompt
        self.llm.set_system_prompt(SYSTEM_PROMPT)
    
    def _register_default_tools(self) -> None:
        """Register the default tools (terminal, editor, browser)."""
        # Only register if not already registered
        if not self.registry.get("terminal"):
            terminal = create_terminal_tool(working_directory=self.working_directory)
            self.registry.register(terminal)
        
        if not self.registry.get("editor"):
            editor = create_editor_tool(working_directory=self.working_directory)
            self.registry.register(editor)
        
        # Browser tools
        if not self.registry.get("browser_search"):
            browser_search = create_search_tool()
            self.registry.register(browser_search)
        
        if not self.registry.get("browser_fetch"):
            browser_fetch = create_fetch_tool()
            self.registry.register(browser_fetch)
        
        if not self.registry.get("browser_interactive"):
            browser_interactive = create_interactive_tool()
            self.registry.register(browser_interactive)
        
        # Citation store for tracking web references
        if not hasattr(self, "_citation_store"):
            self._citation_store = create_citation_store()
    
    def _get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get tool schemas for LLM function calling."""
        schemas = []
        
        # Terminal tool schema
        schemas.append({
            "name": "terminal",
            "description": "Execute a shell command in the terminal. Use this for running builds, tests, git commands, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute",
                    },
                    "working_directory": {
                        "type": "string",
                        "description": "Working directory for the command (default: current directory)",
                    },
                },
                "required": ["command"],
            },
        })
        
        # Editor tool schema (multi-action)
        schemas.append({
            "name": "editor",
            "description": """Perform file operations. Supports multiple actions:
- read_file: Read a file's contents
- write_file: Write content to a file
- search: Search for patterns in files
- list_directory: List directory contents
- apply_patch: Apply a unified diff patch""",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["read_file", "write_file", "search", "list_directory", "apply_patch"],
                        "description": "The action to perform",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory path",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write (for write_file action)",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Search pattern (for search action)",
                    },
                    "patch": {
                        "type": "string",
                        "description": "Unified diff patch (for apply_patch action)",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Start line for reading (for read_file action)",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "End line for reading (for read_file action)",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Recursive listing (for list_directory action)",
                    },
                },
                "required": ["action", "path"],
            },
        })
        
        # Browser search tool schema
        schemas.append({
            "name": "browser_search",
            "description": "Search the web for information. Use this to find documentation, solutions to errors, or research topics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 10)",
                    },
                },
                "required": ["query"],
            },
        })
        
        # Browser fetch tool schema
        schemas.append({
            "name": "browser_fetch",
            "description": "Fetch and read the content of a web page. Use this to read documentation, articles, or any web content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch",
                    },
                    "extract_content": {
                        "type": "boolean",
                        "description": "Extract clean text content (default: true)",
                    },
                },
                "required": ["url"],
            },
        })
        
        # Browser interactive tool schema
        schemas.append({
            "name": "browser_interactive",
            "description": "Interact with web pages that require JavaScript or complex interactions. Use for forms, login pages, or dynamic content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["navigate", "click", "type", "scroll", "screenshot", "get_text", "get_html", "wait"],
                        "description": "The browser action to perform",
                    },
                    "url": {
                        "type": "string",
                        "description": "URL to navigate to (for navigate action)",
                    },
                    "selector": {
                        "type": "string",
                        "description": "CSS selector for the element (for click, type, get_text actions)",
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to type (for type action)",
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down", "top", "bottom"],
                        "description": "Scroll direction (for scroll action)",
                    },
                    "seconds": {
                        "type": "number",
                        "description": "Seconds to wait (for wait action)",
                    },
                },
                "required": ["action"],
            },
        })
        
        return schemas
    
    async def _execute_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool and return the result as a string."""
        import time
        start_time = time.time()
        call_id = str(uuid.uuid4())[:8]
        
        tool = self.registry.get(name)
        if not tool:
            return f"Error: Unknown tool '{name}'"
        
        try:
            if name == "terminal":
                command = arguments.get("command", "")
                
                # Safety check for commands
                violation = self._check_command_safety(command)
                if violation and violation.blocked:
                    self._update_phase(AgentPhase.BLOCKED)
                    return f"BLOCKED: {violation.message}. Task moved to BLOCKED state."
                
                from ..schemas.tools import TerminalInput
                input_data = TerminalInput(
                    command=command,
                    working_directory=arguments.get("working_directory", "."),
                )
                result = await tool.execute(input_data)
                
                # Format terminal output
                output_parts = []
                if result.stdout:
                    output_parts.append(f"STDOUT:\n{result.stdout}")
                if result.stderr:
                    output_parts.append(f"STDERR:\n{result.stderr}")
                output_parts.append(f"Exit code: {result.exit_code}")
                
                output = "\n".join(output_parts)
                
                # Log to artifacts
                if self._artifact_logger:
                    duration_ms = int((time.time() - start_time) * 1000)
                    self._artifact_logger.log_tool_call(
                        call_id=call_id,
                        tool_name="terminal",
                        arguments=arguments,
                        result=output,
                        duration_ms=duration_ms,
                        success=result.exit_code == 0,
                    )
                    self._artifact_logger.add_command_executed(command)
                
                return output
            
            elif name == "editor":
                action = arguments.get("action", "read_file")
                file_path = arguments.get("path", "")
                
                if action == "read_file":
                    from ..schemas.tools import ReadFileInput, FileRange
                    line_range = None
                    if arguments.get("start_line"):
                        line_range = FileRange(
                            start_line=arguments["start_line"],
                            end_line=arguments.get("end_line"),
                        )
                    input_data = ReadFileInput(
                        path=file_path,
                        line_range=line_range,
                    )
                    result = await tool.execute(input_data)
                    if result.status.value == "success":
                        output = f"File: {file_path} ({result.total_lines} lines)\n\n{result.content}"
                    else:
                        output = f"Error: {result.error_message}"
                
                elif action == "write_file":
                    content = arguments.get("content", "")
                    
                    # Safety check for file edits
                    violation = self._check_file_edit_safety(file_path, content)
                    if violation and violation.blocked:
                        self._update_phase(AgentPhase.BLOCKED)
                        return f"BLOCKED: {violation.message}. Task moved to BLOCKED state."
                    
                    # Safety check for dependency files
                    dep_violation = self._check_dependency_safety(file_path)
                    if dep_violation and dep_violation.blocked:
                        self._update_phase(AgentPhase.BLOCKED)
                        return f"BLOCKED: {dep_violation.message}. Task moved to BLOCKED state."
                    
                    from ..schemas.tools import WriteFileInput
                    input_data = WriteFileInput(
                        path=file_path,
                        content=content,
                    )
                    result = await tool.execute(input_data)
                    if result.status.value == "success":
                        output = f"Successfully wrote {result.bytes_written} bytes to {result.path}"
                        if self._artifact_logger:
                            self._artifact_logger.add_file_modified(file_path)
                    else:
                        output = f"Error: {result.error_message}"
                
                elif action == "search":
                    from ..schemas.tools import SearchInput
                    input_data = SearchInput(
                        pattern=arguments.get("pattern", ""),
                        path=file_path,
                        file_pattern=arguments.get("file_pattern"),
                    )
                    result = await tool.execute(input_data)
                    if result.status.value == "success":
                        if not result.matches:
                            output = "No matches found"
                        else:
                            matches_str = "\n".join([
                                f"{m.file_path}:{m.line_number}: {m.line_content}"
                                for m in result.matches[:50]
                            ])
                            output = f"Found {result.total_matches} matches:\n{matches_str}"
                    else:
                        output = f"Error: {result.error_message}"
                
                elif action == "list_directory":
                    from ..schemas.tools import ListDirectoryInput
                    input_data = ListDirectoryInput(
                        path=file_path,
                        recursive=arguments.get("recursive", False),
                    )
                    result = await tool.execute(input_data)
                    if result.status.value == "success":
                        entries_str = "\n".join([
                            f"{'[DIR] ' if e.is_directory else ''}{e.name}"
                            for e in result.entries[:100]
                        ])
                        output = f"Directory: {file_path}\n{result.total_directories} directories, {result.total_files} files\n\n{entries_str}"
                    else:
                        output = f"Error: {result.error_message}"
                
                elif action == "apply_patch":
                    patch = arguments.get("patch", "")
                    
                    # Safety check for patch (estimate lines changed)
                    violation = self._check_file_edit_safety(file_path, patch)
                    if violation and violation.blocked:
                        self._update_phase(AgentPhase.BLOCKED)
                        return f"BLOCKED: {violation.message}. Task moved to BLOCKED state."
                    
                    from ..schemas.tools import ApplyPatchInput
                    input_data = ApplyPatchInput(
                        path=file_path,
                        patch=patch,
                    )
                    result = await tool.execute(input_data)
                    if result.status.value == "success":
                        output = f"Patch applied: {result.hunks_applied} hunks applied, {result.hunks_failed} failed"
                        if self._artifact_logger:
                            self._artifact_logger.add_file_modified(file_path)
                    else:
                        output = f"Error: {result.error_message}"
                
                else:
                    output = f"Error: Unknown editor action '{action}'"
                
                # Log to artifacts
                if self._artifact_logger:
                    duration_ms = int((time.time() - start_time) * 1000)
                    self._artifact_logger.log_tool_call(
                        call_id=call_id,
                        tool_name="editor",
                        arguments=arguments,
                        result=output[:5000],  # Truncate for logging
                        duration_ms=duration_ms,
                        success="Error" not in output and "BLOCKED" not in output,
                    )
                
                return output
            
            elif name == "browser_search":
                query = arguments.get("query", "")
                max_results = arguments.get("max_results", 10)
                
                # Create a simple input object for the search tool
                class SearchInput:
                    def __init__(self, q, mr):
                        self.query = q
                        self.max_results = mr
                
                input_data = SearchInput(query, max_results)
                result = await tool.execute(input_data)
                
                if result.success:
                    search_response = result.data
                    results_str = "\n".join([
                        f"{i+1}. [{r.title}]({r.url})\n   {r.snippet[:200]}..."
                        for i, r in enumerate(search_response.results[:max_results])
                    ])
                    output = f"Search results for '{query}':\n\n{results_str}"
                    
                    # Add citations for search results
                    if hasattr(self, "_citation_store"):
                        for r in search_response.results:
                            self._citation_store.add(
                                url=r.url,
                                title=r.title,
                                snippet=r.snippet,
                            )
                else:
                    output = f"Search failed: {result.message}"
                
                # Log to artifacts
                if self._artifact_logger:
                    duration_ms = int((time.time() - start_time) * 1000)
                    self._artifact_logger.log_tool_call(
                        call_id=call_id,
                        tool_name="browser_search",
                        arguments=arguments,
                        result=output[:5000],
                        duration_ms=duration_ms,
                        success=result.success,
                    )
                
                return output
            
            elif name == "browser_fetch":
                url = arguments.get("url", "")
                extract_content = arguments.get("extract_content", True)
                
                # Create a simple input object for the fetch tool
                class FetchInput:
                    def __init__(self, u, ec):
                        self.url = u
                        self.extract_content = ec
                        self.use_cache = True
                
                input_data = FetchInput(url, extract_content)
                result = await tool.execute(input_data)
                
                if result.success:
                    fetch_response = result.data
                    page = fetch_response.page
                    
                    # Truncate content if too long
                    content = page.content
                    if len(content) > 10000:
                        content = content[:10000] + "\n\n[Content truncated...]"
                    
                    output = f"Fetched: {page.title}\nURL: {page.url}\nWords: {page.word_count}\n\n{content}"
                    
                    # Add citation
                    if hasattr(self, "_citation_store"):
                        self._citation_store.add(
                            url=page.url,
                            title=page.title,
                            snippet=page.content[:500] if page.content else "",
                        )
                else:
                    output = f"Fetch failed: {result.message}"
                
                # Log to artifacts
                if self._artifact_logger:
                    duration_ms = int((time.time() - start_time) * 1000)
                    self._artifact_logger.log_tool_call(
                        call_id=call_id,
                        tool_name="browser_fetch",
                        arguments=arguments,
                        result=output[:5000],
                        duration_ms=duration_ms,
                        success=result.success,
                    )
                
                return output
            
            elif name == "browser_interactive":
                action = arguments.get("action", "navigate")
                
                # Create a simple input object for the interactive tool
                class InteractiveInput:
                    def __init__(self, args):
                        self.action = args.get("action", "navigate")
                        self.url = args.get("url", "")
                        self.selector = args.get("selector", "")
                        self.text = args.get("text", "")
                        self.direction = args.get("direction", "down")
                        self.seconds = args.get("seconds", 1)
                        self.selector_type = args.get("selector_type", "css")
                        self.clear_first = args.get("clear_first", True)
                
                input_data = InteractiveInput(arguments)
                result = await tool.execute(input_data)
                
                if result.success:
                    response = result.data
                    page_state = response.page_state
                    
                    output_parts = [f"Browser action '{action}' completed."]
                    if page_state:
                        output_parts.append(f"URL: {page_state.url}")
                        output_parts.append(f"Title: {page_state.title}")
                        if page_state.text:
                            text = page_state.text
                            if len(text) > 5000:
                                text = text[:5000] + "\n\n[Content truncated...]"
                            output_parts.append(f"\nPage text:\n{text}")
                    
                    output = "\n".join(output_parts)
                else:
                    output = f"Browser action failed: {result.message}"
                
                # Log to artifacts
                if self._artifact_logger:
                    duration_ms = int((time.time() - start_time) * 1000)
                    self._artifact_logger.log_tool_call(
                        call_id=call_id,
                        tool_name="browser_interactive",
                        arguments=arguments,
                        result=output[:5000],
                        duration_ms=duration_ms,
                        success=result.success,
                    )
                
                return output
            
            else:
                return f"Error: Tool '{name}' not implemented"
                
        except Exception as e:
            error_msg = f"Error executing {name}: {str(e)}"
            if self._artifact_logger:
                duration_ms = int((time.time() - start_time) * 1000)
                self._artifact_logger.log_tool_call(
                    call_id=call_id,
                    tool_name=name,
                    arguments=arguments,
                    result=error_msg,
                    duration_ms=duration_ms,
                    success=False,
                    error=str(e),
                )
            return error_msg
    
    def _log(self, message: str) -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(f"[Agent] {message}")
    
    def _init_artifact_logger(self, task_id: str, task_description: str) -> None:
        """Initialize the artifact logger for a task."""
        self._artifact_logger = create_artifact_logger(
            base_dir=self.artifact_dir,
            task_id=task_id,
            task_description=task_description,
        )
        self._artifact_logger.set_model(self.llm.model)
    
    def _check_command_safety(self, command: str) -> SafetyViolation | None:
        """Check if a command is safe to execute."""
        violation = self.safety_guard.check_command(command)
        if violation:
            self._log(f"SAFETY VIOLATION: {violation.message}")
            if self._artifact_logger:
                self._artifact_logger.log_tool_call(
                    call_id=str(uuid.uuid4()),
                    tool_name="terminal",
                    arguments={"command": command},
                    result=f"BLOCKED: {violation.message}",
                    success=False,
                    error=violation.message,
                )
        return violation
    
    def _check_file_edit_safety(
        self,
        file_path: str,
        content: str,
        is_delete: bool = False,
    ) -> SafetyViolation | None:
        """Check if a file edit is safe."""
        lines_changed = len(content.split("\n")) if content else 0
        violation = self.safety_guard.check_file_edit(file_path, lines_changed, is_delete)
        if violation:
            self._log(f"SAFETY VIOLATION: {violation.message}")
        return violation
    
    def _check_dependency_safety(self, file_path: str) -> SafetyViolation | None:
        """Check if a dependency change is allowed."""
        violation = self.safety_guard.check_dependency_change(file_path)
        if violation:
            self._log(f"SAFETY VIOLATION: {violation.message}")
        return violation
    
    def get_artifact_logger(self) -> ArtifactLogger | None:
        """Get the current artifact logger."""
        return self._artifact_logger
    
    def allow_dependency_changes(self, allow: bool = True) -> None:
        """Allow or disallow dependency changes."""
        self.safety_guard.allow_dependency_changes(allow)
        self._log(f"Dependency changes {'allowed' if allow else 'blocked'}")
    
    def _get_verification_runner(self):
        """Get or create the verification runner."""
        if self._verification_runner is None and self.working_directory:
            from ..verification.runner import create_verification_runner
            self._verification_runner = create_verification_runner(
                self.working_directory,
                verbose=self.verbose,
            )
        return self._verification_runner
    
    def _get_git_manager(self):
        """Get or create the git manager."""
        if self._git_manager is None and self.working_directory:
            from ..verification.git_manager import create_git_manager
            self._git_manager = create_git_manager(
                self.working_directory,
                verbose=self.verbose,
            )
        return self._git_manager
    
    def _get_repair_loop(self):
        """Get or create the repair loop."""
        if self._repair_loop is None and self.working_directory:
            from ..verification.repair import RepairLoop
            runner = self._get_verification_runner()
            git_mgr = self._get_git_manager()
            if runner:
                self._repair_loop = RepairLoop(
                    verification_runner=runner,
                    git_manager=git_mgr,
                    max_iterations=self.max_repair_iterations,
                    verbose=self.verbose,
                )
        return self._repair_loop
    
    async def run_verification(
        self,
        suite: VerificationSuite | None = None,
        task_id: str = "default",
    ):
        """
        Run verification checks on the current working directory.
        
        Args:
            suite: Optional verification suite. If not provided, uses Python defaults.
            task_id: Task identifier for tracking.
            
        Returns:
            VerificationResult with check outcomes.
        """
        runner = self._get_verification_runner()
        if not runner:
            self._log("No verification runner available (no working directory)")
            return None
        
        if suite is None:
            suite = create_python_verification_suite(self.working_directory or ".")
        
        self._log("Running verification checks...")
        result = await runner.run_suite(suite, task_id)
        
        if result.passed:
            self._log(f"Verification passed: {result.checks_passed}/{result.total_checks} checks")
        else:
            self._log(f"Verification failed: {len(result.blocking_failures)} blocking failures")
            for failure_id in result.blocking_failures:
                for check_result in result.check_results:
                    if check_result.check_id == failure_id:
                        self._log(f"  - {failure_id}: {check_result.message}")
        
        return result
    
    async def run_repair_loop(
        self,
        suite: VerificationSuite | None = None,
        task_id: str = "default",
        repair_fn=None,
    ):
        """
        Run verification with automatic repair attempts.
        
        Args:
            suite: Optional verification suite.
            task_id: Task identifier.
            repair_fn: Optional custom repair function.
            
        Returns:
            RepairResult with repair outcomes.
        """
        repair_loop = self._get_repair_loop()
        if not repair_loop:
            self._log("No repair loop available (no working directory)")
            return None
        
        if suite is None:
            suite = create_python_verification_suite(self.working_directory or ".")
        
        self._log("Running repair loop...")
        result = await repair_loop.run(suite, task_id, repair_fn)
        
        if result.status.value == "success":
            self._log(f"Repair succeeded after {result.total_attempts} attempt(s)")
        elif result.status.value == "escalated":
            self._log(f"Repair escalated after {result.total_attempts} attempt(s)")
            self._log(f"Remaining issues: {result.remaining_issues}")
        
        return result
    
    async def create_checkpoint(self, checkpoint_id: str, message: str = "Agent checkpoint"):
        """Create a git checkpoint for potential rollback."""
        git_mgr = self._get_git_manager()
        if not git_mgr:
            return None
        return await git_mgr.create_checkpoint(checkpoint_id, message)
    
    async def rollback_to_checkpoint(self, checkpoint_id: str, hard: bool = False):
        """Rollback to a previous checkpoint."""
        git_mgr = self._get_git_manager()
        if not git_mgr:
            return None
        return await git_mgr.rollback_to_checkpoint(checkpoint_id, hard)
    
    async def get_diff(self):
        """Get the current git diff."""
        git_mgr = self._get_git_manager()
        if not git_mgr:
            return None
        return await git_mgr.get_diff()
    
    # Memory System Methods (Phase 4)
    
    def get_symbol_index(self) -> SymbolIndex:
        """Get or create the symbol index."""
        if self._symbol_index is None:
            workspace = self.working_directory or "."
            self._symbol_index = create_symbol_index(workspace)
        return self._symbol_index
    
    def get_vector_store(self) -> VectorStore:
        """Get or create the vector store."""
        if self._vector_store is None:
            persist_path = None
            if self.working_directory:
                persist_path = f"{self.working_directory}/.mini-devin/vector_store.json"
            self._vector_store = create_vector_store(persist_path=persist_path)
        return self._vector_store
    
    def get_retrieval_manager(self) -> RetrievalManager:
        """Get or create the retrieval manager."""
        if self._retrieval_manager is None:
            workspace = self.working_directory or "."
            self._retrieval_manager = create_retrieval_manager(
                workspace_path=workspace,
                symbol_index=self.get_symbol_index(),
                vector_store=self.get_vector_store(),
            )
        return self._retrieval_manager
    
    def get_working_memory(self) -> WorkingMemory:
        """Get or create the working memory."""
        if self._working_memory is None:
            persist_path = None
            if self.working_directory:
                persist_path = f"{self.working_directory}/.mini-devin/working_memory.json"
            self._working_memory = create_working_memory(
                max_tokens=8000,
                persist_path=persist_path,
            )
        return self._working_memory
    
    def index_workspace(self, force: bool = False) -> dict:
        """
        Index the workspace for code search and retrieval.
        
        Args:
            force: Force re-indexing even if already indexed
            
        Returns:
            Statistics about the indexing
        """
        if self._memory_indexed and not force:
            return self.get_retrieval_manager().get_statistics()
        
        retrieval_mgr = self.get_retrieval_manager()
        stats = retrieval_mgr.index_workspace(force=force)
        self._memory_indexed = True
        
        self._log(f"Indexed workspace: {stats}")
        return stats
    
    def search_code(self, query: str, max_results: int = 10) -> list:
        """
        Search the codebase using the retrieval manager.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of search results
        """
        if not self._memory_indexed:
            self.index_workspace()
        
        retrieval_mgr = self.get_retrieval_manager()
        result = retrieval_mgr.retrieve(query, max_results=max_results)
        return result.results
    
    def find_definition(self, name: str):
        """Find the definition of a symbol by name."""
        if not self._memory_indexed:
            self.index_workspace()
        
        return self.get_retrieval_manager().find_definition(name)
    
    def find_similar_code(self, text: str, max_results: int = 5) -> list:
        """Find code similar to the given text."""
        if not self._memory_indexed:
            self.index_workspace()
        
        return self.get_retrieval_manager().find_similar(text, max_results=max_results)
    
    def add_to_memory(self, content: str, item_type: str = "context", priority: str = "medium") -> str:
        """
        Add an item to working memory.
        
        Args:
            content: The content to remember
            item_type: Type of item (plan, constraint, decision, lesson, error, context, goal)
            priority: Priority level (critical, high, medium, low)
            
        Returns:
            The memory item ID
        """
        from ..memory.working_memory import MemoryItemType, MemoryPriority
        
        memory = self.get_working_memory()
        
        type_map = {
            "plan": memory.add_plan,
            "constraint": memory.add_constraint,
            "decision": memory.add_decision,
            "lesson": memory.add_lesson,
            "error": memory.add_error,
            "goal": memory.add_goal,
        }
        
        priority_map = {
            "critical": MemoryPriority.CRITICAL,
            "high": MemoryPriority.HIGH,
            "medium": MemoryPriority.MEDIUM,
            "low": MemoryPriority.LOW,
        }
        
        prio = priority_map.get(priority, MemoryPriority.MEDIUM)
        
        if item_type in type_map:
            return type_map[item_type](content, priority=prio)
        else:
            from ..memory.working_memory import MemoryItem
            item = MemoryItem(
                id="",
                item_type=MemoryItemType.CONTEXT,
                content=content,
                priority=prio,
            )
            return memory.add(item)
    
    def get_memory_context(self, max_tokens: int = 4000) -> str:
        """Get the current working memory context as a string."""
        return self.get_working_memory().get_context(max_tokens=max_tokens)
    
    def get_memory_statistics(self) -> dict:
        """Get statistics about the memory system."""
        return {
            "working_memory": self.get_working_memory().get_statistics(),
            "retrieval": self.get_retrieval_manager().get_statistics() if self._memory_indexed else {},
            "indexed": self._memory_indexed,
            "conversation_memory": self.get_conversation_memory().get_statistics() if self._conversation_memory else {},
        }
    
    # Conversation Memory Methods (Phase 18)
    
    def get_conversation_memory(self) -> ConversationMemory:
        """
        Get or create the conversation memory for cross-session learning.
        
        Returns:
            ConversationMemory instance
        """
        if self._conversation_memory is None:
            from pathlib import Path
            storage_path = Path.home() / ".mini-devin" / "conversation_memory.json"
            self._conversation_memory = create_conversation_memory(
                storage_path=str(storage_path),
                max_entries=1000,
            )
        return self._conversation_memory
    
    def enable_conversation_memory(self, enabled: bool = True) -> None:
        """Enable or disable conversation memory for this agent."""
        self._use_conversation_memory = enabled
    
    def get_context_from_memory(self, task_description: str) -> str:
        """
        Get relevant context from conversation memory for a task.
        
        Args:
            task_description: Description of the current task
            
        Returns:
            Context string with relevant past experiences
        """
        if not self._use_conversation_memory:
            return ""
        
        memory = self.get_conversation_memory()
        return memory.get_context_for_task(task_description, max_entries=5)
    
    def add_lesson_to_memory(
        self,
        lesson: str,
        context: str = "",
        importance: str = "medium",
        tags: list[str] | None = None,
    ) -> str:
        """
        Add a lesson learned to conversation memory.
        
        Args:
            lesson: The lesson content
            context: Context about when this lesson applies
            importance: Importance level (low, medium, high, critical)
            tags: Optional tags for categorization
            
        Returns:
            Entry ID
        """
        if not self._use_conversation_memory:
            return ""
        
        importance_map = {
            "low": Importance.LOW,
            "medium": Importance.MEDIUM,
            "high": Importance.HIGH,
            "critical": Importance.CRITICAL,
        }
        
        memory = self.get_conversation_memory()
        result = memory.add_lesson(
            lesson=lesson,
            context=context or "General",
            importance=importance_map.get(importance, Importance.MEDIUM),
            tags=tags or [],
            session_id=self.state.session_id,
        )
        return result or ""
    
    def add_error_pattern_to_memory(
        self,
        error: str,
        cause: str,
        solution: str,
        tags: list[str] | None = None,
    ) -> str:
        """
        Add an error pattern and its solution to memory.
        
        Args:
            error: The error message or type
            cause: What caused the error
            solution: How the error was solved
            tags: Optional tags for categorization
            
        Returns:
            Entry ID
        """
        if not self._use_conversation_memory:
            return ""
        
        memory = self.get_conversation_memory()
        result = memory.add_error_pattern(
            error=error,
            cause=cause,
            solution=solution,
            tags=tags or [],
            session_id=self.state.session_id,
        )
        return result or ""
    
    def add_solution_pattern_to_memory(
        self,
        problem: str,
        solution: str,
        code_example: str | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """
        Add a solution pattern to memory.
        
        Args:
            problem: Description of the problem
            solution: The solution that worked
            code_example: Optional code example
            tags: Optional tags for categorization
            
        Returns:
            Entry ID
        """
        if not self._use_conversation_memory:
            return ""
        
        memory = self.get_conversation_memory()
        result = memory.add_solution_pattern(
            problem=problem,
            solution=solution,
            code_example=code_example,
            tags=tags or [],
            session_id=self.state.session_id,
        )
        return result or ""
    
    def get_error_solutions(self, error_message: str) -> list[str]:
        """
        Get solutions for similar errors from memory.
        
        Args:
            error_message: The error message to find solutions for
            
        Returns:
            List of solution strings
        """
        if not self._use_conversation_memory:
            return []
        
        memory = self.get_conversation_memory()
        return memory.get_error_solutions(error_message)
    
    def save_task_summary(
        self,
        task: "TaskState",
        summary: str,
        lessons_learned: list[str] | None = None,
    ) -> str:
        """
        Save a task summary to conversation memory.
        
        Args:
            task: The completed task
            summary: Summary of what was done
            lessons_learned: Optional list of lessons learned
            
        Returns:
            Entry ID
        """
        if not self._use_conversation_memory:
            return ""
        
        memory = self.get_conversation_memory()
        
        task_summary = TaskSummary(
            task_id=task.task_id,
            session_id=self.state.session_id,
            description=task.goal.description,
            outcome=summary,
            success=task.status == TaskStatus.COMPLETED,
            duration_seconds=int((task.completed_at - task.started_at).total_seconds()) if task.completed_at and task.started_at else 0,
            tools_used=list(task.commands_executed or []),
            files_modified=[fc.path for fc in task.files_changed] if task.files_changed else [],
            errors_encountered=[task.last_error] if task.last_error else [],
            lessons=lessons_learned or [],
        )
        
        result = memory.add_task_summary(task_summary)
        return result or ""
    
    def get_recent_lessons(self, limit: int = 10) -> list[str]:
        """
        Get recent lessons learned from memory.
        
        Args:
            limit: Maximum number of lessons to return
            
        Returns:
            List of lesson strings
        """
        if not self._use_conversation_memory:
            return []
        
        memory = self.get_conversation_memory()
        entries = memory.get_recent_lessons(limit=limit)
        return [e.content for e in entries]
    
    def search_memory(
        self,
        query: str,
        entry_type: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """
        Search conversation memory.
        
        Args:
            query: Search query
            entry_type: Optional filter by entry type
            limit: Maximum results to return
            
        Returns:
            List of matching entries as dicts
        """
        if not self._use_conversation_memory:
            return []
        
        memory = self.get_conversation_memory()
        
        type_filter = None
        if entry_type:
            type_map = {
                "task_summary": ConversationEntryType.TASK_SUMMARY,
                "lesson": ConversationEntryType.LESSON_LEARNED,
                "error": ConversationEntryType.ERROR_PATTERN,
                "solution": ConversationEntryType.SOLUTION_PATTERN,
                "preference": ConversationEntryType.USER_PREFERENCE,
                "feedback": ConversationEntryType.FEEDBACK,
            }
            type_filter = type_map.get(entry_type)
        
        entry_types = [type_filter] if type_filter else None
        entries = memory.search(query, entry_types=entry_types, limit=limit)
        return [e.to_dict() for e in entries]
    
    def get_reviewer_agent(self, strict_mode: bool = False) -> ReviewerAgent:
        """
        Get or create the reviewer agent.
        
        Args:
            strict_mode: If True, be stricter about diff discipline
            
        Returns:
            ReviewerAgent instance
        """
        if self._reviewer_agent is None:
            self._reviewer_agent = create_reviewer_agent(
                strict_mode=strict_mode,
                auto_suggest_improvements=True,
            )
        return self._reviewer_agent
    
    async def review_changes(
        self,
        context: str | None = None,
        task_description: str | None = None,
    ) -> ReviewFeedback:
        """
        Review the current git diff using the reviewer agent.
        
        Args:
            context: Optional context about the codebase
            task_description: Optional description of what the change is trying to do
            
        Returns:
            ReviewFeedback with the review results
        """
        git_mgr = self._get_git_manager()
        if not git_mgr:
            from ..agents import ReviewFeedback
            return ReviewFeedback(
                approved=False,
                summary="Cannot review: Git manager not available",
            )
        
        diff_result = await git_mgr.get_diff()
        if not diff_result or not diff_result.success:
            return ReviewFeedback(
                approved=False,
                summary="Cannot review: Failed to get git diff",
            )
        
        diff = diff_result.data.get("diff", "")
        if not diff:
            return ReviewFeedback(
                approved=True,
                summary="No changes to review",
                overall_quality_score=10.0,
            )
        
        reviewer = self.get_reviewer_agent()
        return await reviewer.review_diff(diff, context, task_description)
    
    def quick_review_changes(self, diff: str) -> tuple[bool, list[str]]:
        """
        Perform a quick, synchronous review of a diff without LLM.
        
        Args:
            diff: The diff to review
            
        Returns:
            Tuple of (approved, list of issues)
        """
        reviewer = self.get_reviewer_agent()
        return reviewer.quick_review(diff)
    
    async def review_before_commit(
        self,
        task_description: str | None = None,
    ) -> tuple[bool, str]:
        """
        Review changes before committing.
        
        This is a convenience method that reviews the current changes
        and returns whether they should be committed.
        
        Args:
            task_description: Optional description of the task
            
        Returns:
            Tuple of (should_commit, review_report)
        """
        feedback = await self.review_changes(task_description=task_description)
        
        should_commit = feedback.approved and not feedback.has_blocking_issues
        report = feedback.format_report()
        
        if self._artifact_logger:
            self._artifact_logger.log_tool_call(
                call_id="review",
                tool_name="reviewer_agent",
                arguments={"task_description": task_description},
                result=report[:5000],
                duration_ms=0,
                success=should_commit,
            )
        
        return should_commit, report
    
    def get_planner_agent(
        self,
        strategy: PlanningStrategy = PlanningStrategy.ITERATIVE,
    ) -> PlannerAgent:
        """
        Get or create the planner agent.
        
        Args:
            strategy: Default planning strategy to use
            
        Returns:
            PlannerAgent instance
        """
        if self._planner_agent is None:
            self._planner_agent = create_planner_agent(
                default_strategy=strategy,
                max_steps=50,
                include_verification_steps=True,
            )
        return self._planner_agent
    
    async def create_plan(
        self,
        task: TaskState | None = None,
        context: str | None = None,
        strategy: PlanningStrategy | None = None,
    ) -> PlanningResult:
        """
        Create an execution plan for a task using the planner agent.
        
        Args:
            task: The task to plan for (uses current task if None)
            context: Optional context about the codebase
            strategy: Optional strategy override
            
        Returns:
            PlanningResult with the plan and analysis
        """
        task_to_plan = task or self.state.current_task
        if not task_to_plan:
            return PlanningResult(
                success=False,
                plan=None,
                analysis=None,
                validation=None,
                reasoning="No task provided and no current task set",
            )
        
        planner = self.get_planner_agent()
        result = await planner.create_plan(task_to_plan, context, strategy)
        
        if result.success and result.plan:
            self.state.current_plan = result.plan
            
            if self._artifact_logger:
                self._artifact_logger.log_tool_call(
                    call_id="plan",
                    tool_name="planner_agent",
                    arguments={
                        "task": task_to_plan.goal.description,
                        "strategy": strategy.value if strategy else "iterative",
                    },
                    result=result.reasoning[:5000],
                    duration_ms=0,
                    success=True,
                )
        
        return result
    
    async def analyze_task(
        self,
        task: TaskState | None = None,
        context: str | None = None,
    ) -> dict:
        """
        Analyze a task without creating a full plan.
        
        Args:
            task: The task to analyze (uses current task if None)
            context: Optional context about the codebase
            
        Returns:
            Dictionary with task analysis
        """
        task_to_analyze = task or self.state.current_task
        if not task_to_analyze:
            return {"error": "No task provided and no current task set"}
        
        planner = self.get_planner_agent()
        analysis = await planner.analyze_task(task_to_analyze, context)
        return analysis.to_dict()
    
    async def refine_plan(
        self,
        feedback: str,
    ) -> PlanningResult:
        """
        Refine the current plan based on feedback.
        
        Args:
            feedback: Feedback on what to improve
            
        Returns:
            PlanningResult with the refined plan
        """
        if not self.state.current_plan:
            return PlanningResult(
                success=False,
                plan=None,
                analysis=None,
                validation=None,
                reasoning="No current plan to refine",
            )
        
        if not self.state.current_task:
            return PlanningResult(
                success=False,
                plan=None,
                analysis=None,
                validation=None,
                reasoning="No current task set",
            )
        
        planner = self.get_planner_agent()
        result = await planner.refine_plan(
            self.state.current_plan,
            feedback,
            self.state.current_task,
        )
        
        if result.success and result.plan:
            self.state.current_plan = result.plan
        
        return result
    
    async def replan_from_failure(
        self,
        failed_step_id: str,
        error: str,
    ) -> PlanningResult:
        """
        Create a recovery plan after a step failure.
        
        Args:
            failed_step_id: ID of the step that failed
            error: Error message
            
        Returns:
            PlanningResult with a recovery plan
        """
        if not self.state.current_plan:
            return PlanningResult(
                success=False,
                plan=None,
                analysis=None,
                validation=None,
                reasoning="No current plan",
            )
        
        if not self.state.current_task:
            return PlanningResult(
                success=False,
                plan=None,
                analysis=None,
                validation=None,
                reasoning="No current task set",
            )
        
        failed_step = None
        for step in self.state.current_plan.steps:
            if step.step_id == failed_step_id:
                failed_step = step
                break
        
        if not failed_step:
            return PlanningResult(
                success=False,
                plan=None,
                analysis=None,
                validation=None,
                reasoning=f"Step {failed_step_id} not found in plan",
            )
        
        planner = self.get_planner_agent()
        result = await planner.replan_from_failure(
            self.state.current_plan,
            failed_step,
            error,
            self.state.current_task,
        )
        
        if result.success and result.plan:
            self.state.current_plan = result.plan
        
        return result
    
    def get_next_plan_step(self):
        """
        Get the next step to execute in the current plan.
        
        Returns:
            The next PlanStep to execute, or None if done/no plan
        """
        if not self.state.current_plan:
            return None
        
        planner = self.get_planner_agent()
        return planner.get_next_step(self.state.current_plan)
    
    def mark_plan_step_complete(
        self,
        step_id: str,
        result: str | None = None,
    ) -> None:
        """
        Mark a plan step as completed.
        
        Args:
            step_id: ID of the step to mark complete
            result: Optional result description
        """
        if not self.state.current_plan:
            return
        
        planner = self.get_planner_agent()
        self.state.current_plan = planner.mark_step_complete(
            self.state.current_plan,
            step_id,
            result,
        )
    
    def mark_plan_step_failed(
        self,
        step_id: str,
        error: str,
    ) -> None:
        """
        Mark a plan step as failed.
        
        Args:
            step_id: ID of the step to mark failed
            error: Error message
        """
        if not self.state.current_plan:
            return
        
        planner = self.get_planner_agent()
        self.state.current_plan = planner.mark_step_failed(
            self.state.current_plan,
            step_id,
            error,
        )
    
    def get_plan_progress(self) -> dict:
        """
        Get progress statistics for the current plan.
        
        Returns:
            Dictionary with progress statistics
        """
        if not self.state.current_plan:
            return {
                "total_steps": 0,
                "completed_steps": 0,
                "progress_percent": 0,
                "has_plan": False,
            }
        
        planner = self.get_planner_agent()
        progress = planner.get_plan_progress(self.state.current_plan)
        progress["has_plan"] = True
        return progress
    
    def create_minimal_plan(
        self,
        task: TaskState | None = None,
    ) -> None:
        """
        Create a minimal plan without LLM (for simple tasks).
        
        Args:
            task: The task to plan for (uses current task if None)
        """
        task_to_plan = task or self.state.current_task
        if not task_to_plan:
            return
        
        planner = self.get_planner_agent()
        self.state.current_plan = planner.create_minimal_plan(task_to_plan)
    
    def _update_phase(self, new_phase: AgentPhase) -> None:
        """Update the agent phase."""
        old_phase = self.state.phase
        self.state.phase = new_phase
        self._log(f"Phase transition: {old_phase.value} -> {new_phase.value}")
    
    async def run(self, task: TaskState) -> TaskState:
        """
        Run the agent on a task.
        
        Args:
            task: The task to execute
            
        Returns:
            The updated task state
        """
        self.state.current_task = task
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.utcnow()
        
        # Initialize artifact logger
        self._init_artifact_logger(task.task_id, task.goal.description)
        
        # Reset safety guard for new task
        self.safety_guard.reset_all()
        
        self._log(f"Starting task: {task.goal.description}")
        self._update_phase(AgentPhase.INTAKE)
        
        # Add task description to conversation
        task_message = f"""Task: {task.goal.description}

Acceptance Criteria:
{chr(10).join(f'- {c}' for c in task.goal.acceptance_criteria) if task.goal.acceptance_criteria else '- Complete the task successfully'}

Working Directory: {self.working_directory or 'current directory'}

Please start by exploring the codebase if needed, then create a plan and execute it."""
        
        self.llm.add_user_message(task_message)
        
        # Retrieve relevant context from conversation memory (Phase 18)
        memory_context = ""
        if self._use_conversation_memory:
            try:
                memory_context = self.get_context_from_memory(task.goal.description)
                if memory_context:
                    self._log("Retrieved relevant context from conversation memory")
                    self.llm.add_user_message(
                        f"Here is relevant context from past experiences that may help:\n\n{memory_context}"
                    )
            except Exception as e:
                self._log(f"Warning: Failed to retrieve memory context: {e}")
        
        # Main agent loop
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            self.state.iteration = iteration
            self._log(f"Iteration {iteration}/{self.max_iterations}")
            
            # Update artifact logger
            if self._artifact_logger:
                self._artifact_logger.increment_iteration()
            
            # Reset per-iteration safety counters
            self.safety_guard.reset_iteration()
            
            try:
                # Get LLM response with tools
                response = await self.llm.complete(
                    tools=self._get_tool_schemas(),
                    tool_choice="auto",
                )
                
                # Handle tool calls
                if response.tool_calls:
                    self._update_phase(AgentPhase.EXECUTE)
                    
                    # Add assistant message with tool calls
                    self.llm.add_assistant_message(
                        content=response.content,
                        tool_calls=response.tool_calls,
                    )
                    
                    # Execute each tool
                    for tc in response.tool_calls:
                        self._log(f"Executing tool: {tc.name}({json.dumps(tc.arguments)[:100]}...)")
                        result = await self._execute_tool(tc.name, tc.arguments)
                        self._log(f"Tool result: {result[:200]}...")
                        
                        # Add tool result to conversation
                        self.llm.add_tool_result(tc.id, tc.name, result)
                        
                        # Track in task state
                        if tc.name == "terminal":
                            task.commands_executed.append(tc.arguments.get("command", ""))
                
                else:
                    # No tool calls - check if task is complete
                    if response.content:
                        self._log(f"Assistant: {response.content[:200]}...")
                        self.llm.add_assistant_message(content=response.content)
                        
                        # Check for completion signals in the response
                        content_lower = response.content.lower()
                        if any(phrase in content_lower for phrase in [
                            "task complete",
                            "task is complete",
                            "successfully completed",
                            "finished the task",
                            "completed the task",
                            "all done",
                        ]):
                            self._update_phase(AgentPhase.COMPLETE)
                            task.status = TaskStatus.COMPLETED
                            task.completed_at = datetime.utcnow()
                            self._log("Task completed!")
                            break
                    
                    # If finish_reason is "stop" without completion signal, ask for next step
                    if response.finish_reason == "stop":
                        self.llm.add_user_message(
                            "Please continue with the task. If you're done, say 'Task complete' and summarize what you did."
                        )
            
            except Exception as e:
                self._log(f"Error in iteration: {str(e)}")
                task.last_error = str(e)
                task.error_count += 1
                
                if task.error_count >= 3:
                    self._update_phase(AgentPhase.BLOCKED)
                    task.status = TaskStatus.FAILED
                    break
        
        # Max iterations reached
        if iteration >= self.max_iterations:
            self._log("Max iterations reached")
            task.status = TaskStatus.FAILED
            task.last_error = "Max iterations reached"
        
        # Update token usage
        usage = self.llm.get_usage_stats()
        task.total_tokens_used = usage["total_tokens"]
        
        # Save final artifacts
        if self._artifact_logger:
            self._artifact_logger.update_tokens(task.total_tokens_used)
            
            # Get git diff if available
            git_mgr = self._get_git_manager()
            if git_mgr:
                diff_result = await git_mgr.get_diff()
                if diff_result and diff_result.success:
                    self._artifact_logger.set_diff(diff_result.data.get("diff", ""))
            
            # Get final summary from last assistant message
            summary = ""
            for msg in reversed(self.llm.conversation):
                if msg.role == "assistant" and msg.content:
                    summary = msg.content
                    break
            
            # Complete the artifact logger
            status = "completed" if task.status == TaskStatus.COMPLETED else "failed"
            if self.state.phase == AgentPhase.BLOCKED:
                status = "blocked"
            self._artifact_logger.complete(status=status, summary=summary)
            
            self._log(f"Artifacts saved to: {self._artifact_logger.get_run_dir()}")
        
        # Save task summary to conversation memory (Phase 18)
        if self._use_conversation_memory:
            try:
                self.save_task_summary(
                    task=task,
                    summary=summary,
                    lessons_learned=[],
                )
                self._log("Task summary saved to conversation memory")
            except Exception as e:
                self._log(f"Warning: Failed to save task summary to memory: {e}")
        
        return task
    
    async def run_simple(self, task_description: str) -> str:
        """
        Simplified interface to run a task from a description.
        
        Args:
            task_description: Natural language description of the task
            
        Returns:
            Summary of what was done
        """
        task = TaskState(
            task_id=str(uuid.uuid4()),
            goal=TaskGoal(
                description=task_description,
                acceptance_criteria=[],
            ),
        )
        
        result = await self.run(task)
        
        # Get the last assistant message as summary
        for msg in reversed(self.llm.conversation):
            if msg.role == "assistant" and msg.content:
                return msg.content
        
        return f"Task {'completed' if result.status == TaskStatus.COMPLETED else 'failed'}"


async def create_agent(
    model: str = "gpt-4o",
    api_key: str | None = None,
    working_directory: str | None = None,
    verbose: bool = True,
) -> Agent:
    """Create an agent with default configuration."""
    import os
    
    llm = create_llm_client(
        model=model,
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
    )
    
    return Agent(
        llm_client=llm,
        working_directory=working_directory,
        verbose=verbose,
    )
