"""
Agent Orchestrator for Mini-Devin

This module implements the main agent that orchestrates task execution
using a state machine approach with planning, execution, and verification phases.
"""
from __future__ import annotations  # Enable forward references for type hints

import asyncio
import json
import os
import time
import uuid
import re
import inspect
from pathlib import Path
from datetime import datetime, timezone
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
# Selective imports will be handled locally within methods to avoid circular dependencies
from ..core.parallel_executor import (
    ParallelExecutor,
    BatchToolCaller,
    ToolCall,
    ToolCallResult,
    ParallelExecutionResult,
    create_parallel_executor,
    create_batch_caller,
)
from ..reliability.self_correction import SelfCorrectionEngine, ErrorType



# System prompt for the agent
SYSTEM_PROMPT = """You are Mini-Devin, an autonomous AI software engineer agent. You solve software engineering tasks end-to-end using tools.

## CRITICAL RULES — READ FIRST
- **NEVER describe or narrate actions without calling a tool.** If you say "I will create a file", you MUST immediately call the `editor` tool to do it.
- **NEVER write fake outputs.** Do not write "The tests passed" unless you actually ran `pytest` using the `terminal` tool and saw the output.
- **NEVER say TASK COMPLETE unless you have used at least one tool** and verified the result with actual tool output.
- **Do NOT just write a plan as text and stop.** After your brief plan, immediately call the first tool.

## Workflow
1. **Brief plan** (2-3 lines max): State what you will do.
2. **ACT immediately**: Call the first tool right away — do not wait.
3. **Continue**: After each tool result, call the next tool needed.
4. **Verify**: Run tests or check output with real tool calls.
5. **TASK COMPLETE**: Only after you have seen real tool output confirming success.

## Tool Usage
- `terminal` — Run shell commands (pytest, pip, git, etc.)
- `editor` with `read_file` — Read a file
- `editor` with `write_file` — Write/create a file
- `editor` with `search` — Search patterns in files
- `editor` with `list_directory` — List directory contents
- `editor` with `str_replace` — Replace exact text in a file (token-efficient, preferred for edits)
- `editor` with `apply_patch` — Apply a unified diff patch
- `browser_search` — Search the web
- `browser_fetch` — Fetch a web page
- `github` — GitHub workflows
- \monitor\ — Check app health, fetch cloud/docker logs, register for continuous monitoring
- \env_parity\ — Generate Dockerfile/.env.example/docker-compose; diff local vs production env

## Important Rules
- Read a file before editing it (to avoid overwriting changes)
- Always write COMPLETE file content when using `write_file`
- After writing a file, verify with `read_file`
- If a command fails, read the error and fix it — do NOT give up
- When done: write **TASK COMPLETE** followed by a short summary of actual results.

## Environment (critical)
- The agent runs on **Linux/bash** in the cloud/container, **not** on the user's Windows PC.
- **Never** use Windows drive paths (`C:\\`, `G:\\`, `D:\\`) in `terminal` or `editor` paths—they do not exist here.
- Use **paths under the workspace** only: relative paths like `./myself/index.html` or POSIX paths starting with the workspace root shown below.
- Prefer **`editor` `write_file`** to create files and folders; it creates parent directories automatically."""


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
        enable_parallel_execution: bool = True,
        max_parallel_tools: int = 5,
        callbacks: dict[str, Any] | None = None,
        use_sandbox: bool = False,
        auto_git_commit: bool = False,
        git_push: bool = False,
    ):
        self.llm = llm_client or create_llm_client()
        self.registry = tool_registry or get_global_registry()
        self.working_directory = working_directory
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.auto_verify = auto_verify
        self.max_repair_iterations = max_repair_iterations
        self.callbacks = callbacks or {}
        
        # Self-correction specific parameters
        self.max_immediate_retries = 3
        
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
        
        # Parallel execution (Phase 19)
        self._enable_parallel_execution = enable_parallel_execution
        self._max_parallel_tools = max_parallel_tools
        self._parallel_executor: ParallelExecutor | None = None
        self._batch_caller: BatchToolCaller | None = None
        
        # Auto git commit after task completion
        self.auto_git_commit = auto_git_commit
        self.git_push = git_push

        # Self-correction (Phase 24)
        self._correction_engine = SelfCorrectionEngine(
            max_immediate_retries=self.max_immediate_retries
        )
        self._consecutive_failures = 0
        
        # Docker Sandbox (optional isolated execution)
        self.use_sandbox = use_sandbox
        self._sandbox = None  # Initialized lazily per task
        
        # Proactive clarification support
        self._clarification_event: asyncio.Event | None = None
        self._clarification_answer: str | None = None
        
        # Register default tools
        self._register_default_tools()
        
        # Set system prompt (inject workspace so models stop inventing Windows paths)
        wd = self.working_directory or os.getcwd()
        self.llm.set_system_prompt(
            SYSTEM_PROMPT
            + f"\n\n**Workspace root (authoritative):** `{wd}`\n"
        )
    
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
            
        # GitHub tool
        if not self.registry.get("github"):
            from ..tools.github import create_github_tool
            github_tool = create_github_tool()
            self.registry.register(github_tool)
    
    def _get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get tool schemas for LLM function calling."""
        schemas = []
        
        # Terminal tool schema
        schemas.append({
            "name": "terminal",
            "description": "Execute a shell command in Linux/bash under the task workspace. Do not use Windows paths (C:\\\\, G:\\\\). Use relative paths (./...) from the workspace.",
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
- write_file: Write/create a file with full content
- str_replace: Replace an exact string in a file (PREFERRED for edits — token-efficient, precise)
- search: Search for patterns in files
- list_directory: List directory contents
- apply_patch: Apply a unified diff patch

PREFER str_replace over write_file when editing existing files.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["read_file", "write_file", "str_replace", "search", "list_directory", "apply_patch"],
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
                    "old_str": {
                        "type": "string",
                        "description": "Exact string to find and replace (for str_replace action — must be unique in file)",
                    },
                    "new_str": {
                        "type": "string",
                        "description": "Replacement string (for str_replace action — use empty string to delete)",
                    },
                    "allow_multiple": {
                        "type": "boolean",
                        "description": "Replace all occurrences (for str_replace action, default false)",
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
        
        # Ask-user tool for proactive clarification
        schemas.append({
            "name": "ask_user",
            "description": "Ask the user a clarifying question when the task is ambiguous or you need more information before proceeding. Use this sparingly, only when truly necessary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The specific question to ask the user",
                    },
                },
                "required": ["question"],
            },
        })

        # Monitor tool schema
        schemas.append({
            "name": "monitor",
            "description": (
                "Self-healing monitor: check app health, fetch cloud/docker logs, and manage "
                "continuous monitoring with auto-heal on crash."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["status", "health_check", "fetch_logs", "register", "start", "stop"],
                        "description": (
                            "status=show monitor state; health_check=one-shot HTTP check; "
                            "fetch_logs=get logs from platform; register=add an app to continuous monitoring; "
                            "start/stop=control the monitor loop."
                        ),
                    },
                    "url": {"type": "string", "description": "App URL for health_check"},
                    "platform": {
                        "type": "string",
                        "enum": ["digitalocean", "railway", "docker", "generic"],
                        "description": "Platform for fetch_logs or register",
                    },
                    "config": {
                        "type": "object",
                        "description": (
                            "Platform config dict. DO: {do_token, app_id}. "
                            "Railway: {railway_token, service_id}. Docker: {container_name}."
                        ),
                    },
                    "lines": {"type": "integer", "description": "Log lines to fetch (default 50)"},
                    "name": {"type": "string", "description": "App name for register"},
                    "health_url": {"type": "string", "description": "Health check URL for register"},
                    "interval": {"type": "integer", "description": "Poll interval seconds (default 60)"},
                    "failure_threshold": {"type": "integer", "description": "Failures before heal (default 3)"},
                    "platform_config": {"type": "object", "description": "Platform-specific config for register"},
                },
                "required": ["action"],
            },
        })

        # Environment parity tool schema
        schemas.append({
            "name": "env_parity",
            "description": (
                "Ensure local and production environments are identical. "
                "Generate Dockerfiles, .env.example, docker-compose.yml, and diff environments."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["diff", "generate_dockerfile", "generate_env_example", "generate_docker_compose"],
                        "description": (
                            "diff=compare local .env vs production; "
                            "generate_dockerfile=create optimized Dockerfile; "
                            "generate_env_example=create .env.example from .env; "
                            "generate_docker_compose=create local dev compose file."
                        ),
                    },
                    "project_root": {"type": "string", "description": "Project root directory (default: workspace)"},
                    "project_type": {
                        "type": "string",
                        "enum": ["auto", "python", "node", "fullstack"],
                        "description": "Project type for Dockerfile generation (default: auto-detect)",
                    },
                    "frontend_dir": {"type": "string", "description": "Frontend subdirectory (default: frontend)"},
                    "requirements_file": {"type": "string", "description": "Python requirements file (default: requirements.txt)"},
                    "port": {"type": "integer", "description": "App port (default: auto-detect from .env)"},
                    "health_path": {"type": "string", "description": "Healthcheck path (default: /health)"},
                    "output_path": {"type": "string", "description": "Output file path (optional)"},
                    "env_file": {"type": "string", "description": "Source .env file for diff (default: .env)"},
                    "source_env_file": {"type": "string", "description": "Source .env file for .env.example generation"},
                    "include_current_values": {"type": "boolean", "description": "Include non-secret values in .env.example"},
                    "include_redis": {"type": "boolean", "description": "Include Redis in docker-compose"},
                    "include_postgres": {"type": "boolean", "description": "Include Postgres in docker-compose"},
                    "production_env": {"type": "object", "description": "Production env vars dict for diff comparison"},
                },
                "required": ["action"],
            },
        })

        # GitHub tool schema
        schemas.append({
            "name": "github",
            "description": "Perform GitHub operations like creating branches, committing, and creating pull requests.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create_branch", "commit", "create_pr", "automated_workflow"],
                        "description": "The GitHub action to perform",
                    },
                    "branch_name": {
                        "type": "string",
                        "description": "Name of the branch (for create_branch, create_pr, automated_workflow)",
                    },
                    "base_branch": {
                        "type": "string",
                        "description": "Base branch name (default: main)",
                    },
                    "commit_message": {
                        "type": "string",
                        "description": "Commit message (for commit)",
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of files to commit. Empty means all changes.",
                    },
                    "pr_title": {
                        "type": "string",
                        "description": "Pull Request title",
                    },
                    "pr_description": {
                        "type": "string",
                        "description": "Pull Request description",
                    },
                    "task_description": {
                        "type": "string",
                        "description": "Task description (for automated_workflow)",
                    },
                    "repo_path": {
                        "type": "string",
                        "description": "Local repository path (default: .)",
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
        
        # ── ask_user: pause and request clarification from user ──
        if name == "ask_user":
            question = arguments.get("question", "Can you clarify?")
            on_clarification = self.callbacks.get("on_clarification_needed")
            if on_clarification:
                on_clarification(question)
            # Wait up to 5 minutes for user answer
            self._clarification_event = asyncio.Event()
            self._clarification_answer = None
            try:
                await asyncio.wait_for(self._clarification_event.wait(), timeout=300)
            except asyncio.TimeoutError:
                return "User did not answer within 5 minutes. Proceeding with best guess."
            answer = self._clarification_answer or "(no answer)"
            self._clarification_event = None
            return f"User answered: {answer}"
        
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
                
                # Emit command to shell stream
                on_cmd_output = self.callbacks.get("on_command_output")
                if on_cmd_output:
                    on_cmd_output(f"$ {command}")
                
                # ── Docker Sandbox Execution ──
                if self.use_sandbox and self._sandbox is not None:
                    try:
                        if not self._sandbox.is_running():
                            started = await self._sandbox.start()
                            if not started:
                                # Fall through to regular execution
                                self.use_sandbox = False
                        
                        if self._sandbox.is_running():
                            sandbox_result = await self._sandbox.execute(
                                command,
                                working_dir=arguments.get("working_directory"),
                            )
                            output_parts = []
                            if sandbox_result.stdout:
                                output_parts.append(f"STDOUT:\n{sandbox_result.stdout}")
                                if on_cmd_output:
                                    for line in sandbox_result.stdout.splitlines():
                                        on_cmd_output(line)
                            if sandbox_result.stderr:
                                output_parts.append(f"STDERR:\n{sandbox_result.stderr}")
                                if on_cmd_output:
                                    for line in sandbox_result.stderr.splitlines():
                                        on_cmd_output(line)
                            output_parts.append(f"Exit code: {sandbox_result.exit_code}")
                            if on_cmd_output:
                                on_cmd_output(f"Exit code: {sandbox_result.exit_code}")
                            output = "\n".join(output_parts)
                            if self._artifact_logger:
                                duration_ms = int((time.time() - start_time) * 1000)
                                self._artifact_logger.log_tool_call(
                                    call_id=call_id,
                                    tool_name="terminal",
                                    arguments=arguments,
                                    result=output,
                                    duration_ms=duration_ms,
                                    success=sandbox_result.exit_code == 0,
                                )
                                self._artifact_logger.add_command_executed(command)
                            return output
                    except Exception as sandbox_err:
                        self._log(f"Sandbox error, falling back to host: {sandbox_err}")
                
                # ── Regular Host Execution ──
                from ..schemas.tools import TerminalInput
                input_data = TerminalInput(
                    command=command,
                    working_directory=arguments.get("working_directory", "."),
                )
                
                exit_code = -1
                result = None
                try:
                    result = await tool.execute(input_data)
                    exit_code = result.exit_code
                except Exception as e:
                    return f"Error executing terminal command: {str(e)}"
                
                # Format terminal output
                output_parts = []
                if result.stdout:
                    output_parts.append(f"STDOUT:\n{result.stdout}")
                    if on_cmd_output:
                        for line in result.stdout.splitlines():
                            on_cmd_output(line)
                            
                if result.stderr:
                    output_parts.append(f"STDERR:\n{result.stderr}")
                    if on_cmd_output:
                        for line in result.stderr.splitlines():
                            on_cmd_output(line)
                            
                output_parts.append(f"Exit code: {result.exit_code}")
                if on_cmd_output:
                    on_cmd_output(f"Exit code: {result.exit_code}")
                
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

                elif action == "str_replace":
                    old_str = arguments.get("old_str", "")
                    new_str = arguments.get("new_str", "")
                    allow_multiple = arguments.get("allow_multiple", False)

                    dep_violation = self._check_dependency_safety(file_path)
                    if dep_violation and dep_violation.blocked:
                        self._update_phase(AgentPhase.BLOCKED)
                        return f"BLOCKED: {dep_violation.message}. Task moved to BLOCKED state."

                    from ..schemas.tools import StrReplaceInput
                    input_data = StrReplaceInput(
                        path=file_path,
                        old_str=old_str,
                        new_str=new_str,
                        allow_multiple=allow_multiple,
                    )
                    result = await tool.execute(input_data)
                    if result.status.value == "success":
                        output = f"str_replace: {result.replacements_made} replacement(s) in {result.path}"
                        if self._artifact_logger:
                            self._artifact_logger.add_file_modified(file_path)
                        # Notify frontend of file change
                        await self._trigger_callback("on_file_changed", file_path)
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
            
            elif name == "github":
                action = arguments.get("action", "")
                
                from ..tools.github import GitHubToolInput, GitHubAction
                input_data = GitHubToolInput(
                    action=GitHubAction(action),
                    branch_name=arguments.get("branch_name"),
                    base_branch=arguments.get("base_branch", "main"),
                    commit_message=arguments.get("commit_message"),
                    files=arguments.get("files"),
                    pr_title=arguments.get("pr_title"),
                    pr_description=arguments.get("pr_description"),
                    task_description=arguments.get("task_description"),
                    repo_path=arguments.get("repo_path", self.working_directory or ".")
                )
                
                result = await tool.execute(input_data)
                
                if result.success:
                    output = f"GitHub action '{action}' completed successfully.\n{result.message}"
                    if result.pr_url:
                        output += f"\nPR URL: {result.pr_url}"
                else:
                    output = f"GitHub action failed: {result.message}"
                
                if self._artifact_logger:
                    duration_ms = int((time.time() - start_time) * 1000)
                    self._artifact_logger.log_tool_call(
                        call_id=call_id,
                        tool_name="github",
                        arguments=arguments,
                        result=output[:5000],
                        duration_ms=duration_ms,
                        success=result.success,
                    )
                
                return output
            
            elif name == "monitor":
                action = arguments.get("action", "status")
                from ..integrations.monitor import (
                    check_app_health,
                    fetch_app_logs,
                    get_status,
                    register_app,
                    start_monitor,
                    stop_monitor,
                    MonitoredApp,
                )
                if action == "status":
                    return json.dumps(get_status(), indent=2)
                elif action == "health_check":
                    url = arguments.get("url", "")
                    if not url:
                        return "Error: url is required for health_check"
                    result = await check_app_health(url)
                    return json.dumps(result, indent=2)
                elif action == "fetch_logs":
                    platform = arguments.get("platform", "docker")
                    config = arguments.get("config", {})
                    lines = int(arguments.get("lines", 50))
                    logs = await fetch_app_logs(platform, config, lines)
                    return logs or "(no logs returned)"
                elif action == "register":
                    app_cfg = MonitoredApp(
                        name=arguments.get("name", "app"),
                        health_url=arguments.get("health_url", ""),
                        platform=arguments.get("platform", "generic"),
                        platform_config=arguments.get("platform_config", {}),
                        check_interval_seconds=int(arguments.get("interval", 60)),
                        failure_threshold=int(arguments.get("failure_threshold", 3)),
                        session_id=self.session_id if hasattr(self, "session_id") else None,
                    )
                    register_app(app_cfg)
                    start_monitor()
                    return f"Registered {app_cfg.name} and started monitor"
                elif action == "start":
                    start_monitor()
                    return "Monitor started"
                elif action == "stop":
                    stop_monitor()
                    return "Monitor stopped"
                else:
                    return f"Unknown monitor action: {action}"

            elif name == "env_parity":
                action = arguments.get("action", "diff")
                project_root = arguments.get("project_root", self.working_directory or ".")
                from ..integrations.env_parity import (
                    generate_dockerfile,
                    generate_env_example,
                    generate_docker_compose,
                    diff_environments,
                )
                if action == "diff":
                    env_file = arguments.get("env_file", ".env")
                    import os as _os
                    prod_env = arguments.get("production_env")
                    result = diff_environments(
                        local_env_file=str(Path(project_root) / env_file),
                        production_env=prod_env,
                    )
                    return json.dumps(result, indent=2)
                elif action == "generate_dockerfile":
                    content, path = generate_dockerfile(
                        project_root,
                        project_type=arguments.get("project_type", "auto"),
                        frontend_dir=arguments.get("frontend_dir", "frontend"),
                        requirements_file=arguments.get("requirements_file", "requirements.txt"),
                        port=arguments.get("port"),
                        health_path=arguments.get("health_path", "/health"),
                        output_path=arguments.get("output_path"),
                    )
                    return f"Dockerfile generated at {path}:\n\n{content}"
                elif action == "generate_env_example":
                    content, path = generate_env_example(
                        project_root,
                        source_env_file=arguments.get("source_env_file", ".env"),
                        output_path=arguments.get("output_path"),
                        include_current_values=arguments.get("include_current_values", False),
                    )
                    return f".env.example generated at {path}:\n\n{content}"
                elif action == "generate_docker_compose":
                    content, path = generate_docker_compose(
                        project_root,
                        port=arguments.get("port"),
                        include_redis=arguments.get("include_redis", False),
                        include_postgres=arguments.get("include_postgres", False),
                        output_path=arguments.get("output_path"),
                    )
                    return f"docker-compose.yml generated at {path}:\n\n{content}"
                else:
                    return f"Unknown env_parity action: {action}"

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
        self._artifact_logger.set_model(self.llm.config.model)
    
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
            from ..agents import create_reviewer_agent
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
        strategy: Any = None, # Avoid PlanningStrategy type at top level if possible
    ) -> Any:
        # Import inside to avoid circularity
        from ..agents import create_planner_agent, PlanningStrategy
        strategy = strategy or PlanningStrategy.ITERATIVE
        """
        Get or create the planner agent.
        
        Args:
            strategy: Default planning strategy to use
            
        Returns:
            PlannerAgent instance
        """
        if self._planner_agent is None:
            from ..agents import create_planner_agent
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
        strategy: Any = None,
    ) -> Any:
        from ..agents import PlanningResult
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
    ) -> Any:
        from ..agents import PlanningResult
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
    ) -> Any:
        from ..agents import PlanningResult
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
    
    def get_parallel_executor(self) -> ParallelExecutor:
        """
        Get or create the parallel executor.
        
        Returns:
            ParallelExecutor instance
        """
        if self._parallel_executor is None:
            async def execute_tool(tool_name: str, arguments: dict) -> str:
                return await self._execute_tool(tool_name, arguments)
            
            self._parallel_executor = create_parallel_executor(
                execute_fn=execute_tool,
                max_concurrent=self._max_parallel_tools,
                fail_fast=False,
            )
        return self._parallel_executor
    
    def get_batch_caller(self) -> BatchToolCaller:
        """
        Get or create the batch tool caller.
        
        Returns:
            BatchToolCaller instance for fluent API
        """
        if self._batch_caller is None:
            async def execute_tool(tool_name: str, arguments: dict) -> str:
                return await self._execute_tool(tool_name, arguments)
            
            self._batch_caller = create_batch_caller(
                execute_fn=execute_tool,
                max_concurrent=self._max_parallel_tools,
            )
        return self._batch_caller
    
    def enable_parallel_execution(self, enabled: bool = True) -> None:
        """
        Enable or disable parallel tool execution.
        
        Args:
            enabled: Whether to enable parallel execution
        """
        self._enable_parallel_execution = enabled
    
    def set_max_parallel_tools(self, max_tools: int) -> None:
        """
        Set the maximum number of parallel tool executions.
        
        Args:
            max_tools: Maximum number of concurrent tool calls
        """
        self._max_parallel_tools = max_tools
        self._parallel_executor = None
        self._batch_caller = None
    
    async def execute_tools_parallel(
        self,
        tool_calls: list[dict],
    ) -> ParallelExecutionResult:
        """
        Execute multiple tool calls in parallel when possible.
        
        Analyzes dependencies between calls and executes independent
        calls concurrently for improved performance.
        
        Args:
            tool_calls: List of tool call dicts with 'name' and 'arguments'
            
        Returns:
            ParallelExecutionResult with all results and timing info
        """
        if not self._enable_parallel_execution or len(tool_calls) <= 1:
            results = []
            for tc in tool_calls:
                started_at = datetime.now(timezone.utc)
                try:
                    result = await self._execute_tool(tc["name"], tc["arguments"])
                    completed_at = datetime.now(timezone.utc)
                    duration_ms = int((completed_at - started_at).total_seconds() * 1000)
                    success = "Error" not in str(result) and "BLOCKED" not in str(result)
                    results.append(ToolCallResult(
                        call_id=tc.get("id", str(len(results))),
                        tool_name=tc["name"],
                        success=success,
                        result=result,
                        error=None if success else result,
                        started_at=started_at,
                        completed_at=completed_at,
                        duration_ms=duration_ms,
                    ))
                except Exception as e:
                    completed_at = datetime.now(timezone.utc)
                    duration_ms = int((completed_at - started_at).total_seconds() * 1000)
                    results.append(ToolCallResult(
                        call_id=tc.get("id", str(len(results))),
                        tool_name=tc["name"],
                        success=False,
                        result=None,
                        error=str(e),
                        started_at=started_at,
                        completed_at=completed_at,
                        duration_ms=duration_ms,
                    ))
            
            total_duration = sum(r.duration_ms for r in results)
            return ParallelExecutionResult(
                results=results,
                total_duration_ms=total_duration,
                parallel_speedup=1.0,
                execution_order=[[r.call_id for r in results]],
            )
        
        calls = [
            ToolCall.create(
                tool_name=tc["name"],
                arguments=tc["arguments"],
            )
            for tc in tool_calls
        ]
        
        executor = self.get_parallel_executor()
        return await executor.execute(calls)
    
    async def batch_read_files(self, paths: list[str]) -> dict[str, str]:
        """
        Read multiple files in parallel.
        
        Args:
            paths: List of file paths to read
            
        Returns:
            Dict mapping paths to file contents
        """
        tool_calls = [
            {"name": "editor", "arguments": {"action": "read_file", "path": path}}
            for path in paths
        ]
        
        result = await self.execute_tools_parallel(tool_calls)
        
        contents = {}
        for i, path in enumerate(paths):
            if i < len(result.results):
                r = result.results[i]
                contents[path] = r.result if r.success else f"Error: {r.error}"
            else:
                contents[path] = "Error: No result"
        
        return contents
    
    async def batch_search(
        self,
        patterns: list[str],
        path: str,
    ) -> dict[str, str]:
        """
        Search for multiple patterns in parallel.
        
        Args:
            patterns: List of search patterns
            path: Directory to search in
            
        Returns:
            Dict mapping patterns to search results
        """
        tool_calls = [
            {"name": "editor", "arguments": {"action": "search", "pattern": pattern, "path": path}}
            for pattern in patterns
        ]
        
        result = await self.execute_tools_parallel(tool_calls)
        
        results = {}
        for i, pattern in enumerate(patterns):
            if i < len(result.results):
                r = result.results[i]
                results[pattern] = r.result if r.success else f"Error: {r.error}"
            else:
                results[pattern] = "Error: No result"
        
        return results
    
    async def batch_terminal_commands(
        self,
        commands: list[str],
        working_directory: str | None = None,
    ) -> list[ToolCallResult]:
        """
        Execute multiple terminal commands.
        
        Note: Commands are analyzed for dependencies and may be
        executed sequentially if they depend on each other.
        
        Args:
            commands: List of shell commands
            working_directory: Optional working directory
            
        Returns:
            List of ToolCallResult for each command
        """
        tool_calls = [
            {
                "name": "terminal",
                "arguments": {
                    "command": cmd,
                    **({"working_directory": working_directory} if working_directory else {}),
                },
            }
            for cmd in commands
        ]
        
        result = await self.execute_tools_parallel(tool_calls)
        return result.results
    
    def get_parallel_execution_stats(self) -> dict:
        """
        Get statistics about parallel execution.
        
        Returns:
            Dict with parallel execution configuration and stats
        """
        return {
            "enabled": self._enable_parallel_execution,
            "max_parallel_tools": self._max_parallel_tools,
            "executor_initialized": self._parallel_executor is not None,
            "batch_caller_initialized": self._batch_caller is not None,
        }
    
    async def _trigger_callback(self, name: str, *args, **kwargs) -> None:
        """Trigger a callback if it exists, awaiting it if it's a coroutine."""
        if name in self.callbacks:
            callback = self.callbacks[name]
            try:
                import asyncio
                if asyncio.iscoroutine(callback) or inspect.iscoroutinefunction(callback):
                    await callback(*args, **kwargs)
                else:
                    # If it's a lambda or function that returns a coroutine, handle it
                    result = callback(*args, **kwargs)
                    if asyncio.iscoroutine(result):
                        await result
            except Exception as e:
                self._log(f"Error in callback '{name}': {e}")

    async def _update_phase(self, new_phase: AgentPhase) -> None:
        """Update the agent phase."""
        old_phase = self.state.phase
        self.state.phase = new_phase
        self._log(f"Phase transition: {old_phase.value} -> {new_phase.value}")
        await self._trigger_callback("on_phase_change", new_phase.value)
    
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
        task.started_at = datetime.now(timezone.utc)
        
        # Reset plan state for this new task
        self._plan_sent = False
        self._plan_steps = []
        self._current_step_idx = 0
        self._tools_used_count = 0
        self._no_tool_streak = 0
        
        # Initialize artifact logger
        self._init_artifact_logger(task.task_id, task.goal.description)
        
        # Reset safety guard for new task
        self.safety_guard.reset_all()
        
        self._log(f"Starting task: {task.goal.description}")
        await self._update_phase(AgentPhase.INTAKE)
        
        # Add task description to conversation
        task_message = f"""Task: {task.goal.description}

Acceptance Criteria:
{chr(10).join(f'- {c}' for c in task.goal.acceptance_criteria) if task.goal.acceptance_criteria else '- Complete the task successfully'}

Working Directory: {self.working_directory or 'current directory'}

IMPORTANT: You MUST use tools to complete this task. Do NOT just write text descriptions. 
Call a tool (editor or terminal) immediately as your first action."""
        
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
            
            # Notify frontend of iteration update
            await self._trigger_callback("on_iteration", iteration, self.max_iterations)
            
            
            # Reset per-iteration safety counters
            self.safety_guard.reset_iteration()
            
            try:
                # Define token callback
                async def handle_token(token: str):
                    # The callback handles token messages in the frontend
                    await self._trigger_callback("on_message", token, is_token=True)

                # Get LLM response with tools and streaming
                response = await self.llm.complete(
                    tools=self._get_tool_schemas(),
                    tool_choice="auto",
                    stream=True,
                    on_token=handle_token,
                )
                # Detect step progression in the message content
                if response.content:
                    # --- Plan detection: fire on_plan_created when agent produces a numbered plan ---
                    if not getattr(self, '_plan_sent', False):
                        # Look for numbered lists that indicate a plan
                        import re as _re
                        plan_lines = _re.findall(
                            r'^\s*(?:\d+\.|-|\*|•)\s+(.+)', response.content, _re.MULTILINE
                        )
                        # Only treat as a plan if we have 2+ distinct bullet/number points
                        if len(plan_lines) >= 2:
                            steps = [l.strip() for l in plan_lines[:20] if l.strip()]
                            if steps:
                                self._plan_sent = True
                                self._plan_steps = steps
                                self._current_step_idx = 0
                                await self._trigger_callback("on_plan_created", steps)
                                await self._trigger_callback("on_step_started", 0, steps[0])

                    # --- Step progression detection when a plan is already active ---
                    if getattr(self, '_plan_sent', False):
                        new_step_reached = -1
                        # Look for patterns like "Moving to milestone 2" or "Step 2:""
                        step_patterns = [
                            r"move to step (\d+)",
                            r"moving to step (\d+)",
                            r"starting step (\d+)",
                            r"now on step (\d+)",
                            r"proceed to step (\d+)",
                            r"step (\d+):",
                            r"milestone (\d+):",
                            r"^\s*(\d+)\.", # Numbered line
                        ]

                        content_lower = response.content.lower()
                        for pattern in step_patterns:
                            match = _re.search(pattern, content_lower, _re.MULTILINE)
                            if match:
                                try:
                                    new_step_reached = int(match.group(1)) - 1 # 0-indexed
                                    break
                                except (ValueError, IndexError):
                                    continue

                        if 0 <= new_step_reached < len(self._plan_steps):
                            if new_step_reached > self._current_step_idx:
                                # Complete previous step
                                await self._trigger_callback("on_step_completed", self._current_step_idx, self._plan_steps[self._current_step_idx])
                                # Start new step
                                self._current_step_idx = new_step_reached
                                await self._trigger_callback("on_step_started", self._current_step_idx, self._plan_steps[self._current_step_idx])

                # Handle tool calls
                if response.tool_calls:
                    await self._update_phase(AgentPhase.EXECUTE)
                    
                    # Add assistant message with tool calls
                    self.llm.add_assistant_message(
                        content=response.content,
                        tool_calls=response.tool_calls,
                    )
                    
                    # Execute each tool
                    for tc in response.tool_calls:
                        if "on_tool_start" in self.callbacks:
                            await self._trigger_callback("on_tool_start", tc.name, tc.arguments)
                        
                        tool_success = False
                        retry_count = 0
                        final_result = ""
                        # Track total tool uses this run (used to gate TASK COMPLETE)
                        self._tools_used_count = getattr(self, '_tools_used_count', 0) + 1
                        self._no_tool_streak = 0  # reset streak on real tool call
                        
                        # Self-correction retry loop
                        while retry_count <= self.max_immediate_retries:
                            self._log(f"Executing tool {tc.name} ({retry_count}/{self.max_immediate_retries}): {json.dumps(tc.arguments)[:100]}...")
                            import time
                            start_time = time.time()
                            
                            # For tool execution classification
                            exit_code = None
                            
                            result = await self._execute_tool(tc.name, tc.arguments)
                            duration_ms = (time.time() - start_time) * 1000
                            self._log(f"Tool result: {result[:200]}...")
                            final_result = result
                            
                            # Parse exit code from terminal output if available
                            if tc.name == "terminal" and "Exit code:" in result:
                                try:
                                    exit_code_str = result.split("Exit code:")[-1].strip()
                                    exit_code = int(exit_code_str)
                                except (ValueError, TypeError):
                                    pass
                                    
                            # Classify the result
                            error_type = self._correction_engine.classify_error(tc.name, result, exit_code)
                            
                            if error_type == ErrorType.SUCCESS:
                                tool_success = True
                                # If we had retried, record the successful correction
                                if retry_count > 0:
                                    self._log(f"Self-correction successful after {retry_count} retries")
                                    # Reset failures and notify UI about success
                                    self._consecutive_failures = 0
                                    await self._trigger_callback("on_message", f"✅ Successfully corrected error in {tc.name}", is_token=False)
                                break
                                
                            # Tool failed
                            self._log(f"Tool {tc.name} failed with {error_type.value}")
                            
                            if not self._correction_engine.should_retry(error_type, retry_count):
                                self._log("Max retries reached or error not retryable immediately")
                                break
                                
                            # Need to retry with hint
                            retry_count += 1
                            hint = self._correction_engine.get_retry_hint(error_type, tc.name, tc.arguments, result)
                            
                            await self._trigger_callback("on_message", f"🔄 **Self-Correcting**: Tool '{tc.name}' failed. Retrying... (Attempt {retry_count}/{self.max_immediate_retries})\n*Hint:* {hint}", is_token=False)
                            
                            # We feed the failed result and hint to LLM to get a corrected tool call
                            self.llm.add_tool_result(tc.id, tc.name, result)
                            self.llm.add_user_message(f"Your tool call failed: {hint}\nPlease review the error and provide a corrected tool call.")
                            
                            # Ask LLM again
                            retry_response = await self.llm.complete(
                                tools=self._get_tool_schemas(),
                                tool_choice="required", # Force tool use
                                stream=False
                            )
                            
                            if retry_response.tool_calls:
                                # Update the current tool call with LLM's new attempt
                                tc = retry_response.tool_calls[0]
                                self.llm.add_assistant_message(
                                    content=retry_response.content,
                                    tool_calls=[tc],
                                )
                            else:
                                # LLM didn't provide a tool call, break retry loop
                                break
                        
                        if "on_tool_result" in self.callbacks:
                            await self._trigger_callback("on_tool_result", tc.name, tc.arguments, final_result, duration_ms)
                        
                        # Add final tool result to conversation
                        self.llm.add_tool_result(tc.id, tc.name, final_result)
                        
                        # Track in task state
                        if tc.name == "terminal":
                            task.commands_executed.append(tc.arguments.get("command", ""))
                            
                        # Handle Escalation to Planner
                        if not tool_success:
                            self._consecutive_failures += 1
                            if self._correction_engine.should_replan(self._consecutive_failures):
                                await self._trigger_callback("on_message", "🔁 **Replanning needed**: Consecutive failures exceeded limit.", is_token=False)
                                self._consecutive_failures = 0
                                
                                # trigger replanning
                                if self.state.current_plan:
                                    # Find current step
                                    failed_step_id = None
                                    from ..schemas.state import StepStatus
                                    for step in self.state.current_plan.steps:
                                        if step.status == StepStatus.PENDING or step.status == StepStatus.IN_PROGRESS:
                                            failed_step_id = step.step_id
                                            break
                                            
                                    if failed_step_id:
                                        await self.replan_from_failure(failed_step_id, final_result)
                                        # Force breakdown of iteration loop to let new plan take over
                                        task.error_count += 1
                                        break
                                
                else:
                    # No tool calls - check if task is complete
                    if response.content:
                        self._log(f"Assistant: {response.content[:200]}...")
                        self.llm.add_assistant_message(content=response.content)
                        
                        # Detect numbered plan in LLM response (e.g. "1. Step one\n2. Step two")
                        content = response.content or ""
                        import re
                        plan_lines = re.findall(r'^\s*(\d+)\.\s+(.+)', content, re.MULTILINE)
                        if len(plan_lines) >= 2 and not getattr(self, '_plan_sent', False):
                            steps = [text.strip() for _, text in plan_lines]
                            self._plan_sent = True
                            self._plan_steps = steps
                            self._current_step_idx = 0
                            await self._trigger_callback("on_plan_created", steps)
                            await self._update_phase(AgentPhase.PLAN)
                            
                            # Start first step
                            if steps:
                                await self._trigger_callback("on_step_started", 0, steps[0])
                        
                        # Only allow TASK COMPLETE if at least one tool was used this run
                        _tools_used = getattr(self, '_tools_used_count', 0)
                        content_lower = response.content.lower()
                        is_completion = any(phrase in content_lower for phrase in [
                            "task complete",
                            "task is complete",
                            "successfully completed",
                            "finished the task",
                            "completed the task",
                            "all done",
                        ])

                        if is_completion and _tools_used > 0:
                            await self._update_phase(AgentPhase.COMPLETE)
                            task.status = TaskStatus.COMPLETED
                            task.completed_at = datetime.now(timezone.utc)
                            self._log("Task completed!")
                            break
                        elif is_completion and _tools_used == 0:
                            # Agent said complete without using any tools — force it to act
                            self._log("Completion signal with no tool use — forcing tool call.")
                            self.llm.add_user_message(
                                "You have not used any tools yet. You MUST use the available tools (terminal, editor, etc.) "
                                "to actually perform the task. Do NOT just describe what you would do — call a tool now."
                            )
                        elif response.finish_reason == "stop":
                            # Nudge with forced tool use on the next iteration
                            self._no_tool_streak = getattr(self, '_no_tool_streak', 0) + 1
                            if self._no_tool_streak >= 2:
                                # After 2 text-only turns, force a tool call
                                self._log("Multiple text-only turns — injecting forced tool call.")
                                forced_response = await self.llm.complete(
                                    tools=self._get_tool_schemas(),
                                    tool_choice="required",
                                    stream=False,
                                )
                                self._no_tool_streak = 0
                                if forced_response.tool_calls:
                                    # Re-inject as if it was a normal tool call turn
                                    self.llm.add_assistant_message(
                                        content=forced_response.content,
                                        tool_calls=forced_response.tool_calls,
                                    )
                                    for tc in forced_response.tool_calls:
                                        result = await self._execute_tool(tc.name, tc.arguments)
                                        self._tools_used_count = getattr(self, '_tools_used_count', 0) + 1
                                        self.llm.add_tool_result(tc.id, tc.name, result)
                            else:
                                self.llm.add_user_message(
                                    "Please continue with the task using the tools. "
                                    "Call a tool (terminal or editor) to take the next action."
                                )

            
            except Exception as e:
                error_msg = f"Error in iteration: {str(e)}"
                self._log(error_msg)
                
                # Report error to UI if callback exists
                await self._trigger_callback("on_message", f"⚠️ **Error**: {str(e)}", is_token=False)
                
                task.last_error = str(e)
                task.error_count += 1
                
                if task.error_count >= 3:
                    await self._update_phase(AgentPhase.BLOCKED)
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
                if diff_result and not diff_result.is_empty:
                    self._artifact_logger.set_diff(diff_result.diff_text)
            
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

        # ── Checkpoint → Verify → Rollback-on-failure ──────────────────────────
        if task.status == TaskStatus.COMPLETED and self.working_directory:
            git_mgr = self._get_git_manager()
            if git_mgr:
                checkpoint_id = f"post-task-{task.task_id}"
                try:
                    # Create a checkpoint with the current changes
                    await git_mgr.create_checkpoint(checkpoint_id, f"agent: {task.goal.description[:60]}")
                    self._log(f"Checkpoint created: {checkpoint_id}")
                    await self._trigger_callback(
                        "on_message",
                        "📌 **Git**: Checkpoint created — running verification…",
                        is_token=False,
                    )

                    # Run verification (tests, lints)
                    verification = await self.run_verification(task_id=task.task_id)

                    if verification and not verification.passed and verification.blocking_failures:
                        # Tests failed — rollback to checkpoint
                        await self._trigger_callback(
                            "on_message",
                            f"⚠️ **Verification failed** ({len(verification.blocking_failures)} issue(s)). "
                            "Rolling back to last checkpoint…",
                            is_token=False,
                        )
                        self._log(f"Verification failed. Rolling back to {checkpoint_id}")
                        await git_mgr.rollback_to_checkpoint(checkpoint_id, hard=True)
                        task.status = TaskStatus.FAILED
                        task.last_error = (
                            f"Verification failed: {', '.join(verification.blocking_failures)}"
                        )
                        await self._trigger_callback(
                            "on_message",
                            "🔄 **Rolled back** to pre-task state. Please review errors and retry.",
                            is_token=False,
                        )
                    elif verification and verification.passed:
                        self._log("Verification passed — keeping changes")
                        await self._trigger_callback(
                            "on_message",
                            f"✅ **Verification passed** ({verification.checks_passed}/{verification.total_checks} checks)",
                            is_token=False,
                        )
                except Exception as e:
                    self._log(f"Checkpoint/verify step skipped: {e}")

        # Auto git commit after task completion (only if still COMPLETED after verification)
        if self.auto_git_commit and task.status == TaskStatus.COMPLETED and self.working_directory:
            await self._auto_commit_changes(task)

        return task

    async def _auto_commit_changes(self, task: TaskState) -> None:
        """Automatically commit and optionally push changes after task completion."""
        import os
        from pathlib import Path
        cwd = self.working_directory
        git_dir = Path(cwd) / ".git"
        try:
            # Init repo if not a git repo yet
            if not git_dir.exists():
                import asyncio
                proc = await asyncio.create_subprocess_shell(
                    "git init && git add -A",
                    cwd=cwd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await proc.communicate()
            else:
                proc = await asyncio.create_subprocess_shell(
                    "git add -A",
                    cwd=cwd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await proc.communicate()

            # Build commit message from task description
            short_desc = (task.goal.description[:72] if task.goal.description else "task")
            commit_msg = f"feat: {short_desc}"
            proc = await asyncio.create_subprocess_shell(
                f'git commit -m "{commit_msg}"',
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode == 0:
                self._log(f"Auto-committed changes: {commit_msg}")
                await self._trigger_callback("on_message", f"✅ **Git**: Committed — `{commit_msg}`", is_token=False)
            else:
                out = (stdout or b"").decode() + (stderr or b"").decode()
                if "nothing to commit" in out:
                    self._log("Auto-commit: nothing to commit")
                else:
                    self._log(f"Auto-commit warning: {out[:200]}")

            # Push if enabled and remote exists
            if self.git_push:
                proc = await asyncio.create_subprocess_shell(
                    "git push",
                    cwd=cwd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()
                if proc.returncode == 0:
                    self._log("Auto-pushed to remote")
                    await self._trigger_callback("on_message", "✅ **Git**: Pushed to remote", is_token=False)
                else:
                    err = (stderr or b"").decode()
                    self._log(f"Auto-push failed: {err[:200]}")
        except Exception as e:
            self._log(f"Auto-commit failed: {e}")
    
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
        summary = f"Task {'completed' if result.status == TaskStatus.COMPLETED else 'failed'}"
        for msg in reversed(self.llm.conversation):
            if msg.role == "assistant" and msg.content:
                summary = msg.content
                break
                
        if result.status == TaskStatus.COMPLETED:
            await self._trigger_callback("on_task_complete", summary)
        elif result.status == TaskStatus.FAILED:
            await self._trigger_callback("on_task_failed", result.last_error)
        
        return summary


async def create_agent(
    model: str = "gpt-4o",
    api_key: str | None = None,
    working_directory: str | None = None,
    verbose: bool = True,
    callbacks: dict[str, Any] | None = None,
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
        callbacks=callbacks
    )
