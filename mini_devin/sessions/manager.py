"""
Session Manager for Mini-Devin

This module provides multi-session support for:
- Creating and managing agent sessions
- Running tasks concurrently
- Tracking task progress and results
- Resource management and cleanup
"""

import asyncio
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field

from ..orchestrator.agent import Agent
from ..api.websocket import ConnectionManager
from ..schemas.state import TaskState, TaskGoal, TaskStatus as AgentTaskStatus


class SessionStatus(str, Enum):
    """Status of a session."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class TaskStatus(str, Enum):
    """Status of a task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    """Result of a completed task."""
    status: str
    summary: str
    files_modified: list[str] = field(default_factory=list)
    commands_executed: list[str] = field(default_factory=list)
    total_tokens: int = 0
    duration_seconds: float = 0.0


@dataclass
class Task:
    """A task to be executed by an agent."""
    task_id: str
    description: str
    acceptance_criteria: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    iteration: int = 0
    error_message: str | None = None
    result: TaskResult | None = None
    
    # Artifacts
    artifacts_dir: Path | None = None


@dataclass
class Session:
    """An agent session."""
    session_id: str
    working_directory: str
    model: str
    max_iterations: int
    status: SessionStatus = SessionStatus.IDLE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Task tracking
    tasks: dict[str, Task] = field(default_factory=dict)
    current_task_id: str | None = None
    total_tasks: int = 0
    iteration: int = 0
    
    # Agent instance
    agent: Agent | None = None
    
    # Cancellation
    cancel_event: asyncio.Event | None = None


class SessionManager:
    """
    Manages multiple agent sessions.
    
    Supports:
    - Creating and deleting sessions
    - Running tasks in sessions
    - Concurrent task execution across sessions
    - Task cancellation
    - Artifact management
    """
    
    def __init__(
        self,
        artifacts_base_dir: str = "./runs",
        max_concurrent_sessions: int = 10,
    ):
        self.artifacts_base_dir = Path(artifacts_base_dir)
        self.max_concurrent_sessions = max_concurrent_sessions
        
        # Session storage
        self.sessions: dict[str, Session] = {}
        
        # Statistics
        self._start_time = datetime.now(timezone.utc)
        self._total_tasks_completed = 0
        
        # Locks for thread safety
        self._session_lock = asyncio.Lock()
    
    async def create_session(
        self,
        working_directory: str = ".",
        model: str = "gpt-4o",
        max_iterations: int = 50,
    ) -> Session:
        """
        Create a new agent session.
        
        Args:
            working_directory: Working directory for the agent
            model: LLM model to use
            max_iterations: Maximum iterations per task
            
        Returns:
            The created session
        """
        async with self._session_lock:
            if len(self.sessions) >= self.max_concurrent_sessions:
                raise RuntimeError(f"Maximum concurrent sessions ({self.max_concurrent_sessions}) reached")
            
            session_id = str(uuid.uuid4())[:8]
            
            # Create LLM client with specified model
            from ..core.llm_client import create_llm_client
            llm_client = create_llm_client(model=model)
            
            # Create agent
            agent = Agent(
                llm_client=llm_client,
                working_directory=working_directory,
                max_iterations=max_iterations,
            )
            
            session = Session(
                session_id=session_id,
                working_directory=working_directory,
                model=model,
                max_iterations=max_iterations,
                agent=agent,
                cancel_event=asyncio.Event(),
            )
            
            self.sessions[session_id] = session
            
            return session
    
    def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID."""
        return self.sessions.get(session_id)
    
    def list_sessions(self) -> list[Session]:
        """List all sessions."""
        return list(self.sessions.values())
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: The session ID
            
        Returns:
            True if deleted, False if not found
        """
        async with self._session_lock:
            if session_id not in self.sessions:
                return False
            
            session = self.sessions[session_id]
            
            # Cancel any running task
            if session.cancel_event:
                session.cancel_event.set()
            
            # Clean up agent
            if session.agent:
                await session.agent.cleanup()
            
            del self.sessions[session_id]
            return True
    
    async def create_task(
        self,
        session_id: str,
        description: str,
        acceptance_criteria: list[str] | None = None,
    ) -> Task:
        """
        Create a new task in a session.
        
        Args:
            session_id: The session ID
            description: Task description
            acceptance_criteria: Optional acceptance criteria
            
        Returns:
            The created task
        """
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        task_id = str(uuid.uuid4())[:8]
        
        # Create artifacts directory
        artifacts_dir = self.artifacts_base_dir / session_id / task_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        task = Task(
            task_id=task_id,
            description=description,
            acceptance_criteria=acceptance_criteria or [],
            artifacts_dir=artifacts_dir,
        )
        
        session.tasks[task_id] = task
        session.total_tasks += 1
        
        return task
    
    def get_task(self, session_id: str, task_id: str) -> Task | None:
        """Get a task by ID."""
        session = self.sessions.get(session_id)
        if not session:
            return None
        return session.tasks.get(task_id)
    
    def list_tasks(self, session_id: str) -> list[Task]:
        """List all tasks in a session."""
        session = self.sessions.get(session_id)
        if not session:
            return []
        return list(session.tasks.values())
    
    def get_task_result(self, session_id: str, task_id: str) -> TaskResult | None:
        """Get the result of a task."""
        task = self.get_task(session_id, task_id)
        if not task:
            return None
        return task.result
    
    async def run_task(
        self,
        session_id: str,
        task_id: str,
        connection_manager: ConnectionManager | None = None,
    ) -> TaskResult:
        """
        Run a task in a session.
        
        Args:
            session_id: The session ID
            task_id: The task ID
            connection_manager: Optional WebSocket connection manager for streaming
            
        Returns:
            The task result
        """
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        task = session.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        if not session.agent:
            raise ValueError(f"Session {session_id} has no agent")
        
        # Update status
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now(timezone.utc)
        session.status = SessionStatus.RUNNING
        session.current_task_id = task_id
        
        # Notify via WebSocket
        if connection_manager:
            await connection_manager.send_task_started(
                session_id, task_id, task.description
            )
        
        start_time = datetime.now(timezone.utc)
        
        try:
            # Create callback for streaming updates
            async def on_update(update_type: str, data: dict[str, Any]) -> None:
                if not connection_manager:
                    return
                
                if update_type == "phase_changed":
                    await connection_manager.send_phase_changed(
                        session_id, task_id, data.get("phase", "")
                    )
                elif update_type == "tool_started":
                    await connection_manager.send_tool_started(
                        session_id, task_id,
                        data.get("tool", ""),
                        data.get("input", {}),
                    )
                elif update_type == "tool_completed":
                    await connection_manager.send_tool_completed(
                        session_id, task_id,
                        data.get("tool", ""),
                        data.get("output", {}),
                        data.get("duration_ms", 0),
                    )
                elif update_type == "tokens":
                    await connection_manager.send_tokens(
                        session_id, task_id, data.get("content", "")
                    )
                
                # Update iteration count
                task.iteration = data.get("iteration", task.iteration)
                session.iteration = task.iteration
            
            # Create TaskState object for the agent
            task_state = TaskState(
                task_id=task_id,
                goal=TaskGoal(
                    description=task.description,
                    acceptance_criteria=task.acceptance_criteria or [],
                ),
                status=AgentTaskStatus.PENDING,
            )

            # Set up callbacks on the agent instance
            session.agent.callbacks = {
                "on_message": lambda token, is_token=False: asyncio.create_task(on_update("tokens", {"content": token})) if is_token else None,
                "on_tool_start": lambda name, args: asyncio.create_task(on_update("tool_started", {"tool": name, "input": args})),
                "on_tool_result": lambda name, args, output, duration: asyncio.create_task(on_update("tool_completed", {"tool": name, "output": output, "duration_ms": duration})),
                "on_phase_change": lambda phase: asyncio.create_task(on_update("phase_changed", {"phase": phase})),
            }
            
            # Run the agent with the TaskState object
            final_task_state = await session.agent.run(task_state)
            
            # Extract result info from final_task_state and agent's conversation
            success = final_task_state.status == AgentTaskStatus.COMPLETE or final_task_state.status == AgentTaskStatus.COMPLETED
            
            # Get summary from last assistant message
            summary = ""
            if session.agent.llm and hasattr(session.agent.llm, 'conversation'):
                for msg in reversed(session.agent.llm.conversation):
                    if msg.role == "assistant" and msg.content:
                        summary = msg.content
                        break
            
            # Create a result-like object for compatibility
            class AgentResult:
                def __init__(self, success, summary, files_modified, commands_executed, total_tokens, error_message):
                    self.success = success
                    self.summary = summary
                    self.files_modified = files_modified
                    self.commands_executed = commands_executed
                    self.total_tokens = total_tokens
                    self.error_message = error_message

            result = AgentResult(
                success=success,
                summary=summary,
                files_modified=[f.path for f in final_task_state.files_changed],
                commands_executed=final_task_state.commands_executed,
                total_tokens=final_task_state.total_tokens_used,
                error_message=final_task_state.last_error
            )
            
            # Calculate duration
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            # Create task result
            task_result = TaskResult(
                status="completed" if result.success else "failed",
                summary=result.summary or "",
                files_modified=result.files_modified or [],
                commands_executed=result.commands_executed or [],
                total_tokens=result.total_tokens or 0,
                duration_seconds=duration,
            )
            
            # Update task
            task.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
            task.completed_at = end_time
            task.result = task_result
            
            if not result.success:
                task.error_message = result.error_message
            
            # Save artifacts
            if task.artifacts_dir:
                self._save_artifacts(task, result)
            
            # Update statistics
            if result.success:
                self._total_tasks_completed += 1
            
            # Notify via WebSocket
            if connection_manager:
                if result.success:
                    await connection_manager.send_task_completed(
                        session_id, task_id, task_result.summary
                    )
                else:
                    await connection_manager.send_task_failed(
                        session_id, task_id, task.error_message or "Unknown error"
                    )
            
            return task_result
            
        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now(timezone.utc)
            task.error_message = "Task cancelled"
            
            return TaskResult(
                status="cancelled",
                summary="Task was cancelled",
                duration_seconds=(datetime.now(timezone.utc) - start_time).total_seconds(),
            )
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now(timezone.utc)
            task.error_message = str(e)
            
            if connection_manager:
                await connection_manager.send_task_failed(
                    session_id, task_id, str(e)
                )
            
            return TaskResult(
                status="failed",
                summary=f"Task failed: {e}",
                duration_seconds=(datetime.now(timezone.utc) - start_time).total_seconds(),
            )
            
        finally:
            session.status = SessionStatus.IDLE
            session.current_task_id = None
    
    async def cancel_task(self, session_id: str, task_id: str) -> bool:
        """
        Cancel a running task.
        
        Args:
            session_id: The session ID
            task_id: The task ID
            
        Returns:
            True if cancelled, False if not found or not running
        """
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        task = session.tasks.get(task_id)
        if not task or task.status != TaskStatus.RUNNING:
            return False
        
        # Signal cancellation
        if session.cancel_event:
            session.cancel_event.set()
        
        return True
    
    def _save_artifacts(self, task: Task, result: Any) -> None:
        """Save task artifacts to disk."""
        if not task.artifacts_dir:
            return
        
        import json
        
        # Save plan
        if hasattr(result, 'plan') and result.plan:
            plan_file = task.artifacts_dir / "plan.json"
            plan_file.write_text(json.dumps(result.plan, indent=2, default=str))
        
        # Save tool calls
        if hasattr(result, 'tool_calls') and result.tool_calls:
            calls_file = task.artifacts_dir / "tool_calls.json"
            calls_file.write_text(json.dumps(result.tool_calls, indent=2, default=str))
        
        # Save verification results
        if hasattr(result, 'verification_results') and result.verification_results:
            verify_file = task.artifacts_dir / "verification_results.json"
            verify_file.write_text(json.dumps(result.verification_results, indent=2, default=str))
        
        # Save diff
        if hasattr(result, 'diff') and result.diff:
            diff_file = task.artifacts_dir / "diff.patch"
            diff_file.write_text(result.diff)
        
        # Save summary
        summary_file = task.artifacts_dir / "final_summary.md"
        summary_content = f"""# Task Summary

**Task ID:** {task.task_id}
**Description:** {task.description}
**Status:** {task.status.value}
**Duration:** {task.result.duration_seconds:.2f}s

## Result

{task.result.summary if task.result else 'No result'}

## Files Modified

{chr(10).join(f'- {f}' for f in (task.result.files_modified if task.result else []))}

## Commands Executed

{chr(10).join(f'- {c}' for c in (task.result.commands_executed if task.result else []))}
"""
        summary_file.write_text(summary_content)
    
    def list_artifacts(self, session_id: str, task_id: str) -> list[str] | None:
        """List artifacts for a task."""
        task = self.get_task(session_id, task_id)
        if not task or not task.artifacts_dir:
            return None
        
        if not task.artifacts_dir.exists():
            return []
        
        return [f.name for f in task.artifacts_dir.iterdir() if f.is_file()]
    
    def get_artifact(self, session_id: str, task_id: str, artifact_name: str) -> str | None:
        """Get artifact content."""
        task = self.get_task(session_id, task_id)
        if not task or not task.artifacts_dir:
            return None
        
        artifact_path = task.artifacts_dir / artifact_name
        if not artifact_path.exists():
            return None
        
        return artifact_path.read_text()
    
    @property
    def active_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self.sessions)
    
    @property
    def total_tasks_completed(self) -> int:
        """Get total number of completed tasks."""
        return self._total_tasks_completed
    
    @property
    def uptime_seconds(self) -> float:
        """Get uptime in seconds."""
        return (datetime.now(timezone.utc) - self._start_time).total_seconds()
    
    async def shutdown(self) -> None:
        """Shutdown all sessions."""
        for session_id in list(self.sessions.keys()):
            await self.delete_session(session_id)
