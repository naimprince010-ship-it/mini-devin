"""
Database-backed Session Manager for Mini-Devin

This module provides persistent session management using PostgreSQL.
It wraps the existing SessionManager functionality while adding database persistence.
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from ..database.models import (
    SessionModel,
    TaskModel,
    SessionStatus as DBSessionStatus,
    TaskStatus as DBTaskStatus,
)
from ..database.repository import (
    SessionRepository,
    TaskRepository,
    ArtifactRepository,
)
from ..database.config import get_session_maker
from ..orchestrator.agent import Agent
from ..api.websocket import ConnectionManager
from .manager import SessionStatus, TaskStatus, TaskResult, Task, Session
from ..schemas.state import TaskState, TaskGoal, TaskStatus as AgentTaskStatus


class DatabaseSessionManager:
    """
    Database-backed session manager.
    
    Provides persistent storage for sessions, tasks, and results using PostgreSQL.
    Maintains in-memory agent instances for active sessions.
    """
    
    def __init__(
        self,
        artifacts_base_dir: str = "./runs",
        max_concurrent_sessions: int = 10,
    ):
        self.artifacts_base_dir = Path(artifacts_base_dir)
        self.max_concurrent_sessions = max_concurrent_sessions
        
        # In-memory agent instances (not persisted)
        self._agents: dict[str, Agent] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._running_tasks: dict[str, asyncio.Task] = {}  # Track asyncio Tasks for cancellation
        self._session_titles: dict[str, str] = {}  # Cache titles in-memory
        self._sandboxes: dict[str, Any] = {}  # Active DockerSandbox instances per session
        
        # Statistics
        self._start_time = datetime.now(timezone.utc)
        
        # Session maker
        self._session_maker = get_session_maker()
        
        # Locks for thread safety
        self._session_lock = asyncio.Lock()
    
    async def _get_db_session(self) -> AsyncSession:
        """Get a database session."""
        return self._session_maker()
    
    async def create_session(
        self,
        working_directory: str = ".",
        model: str = "gpt-4o",
        max_iterations: int = 50,
        session_id: str | None = None,
        use_sandbox: bool = False,
        auto_git_commit: bool = False,
        git_push: bool = False,
    ) -> Session:
        """Create a new agent session with database persistence."""
        async with self._session_lock:
            async with self._session_maker() as db:
                repo = SessionRepository(db)
                
                # Check concurrent session limit
                active_count = await repo.count_active()
                if active_count >= self.max_concurrent_sessions:
                    raise RuntimeError(f"Maximum concurrent sessions ({self.max_concurrent_sessions}) reached")
                
                # Create database record
                db_session = await repo.create(
                    working_directory=working_directory,
                    model=model,
                    max_iterations=max_iterations,
                    session_id=session_id,
                )
                await db.commit()
                
                session_id = db_session.id
                
                # Create LLM client and agent (in-memory)
                from ..core.llm_client import create_llm_client
                llm_client = create_llm_client(model=model)
                
                # Build Docker sandbox if requested
                sandbox = None
                if use_sandbox:
                    try:
                        from ..sandbox.factory import create_execution_sandbox

                        sandbox = create_execution_sandbox(repo_path=working_directory)
                    except Exception as e:
                        print(f"[Session] Sandbox creation failed, running without sandbox: {e}")
                
                agent = Agent(
                    llm_client=llm_client,
                    working_directory=working_directory,
                    max_iterations=max_iterations,
                    use_sandbox=use_sandbox,
                    auto_git_commit=auto_git_commit,
                    git_push=git_push,
                )
                if sandbox:
                    agent._sandbox = sandbox
                
                self._agents[session_id] = agent
                self._cancel_events[session_id] = asyncio.Event()
                
                # Return Session object for API compatibility
                return Session(
                    session_id=session_id,
                    working_directory=working_directory,
                    model=model,
                    max_iterations=max_iterations,
                    agent=agent,
                    cancel_event=self._cancel_events[session_id],
                )
    
    async def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID."""
        async with self._session_maker() as db:
            repo = SessionRepository(db)
            db_session = await repo.get(session_id)
            
            if not db_session:
                return None
            
            # Reconstruct Session object
            return self._db_to_session(db_session)
    
    async def list_sessions(self) -> list[Session]:
        """List all sessions with their tasks eagerly loaded."""
        async with self._session_maker() as db:
            # Use selectinload to avoid MissingGreenlet error when accessing db_session.tasks
            from sqlalchemy import select
            from sqlalchemy.orm import selectinload
            from ..database.models import SessionModel, TaskModel
            
            result = await db.execute(
                select(SessionModel)
                .options(
                    selectinload(SessionModel.tasks).selectinload(TaskModel.result),
                    selectinload(SessionModel.tasks).selectinload(TaskModel.artifacts)
                )
                .order_by(SessionModel.created_at.desc())
            )
            db_sessions = result.scalars().all()
            
            # Explicitly initialize properties to avoid lazy loading issues
            for s in db_sessions:
                if not hasattr(s, 'tasks') or s.tasks is None:
                    s.tasks = []
                    
            return [self._db_to_session(s) for s in db_sessions]
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        async with self._session_lock:
            # Cancel any running task
            if session_id in self._cancel_events:
                self._cancel_events[session_id].set()
                del self._cancel_events[session_id]
            
            # Clean up agent
            if session_id in self._agents:
                agent = self._agents[session_id]
                if hasattr(agent, 'cleanup') and callable(getattr(agent, 'cleanup')):
                    await agent.cleanup()
                del self._agents[session_id]
            
            # Delete from database
            async with self._session_maker() as db:
                repo = SessionRepository(db)
                result = await repo.delete(session_id)
                await db.commit()
                return result
    
    async def create_task(
        self,
        session_id: str,
        description: str,
        acceptance_criteria: list[str] | None = None,
        connection_manager: 'ConnectionManager | None' = None,
    ) -> Task:
        """Create a new task in a session."""
        async with self._session_maker() as db:
            session_repo = SessionRepository(db)
            task_repo = TaskRepository(db)
            
            # Verify session exists
            db_session = await session_repo.get(session_id)
            if not db_session:
                raise ValueError(f"Session {session_id} not found")
            
            # Create task in database
            db_task = await task_repo.create(
                session_id=session_id,
                description=description,
                acceptance_criteria=acceptance_criteria,
            )
            await db.commit()
            
            # Create artifacts directory
            artifacts_dir = self.artifacts_base_dir / session_id / db_task.id
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            
            # Auto-generate title from first task
            session_title = self._session_titles.get(session_id, '')
            if not session_title and description:
                raw = description.strip().replace('\n', ' ')
                session_title = raw[:60] + ('\u2026' if len(raw) > 60 else '')
                self._session_titles[session_id] = session_title
                # Emit WS event if connection_manager provided
                if connection_manager:
                    try:
                        await connection_manager.send_session_title_updated(session_id, session_title)
                    except Exception:
                        pass

            return self._db_to_task(db_task, artifacts_dir)

    async def inject_followup(self, session_id: str, message: str) -> None:
        """Inject a follow-up message into the running agent's conversation."""
        agent = self._agents.get(session_id)
        if not agent:
            return
        # Inject into the LLM conversation history so the next LLM call sees it
        try:
            if hasattr(agent, 'llm') and agent.llm and hasattr(agent.llm, 'conversation'):
                from ..core.llm_client import LLMMessage
                agent.llm.conversation.append(LLMMessage(role='user', content=f'[User follow-up]: {message}'))
        except Exception as e:
            print(f'inject_followup error: {e}')

    async def answer_clarification(self, session_id: str, answer: str) -> bool:
        """Provide an answer to a clarification question asked by the agent."""
        agent = self._agents.get(session_id)
        if not agent:
            return False
        try:
            agent._clarification_answer = answer
            if agent._clarification_event is not None:
                agent._clarification_event.set()
            return True
        except Exception as e:
            print(f'answer_clarification error: {e}')
            return False

    async def stop_session(self, session_id: str) -> bool:
        """Stop the running task in a session."""
        if session_id in self._cancel_events:
            self._cancel_events[session_id].set()
        running = self._running_tasks.get(session_id)
        if running and not running.done():
            running.cancel()
            return True
        return False

    async def start_sandbox(self, session_id: str) -> dict:
        """Start a Docker sandbox for the given session."""
        session = await self.get_session(session_id)
        if not session:
            return {"started": False, "error": "Session not found"}

        # If a sandbox is already running, return its status
        existing = self._sandboxes.get(session_id)
        if existing and existing.is_running():
            return {
                "started": True,
                "session_id": session_id,
                "container_id": existing.container_id,
                "status": existing.status.value,
                "backend": getattr(existing, "backend", "docker"),
            }

        try:
            from ..sandbox.factory import create_execution_sandbox

            working_dir = session.working_directory or "."
            sandbox = create_execution_sandbox(repo_path=working_dir)
            ok = await sandbox.start()
            if ok:
                self._sandboxes[session_id] = sandbox
                # Also wire it to the agent if one exists
                agent = self._agents.get(session_id)
                if agent:
                    agent._sandbox = sandbox
                    agent.use_sandbox = True
                return {
                    "started": True,
                    "session_id": session_id,
                    "container_id": sandbox.container_id,
                    "status": sandbox.status.value,
                    "backend": getattr(sandbox, "backend", "docker"),
                }
            else:
                return {"started": False, "error": "Docker failed to start the container"}
        except Exception as e:
            return {"started": False, "error": str(e)}

    async def stop_sandbox(self, session_id: str) -> dict:
        """Stop the Docker sandbox for the given session."""
        sandbox = self._sandboxes.get(session_id)
        if not sandbox:
            return {"stopped": False, "error": "No active sandbox for this session"}

        try:
            ok = await sandbox.stop()
            if ok:
                del self._sandboxes[session_id]
                # Detach sandbox from agent
                agent = self._agents.get(session_id)
                if agent:
                    agent._sandbox = None
                    agent.use_sandbox = False
                return {"stopped": True, "session_id": session_id}
            else:
                return {"stopped": False, "error": "Failed to stop Docker container"}
        except Exception as e:
            return {"stopped": False, "error": str(e)}
    
    async def get_task(self, session_id: str, task_id: str) -> Task | None:
        """Get a task by ID."""
        async with self._session_maker() as db:
            repo = TaskRepository(db)
            db_task = await repo.get(task_id)
            
            if not db_task or db_task.session_id != session_id:
                return None
            
            artifacts_dir = self.artifacts_base_dir / session_id / task_id
            return self._db_to_task(db_task, artifacts_dir)
    
    async def list_tasks(self, session_id: str) -> list[Task]:
        """List all tasks in a session."""
        async with self._session_maker() as db:
            repo = TaskRepository(db)
            db_tasks = await repo.list_by_session(session_id)
            
            return [
                self._db_to_task(t, self.artifacts_base_dir / session_id / t.id)
                for t in db_tasks
            ]
    
    async def get_task_result(self, session_id: str, task_id: str) -> TaskResult | None:
        """Get the result of a task."""
        async with self._session_maker() as db:
            repo = TaskRepository(db)
            db_result = await repo.get_result(task_id)
            
            if not db_result:
                return None
            
            return TaskResult(
                status="completed" if db_result.success else "failed",
                summary=db_result.summary or "",
                files_modified=db_result.files_modified or [],
                commands_executed=db_result.commands_executed or [],
                total_tokens=0,
                duration_seconds=db_result.duration_seconds or 0.0,
            )
    
    async def run_task(
        self,
        session_id: str,
        task_id: str,
        connection_manager: ConnectionManager | None = None,
    ) -> TaskResult:
        """Run a task in a session."""
        # Get agent
        agent = self._agents.get(session_id)
        if not agent:
            # Try to recreate agent from database
            async with self._session_maker() as db:
                repo = SessionRepository(db)
                db_session = await repo.get(session_id)
                
                if not db_session:
                    raise ValueError(f"Session {session_id} not found")
                
                from ..core.llm_client import create_llm_client
                llm_client = create_llm_client(model=db_session.model)
                
                agent = Agent(
                    llm_client=llm_client,
                    working_directory=db_session.working_directory,
                    max_iterations=db_session.max_iterations,
                )
                
                self._agents[session_id] = agent
                self._cancel_events[session_id] = asyncio.Event()
        
        # Get task
        async with self._session_maker() as db:
            task_repo = TaskRepository(db)
            session_repo = SessionRepository(db)
            
            db_task = await task_repo.get(task_id)
            if not db_task or db_task.session_id != session_id:
                raise ValueError(f"Task {task_id} not found")
            
            # Update status to running
            await task_repo.update_status(task_id, DBTaskStatus.RUNNING)
            await session_repo.update_status(session_id, DBSessionStatus.RUNNING)
            await db.commit()
        
        # Notify via WebSocket
        if connection_manager:
            await connection_manager.send_task_started(
                session_id, task_id, db_task.description
            )
        
        start_time = datetime.now(timezone.utc)
        cancel_event = self._cancel_events.get(session_id, asyncio.Event())
        
        # ── Auto Dependency Detection & Installation ──
        try:
            ws_dir = Path(db_task.description and agent.working_directory or ".")
            if agent.working_directory:
                ws_dir = Path(agent.working_directory)
            dep_cmds = []
            if (ws_dir / "requirements.txt").exists():
                dep_cmds.append("pip install -r requirements.txt")
            if (ws_dir / "pyproject.toml").exists() and not (ws_dir / "requirements.txt").exists():
                dep_cmds.append("pip install -e . 2>/dev/null || true")
            if (ws_dir / "package.json").exists():
                dep_cmds.append("npm install --prefer-offline 2>/dev/null || true")
            if dep_cmds and connection_manager:
                await connection_manager.broadcast_to_session(session_id, WebSocketMessage(
                    type=MessageType.TOOL_OUTPUT,
                    data={"line": "[Auto] Detecting and installing dependencies..."},
                    task_id=task_id,
                ))
                import asyncio as _asyncio
                for dep_cmd in dep_cmds:
                    proc = await _asyncio.create_subprocess_shell(
                        dep_cmd, cwd=str(ws_dir),
                        stdout=_asyncio.subprocess.PIPE,
                        stderr=_asyncio.subprocess.STDOUT,
                    )
                    stdout_bytes, _ = await proc.communicate()
                    out = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
                    for line in out.splitlines():
                        if connection_manager:
                            await connection_manager.broadcast_to_session(session_id, WebSocketMessage(
                                type=MessageType.TOOL_OUTPUT,
                                data={"line": line},
                                task_id=task_id,
                            ))
        except Exception as dep_err:
            print(f"[Auto-deps] error: {dep_err}")
        
        try:
            # Lock for serialized database updates in background tasks
            update_lock = asyncio.Lock()

            # Create callback for streaming updates
            async def on_update(update_type: str, data: dict[str, Any]) -> None:
                if not connection_manager:
                    return
                
                try:
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
                        # Also emit accumulated token usage
                        if agent and hasattr(agent, 'llm') and agent.llm:
                            usage = agent.llm.get_usage_stats()
                            await connection_manager.broadcast_to_session(session_id, WebSocketMessage(
                                type=MessageType.TOKEN_USAGE,
                                data=usage,
                                task_id=task_id,
                            ))
                    elif update_type == "tool_output":
                        # Shell live streaming — emit a token-style line for the Shell tab
                        line = data.get("line", "")
                        if line:
                            await connection_manager.broadcast_to_session(session_id, WebSocketMessage(
                                type=MessageType.TOOL_OUTPUT,
                                data={"line": line},
                                task_id=task_id,
                            ))
                    
                    # Update iteration and status in database with locking
                    # Skip for tokens to reduce database pressure during streaming
                    if update_type != "tokens":
                        iteration = agent.state.iteration if agent and hasattr(agent, 'state') else 0
                        async with update_lock:
                            async with self._session_maker() as db:
                                task_repo = TaskRepository(db)
                                session_repo = SessionRepository(db)
                                # Update status and iteration
                                await task_repo.update_status(task_id, DBTaskStatus.RUNNING, iteration=iteration)
                                await session_repo.update_status(session_id, DBSessionStatus.RUNNING, iteration=iteration)
                                await db.commit()
                except Exception as e:
                    print(f"Error in background update: {e}")
            
            # Create TaskState object for the agent
            task_state = TaskState(
                task_id=task_id,
                goal=TaskGoal(
                    description=db_task.description,
                    acceptance_criteria=db_task.acceptance_criteria or [],
                ),
                status=AgentTaskStatus.PENDING,
            )

            # Set up callbacks on the agent instance.
            # IMPORTANT: The callbacks are called from synchronous context inside the agent,
            # so we use asyncio.ensure_future() to schedule the async on_update coroutine.
            def _fire(coro):
                """Schedule an async coroutine from a sync lambda context."""
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.ensure_future(coro)
                    else:
                        loop.run_until_complete(coro)
                except Exception:
                    pass

            agent.callbacks = {
                "on_message": lambda token, is_token=False: _fire(on_update("tokens", {"content": token})),
                "on_tool_start": lambda name, args: _fire(on_update("tool_started", {"tool": name, "input": args})),
                "on_tool_result": lambda name, args, output, duration: _fire(on_update("tool_completed", {"tool": name, "output": output, "duration_ms": duration})),
                "on_phase_change": lambda phase: _fire(on_update("phase_changed", {"phase": phase})),
                # on_clarification_needed: emit WS event so UI can show modal
                "on_clarification_needed": lambda question: asyncio.ensure_future(
                    connection_manager.broadcast_to_session(session_id, WebSocketMessage(
                        type=MessageType.CLARIFICATION_NEEDED,
                        data={"question": question},
                        task_id=task_id,
                    ))
                ) if connection_manager else None,
                # Plan events
                "on_plan_created": lambda steps: _fire(on_update("plan_created", {"steps": steps})),
                "on_step_started": lambda idx, text="": _fire(on_update("step_started", {"index": idx, "text": text})),
                "on_step_completed": lambda idx, text="": _fire(on_update("step_completed", {"index": idx, "text": text})),
                "on_iteration": lambda iteration, max_iter: _fire(on_update("iteration", {"iteration": iteration, "max": max_iter})),
                # Shell live streaming: each stdout line becomes a tool_output event
                "on_command_output": lambda line: _fire(on_update("tool_output", {"line": line})),
            }

            # Run the agent — track asyncio.Task for cancellation
            coro = agent.run(task_state)
            running = asyncio.ensure_future(coro)
            self._running_tasks[session_id] = running
            final_task_state = await running
            
            # Check success: only COMPLETED = success, anything else = not success
            success = final_task_state.status == AgentTaskStatus.COMPLETED
            
            # Get summary from last assistant message
            summary = ""
            if agent.llm and hasattr(agent.llm, 'conversation'):
                for msg in reversed(agent.llm.conversation):
                    if msg.role == "assistant" and msg.content:
                        summary = msg.content
                        break
            
            # Create a simple result object that matches the code's expectations
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
                # Only treat last_error as an error if the task actually failed
                error_message=final_task_state.last_error if not success else None
            )
            
            # Calculate duration
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            # Save result to database
            async with self._session_maker() as db:
                task_repo = TaskRepository(db)
                session_repo = SessionRepository(db)
                artifact_repo = ArtifactRepository(db)
                
                # Update task status
                status = DBTaskStatus.COMPLETED if result.success else DBTaskStatus.FAILED
                error_msg = None if result.success else result.error_message
                await task_repo.update_status(task_id, status, error_message=error_msg)
                
                # Create result record
                await task_repo.create_result(
                    task_id=task_id,
                    success=result.success,
                    summary=result.summary,
                    files_modified=result.files_modified or [],
                    commands_executed=result.commands_executed or [],
                    verification_passed=getattr(result, 'verification_passed', None),
                    total_iterations=getattr(result, 'total_iterations', 0),
                    total_tool_calls=getattr(result, 'total_tool_calls', 0),
                    duration_seconds=duration,
                )
                
                # Update session status
                await session_repo.update_status(session_id, DBSessionStatus.IDLE)
                
                # Save artifacts metadata
                artifacts_dir = self.artifacts_base_dir / session_id / task_id
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                
                await self._save_artifacts_to_db(
                    artifact_repo, task_id, artifacts_dir, result
                )
                
                await db.commit()
            
            # Create task result
            task_result = TaskResult(
                status="completed" if result.success else "failed",
                summary=result.summary or "",
                files_modified=result.files_modified or [],
                commands_executed=result.commands_executed or [],
                total_tokens=result.total_tokens or 0,
                duration_seconds=duration,
            )
            
            # Notify via WebSocket
            if connection_manager:
                if result.success:
                    await connection_manager.send_task_completed(
                        session_id, task_id, task_result.summary
                    )
                else:
                    await connection_manager.send_task_failed(
                        session_id, task_id, result.error_message or "Unknown error"
                    )
            
            return task_result
            
        except asyncio.CancelledError:
            async with self._session_maker() as db:
                task_repo = TaskRepository(db)
                session_repo = SessionRepository(db)
                await task_repo.update_status(task_id, DBTaskStatus.CANCELLED, error_message="Task cancelled")
                await session_repo.update_status(session_id, DBSessionStatus.IDLE)
                await db.commit()
            
            return TaskResult(
                status="cancelled",
                summary="Task was cancelled",
                duration_seconds=(datetime.now(timezone.utc) - start_time).total_seconds(),
            )
            
        except Exception as e:
            async with self._session_maker() as db:
                task_repo = TaskRepository(db)
                session_repo = SessionRepository(db)
                await task_repo.update_status(task_id, DBTaskStatus.FAILED, error_message=str(e))
                await session_repo.update_status(session_id, DBSessionStatus.IDLE)
                await db.commit()
            
            if connection_manager:
                await connection_manager.send_task_failed(
                    session_id, task_id, str(e)
                )
            
            return TaskResult(
                status="failed",
                summary=f"Task failed: {e}",
                duration_seconds=(datetime.now(timezone.utc) - start_time).total_seconds(),
            )
    
    async def cancel_task(self, session_id: str, task_id: str) -> bool:
        """Cancel a running task."""
        if session_id not in self._cancel_events:
            return False
        
        async with self._session_maker() as db:
            repo = TaskRepository(db)
            db_task = await repo.get(task_id)
            
            if not db_task or db_task.status != DBTaskStatus.RUNNING:
                return False
        
        # Signal cancellation
        self._cancel_events[session_id].set()
        return True
    
    async def _save_artifacts_to_db(
        self,
        repo: ArtifactRepository,
        task_id: str,
        artifacts_dir: Path,
        result: Any,
    ) -> None:
        """Save task artifacts to disk and database."""
        import json
        
        # Save plan
        if hasattr(result, 'plan') and result.plan:
            plan_file = artifacts_dir / "plan.json"
            content = json.dumps(result.plan, indent=2, default=str)
            plan_file.write_text(content)
            await repo.create(
                task_id=task_id,
                name="plan.json",
                artifact_type="json",
                file_path=str(plan_file),
                size_bytes=len(content),
            )
        
        # Save tool calls
        if hasattr(result, 'tool_calls') and result.tool_calls:
            calls_file = artifacts_dir / "tool_calls.json"
            content = json.dumps(result.tool_calls, indent=2, default=str)
            calls_file.write_text(content)
            await repo.create(
                task_id=task_id,
                name="tool_calls.json",
                artifact_type="json",
                file_path=str(calls_file),
                size_bytes=len(content),
            )
        
        # Save verification results
        if hasattr(result, 'verification_results') and result.verification_results:
            verify_file = artifacts_dir / "verification_results.json"
            content = json.dumps(result.verification_results, indent=2, default=str)
            verify_file.write_text(content)
            await repo.create(
                task_id=task_id,
                name="verification_results.json",
                artifact_type="json",
                file_path=str(verify_file),
                size_bytes=len(content),
            )
        
        # Save diff
        if hasattr(result, 'diff') and result.diff:
            diff_file = artifacts_dir / "diff.patch"
            diff_file.write_text(result.diff)
            await repo.create(
                task_id=task_id,
                name="diff.patch",
                artifact_type="patch",
                file_path=str(diff_file),
                size_bytes=len(result.diff),
            )
        
        # Save summary
        summary_content = f"""# Task Summary

**Task ID:** {task_id}
**Status:** {"completed" if result.success else "failed"}

## Result

{result.summary or 'No summary'}

## Files Modified

{chr(10).join(f'- {f}' for f in (result.files_modified or []))}

## Commands Executed

{chr(10).join(f'- {c}' for c in (result.commands_executed or []))}
"""
        summary_file = artifacts_dir / "final_summary.md"
        summary_file.write_text(summary_content)
        await repo.create(
            task_id=task_id,
            name="final_summary.md",
            artifact_type="markdown",
            file_path=str(summary_file),
            size_bytes=len(summary_content),
        )
    
    async def list_directory(self, session_id: str, directory: str = ".") -> list[dict]:
        """List files and directories in the session's working directory."""
        async with self._session_maker() as db:
            repo = SessionRepository(db)
            db_session = await repo.get(session_id)
            
            if not db_session:
                raise ValueError(f"Session {session_id} not found")
            
            base_path = Path(db_session.working_directory).resolve()
            target_path = (base_path / directory).resolve()
            
            # Security check: ensure target_path is within base_path
            if not str(target_path).startswith(str(base_path)):
                raise ValueError("Cannot access directories outside of workspace")
            
            if not target_path.exists():
                return []
            
            nodes = []
            for item in target_path.iterdir():
                # Skip hidden files and common exclusions
                if item.name.startswith('.') and item.name != '.env':
                    continue
                if item.name == '__pycache__' or item.name == 'node_modules':
                    continue
                
                rel_path = str(item.relative_to(base_path))
                nodes.append({
                    "name": item.name,
                    "path": rel_path,
                    "isDir": item.is_dir(),
                    "children": [] if item.is_dir() else None
                })
            
            # Sort: directories first, then by name
            nodes.sort(key=lambda x: (not x["isDir"], x["name"].lower()))
            return nodes

    async def list_artifacts(self, session_id: str, task_id: str) -> list[str] | None:
        """List artifacts for a task."""
        async with self._session_maker() as db:
            repo = ArtifactRepository(db)
            artifacts = await repo.list_by_task(task_id)
            return [a.name for a in artifacts]
    
    async def get_artifact(self, session_id: str, task_id: str, artifact_name: str) -> str | None:
        """Get artifact content."""
        async with self._session_maker() as db:
            repo = ArtifactRepository(db)
            artifact = await repo.get_by_name(task_id, artifact_name)
            
            if not artifact:
                return None
            
            if artifact.content:
                return artifact.content
            
            if artifact.file_path:
                path = Path(artifact.file_path)
                if path.exists():
                    return path.read_text()
            
            return None
    
    async def get_active_session_count(self) -> int:
        """Get number of active sessions."""
        async with self._session_maker() as db:
            repo = SessionRepository(db)
            return await repo.count_active()
    
    async def get_total_tasks_completed(self) -> int:
        """Get total number of completed tasks."""
        async with self._session_maker() as db:
            repo = TaskRepository(db)
            return await repo.count_completed()
    
    def get_uptime_seconds(self) -> float:
        """Get uptime in seconds."""
        return (datetime.now(timezone.utc) - self._start_time).total_seconds()
    
    async def shutdown(self) -> None:
        """Shutdown all sessions."""
        for session_id in list(self._agents.keys()):
            await self.delete_session(session_id)
    
    def _db_to_session(self, db_session: SessionModel) -> Session:
        """Convert database model to Session object."""
        status_map = {
            DBSessionStatus.IDLE: SessionStatus.IDLE,
            DBSessionStatus.RUNNING: SessionStatus.RUNNING,
            DBSessionStatus.ERROR: SessionStatus.ERROR,
            DBSessionStatus.TERMINATED: SessionStatus.STOPPED,
        }
        
        session = Session(
            session_id=db_session.id,
            working_directory=db_session.working_directory,
            model=db_session.model,
            max_iterations=db_session.max_iterations,
            status=status_map.get(db_session.status, SessionStatus.IDLE),
            created_at=db_session.created_at,
            iteration=db_session.iteration,
            total_tasks=len(db_session.tasks) if db_session.tasks else 0,
            agent=self._agents.get(db_session.id),
            cancel_event=self._cancel_events.get(db_session.id),
        )
        
        # Add tasks
        try:
            # Check if tasks relationship is loaded to avoid MissingGreenlet
            from sqlalchemy import inspect
            insp = inspect(db_session)
            if "tasks" not in insp.unloaded:
                # Use getattr with None default for extra safety
                tasks_list = getattr(db_session, "tasks", [])
                if tasks_list:
                    for db_task in tasks_list:
                        artifacts_dir = self.artifacts_base_dir / db_session.id / db_task.id
                        session.tasks[db_task.id] = self._db_to_task(db_task, artifacts_dir)
        except Exception as e:
            print(f"Warning: Failed to load tasks for session {db_session.id}: {e}")
        
        return session
    
    def _db_to_task(self, db_task: TaskModel, artifacts_dir: Path) -> Task:
        """Convert database model to Task object."""
        status_map = {
            DBTaskStatus.PENDING: TaskStatus.PENDING,
            DBTaskStatus.RUNNING: TaskStatus.RUNNING,
            DBTaskStatus.COMPLETED: TaskStatus.COMPLETED,
            DBTaskStatus.FAILED: TaskStatus.FAILED,
            DBTaskStatus.CANCELLED: TaskStatus.CANCELLED,
        }
        
        task = Task(
            task_id=db_task.id,
            description=db_task.description,
            acceptance_criteria=db_task.acceptance_criteria or [],
            status=status_map.get(db_task.status, TaskStatus.PENDING),
            created_at=db_task.created_at,
            started_at=db_task.started_at,
            completed_at=db_task.completed_at,
            iteration=db_task.iteration,
            error_message=db_task.error_message,
            artifacts_dir=artifacts_dir,
        )
        
        # Add result if exists
        try:
            # Check if result relationship is loaded to avoid MissingGreenlet
            from sqlalchemy import inspect
            insp = inspect(db_task)
            if "result" not in insp.unloaded:
                res = getattr(db_task, "result", None)
                if res:
                    task.result = TaskResult(
                        status="completed" if getattr(res, "success", False) else "failed",
                        summary=getattr(res, "summary", "") or "",
                        files_modified=getattr(res, "files_modified", []) or [],
                        commands_executed=getattr(res, "commands_executed", []) or [],
                        total_tokens=0,
                        duration_seconds=getattr(res, "duration_seconds", 0.0) or 0.0,
                    )
        except Exception as e:
            print(f"Warning: Failed to load result for task {db_task.id}: {e}")
        
        return task
