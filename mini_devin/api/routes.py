"""
API Routes for Mini-Devin

This module provides REST API endpoints for:
- Session management (create, list, get, delete)
- Task management (create, run, status, cancel)
- Artifact retrieval
- System status
"""


from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field


router = APIRouter(tags=["api"])


# Request/Response Models

class CreateSessionRequest(BaseModel):
    """Request to create a new session."""
    working_directory: str = Field(default=".", description="Working directory for the session")
    model: str = Field(default="gpt-4o", description="LLM model to use")
    max_iterations: int = Field(default=50, description="Maximum iterations per task")


class CreateSessionResponse(BaseModel):
    """Response from creating a session."""
    session_id: str
    created_at: str
    status: str


class SessionInfo(BaseModel):
    """Information about a session."""
    session_id: str
    created_at: str
    status: str
    working_directory: str
    current_task: str | None
    iteration: int
    total_tasks: int


class CreateTaskRequest(BaseModel):
    """Request to create a new task."""
    description: str = Field(..., description="Task description")
    acceptance_criteria: list[str] = Field(default_factory=list, description="Acceptance criteria")


class CreateTaskResponse(BaseModel):
    """Response from creating a task."""
    task_id: str
    session_id: str
    status: str
    created_at: str


class TaskInfo(BaseModel):
    """Information about a task."""
    task_id: str
    session_id: str
    description: str
    status: str
    created_at: str
    started_at: str | None
    completed_at: str | None
    iteration: int
    error_message: str | None


class TaskResult(BaseModel):
    """Result of a completed task."""
    task_id: str
    status: str
    summary: str
    files_modified: list[str]
    commands_executed: list[str]
    total_tokens: int
    duration_seconds: float


class SystemStatus(BaseModel):
    """System status information."""
    status: str
    version: str
    active_sessions: int
    total_tasks_completed: int
    uptime_seconds: float


# Session Endpoints

@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session(request: CreateSessionRequest, req: Request):
    """Create a new agent session."""
    session_manager = req.app.state.session_manager
    
    session = await session_manager.create_session(
        working_directory=request.working_directory,
        model=request.model,
        max_iterations=request.max_iterations,
    )
    
    return CreateSessionResponse(
        session_id=session.session_id,
        created_at=session.created_at.isoformat(),
        status=session.status,
    )


@router.get("/sessions", response_model=list[SessionInfo])
async def list_sessions(req: Request):
    """List all active sessions."""
    session_manager = req.app.state.session_manager
    sessions = await session_manager.list_sessions()
    
    return [
        SessionInfo(
            session_id=s.session_id,
            created_at=s.created_at.isoformat(),
            status=s.status.value if hasattr(s.status, 'value') else str(s.status),
            working_directory=s.working_directory,
            current_task=getattr(s, 'current_task_id', None),
            iteration=s.iteration,
            total_tasks=s.total_tasks,
        )
        for s in sessions
    ]


@router.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str, req: Request):
    """Get information about a specific session."""
    session_manager = req.app.state.session_manager
    session = await session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SessionInfo(
        session_id=session.session_id,
        created_at=session.created_at.isoformat(),
        status=session.status.value if hasattr(session.status, 'value') else str(session.status),
        working_directory=session.working_directory,
        current_task=getattr(session, 'current_task_id', None),
        iteration=session.iteration,
        total_tasks=session.total_tasks,
    )


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, req: Request):
    """Delete a session."""
    session_manager = req.app.state.session_manager
    success = await session_manager.delete_session(session_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"status": "deleted", "session_id": session_id}


# Task Endpoints

@router.post("/sessions/{session_id}/tasks", response_model=CreateTaskResponse)
async def create_task(
    session_id: str,
    request: CreateTaskRequest,
    req: Request,
    background_tasks: BackgroundTasks,
):
    """Create and start a new task in a session."""
    session_manager = req.app.state.session_manager
    connection_manager = req.app.state.connection_manager
    
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Create the task
    task = await session_manager.create_task(
        session_id=session_id,
        description=request.description,
        acceptance_criteria=request.acceptance_criteria,
    )
    
    # Start the task in the background
    background_tasks.add_task(
        session_manager.run_task,
        session_id,
        task.task_id,
        connection_manager,
    )
    
    return CreateTaskResponse(
        task_id=task.task_id,
        session_id=session_id,
        status=task.status.value if hasattr(task.status, 'value') else str(task.status),
        created_at=task.created_at.isoformat(),
    )


@router.get("/sessions/{session_id}/tasks", response_model=list[TaskInfo])
async def list_tasks(session_id: str, req: Request):
    """List all tasks in a session."""
    session_manager = req.app.state.session_manager
    
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    tasks = await session_manager.list_tasks(session_id)
    
    return [
        TaskInfo(
            task_id=t.task_id,
            session_id=session_id,
            description=t.description,
            status=t.status.value if hasattr(t.status, 'value') else str(t.status),
            created_at=t.created_at.isoformat(),
            started_at=t.started_at.isoformat() if t.started_at else None,
            completed_at=t.completed_at.isoformat() if t.completed_at else None,
            iteration=t.iteration,
            error_message=t.error_message,
        )
        for t in tasks
    ]


@router.get("/sessions/{session_id}/tasks/{task_id}", response_model=TaskInfo)
async def get_task(session_id: str, task_id: str, req: Request):
    """Get information about a specific task."""
    session_manager = req.app.state.session_manager
    
    task = await session_manager.get_task(session_id, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return TaskInfo(
        task_id=task.task_id,
        session_id=session_id,
        description=task.description,
        status=task.status.value if hasattr(task.status, 'value') else str(task.status),
        created_at=task.created_at.isoformat(),
        started_at=task.started_at.isoformat() if task.started_at else None,
        completed_at=task.completed_at.isoformat() if task.completed_at else None,
        iteration=task.iteration,
        error_message=task.error_message,
    )


@router.get("/sessions/{session_id}/tasks/{task_id}/result", response_model=TaskResult)
async def get_task_result(session_id: str, task_id: str, req: Request):
    """Get the result of a completed task."""
    session_manager = req.app.state.session_manager
    
    result = await session_manager.get_task_result(session_id, task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Task result not found")
    
    return TaskResult(
        task_id=task_id,
        status=result.status,
        summary=result.summary,
        files_modified=result.files_modified,
        commands_executed=result.commands_executed,
        total_tokens=result.total_tokens,
        duration_seconds=result.duration_seconds,
    )


@router.post("/sessions/{session_id}/tasks/{task_id}/cancel")
async def cancel_task(session_id: str, task_id: str, req: Request):
    """Cancel a running task."""
    session_manager = req.app.state.session_manager
    
    success = await session_manager.cancel_task(session_id, task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found or not running")
    
    return {"status": "cancelled", "task_id": task_id}


# Artifact Endpoints

@router.get("/sessions/{session_id}/tasks/{task_id}/artifacts")
async def list_artifacts(session_id: str, task_id: str, req: Request):
    """List artifacts for a task."""
    session_manager = req.app.state.session_manager
    
    artifacts = await session_manager.list_artifacts(session_id, task_id)
    if artifacts is None:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return {"artifacts": artifacts}


@router.get("/sessions/{session_id}/tasks/{task_id}/artifacts/{artifact_name}")
async def get_artifact(session_id: str, task_id: str, artifact_name: str, req: Request):
    """Get a specific artifact."""
    session_manager = req.app.state.session_manager
    
    content = await session_manager.get_artifact(session_id, task_id, artifact_name)
    if content is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    
    return {"name": artifact_name, "content": content}


# System Endpoints

@router.get("/status", response_model=SystemStatus)
async def get_status(req: Request):
    """Get system status."""
    session_manager = req.app.state.session_manager
    
    active_sessions = await session_manager.get_active_session_count()
    total_completed = await session_manager.get_total_tasks_completed()
    
    return SystemStatus(
        status="running",
        version="1.0.0",
        active_sessions=active_sessions,
        total_tasks_completed=total_completed,
        uptime_seconds=session_manager.get_uptime_seconds(),
    )


@router.get("/models")
async def list_models(
    provider: str | None = None,
    supports_tools: bool | None = None,
    only_configured: bool = False,
):
    """
    List available LLM models.
    
    Args:
        provider: Filter by provider (openai, anthropic, ollama, azure)
        supports_tools: Filter by tool support capability
        only_configured: Only return models from configured providers
    """
    from mini_devin.core.providers import get_model_registry, Provider
    
    registry = get_model_registry()
    
    provider_enum = None
    if provider:
        try:
            provider_enum = Provider(provider)
        except ValueError:
            pass
    
    models = registry.list_models(
        provider=provider_enum,
        supports_tools=supports_tools,
        only_configured=only_configured,
    )
    
    return {
        "models": [
            {
                "id": m.id,
                "name": m.name,
                "provider": m.provider.value,
                "context_window": m.context_window,
                "supports_tools": m.supports_tools,
                "supports_vision": m.supports_vision,
                "max_output_tokens": m.max_output_tokens,
                "description": m.description,
            }
            for m in models
        ]
    }


@router.get("/providers")
async def list_providers():
    """List all providers and their configuration status."""
    from mini_devin.core.providers import get_model_registry, Provider
    
    registry = get_model_registry()
    
    providers = []
    for p in Provider:
        config = registry.get_provider_config(p)
        providers.append({
            "id": p.value,
            "name": p.name,
            "configured": registry.is_provider_configured(p),
            "enabled": config.enabled if config else False,
        })
    
    return {
        "providers": providers,
        "default_model": registry.get_default_model(),
    }
