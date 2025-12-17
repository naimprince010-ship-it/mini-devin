"""
API Routes for Mini-Devin

This module provides REST API endpoints for:
- Session management (create, list, get, delete)
- Task management (create, run, status, cancel)
- Artifact retrieval
- System status
"""


import json
from pathlib import Path

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


# Skills Endpoints

class CreateSkillRequest(BaseModel):
    """Request to create a custom skill."""
    name: str = Field(..., description="Unique skill name")
    description: str = Field(..., description="Skill description")
    steps: list[dict] = Field(..., description="List of skill steps")
    parameters: list[dict] = Field(default_factory=list, description="Skill parameters")
    tags: list[str] = Field(default_factory=list, description="Skill tags")


class UpdateSkillRequest(BaseModel):
    """Request to update a custom skill."""
    description: str | None = Field(None, description="Skill description")
    steps: list[dict] | None = Field(None, description="List of skill steps")
    parameters: list[dict] | None = Field(None, description="Skill parameters")
    tags: list[str] | None = Field(None, description="Skill tags")


class SkillInfo(BaseModel):
    """Information about a skill."""
    name: str
    description: str
    version: str
    tags: list[str]
    required_tools: list[str]
    parameters: list[dict]
    is_custom: bool = False


class ExecuteSkillRequest(BaseModel):
    """Request to execute a skill."""
    parameters: dict = Field(default_factory=dict, description="Skill parameters")
    workspace_path: str = Field(default=".", description="Workspace path")
    dry_run: bool = Field(default=False, description="Preview changes without executing")


@router.get("/skills")
async def list_skills(
    tag: str | None = None,
    search: str | None = None,
    include_builtin: bool = True,
    include_custom: bool = True,
):
    """
    List available skills.
    
    Args:
        tag: Filter by tag
        search: Search by name or description
        include_builtin: Include built-in skills
        include_custom: Include custom user-created skills
    """
    from mini_devin.skills.registry import get_registry
    
    registry = get_registry()
    
    if tag:
        skill_names = registry.list_by_tag(tag)
    elif search:
        skill_names = registry.search(search)
    else:
        skill_names = registry.list_skills()
    
    skills = []
    custom_skills = _get_custom_skills()
    
    for name in skill_names:
        info = registry.get_skill_info(name)
        if info:
            if include_builtin:
                skills.append({**info, "is_custom": False})
    
    if include_custom:
        for skill in custom_skills.values():
            if tag and tag not in skill.get("tags", []):
                continue
            if search and search.lower() not in skill["name"].lower() and search.lower() not in skill.get("description", "").lower():
                continue
            skills.append({**skill, "is_custom": True})
    
    return {"skills": skills, "total": len(skills)}


@router.get("/skills/tags")
async def list_skill_tags():
    """List all available skill tags."""
    from mini_devin.skills.registry import get_registry
    
    registry = get_registry()
    builtin_tags = registry.list_tags()
    
    custom_skills = _get_custom_skills()
    custom_tags = set()
    for skill in custom_skills.values():
        custom_tags.update(skill.get("tags", []))
    
    all_tags = list(set(builtin_tags) | custom_tags)
    return {"tags": sorted(all_tags)}


@router.get("/skills/{skill_name}")
async def get_skill(skill_name: str):
    """Get detailed information about a skill."""
    from mini_devin.skills.registry import get_registry
    
    registry = get_registry()
    
    custom_skills = _get_custom_skills()
    if skill_name in custom_skills:
        return {**custom_skills[skill_name], "is_custom": True}
    
    info = registry.get_skill_info(skill_name)
    if not info:
        raise HTTPException(status_code=404, detail="Skill not found")
    
    return {**info, "is_custom": False}


@router.post("/skills", response_model=SkillInfo)
async def create_skill(request: CreateSkillRequest):
    """Create a new custom skill."""
    from mini_devin.skills.registry import get_registry
    
    registry = get_registry()
    
    if registry.get(request.name):
        raise HTTPException(status_code=400, detail="A built-in skill with this name already exists")
    
    custom_skills = _get_custom_skills()
    if request.name in custom_skills:
        raise HTTPException(status_code=400, detail="A custom skill with this name already exists")
    
    skill_data = {
        "name": request.name,
        "description": request.description,
        "version": "1.0.0",
        "steps": request.steps,
        "parameters": request.parameters,
        "tags": request.tags,
        "required_tools": _extract_required_tools(request.steps),
    }
    
    custom_skills[request.name] = skill_data
    _save_custom_skills(custom_skills)
    
    return SkillInfo(
        name=skill_data["name"],
        description=skill_data["description"],
        version=skill_data["version"],
        tags=skill_data["tags"],
        required_tools=skill_data["required_tools"],
        parameters=skill_data["parameters"],
        is_custom=True,
    )


@router.put("/skills/{skill_name}")
async def update_skill(skill_name: str, request: UpdateSkillRequest):
    """Update a custom skill."""
    custom_skills = _get_custom_skills()
    
    if skill_name not in custom_skills:
        from mini_devin.skills.registry import get_registry
        registry = get_registry()
        if registry.get(skill_name):
            raise HTTPException(status_code=400, detail="Cannot modify built-in skills")
        raise HTTPException(status_code=404, detail="Skill not found")
    
    skill = custom_skills[skill_name]
    
    if request.description is not None:
        skill["description"] = request.description
    if request.steps is not None:
        skill["steps"] = request.steps
        skill["required_tools"] = _extract_required_tools(request.steps)
    if request.parameters is not None:
        skill["parameters"] = request.parameters
    if request.tags is not None:
        skill["tags"] = request.tags
    
    version_parts = skill.get("version", "1.0.0").split(".")
    version_parts[-1] = str(int(version_parts[-1]) + 1)
    skill["version"] = ".".join(version_parts)
    
    _save_custom_skills(custom_skills)
    
    return {**skill, "is_custom": True}


@router.delete("/skills/{skill_name}")
async def delete_skill(skill_name: str):
    """Delete a custom skill."""
    custom_skills = _get_custom_skills()
    
    if skill_name not in custom_skills:
        from mini_devin.skills.registry import get_registry
        registry = get_registry()
        if registry.get(skill_name):
            raise HTTPException(status_code=400, detail="Cannot delete built-in skills")
        raise HTTPException(status_code=404, detail="Skill not found")
    
    del custom_skills[skill_name]
    _save_custom_skills(custom_skills)
    
    return {"status": "deleted", "skill_name": skill_name}


@router.post("/skills/{skill_name}/execute")
async def execute_skill(
    skill_name: str,
    request: ExecuteSkillRequest,
    req: Request,
    background_tasks: BackgroundTasks,
):
    """Execute a skill."""
    from mini_devin.skills.registry import get_registry
    from mini_devin.skills.base import SkillContext
    
    registry = get_registry()
    
    custom_skills = _get_custom_skills()
    if skill_name in custom_skills:
        execution_id = _execute_custom_skill(
            custom_skills[skill_name],
            request.parameters,
            request.workspace_path,
            request.dry_run,
        )
        return {
            "execution_id": execution_id,
            "skill_name": skill_name,
            "status": "started",
            "dry_run": request.dry_run,
        }
    
    skill = registry.get(skill_name)
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    
    context = SkillContext(
        workspace_path=request.workspace_path,
        dry_run=request.dry_run,
    )
    
    import uuid
    execution_id = str(uuid.uuid4())
    
    async def run_skill():
        try:
            result = await registry.execute(skill_name, context, **request.parameters)
            _store_execution_result(execution_id, result)
        except Exception as e:
            _store_execution_error(execution_id, str(e))
    
    background_tasks.add_task(run_skill)
    
    return {
        "execution_id": execution_id,
        "skill_name": skill_name,
        "status": "started",
        "dry_run": request.dry_run,
    }


@router.get("/skills/executions/{execution_id}")
async def get_skill_execution(execution_id: str):
    """Get the status and result of a skill execution."""
    result = _get_execution_result(execution_id)
    if not result:
        raise HTTPException(status_code=404, detail="Execution not found")
    return result


# Helper functions for custom skills storage

_CUSTOM_SKILLS_FILE = Path.home() / ".mini-devin" / "custom_skills.json"
_EXECUTIONS: dict = {}


def _get_custom_skills() -> dict:
    """Load custom skills from storage."""
    if not _CUSTOM_SKILLS_FILE.exists():
        return {}
    try:
        return json.loads(_CUSTOM_SKILLS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_custom_skills(skills: dict) -> None:
    """Save custom skills to storage."""
    _CUSTOM_SKILLS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _CUSTOM_SKILLS_FILE.write_text(json.dumps(skills, indent=2))


def _extract_required_tools(steps: list[dict]) -> list[str]:
    """Extract required tools from skill steps."""
    tools = set()
    for step in steps:
        tool = step.get("tool")
        if tool:
            tools.add(tool)
    return list(tools)


def _execute_custom_skill(
    skill: dict,
    parameters: dict,
    workspace_path: str,
    dry_run: bool,
) -> str:
    """Execute a custom skill and return execution ID."""
    import uuid
    execution_id = str(uuid.uuid4())
    
    _EXECUTIONS[execution_id] = {
        "execution_id": execution_id,
        "skill_name": skill["name"],
        "status": "running" if not dry_run else "dry_run",
        "steps": skill.get("steps", []),
        "parameters": parameters,
        "workspace_path": workspace_path,
        "dry_run": dry_run,
        "result": None,
        "error": None,
    }
    
    if dry_run:
        _EXECUTIONS[execution_id]["status"] = "completed"
        _EXECUTIONS[execution_id]["result"] = {
            "success": True,
            "message": "Dry run completed - no changes made",
            "steps_preview": skill.get("steps", []),
        }
    
    return execution_id


def _store_execution_result(execution_id: str, result) -> None:
    """Store execution result."""
    if execution_id in _EXECUTIONS:
        _EXECUTIONS[execution_id]["status"] = "completed" if result.success else "failed"
        _EXECUTIONS[execution_id]["result"] = {
            "success": result.success,
            "message": result.message,
            "outputs": result.outputs,
            "files_created": result.files_created,
            "files_modified": result.files_modified,
        }


def _store_execution_error(execution_id: str, error: str) -> None:
    """Store execution error."""
    if execution_id in _EXECUTIONS:
        _EXECUTIONS[execution_id]["status"] = "failed"
        _EXECUTIONS[execution_id]["error"] = error


def _get_execution_result(execution_id: str) -> dict | None:
    """Get execution result."""
    return _EXECUTIONS.get(execution_id)
