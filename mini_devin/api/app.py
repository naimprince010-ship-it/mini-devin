"""
Lightweight Mini-Devin API for production deployment.

This is a minimal version that doesn't load heavy dependencies
to work within free tier memory constraints.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any, List
import uuid
import json
from datetime import datetime, timezone

import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

from .websocket import ConnectionManager, WebSocketMessage, MessageType
from ..database.config import init_db
from ..sessions.db_manager import DatabaseSessionManager

# Database-backed session management
session_manager = DatabaseSessionManager()
connection_manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    print(f"[API] Starting Mini-Devin API at {datetime.now(timezone.utc).isoformat()}")
    print(f"[API] Environment: PORT={os.getenv('PORT')}, DATABASE_URL={'Set' if os.getenv('DATABASE_URL') else 'Default SLite'}")
    
    try:
        print("[API] Initializing Database...")
        await init_db()
        print("[API] Database initialized successfully.")
    except Exception as e:
        print(f"[API] Error initializing database: {e}")
        import traceback
        traceback.print_exc()
        # We don't exit here to allow health checks to potentially still pass if DB isn't critical
    
    yield
    print(f"[API] Shutting down Mini-Devin API at {datetime.now(timezone.utc).isoformat()}")


app = FastAPI(
    title="Mini-Devin API",
    version="1.0.0",
    description="Autonomous AI Software Engineer Agent API (Lightweight Mode)",
    lifespan=lifespan,
)


def create_app() -> FastAPI:
    """Factory function to create the app instance."""
    return app

# Configure CORS - allow all origins for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "name": "Mini-Devin API",
        "version": "1.0.0",
        "status": "running",
        "mode": "lightweight",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/api/health")
async def api_health():
    return {"status": "healthy", "mode": "lightweight"}


@app.get("/api/browse")
async def browse_directory(path: str = "."):
    """Browse server filesystem directories for the folder picker UI."""
    import pathlib
    try:
        target = pathlib.Path(path).resolve()
        if not target.exists() or not target.is_dir():
            raise HTTPException(status_code=404, detail="Directory not found")

        entries = []
        # Add parent navigation (go up one level)
        parent = str(target.parent) if target != target.parent else None

        for entry in sorted(target.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower())):
            if entry.name.startswith('.') and entry.name not in ('.env',):
                continue
            if entry.name in ('__pycache__', 'node_modules', '.git'):
                continue
            entries.append({
                "name": entry.name,
                "path": str(entry),
                "is_directory": entry.is_dir(),
            })

        return {
            "current": str(target),
            "parent": parent,
            "entries": entries,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions")
@app.get("/sessions")
async def list_sessions():
    sessions = await session_manager.list_sessions()
    return [
        {
            "session_id": s.session_id,
            "created_at": s.created_at.isoformat(),
            "status": s.status.value,
            "working_directory": s.working_directory,
            "current_task": s.current_task_id,
            "iteration": s.iteration,
            "total_tasks": s.total_tasks,
            "title": getattr(s, 'title', ''),
        }
        for s in sessions
    ]

class CreateSessionRequest(BaseModel):
    working_directory: str = "."
    model: str = "gpt-4o"
    max_iterations: int = 50
    auto_git_commit: bool = False
    git_push: bool = False

@app.post("/api/sessions")
@app.post("/sessions")
async def create_session(raw_request: Request):
    working_dir = "."
    model = "gpt-4o"
    max_iterations = 50
    auto_git_commit = False
    git_push = False
    try:
        body = await raw_request.json()
        working_dir = body.get("working_directory", ".") or "."
        model = body.get("model", "gpt-4o") or "gpt-4o"
        max_iterations = int(body.get("max_iterations", 50) or 50)
        auto_git_commit = bool(body.get("auto_git_commit", False))
        git_push = bool(body.get("git_push", False))
    except Exception:
        pass
    session = await session_manager.create_session(
        working_directory=working_dir,
        model=model,
        max_iterations=max_iterations,
        auto_git_commit=auto_git_commit,
        git_push=git_push,
    )
    return {
        "session_id": session.session_id,
        "created_at": session.created_at.isoformat(),
        "status": session.status.value,
        "working_directory": session.working_directory,
        "model": getattr(session, 'model', model),
        "auto_git_commit": auto_git_commit,
        "git_push": git_push,
        "iteration": session.iteration,
        "total_tasks": session.total_tasks
    }

@app.get("/api/sessions/{session_id}")
@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    s = await session_manager.get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": s.session_id,
        "created_at": s.created_at.isoformat(),
        "status": s.status.value,
        "working_directory": s.working_directory,
        "current_task": s.current_task_id,
        "iteration": s.iteration,
        "total_tasks": s.total_tasks,
        "title": getattr(s, 'title', ''),
    }

@app.delete("/api/sessions/{session_id}")
@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    success = await session_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}

@app.post("/api/sessions/{session_id}/answer")
@app.post("/sessions/{session_id}/answer")
async def answer_clarification(session_id: str, request: Request):
    """Provide a user answer to a clarification question from the agent."""
    body = await request.json()
    answer = body.get("answer", "")
    if not answer:
        raise HTTPException(status_code=400, detail="Missing 'answer' field")
    ok = await session_manager.answer_clarification(session_id, answer)
    if not ok:
        raise HTTPException(status_code=404, detail="No pending clarification for this session")
    return {"status": "answered", "session_id": session_id}


@app.post("/api/sessions/{session_id}/sandbox/start")
@app.post("/sessions/{session_id}/sandbox/start")
async def start_sandbox(session_id: str):
    """Start a Docker sandbox for the given session."""
    result = await session_manager.start_sandbox(session_id)
    if not result.get("started"):
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to start sandbox"))
    # Broadcast sandbox_started event over WebSocket
    await connection_manager.broadcast_to_session(session_id, WebSocketMessage(
        type=MessageType.STATUS,
        data={"event": "sandbox_started", "container_id": result.get("container_id"), "status": result.get("status")},
    ))
    return result


@app.post("/api/sessions/{session_id}/sandbox/stop")
@app.post("/sessions/{session_id}/sandbox/stop")
async def stop_sandbox(session_id: str):
    """Stop the Docker sandbox for the given session."""
    result = await session_manager.stop_sandbox(session_id)
    if not result.get("stopped"):
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to stop sandbox"))
    # Broadcast sandbox_stopped event over WebSocket
    await connection_manager.broadcast_to_session(session_id, WebSocketMessage(
        type=MessageType.STATUS,
        data={"event": "sandbox_stopped"},
    ))
    return result



@app.post("/api/sandbox/build")
async def build_sandbox_image_endpoint():
    """Trigger a build of the mini-devin-sandbox Docker image (runs in background)."""
    async def _do_build():
        try:
            from ..sandbox.docker_sandbox import build_sandbox_from_repo
            ok = await build_sandbox_from_repo()
            print(f"[API] Sandbox image build {'succeeded' if ok else 'failed'}")
        except Exception as e:
            print(f"[API] Sandbox image build error: {e}")

    asyncio.create_task(_do_build())
    return {
        "status": "building",
        "image": "mini-devin-sandbox:latest",
        "message": "Build started in background. Check server logs for progress.",
    }


@app.get("/api/sessions/{session_id}/ls")
@app.get("/sessions/{session_id}/ls")
async def list_workspace_files(session_id: str, directory: str = "."):
    """List directory contents for a session."""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    base_dir = session.working_directory or "."
    target_dir = os.path.join(base_dir, directory)
    
    try:
        abs_target = os.path.abspath(target_dir)
        if not os.path.exists(abs_target):
            raise HTTPException(status_code=404, detail=f"Directory not found: {directory}")
        
        entries = []
        for entry in os.scandir(abs_target):
            if entry.name.startswith("."):
                continue
                
            is_dir = entry.is_dir()
            stat = entry.stat()
            entries.append({
                "name": entry.name,
                "path": os.path.join(directory, entry.name),
                "is_directory": is_dir,
                "size": stat.st_size if not is_dir else 0,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
            
        entries.sort(key=lambda x: (not x["is_directory"], x["name"].lower()))
        return entries
        
    except Exception as e:
        print(f"Error listing directory {target_dir}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions/{session_id}/file")
@app.get("/sessions/{session_id}/file")
async def read_file_content(session_id: str, path: str):
    """Read a file from the session's workspace."""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    base_dir = os.path.abspath(session.working_directory or ".")
    target = os.path.abspath(os.path.join(base_dir, path))
    if not target.startswith(base_dir):
        raise HTTPException(status_code=403, detail="Access denied")
    if not os.path.exists(target) or os.path.isdir(target):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        with open(target, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        return {"path": path, "content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/sessions/{session_id}/file")
@app.put("/sessions/{session_id}/file")
async def write_file_content(session_id: str, req: Request):
    """Write content to a file in the session's workspace."""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    body = await req.json()
    path = body.get("path", "")
    content = body.get("content", "")
    
    base_dir = os.path.abspath(session.working_directory or ".")
    target = os.path.abspath(os.path.join(base_dir, path))
    if not target.startswith(base_dir):
        raise HTTPException(status_code=403, detail="Access denied")
    
    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, 'w', encoding='utf-8') as f:
            f.write(content)
        return {"path": path, "saved": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sessions/{session_id}/stop")
@app.post("/sessions/{session_id}/stop")
async def stop_session_app(session_id: str):
    """Stop the running task in a session."""
    success = await session_manager.stop_session(session_id)
    return {"stopped": True, "session_id": session_id}


@app.get("/api/sessions/{session_id}/history")
@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """Get chat conversation history for a session (agent messages)."""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    agent = session_manager._agents.get(session_id)
    if not agent:
        return {"messages": [], "session_id": session_id}
    # Return LLM conversation history, excluding system prompt
    messages = []
    for msg in agent.llm.conversation:
        if msg.role == "system":
            continue
        entry = {
            "role": msg.role,
            "content": msg.content or "",
        }
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            entry["tool_calls"] = [
                {"name": tc.get("function", {}).get("name", ""), "args": tc.get("function", {}).get("arguments", {})}
                for tc in msg.tool_calls
            ]
        messages.append(entry)
    return {"messages": messages, "session_id": session_id, "total": len(messages)}

@app.get("/api/sessions/{session_id}/tasks")
@app.get("/sessions/{session_id}/tasks")
async def list_tasks(session_id: str):
    tasks = await session_manager.list_tasks(session_id)
    return [
        {
            "task_id": t.task_id,
            "session_id": session_id,
            "description": t.description,
            "status": t.status.value,
            "iteration": t.iteration,
            "created_at": t.created_at.isoformat() if t.created_at else None,
            "started_at": t.started_at.isoformat() if t.started_at else None,
            "completed_at": t.completed_at.isoformat() if t.completed_at else None,
        }
        for t in tasks
    ]

@app.get("/api/sessions/{session_id}/tasks/{task_id}")
@app.get("/sessions/{session_id}/tasks/{task_id}")
async def get_task(session_id: str, task_id: str):
    t = await session_manager.get_task(session_id, task_id)
    if not t:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "task_id": t.task_id,
        "session_id": session_id,
        "description": t.goal.description,
        "status": t.status.value,
        "iteration": t.iteration,
        "last_error": t.last_error,
        "created_at": t.created_at.isoformat() if t.created_at else None,
        "started_at": t.started_at.isoformat() if t.started_at else None,
        "completed_at": t.completed_at.isoformat() if t.completed_at else None,
    }

@app.get("/api/sessions/{session_id}/tasks/{task_id}/result")
@app.get("/sessions/{session_id}/tasks/{task_id}/result")
async def get_task_result(session_id: str, task_id: str):
    result = await session_manager.get_task_result(session_id, task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    return {
        "success": result.success,
        "summary": result.summary,
        "files_modified": result.files_modified,
        "commands_executed": result.commands_executed,
        "total_tokens": result.total_tokens,
        "duration_ms": result.duration_ms if hasattr(result, "duration_ms") else 0,
    }

@app.get("/api/sessions/{session_id}/tasks/{task_id}/artifacts")
@app.get("/sessions/{session_id}/tasks/{task_id}/artifacts")
async def list_artifacts(session_id: str, task_id: str):
    artifacts = await session_manager.list_artifacts(session_id, task_id)
    return [
        {
            "name": a.name,
            "type": a.type,
            "file_path": a.file_path,
            "size": a.size,
            "created_at": a.created_at.isoformat() if a.created_at else None,
        }
        for a in artifacts
    ]

@app.get("/api/providers")
@app.get("/providers")
async def list_providers():
    """List available LLM providers."""
    # Simplified version for lightweight mode
    providers = []
    if os.getenv("OPENAI_API_KEY") or os.getenv("MiniDevin"):
        providers.append({
            "id": "openai",
            "name": "OpenAI",
            "models": ["gpt-4o", "gpt-4o-mini"]
        })
    if os.getenv("ANTHROPIC_API_KEY"):
        providers.append({
            "id": "anthropic",
            "name": "Anthropic",
            "models": ["claude-3-5-sonnet-latest"]
        })
    return {"providers": providers}

@app.get("/api/models")
@app.get("/models")
async def list_models():
    """List available LLM models."""
    models = []
    if os.getenv("OPENAI_API_KEY") or os.getenv("MiniDevin"):
        models.extend([
            {"id": "gpt-4o", "name": "GPT-4o", "provider": "openai"},
            {"id": "gpt-4o-mini", "name": "GPT-4o Mini", "provider": "openai"}
        ])
    if os.getenv("ANTHROPIC_API_KEY"):
        models.append({"id": "claude-3-5-sonnet-latest", "name": "Claude 3.5 Sonnet", "provider": "anthropic"})
    return {"models": models}

@app.get("/api/status")
@app.get("/status")
async def get_system_status():
    """Get system status with uptime and metrics."""
    return {
        "status": "running",
        "mode": "lightweight",
        "version": "1.0.0",
        "active_sessions": await session_manager.get_active_session_count(),
        "total_tasks_completed": await session_manager.get_total_tasks_completed(),
        "uptime_seconds": session_manager.get_uptime_seconds(),
        "llm_configured": bool(os.getenv("OPENAI_API_KEY") or os.getenv("MiniDevin") or os.getenv("ANTHROPIC_API_KEY")),
    }


@app.get("/api/skills")
async def list_skills():
    return {
        "skills": [
            {
                "id": "add-endpoint",
                "name": "Add API Endpoint",
                "description": "Add a new REST API endpoint to a FastAPI application",
                "tags": ["api", "fastapi", "backend"],
            },
            {
                "id": "add-test",
                "name": "Add Unit Test",
                "description": "Add a unit test for a function or class",
                "tags": ["testing", "pytest"],
            },
            {
                "id": "refactor-function",
                "name": "Refactor Function",
                "description": "Refactor a function to improve readability and maintainability",
                "tags": ["refactoring", "code-quality"],
            },
        ],
        "message": "Demo skills - full functionality requires local deployment with OPENAI_API_KEY",
    }


@app.websocket("/api/ws/{session_id}")
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await connection_manager.connect(websocket, session_id)
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received from {session_id}: {data}")
            
            # Check if session exists, create if not
            session = await session_manager.get_session(session_id)
            if not session:
                session = await session_manager.create_session(session_id=session_id)
                session_id = session.session_id
            
            # If agent is currently running -> inject as follow-up
            if session.status.value == 'running':
                await session_manager.inject_followup(session_id, data)
                # Acknowledge to UI
                await connection_manager.broadcast_to_session(session_id, WebSocketMessage(
                    type=MessageType.TOKEN,
                    data={"content": f"\n\n**[Follow-up received]** {data}\n\n"},
                    task_id=session.current_task_id,
                ))
                continue
            
            # Create a new task
            task = await session_manager.create_task(
                session_id=session_id,
                description=data,
                connection_manager=connection_manager,
            )
            
            asyncio.create_task(session_manager.run_task(
                session_id=session_id,
                task_id=task.task_id,
                connection_manager=connection_manager
            ))

    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        connection_manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
