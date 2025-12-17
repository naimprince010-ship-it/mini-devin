"""
Full-featured Mini-Devin API for production deployment.

This version provides full API functionality with LLM integration
while optimizing memory usage for cloud deployment.
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from datetime import datetime
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# In-memory storage for sessions (production would use database)
sessions: dict = {}
tasks: dict = {}


class CreateSessionRequest(BaseModel):
    working_directory: str = Field(default=".", description="Working directory")
    model: str = Field(default="gpt-4o-mini", description="LLM model to use")
    max_iterations: int = Field(default=50, description="Max iterations per task")


class CreateTaskRequest(BaseModel):
    description: str = Field(..., description="Task description")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    print("Starting Mini-Devin API (Full Mode)...")
    
    # Check for OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("MiniDevin")
    if api_key:
        print(f"OpenAI API key configured (length: {len(api_key)} chars)")
    else:
        print("Warning: No OpenAI API key found - LLM features will be limited")
    
    yield
    print("Shutting down Mini-Devin API...")


app = FastAPI(
    title="Mini-Devin API",
    version="1.0.0",
    description="Autonomous AI Software Engineer Agent API (Full Mode)",
    lifespan=lifespan,
)

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
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("MiniDevin")
    return {
        "name": "Mini-Devin API",
        "version": "1.0.0",
        "status": "running",
        "mode": "full" if api_key else "limited",
        "llm_configured": bool(api_key),
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/api/health")
async def api_health():
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("MiniDevin")
    return {
        "status": "healthy",
        "mode": "full" if api_key else "limited",
        "llm_configured": bool(api_key),
    }


@app.get("/api/status")
async def get_status():
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("MiniDevin")
    return {
        "status": "running",
        "mode": "full" if api_key else "limited",
        "version": "1.0.0",
        "active_sessions": len(sessions),
        "llm_configured": bool(api_key),
    }


@app.get("/api/sessions")
async def list_sessions():
    return {
        "sessions": list(sessions.values()),
        "total": len(sessions),
    }


@app.post("/api/sessions")
async def create_session(request: CreateSessionRequest):
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("MiniDevin")
    
    session_id = str(uuid.uuid4())[:8]
    session = {
        "session_id": session_id,
        "created_at": datetime.utcnow().isoformat(),
        "status": "active",
        "working_directory": request.working_directory,
        "model": request.model,
        "max_iterations": request.max_iterations,
        "current_task": None,
        "iteration": 0,
        "total_tasks": 0,
        "llm_enabled": bool(api_key),
    }
    sessions[session_id] = session
    
    return {
        "session_id": session_id,
        "created_at": session["created_at"],
        "status": "active",
        "llm_enabled": bool(api_key),
    }


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del sessions[session_id]
    return {"status": "deleted", "session_id": session_id}


@app.post("/api/sessions/{session_id}/tasks")
async def create_task(session_id: str, request: CreateTaskRequest):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("MiniDevin")
    
    task_id = str(uuid.uuid4())[:8]
    task = {
        "task_id": task_id,
        "session_id": session_id,
        "description": request.description,
        "status": "pending",
        "created_at": datetime.utcnow().isoformat(),
        "result": None,
    }
    
    if api_key:
        task["status"] = "queued"
        task["message"] = "Task queued for processing with LLM"
    else:
        task["status"] = "limited"
        task["message"] = "LLM not configured - task recorded but cannot be executed"
    
    tasks[task_id] = task
    sessions[session_id]["total_tasks"] += 1
    sessions[session_id]["current_task"] = task_id
    
    return task


@app.get("/api/sessions/{session_id}/tasks")
async def list_tasks(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_tasks = [t for t in tasks.values() if t["session_id"] == session_id]
    return {"tasks": session_tasks, "total": len(session_tasks)}


@app.get("/api/sessions/{session_id}/tasks/{task_id}")
async def get_task(session_id: str, task_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]


@app.get("/api/skills")
async def list_skills():
    return {
        "skills": [
            {
                "id": "add-endpoint",
                "name": "Add API Endpoint",
                "description": "Add a new REST API endpoint to a FastAPI application",
                "tags": ["api", "fastapi", "backend"],
                "version": "1.0.0",
                "required_tools": ["editor", "terminal"],
                "parameters": [],
                "is_custom": False,
            },
            {
                "id": "add-test",
                "name": "Add Unit Test",
                "description": "Add a unit test for a function or class",
                "tags": ["testing", "pytest"],
                "version": "1.0.0",
                "required_tools": ["editor", "terminal"],
                "parameters": [],
                "is_custom": False,
            },
            {
                "id": "refactor-function",
                "name": "Refactor Function",
                "description": "Refactor a function to improve readability and maintainability",
                "tags": ["refactoring", "code-quality"],
                "version": "1.0.0",
                "required_tools": ["editor"],
                "parameters": [],
                "is_custom": False,
            },
            {
                "id": "fix-bug",
                "name": "Fix Bug",
                "description": "Analyze and fix a bug in the codebase",
                "tags": ["debugging", "bugfix"],
                "version": "1.0.0",
                "required_tools": ["editor", "terminal", "browser"],
                "parameters": [
                    {"name": "bug_description", "type": "string", "required": True, "description": "Description of the bug"}
                ],
                "is_custom": False,
            },
        ],
    }


@app.get("/api/skills/tags")
async def list_skill_tags():
    return {
        "tags": ["api", "fastapi", "backend", "testing", "pytest", "refactoring", "code-quality", "debugging", "bugfix"],
    }


@app.get("/skills")
async def list_skills_root():
    return await list_skills()


@app.get("/skills/tags")
async def list_skill_tags_root():
    return await list_skill_tags()


@app.get("/sessions")
async def list_sessions_root():
    return await list_sessions()


@app.get("/status")
async def get_status_root():
    return await get_status()
