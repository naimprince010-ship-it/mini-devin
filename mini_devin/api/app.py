"""
Lightweight Mini-Devin API for production deployment.

This is a minimal version that doesn't load heavy dependencies
to work within free tier memory constraints.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any, List
import uuid
import json
from datetime import datetime

import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
    print("Starting Mini-Devin API (persistent mode)...")
    try:
        await init_db()
        print("Database initialized successfully.")
    except Exception as e:
        print(f"Error initializing database: {e}")
    yield
    print("Shutting down Mini-Devin API...")


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
            "total_tasks": s.total_tasks
        }
        for s in sessions
    ]

@app.post("/api/sessions")
@app.post("/sessions")
async def create_session():
    session = await session_manager.create_session()
    return {
        "session_id": session.session_id,
        "created_at": session.created_at.isoformat(),
        "status": session.status.value,
        "working_directory": session.working_directory,
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
        "total_tasks": s.total_tasks
    }

@app.delete("/api/sessions/{session_id}")
@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    success = await session_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}

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
        # Prevent concurrent agents overlapping and interleaving tokens
        # The session_manager or global lock should handle this.
        # For simplicity, we track it via DatabaseSessionManager's status.
        
        while True:
            data = await websocket.receive_text()
            print(f"Received from {session_id}: {data}")
            
            # Check if session exists, create if not (for UI simplicity)
            session = await session_manager.get_session(session_id)
            if not session:
                session = await session_manager.create_session() # Use default settings
                session_id = session.session_id
            
            # Create a task in the session
            task = await session_manager.create_task(
                session_id=session_id,
                description=data
            )
            
            # Run the task through the session manager
            # This handles persistence and agent execution in a separate task
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
