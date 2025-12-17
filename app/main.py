"""
Lightweight Mini-Devin API for production deployment.

This is a minimal version that doesn't load heavy dependencies
to work within free tier memory constraints.
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    print("Starting Mini-Devin API (lightweight mode)...")
    yield
    print("Shutting down Mini-Devin API...")


app = FastAPI(
    title="Mini-Devin API",
    version="1.0.0",
    description="Autonomous AI Software Engineer Agent API (Lightweight Mode)",
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
async def list_sessions():
    return {"sessions": [], "message": "Running in lightweight mode - full functionality requires local deployment"}


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


@app.post("/api/sessions")
async def create_session():
    return {
        "error": "Session creation requires full deployment with database",
        "message": "Please run Mini-Devin locally for full functionality",
        "docs": "https://github.com/naimprince010-ship-it/mini-devin#readme",
    }


@app.get("/api/skills/tags")
async def list_skill_tags():
    return {
        "tags": ["api", "fastapi", "backend", "testing", "pytest", "refactoring", "code-quality"],
    }


@app.get("/api/status")
async def get_status():
    return {
        "status": "running",
        "mode": "lightweight",
        "version": "1.0.0",
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
