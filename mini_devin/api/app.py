"""
FastAPI Application for Mini-Devin

This module provides the main FastAPI application with:
- REST API endpoints for task management
- WebSocket support for real-time streaming
- CORS middleware for web UI
- Health checks and status endpoints
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routes import router
from .websocket import ConnectionManager
from ..sessions.db_manager import DatabaseSessionManager
from ..database.config import init_db, close_db


# Global instances
session_manager: DatabaseSessionManager | None = None
connection_manager: ConnectionManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    global session_manager, connection_manager
    
    # Startup - Initialize database
    try:
        await init_db()
    except Exception as e:
        print(f"Warning: Could not initialize database: {e}")
        print("Running with in-memory storage only")
    
    # Create managers
    session_manager = DatabaseSessionManager()
    connection_manager = ConnectionManager()
    
    # Store in app state for access in routes
    app.state.session_manager = session_manager
    app.state.connection_manager = connection_manager
    
    yield
    
    # Shutdown
    if session_manager:
        await session_manager.shutdown()
    
    # Close database connection
    await close_db()


def create_app(
    title: str = "Mini-Devin API",
    version: str = "1.0.0",
    debug: bool = False,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        title: API title
        version: API version
        debug: Enable debug mode
        cors_origins: Allowed CORS origins
        
    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title=title,
        version=version,
        description="Autonomous AI Software Engineer Agent API",
        debug=debug,
        lifespan=lifespan,
    )
    
    # Configure CORS
    if cors_origins is None:
        cors_origins = [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
        ]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routes
    app.include_router(router, prefix="/api")
    
    # Serve static files if they exist
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "name": "Mini-Devin API",
            "version": version,
            "status": "running",
            "docs": "/docs",
        }
    
    # Health check
    @app.get("/health")
    async def health():
        return {"status": "healthy"}
    
    # WebSocket endpoint for real-time updates
    @app.websocket("/ws/{session_id}")
    async def websocket_endpoint(websocket: WebSocket, session_id: str):
        """WebSocket endpoint for real-time session updates."""
        connection_mgr = app.state.connection_manager
        
        await connection_mgr.connect(websocket, session_id)
        
        try:
            while True:
                # Keep connection alive and handle incoming messages
                data = await websocket.receive_text()
                # Echo back or handle commands if needed
                await websocket.send_text(f"Received: {data}")
        except WebSocketDisconnect:
            connection_mgr.disconnect(websocket)
    
    # WebSocket endpoint for global updates (no session filter)
    @app.websocket("/ws")
    async def websocket_global(websocket: WebSocket):
        """WebSocket endpoint for global updates."""
        connection_mgr = app.state.connection_manager
        
        await connection_mgr.connect(websocket)
        
        try:
            while True:
                data = await websocket.receive_text()
                await websocket.send_text(f"Received: {data}")
        except WebSocketDisconnect:
            connection_mgr.disconnect(websocket)
    
    return app


# Default app instance
app = create_app()
