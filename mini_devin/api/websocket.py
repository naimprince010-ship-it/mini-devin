"""
WebSocket Manager for Mini-Devin

This module provides WebSocket support for:
- Real-time token streaming
- Live tool execution logs
- Task status updates
- Error notifications
"""

import asyncio
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from dataclasses import dataclass, field

from fastapi import WebSocket


class MessageType(str, Enum):
    """Types of WebSocket messages."""
    # Connection
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    
    # Task lifecycle
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"
    
    # Execution
    PHASE_CHANGED = "phase_changed"
    ITERATION_STARTED = "iteration_started"
    ITERATION_COMPLETED = "iteration_completed"
    
    # Streaming
    TOKEN = "token"
    TOKENS_BATCH = "tokens_batch"
    
    # Tool execution
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    TOOL_FAILED = "tool_failed"
    TOOL_OUTPUT = "tool_output"
    
    # Plan updates
    PLAN_CREATED = "plan_created"
    PLAN_UPDATED = "plan_updated"
    STEP_COMPLETED = "step_completed"
    
    # Verification
    VERIFICATION_STARTED = "verification_started"
    VERIFICATION_COMPLETED = "verification_completed"
    REPAIR_STARTED = "repair_started"
    REPAIR_COMPLETED = "repair_completed"


@dataclass
class WebSocketMessage:
    """A WebSocket message."""
    type: MessageType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    session_id: str | None = None
    task_id: str | None = None
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps({
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "task_id": self.task_id,
        })


class ConnectionManager:
    """
    Manages WebSocket connections for real-time updates.
    
    Supports:
    - Multiple connections per session
    - Broadcast to all connections
    - Session-specific broadcasts
    - Task-specific broadcasts
    """
    
    def __init__(self):
        # All active connections
        self.active_connections: list[WebSocket] = []
        
        # Connections by session ID
        self.session_connections: dict[str, list[WebSocket]] = {}
        
        # Connection metadata
        self.connection_metadata: dict[WebSocket, dict[str, Any]] = {}
        
        # Message queue for buffering
        self._message_queue: asyncio.Queue[tuple[WebSocket, WebSocketMessage]] = asyncio.Queue()
        
        # Background task for processing queue
        self._processor_task: asyncio.Task | None = None
    
    async def connect(
        self,
        websocket: WebSocket,
        session_id: str | None = None,
    ) -> None:
        """
        Accept a new WebSocket connection.
        
        Args:
            websocket: The WebSocket connection
            session_id: Optional session ID to subscribe to
        """
        await websocket.accept()
        
        self.active_connections.append(websocket)
        
        # Store metadata
        self.connection_metadata[websocket] = {
            "session_id": session_id,
            "connected_at": datetime.now(timezone.utc).isoformat(),
        }
        
        # Add to session connections if specified
        if session_id:
            if session_id not in self.session_connections:
                self.session_connections[session_id] = []
            self.session_connections[session_id].append(websocket)
        
        # Send connected message
        await self.send_personal(
            websocket,
            WebSocketMessage(
                type=MessageType.CONNECTED,
                data={"message": "Connected to Mini-Devin"},
                session_id=session_id,
            ),
        )
    
    def disconnect(self, websocket: WebSocket) -> None:
        """
        Handle WebSocket disconnection.
        
        Args:
            websocket: The WebSocket connection
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        # Remove from session connections
        metadata = self.connection_metadata.get(websocket, {})
        session_id = metadata.get("session_id")
        if session_id and session_id in self.session_connections:
            if websocket in self.session_connections[session_id]:
                self.session_connections[session_id].remove(websocket)
            if not self.session_connections[session_id]:
                del self.session_connections[session_id]
        
        # Remove metadata
        if websocket in self.connection_metadata:
            del self.connection_metadata[websocket]
    
    async def send_personal(
        self,
        websocket: WebSocket,
        message: WebSocketMessage,
    ) -> bool:
        """
        Send a message to a specific connection.
        
        Args:
            websocket: The WebSocket connection
            message: The message to send
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            await websocket.send_text(message.to_json())
            return True
        except Exception:
            return False
    
    async def broadcast(self, message: WebSocketMessage) -> int:
        """
        Broadcast a message to all connections.
        
        Args:
            message: The message to broadcast
            
        Returns:
            Number of successful sends
        """
        success_count = 0
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message.to_json())
                success_count += 1
            except Exception:
                disconnected.append(connection)
        
        # Clean up disconnected connections
        for conn in disconnected:
            self.disconnect(conn)
        
        return success_count
    
    async def broadcast_to_session(
        self,
        session_id: str,
        message: WebSocketMessage,
    ) -> int:
        """
        Broadcast a message to all connections subscribed to a session.
        
        Args:
            session_id: The session ID
            message: The message to broadcast
            
        Returns:
            Number of successful sends
        """
        message.session_id = session_id
        
        if session_id not in self.session_connections:
            return 0
        
        success_count = 0
        disconnected = []
        
        for connection in self.session_connections[session_id]:
            try:
                await connection.send_text(message.to_json())
                success_count += 1
            except Exception:
                disconnected.append(connection)
        
        # Clean up disconnected connections
        for conn in disconnected:
            self.disconnect(conn)
        
        return success_count
    
    # Convenience methods for common message types
    
    async def send_task_started(
        self,
        session_id: str,
        task_id: str,
        description: str,
    ) -> int:
        """Send task started notification."""
        return await self.broadcast_to_session(
            session_id,
            WebSocketMessage(
                type=MessageType.TASK_STARTED,
                data={"description": description},
                task_id=task_id,
            ),
        )
    
    async def send_task_completed(
        self,
        session_id: str,
        task_id: str,
        summary: str,
    ) -> int:
        """Send task completed notification."""
        return await self.broadcast_to_session(
            session_id,
            WebSocketMessage(
                type=MessageType.TASK_COMPLETED,
                data={"summary": summary},
                task_id=task_id,
            ),
        )
    
    async def send_task_failed(
        self,
        session_id: str,
        task_id: str,
        error: str,
    ) -> int:
        """Send task failed notification."""
        return await self.broadcast_to_session(
            session_id,
            WebSocketMessage(
                type=MessageType.TASK_FAILED,
                data={"error": error},
                task_id=task_id,
            ),
        )
    
    async def send_phase_changed(
        self,
        session_id: str,
        task_id: str,
        phase: str,
    ) -> int:
        """Send phase changed notification."""
        return await self.broadcast_to_session(
            session_id,
            WebSocketMessage(
                type=MessageType.PHASE_CHANGED,
                data={"phase": phase},
                task_id=task_id,
            ),
        )
    
    async def send_tokens(
        self,
        session_id: str,
        task_id: str,
        tokens: str,
    ) -> int:
        """Send streaming tokens."""
        return await self.broadcast_to_session(
            session_id,
            WebSocketMessage(
                type=MessageType.TOKEN,
                data={"content": tokens},
                task_id=task_id,
            ),
        )
    
    async def send_tool_started(
        self,
        session_id: str,
        task_id: str,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> int:
        """Send tool started notification."""
        return await self.broadcast_to_session(
            session_id,
            WebSocketMessage(
                type=MessageType.TOOL_STARTED,
                data={"tool": tool_name, "input": tool_input},
                task_id=task_id,
            ),
        )
    
    async def send_tool_completed(
        self,
        session_id: str,
        task_id: str,
        tool_name: str,
        tool_output: dict[str, Any],
        duration_ms: float,
    ) -> int:
        """Send tool completed notification."""
        return await self.broadcast_to_session(
            session_id,
            WebSocketMessage(
                type=MessageType.TOOL_COMPLETED,
                data={
                    "tool": tool_name,
                    "output": tool_output,
                    "duration_ms": duration_ms,
                },
                task_id=task_id,
            ),
        )
    
    async def send_tool_output(
        self,
        session_id: str,
        task_id: str,
        tool_name: str,
        output: str,
    ) -> int:
        """Send tool output (for streaming terminal output)."""
        return await self.broadcast_to_session(
            session_id,
            WebSocketMessage(
                type=MessageType.TOOL_OUTPUT,
                data={"tool": tool_name, "output": output},
                task_id=task_id,
            ),
        )
    
    async def send_plan_created(
        self,
        session_id: str,
        task_id: str,
        plan: dict[str, Any],
    ) -> int:
        """Send plan created notification."""
        return await self.broadcast_to_session(
            session_id,
            WebSocketMessage(
                type=MessageType.PLAN_CREATED,
                data={"plan": plan},
                task_id=task_id,
            ),
        )
    
    async def send_verification_result(
        self,
        session_id: str,
        task_id: str,
        passed: bool,
        results: list[dict[str, Any]],
    ) -> int:
        """Send verification result notification."""
        return await self.broadcast_to_session(
            session_id,
            WebSocketMessage(
                type=MessageType.VERIFICATION_COMPLETED,
                data={"passed": passed, "results": results},
                task_id=task_id,
            ),
        )
    
    def get_connection_count(self) -> int:
        """Get total number of active connections."""
        return len(self.active_connections)
    
    def get_session_connection_count(self, session_id: str) -> int:
        """Get number of connections for a session."""
        return len(self.session_connections.get(session_id, []))
