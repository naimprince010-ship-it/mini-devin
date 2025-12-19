"""
Mini-Devin API Server v5.4
Phase 34: Better Error Handling
Phase 35: Task History & Export
Phase 36: Multi-Model Support (OpenAI, Anthropic, Ollama)
Phase 37: File Upload
Phase 38: Agent Memory (cross-session)
Phase 42: Authentication & Security (JWT, Rate Limiting, Session Timeout)
Phase 43: Database Connection Pooling (SQLAlchemy)
Phase 44: Background Task Queue with Retry/Failure Recovery
Phase 45: Logging & Monitoring (Structured JSON, Prometheus, Sentry)
"""

import os
import asyncio
import subprocess
import sqlite3
import re
import json
import uuid
import traceback
import hashlib
import secrets
import time
import logging
import sys
from collections import defaultdict
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Optional, List, Dict, Any, Generator, Callable
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Depends, Request, Header, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

# SQLAlchemy for connection pooling
from sqlalchemy import create_engine, text, event
from sqlalchemy.pool import QueuePool
from sqlalchemy.engine import Engine

# ============================================================================
# Phase 45: Logging & Monitoring Configuration
# ============================================================================
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.environ.get("LOG_FORMAT", "json")  # json or text
SENTRY_DSN = os.environ.get("SENTRY_DSN", "")
ENABLE_METRICS = os.environ.get("ENABLE_METRICS", "true").lower() == "true"

# Initialize Sentry if DSN is provided
sentry_sdk = None
if SENTRY_DSN:
    try:
        import sentry_sdk as _sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration
        _sentry_sdk.init(
            dsn=SENTRY_DSN,
            integrations=[FastApiIntegration(), StarletteIntegration()],
            traces_sample_rate=0.1,
            profiles_sample_rate=0.1,
            environment=os.environ.get("ENVIRONMENT", "production"),
            release=os.environ.get("RELEASE_VERSION", "5.4.0")
        )
        sentry_sdk = _sentry_sdk
        print(f"Sentry initialized with DSN: {SENTRY_DSN[:20]}...")
    except ImportError:
        print("Warning: sentry-sdk not installed, error tracking disabled")
    except Exception as e:
        print(f"Warning: Failed to initialize Sentry: {e}")

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        if hasattr(record, "session_id"):
            log_data["session_id"] = record.session_id
        if hasattr(record, "task_id"):
            log_data["task_id"] = record.task_id
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms
        if hasattr(record, "status_code"):
            log_data["status_code"] = record.status_code
        if hasattr(record, "method"):
            log_data["method"] = record.method
        if hasattr(record, "path"):
            log_data["path"] = record.path
        if hasattr(record, "client_ip"):
            log_data["client_ip"] = record.client_ip
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info)
            }
        
        return json.dumps(log_data)

def setup_logging() -> logging.Logger:
    """Setup structured logging with JSON or text format."""
    logger = logging.getLogger("mini_devin")
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    
    if LOG_FORMAT == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
    
    logger.addHandler(handler)
    return logger

# Initialize logger
logger = setup_logging()

# ============================================================================
# Phase 45: Prometheus Metrics
# ============================================================================
class PrometheusMetrics:
    """Simple Prometheus metrics collector."""
    
    def __init__(self):
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.labels: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._lock = asyncio.Lock()
    
    def inc(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """Increment a counter."""
        key = self._make_key(name, labels)
        self.counters[key] += value
        if labels:
            self.labels[key] = labels
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge value."""
        key = self._make_key(name, labels)
        self.gauges[key] = value
        if labels:
            self.labels[key] = labels
    
    def observe(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram observation."""
        key = self._make_key(name, labels)
        self.histograms[key].append(value)
        # Keep only last 1000 observations per metric
        if len(self.histograms[key]) > 1000:
            self.histograms[key] = self.histograms[key][-1000:]
        if labels:
            self.labels[key] = labels
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Create a unique key for a metric with labels."""
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def _format_histogram(self, name: str, values: List[float], labels_str: str) -> str:
        """Format histogram data in Prometheus format."""
        if not values:
            return ""
        
        lines = []
        buckets = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, float('inf')]
        
        for bucket in buckets:
            count = sum(1 for v in values if v <= bucket)
            bucket_label = f'le="{bucket}"' if bucket != float('inf') else 'le="+Inf"'
            if labels_str:
                lines.append(f'{name}_bucket{{{labels_str},{bucket_label}}} {count}')
            else:
                lines.append(f'{name}_bucket{{{bucket_label}}} {count}')
        
        if labels_str:
            lines.append(f'{name}_sum{{{labels_str}}} {sum(values)}')
            lines.append(f'{name}_count{{{labels_str}}} {len(values)}')
        else:
            lines.append(f'{name}_sum {sum(values)}')
            lines.append(f'{name}_count {len(values)}')
        
        return "\n".join(lines)
    
    def export(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        
        # Export counters
        for key, value in self.counters.items():
            lines.append(f"# TYPE {key.split('{')[0]} counter")
            lines.append(f"{key} {value}")
        
        # Export gauges
        for key, value in self.gauges.items():
            lines.append(f"# TYPE {key.split('{')[0]} gauge")
            lines.append(f"{key} {value}")
        
        # Export histograms
        exported_histograms = set()
        for key, values in self.histograms.items():
            base_name = key.split('{')[0]
            if base_name not in exported_histograms:
                lines.append(f"# TYPE {base_name} histogram")
                exported_histograms.add(base_name)
            
            labels_str = ""
            if '{' in key:
                labels_str = key[key.index('{')+1:key.index('}')]
            
            lines.append(self._format_histogram(base_name, values, labels_str))
        
        return "\n".join(lines)

# Global metrics instance
metrics = PrometheusMetrics() if ENABLE_METRICS else None

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and collecting metrics."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        # Add request_id to request state
        request.state.request_id = request_id
        
        # Log request start
        client_ip = request.client.host if request.client else "unknown"
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client_ip": client_ip
            }
        )
        
        # Process request
        try:
            response = await call_next(request)
            duration_ms = (time.time() - start_time) * 1000
            
            # Log request completion
            logger.info(
                f"Request completed: {request.method} {request.url.path} - {response.status_code}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round(duration_ms, 2),
                    "client_ip": client_ip
                }
            )
            
            # Record metrics
            if metrics:
                metrics.inc("http_requests_total", labels={
                    "method": request.method,
                    "path": request.url.path.split("/")[1] if "/" in request.url.path else request.url.path,
                    "status": str(response.status_code)
                })
                metrics.observe("http_request_duration_seconds", duration_ms / 1000, labels={
                    "method": request.method,
                    "path": request.url.path.split("/")[1] if "/" in request.url.path else request.url.path
                })
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            return response
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            # Log error
            logger.error(
                f"Request failed: {request.method} {request.url.path} - {str(e)}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round(duration_ms, 2),
                    "client_ip": client_ip
                },
                exc_info=True
            )
            
            # Record error metrics
            if metrics:
                metrics.inc("http_requests_total", labels={
                    "method": request.method,
                    "path": request.url.path.split("/")[1] if "/" in request.url.path else request.url.path,
                    "status": "500"
                })
            
            # Report to Sentry
            if sentry_sdk:
                sentry_sdk.capture_exception(e)
            
            raise

def log_task_event(event_type: str, task_id: str, session_id: str, **kwargs):
    """Log task-related events with structured data."""
    logger.info(
        f"Task event: {event_type}",
        extra={
            "task_id": task_id,
            "session_id": session_id,
            **kwargs
        }
    )
    
    if metrics:
        metrics.inc(f"task_events_total", labels={"event": event_type})

def log_error(error: Exception, context: Optional[Dict[str, Any]] = None):
    """Log an error with context and optionally report to Sentry."""
    logger.error(
        f"Error: {str(error)}",
        extra=context or {},
        exc_info=True
    )
    
    if metrics:
        metrics.inc("errors_total", labels={"type": type(error).__name__})
    
    if sentry_sdk:
        with sentry_sdk.push_scope() as scope:
            if context:
                for key, value in context.items():
                    scope.set_extra(key, value)
            sentry_sdk.capture_exception(error)

# JWT Configuration
JWT_SECRET = os.environ.get("JWT_SECRET", secrets.token_hex(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = int(os.environ.get("JWT_EXPIRY_HOURS", "24"))
SESSION_TIMEOUT_MINUTES = int(os.environ.get("SESSION_TIMEOUT_MINUTES", "60"))

# Rate Limiting Configuration
RATE_LIMIT_REQUESTS = int(os.environ.get("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60"))

# Security
security = HTTPBearer(auto_error=False)

# Rate limiting storage (in-memory, use Redis for production scaling)
rate_limit_storage: Dict[str, List[float]] = defaultdict(list)

DB_PATH = os.environ.get("MINI_DEVIN_DB_PATH", "/root/mini-devin/mini_devin.db")
WORKSPACE_DIR = os.environ.get("MINI_DEVIN_WORKSPACE", "/root/mini-devin/workspace")
UPLOADS_DIR = os.environ.get("MINI_DEVIN_UPLOADS", "/root/mini-devin/uploads")
MEMORY_DIR = os.environ.get("MINI_DEVIN_MEMORY", "/root/mini-devin/memory")

# ============================================================================
# Phase 43: Database Connection Pooling Configuration
# ============================================================================
DB_POOL_SIZE = int(os.environ.get("DB_POOL_SIZE", "5"))
DB_MAX_OVERFLOW = int(os.environ.get("DB_MAX_OVERFLOW", "10"))
DB_POOL_TIMEOUT = int(os.environ.get("DB_POOL_TIMEOUT", "30"))
DB_POOL_RECYCLE = int(os.environ.get("DB_POOL_RECYCLE", "3600"))

# Create SQLAlchemy engine with connection pooling
db_engine: Optional[Engine] = None

def init_db_engine() -> Engine:
    """Initialize SQLAlchemy engine with connection pooling."""
    global db_engine
    if db_engine is None:
        # Ensure database directory exists
        db_dir = os.path.dirname(DB_PATH)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        
        # Create engine with QueuePool for connection pooling
        db_engine = create_engine(
            f"sqlite:///{DB_PATH}",
            poolclass=QueuePool,
            pool_size=DB_POOL_SIZE,
            max_overflow=DB_MAX_OVERFLOW,
            pool_timeout=DB_POOL_TIMEOUT,
            pool_recycle=DB_POOL_RECYCLE,
            pool_pre_ping=True,  # Verify connections before use
            connect_args={"check_same_thread": False}  # Allow multi-threaded access
        )
        
        # Enable WAL mode for better concurrent access
        @event.listens_for(db_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()
    
    return db_engine

@contextmanager
def get_db_connection() -> Generator:
    """Get a database connection from the pool with automatic cleanup."""
    engine = init_db_engine()
    connection = engine.connect()
    try:
        yield connection
    finally:
        connection.close()

def get_db_pool_stats() -> Dict[str, Any]:
    """Get current connection pool statistics."""
    if db_engine is None:
        return {"status": "not_initialized"}
    
    pool = db_engine.pool
    return {
        "status": "active",
        "pool_size": DB_POOL_SIZE,
        "max_overflow": DB_MAX_OVERFLOW,
        "pool_timeout": DB_POOL_TIMEOUT,
        "pool_recycle": DB_POOL_RECYCLE,
        "checked_in": pool.checkedin(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "invalid": pool.invalidatedcount() if hasattr(pool, 'invalidatedcount') else 0
    }

# ============================================================================
# Phase 44: Background Task Queue with Retry/Failure Recovery
# ============================================================================
TASK_QUEUE_MAX_WORKERS = int(os.environ.get("TASK_QUEUE_MAX_WORKERS", "3"))
TASK_QUEUE_MAX_RETRIES = int(os.environ.get("TASK_QUEUE_MAX_RETRIES", "3"))
TASK_QUEUE_RETRY_DELAY = float(os.environ.get("TASK_QUEUE_RETRY_DELAY", "5.0"))
TASK_QUEUE_RETRY_BACKOFF = float(os.environ.get("TASK_QUEUE_RETRY_BACKOFF", "2.0"))
TASK_QUEUE_MAX_RETRY_DELAY = float(os.environ.get("TASK_QUEUE_MAX_RETRY_DELAY", "300.0"))

class TaskStatus:
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"

class QueuedTask:
    """Represents a task in the background queue."""
    def __init__(self, task_id: str, session_id: str, description: str, 
                 model: str, provider: str = "openai", max_iterations: int = 10,
                 priority: int = 0):
        self.task_id = task_id
        self.session_id = session_id
        self.description = description
        self.model = model
        self.provider = provider
        self.max_iterations = max_iterations
        self.priority = priority
        self.retry_count = 0
        self.last_error: Optional[str] = None
        self.queued_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.next_retry_at: Optional[datetime] = None

class BackgroundTaskQueue:
    """In-memory background task queue with retry and failure recovery."""
    
    def __init__(self, max_workers: int = TASK_QUEUE_MAX_WORKERS):
        self.max_workers = max_workers
        self.queue: asyncio.Queue[QueuedTask] = asyncio.Queue()
        self.active_tasks: Dict[str, QueuedTask] = {}
        self.completed_tasks: Dict[str, Dict[str, Any]] = {}
        self.failed_tasks: Dict[str, Dict[str, Any]] = {}
        self.cancelled_tasks: set = set()
        self.workers: List[asyncio.Task] = []
        self.running = False
        self.stats = {
            "total_queued": 0,
            "total_completed": 0,
            "total_failed": 0,
            "total_retried": 0,
            "total_cancelled": 0
        }
        self._lock = asyncio.Lock()
    
    async def start(self):
        """Start the background task queue workers."""
        if self.running:
            return
        self.running = True
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(i))
            self.workers.append(worker)
        print(f"Background task queue started with {self.max_workers} workers")
    
    async def stop(self):
        """Stop all workers gracefully."""
        self.running = False
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        print("Background task queue stopped")
    
    async def enqueue(self, task: QueuedTask) -> str:
        """Add a task to the queue."""
        async with self._lock:
            self.stats["total_queued"] += 1
            await self.queue.put(task)
            # Update task status in database
            self._update_task_status(task.task_id, TaskStatus.QUEUED, 
                                    queue_position=self.queue.qsize())
        return task.task_id
    
    async def cancel(self, task_id: str) -> bool:
        """Cancel a task if it's still pending or queued."""
        async with self._lock:
            self.cancelled_tasks.add(task_id)
            if task_id in self.active_tasks:
                # Task is running, mark for cancellation
                self.stats["total_cancelled"] += 1
                self._update_task_status(task_id, TaskStatus.CANCELLED)
                return True
            # Check if in queue (can't easily remove from asyncio.Queue)
            self.stats["total_cancelled"] += 1
            self._update_task_status(task_id, TaskStatus.CANCELLED)
            return True
    
    async def retry(self, task_id: str) -> bool:
        """Manually retry a failed task."""
        if task_id not in self.failed_tasks:
            return False
        
        task_info = self.failed_tasks.pop(task_id)
        new_task = QueuedTask(
            task_id=task_id,
            session_id=task_info["session_id"],
            description=task_info["description"],
            model=task_info["model"],
            provider=task_info["provider"],
            max_iterations=task_info["max_iterations"]
        )
        new_task.retry_count = task_info.get("retry_count", 0)
        await self.enqueue(new_task)
        return True
    
    def _update_task_status(self, task_id: str, status: str, 
                           queue_position: Optional[int] = None,
                           retry_count: Optional[int] = None,
                           error: Optional[str] = None,
                           next_retry_at: Optional[str] = None):
        """Update task status in database."""
        try:
            conn = get_db()
            c = conn.cursor()
            updates = ["status=?"]
            params = [status]
            
            if queue_position is not None:
                updates.append("queue_position=?")
                params.append(queue_position)
            if retry_count is not None:
                updates.append("retry_count=?")
                params.append(retry_count)
            if error is not None:
                updates.append("error_message=?")
                params.append(error)
            if next_retry_at is not None:
                updates.append("next_retry_at=?")
                params.append(next_retry_at)
            
            params.append(task_id)
            c.execute(f"UPDATE tasks SET {', '.join(updates)} WHERE task_id=?", params)
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error updating task status: {e}")
    
    async def _worker(self, worker_id: int):
        """Worker coroutine that processes tasks from the queue."""
        print(f"Worker {worker_id} started")
        while self.running:
            try:
                # Wait for a task with timeout
                try:
                    task = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Check if task was cancelled
                if task.task_id in self.cancelled_tasks:
                    self.cancelled_tasks.discard(task.task_id)
                    self.queue.task_done()
                    continue
                
                # Check if task should wait for retry
                if task.next_retry_at and datetime.utcnow() < task.next_retry_at:
                    wait_time = (task.next_retry_at - datetime.utcnow()).total_seconds()
                    if wait_time > 0:
                        await asyncio.sleep(min(wait_time, 1.0))
                        await self.queue.put(task)
                        self.queue.task_done()
                        continue
                
                # Process the task
                async with self._lock:
                    self.active_tasks[task.task_id] = task
                    task.started_at = datetime.utcnow()
                
                self._update_task_status(task.task_id, TaskStatus.RUNNING, 
                                        retry_count=task.retry_count)
                
                try:
                    # Execute the actual task
                    await execute_agent_task(
                        task.session_id, 
                        task.task_id, 
                        task.description,
                        task.model,
                        task.provider,
                        task.max_iterations
                    )
                    
                    # Task completed successfully
                    async with self._lock:
                        self.active_tasks.pop(task.task_id, None)
                        self.completed_tasks[task.task_id] = {
                            "session_id": task.session_id,
                            "completed_at": datetime.utcnow().isoformat(),
                            "retry_count": task.retry_count
                        }
                        self.stats["total_completed"] += 1
                    
                except Exception as e:
                    error_msg = str(e)
                    task.last_error = error_msg
                    task.retry_count += 1
                    
                    async with self._lock:
                        self.active_tasks.pop(task.task_id, None)
                    
                    # Check if we should retry
                    if task.retry_count < TASK_QUEUE_MAX_RETRIES:
                        # Calculate retry delay with exponential backoff
                        delay = min(
                            TASK_QUEUE_RETRY_DELAY * (TASK_QUEUE_RETRY_BACKOFF ** (task.retry_count - 1)),
                            TASK_QUEUE_MAX_RETRY_DELAY
                        )
                        task.next_retry_at = datetime.utcnow() + timedelta(seconds=delay)
                        
                        self._update_task_status(
                            task.task_id, 
                            TaskStatus.RETRYING,
                            retry_count=task.retry_count,
                            error=error_msg,
                            next_retry_at=task.next_retry_at.isoformat()
                        )
                        
                        async with self._lock:
                            self.stats["total_retried"] += 1
                        
                        # Re-queue for retry
                        await self.queue.put(task)
                        print(f"Task {task.task_id} scheduled for retry {task.retry_count}/{TASK_QUEUE_MAX_RETRIES} in {delay}s")
                    else:
                        # Max retries exceeded, mark as failed
                        async with self._lock:
                            self.failed_tasks[task.task_id] = {
                                "session_id": task.session_id,
                                "description": task.description,
                                "model": task.model,
                                "provider": task.provider,
                                "max_iterations": task.max_iterations,
                                "error": error_msg,
                                "retry_count": task.retry_count,
                                "failed_at": datetime.utcnow().isoformat()
                            }
                            self.stats["total_failed"] += 1
                        
                        self._update_task_status(
                            task.task_id, 
                            TaskStatus.FAILED,
                            retry_count=task.retry_count,
                            error=f"Max retries exceeded: {error_msg}"
                        )
                        print(f"Task {task.task_id} failed after {task.retry_count} retries: {error_msg}")
                
                self.queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1)
        
        print(f"Worker {worker_id} stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            "running": self.running,
            "max_workers": self.max_workers,
            "queue_size": self.queue.qsize(),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "stats": self.stats.copy(),
            "config": {
                "max_retries": TASK_QUEUE_MAX_RETRIES,
                "retry_delay": TASK_QUEUE_RETRY_DELAY,
                "retry_backoff": TASK_QUEUE_RETRY_BACKOFF,
                "max_retry_delay": TASK_QUEUE_MAX_RETRY_DELAY
            }
        }
    
    def get_task_info(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific task."""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return {
                "status": TaskStatus.RUNNING,
                "session_id": task.session_id,
                "retry_count": task.retry_count,
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "last_error": task.last_error
            }
        if task_id in self.completed_tasks:
            return {"status": TaskStatus.COMPLETED, **self.completed_tasks[task_id]}
        if task_id in self.failed_tasks:
            return {"status": TaskStatus.FAILED, **self.failed_tasks[task_id]}
        if task_id in self.cancelled_tasks:
            return {"status": TaskStatus.CANCELLED}
        return None

# Global task queue instance
task_queue: Optional[BackgroundTaskQueue] = None

def get_task_queue() -> BackgroundTaskQueue:
    """Get the global task queue instance."""
    global task_queue
    if task_queue is None:
        task_queue = BackgroundTaskQueue(max_workers=TASK_QUEUE_MAX_WORKERS)
    return task_queue

class APIError(Exception):
    def __init__(self, message: str, code: str, status_code: int = 400, details: Optional[Dict] = None):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)

ERROR_SUGGESTIONS = {
    "file_not_found": ["Check if the file path is correct", "Use list_files to see available files", "Create the file first using file_write"],
    "permission_denied": ["File is in a restricted directory", "Use /tmp or workspace directory instead"],
    "command_timeout": ["Command took too long (60s limit)", "Try breaking into smaller commands", "Check if command is stuck"],
    "command_blocked": ["Command contains dangerous patterns", "Use safer alternatives"],
    "api_error": ["Check API key configuration", "Verify model name is correct", "Try again later"],
    "rate_limit": ["Too many requests", "Wait a moment and try again"],
    "invalid_json": ["Tool call JSON is malformed", "Check JSON syntax"],
    "unknown_tool": ["Tool name not recognized", "Check available tools list"],
}

def get_error_suggestions(error_code: str) -> List[str]:
    return ERROR_SUGGESTIONS.get(error_code, ["Try a different approach", "Check the error message for details"])

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    os.makedirs(MEMORY_DIR, exist_ok=True)
    os.makedirs(REPOS_DIR, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        created_at TEXT,
        status TEXT,
        working_directory TEXT,
        model TEXT,
        provider TEXT DEFAULT 'openai',
        max_iterations INTEGER,
        current_task TEXT,
        iteration INTEGER DEFAULT 0,
        total_tasks INTEGER DEFAULT 0
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS tasks (
        task_id TEXT PRIMARY KEY,
        session_id TEXT,
        description TEXT,
        status TEXT,
        created_at TEXT,
        started_at TEXT,
        completed_at TEXT,
        result TEXT,
        error_message TEXT,
        error_code TEXT,
        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS task_outputs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT,
        output_type TEXT,
        content TEXT,
        created_at TEXT,
        FOREIGN KEY (task_id) REFERENCES tasks(task_id)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS uploaded_files (
        file_id TEXT PRIMARY KEY,
        session_id TEXT,
        original_name TEXT,
        stored_path TEXT,
        file_size INTEGER,
        mime_type TEXT,
        uploaded_at TEXT,
        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS agent_memory (
        memory_id TEXT PRIMARY KEY,
        session_id TEXT,
        memory_type TEXT,
        content TEXT,
        created_at TEXT,
        last_accessed TEXT,
        access_count INTEGER DEFAULT 0,
        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
    )''')
    # Phase 41: GitHub Repo Integration
    c.execute('''CREATE TABLE IF NOT EXISTS github_repos (
        repo_id TEXT PRIMARY KEY,
        repo_url TEXT NOT NULL,
        repo_name TEXT NOT NULL,
        owner TEXT NOT NULL,
        github_token TEXT,
        local_path TEXT,
        default_branch TEXT DEFAULT 'main',
        created_at TEXT,
        last_synced TEXT,
        status TEXT DEFAULT 'pending'
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS session_repos (
        session_id TEXT,
        repo_id TEXT,
        branch TEXT DEFAULT 'main',
        linked_at TEXT,
        PRIMARY KEY (session_id, repo_id),
        FOREIGN KEY (session_id) REFERENCES sessions(session_id),
        FOREIGN KEY (repo_id) REFERENCES github_repos(repo_id)
    )''')
    # Phase 42: Users table for authentication
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE,
        password_hash TEXT NOT NULL,
        created_at TEXT,
        last_login TEXT,
        is_active INTEGER DEFAULT 1,
        role TEXT DEFAULT 'user'
    )''')
    # Phase 42: User sessions for JWT token management
    c.execute('''CREATE TABLE IF NOT EXISTS user_sessions (
        token_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        created_at TEXT,
        expires_at TEXT,
        last_activity TEXT,
        is_revoked INTEGER DEFAULT 0,
        ip_address TEXT,
        user_agent TEXT,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    )''')
    try:
        c.execute("ALTER TABLE sessions ADD COLUMN provider TEXT DEFAULT 'openai'")
    except:
        pass
    try:
        c.execute("ALTER TABLE tasks ADD COLUMN error_message TEXT")
    except:
        pass
    try:
        c.execute("ALTER TABLE tasks ADD COLUMN error_code TEXT")
    except:
        pass
    try:
        c.execute("ALTER TABLE sessions ADD COLUMN user_id TEXT")
    except:
        pass
    # Phase 44: Add retry tracking columns to tasks table
    try:
        c.execute("ALTER TABLE tasks ADD COLUMN retry_count INTEGER DEFAULT 0")
    except:
        pass
    try:
        c.execute("ALTER TABLE tasks ADD COLUMN queue_position INTEGER")
    except:
        pass
    try:
        c.execute("ALTER TABLE tasks ADD COLUMN next_retry_at TEXT")
    except:
        pass
    conn.commit()
    conn.close()

def get_db():
    """Get a database connection from the pool.
    
    This function now uses SQLAlchemy connection pooling for better performance.
    The connection is automatically returned to the pool when closed.
    """
    engine = init_db_engine()
    # Get raw DBAPI connection from pool
    raw_conn = engine.raw_connection()
    raw_conn.row_factory = sqlite3.Row
    return raw_conn

def get_db_pooled():
    """Get a pooled database connection as a context manager.
    
    Usage:
        with get_db_pooled() as conn:
            result = conn.execute(text("SELECT * FROM sessions"))
            rows = result.fetchall()
    """
    return get_db_connection()

active_websockets: Dict[str, List[WebSocket]] = {}
llm_clients: Dict[str, Any] = {}

# ============================================================================
# Phase 42: Authentication & Security Functions
# ============================================================================

def hash_password(password: str) -> str:
    """Hash password using SHA-256 with salt."""
    salt = secrets.token_hex(16)
    password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return f"{salt}:{password_hash}"

def verify_password(password: str, stored_hash: str) -> bool:
    """Verify password against stored hash."""
    try:
        salt, password_hash = stored_hash.split(":")
        return hashlib.sha256((password + salt).encode()).hexdigest() == password_hash
    except ValueError:
        return False

def create_jwt_token(user_id: str, username: str, role: str = "user") -> dict:
    """Create a JWT token with expiry."""
    import base64
    
    now = datetime.utcnow()
    expires_at = now + timedelta(hours=JWT_EXPIRY_HOURS)
    token_id = str(uuid.uuid4())
    
    # Create payload
    payload = {
        "token_id": token_id,
        "user_id": user_id,
        "username": username,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int(expires_at.timestamp())
    }
    
    # Simple JWT encoding (header.payload.signature)
    header = base64.urlsafe_b64encode(json.dumps({"alg": JWT_ALGORITHM, "typ": "JWT"}).encode()).decode().rstrip("=")
    payload_encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    signature_input = f"{header}.{payload_encoded}"
    signature = hashlib.sha256((signature_input + JWT_SECRET).encode()).hexdigest()
    
    token = f"{header}.{payload_encoded}.{signature}"
    
    return {
        "token": token,
        "token_id": token_id,
        "expires_at": expires_at.isoformat(),
        "expires_in": JWT_EXPIRY_HOURS * 3600
    }

def decode_jwt_token(token: str) -> Optional[dict]:
    """Decode and verify a JWT token."""
    import base64
    
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        
        header, payload_encoded, signature = parts
        
        # Verify signature
        signature_input = f"{header}.{payload_encoded}"
        expected_signature = hashlib.sha256((signature_input + JWT_SECRET).encode()).hexdigest()
        if signature != expected_signature:
            return None
        
        # Decode payload (add padding if needed)
        padding = 4 - len(payload_encoded) % 4
        if padding != 4:
            payload_encoded += "=" * padding
        
        payload = json.loads(base64.urlsafe_b64decode(payload_encoded).decode())
        
        # Check expiry
        if payload.get("exp", 0) < int(datetime.utcnow().timestamp()):
            return None
        
        return payload
    except Exception:
        return None

def check_rate_limit(identifier: str) -> tuple[bool, dict]:
    """Check if request is within rate limit. Returns (allowed, info)."""
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    
    # Clean old entries
    rate_limit_storage[identifier] = [t for t in rate_limit_storage[identifier] if t > window_start]
    
    # Check limit
    current_count = len(rate_limit_storage[identifier])
    if current_count >= RATE_LIMIT_REQUESTS:
        retry_after = int(rate_limit_storage[identifier][0] - window_start) + 1
        return False, {
            "allowed": False,
            "limit": RATE_LIMIT_REQUESTS,
            "remaining": 0,
            "reset": int(window_start + RATE_LIMIT_WINDOW_SECONDS),
            "retry_after": retry_after
        }
    
    # Add current request
    rate_limit_storage[identifier].append(now)
    
    return True, {
        "allowed": True,
        "limit": RATE_LIMIT_REQUESTS,
        "remaining": RATE_LIMIT_REQUESTS - current_count - 1,
        "reset": int(now + RATE_LIMIT_WINDOW_SECONDS)
    }

async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[dict]:
    """Get current user from JWT token. Returns None if not authenticated."""
    if not credentials:
        return None
    
    token = credentials.credentials
    payload = decode_jwt_token(token)
    
    if not payload:
        return None
    
    # Check if token is revoked
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("SELECT is_revoked FROM user_sessions WHERE token_id = ?", (payload.get("token_id"),))
        row = c.fetchone()
        if row and row["is_revoked"]:
            return None
        
        # Update last activity
        c.execute("UPDATE user_sessions SET last_activity = ? WHERE token_id = ?",
                  (datetime.utcnow().isoformat(), payload.get("token_id")))
        conn.commit()
    finally:
        conn.close()
    
    return payload

async def require_auth(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> dict:
    """Require authentication. Raises 401 if not authenticated."""
    user = await get_current_user(request, credentials)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated. Please login first.",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return user

async def require_admin(
    user: dict = Depends(require_auth)
) -> dict:
    """Require admin role. Raises 403 if not admin."""
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user

def get_client_ip(request: Request) -> str:
    """Get client IP address from request."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

# Authentication request/response models
class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    password: str = Field(..., min_length=8, description="Password (min 8 characters)")
    email: Optional[str] = Field(default=None, description="Email address")

class LoginRequest(BaseModel):
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")

class TokenRefreshRequest(BaseModel):
    token: str = Field(..., description="Current valid token to refresh")

class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password (min 8 characters)")

class CreateSessionRequest(BaseModel):
    working_directory: str = Field(default=".", description="Working directory")
    model: str = Field(default="gpt-4o-mini", description="LLM model to use")
    provider: str = Field(default="openai", description="LLM provider: openai, anthropic, ollama")
    max_iterations: int = Field(default=10, description="Max iterations per task")

class CreateTaskRequest(BaseModel):
    description: str = Field(..., description="Task description")
    files: Optional[List[str]] = Field(default=None, description="List of uploaded file IDs")

class MemoryEntry(BaseModel):
    key: str
    value: str
    memory_type: str = "fact"

class ToolResult(BaseModel):
    tool: str
    success: bool
    output: str
    error: Optional[str] = None
    error_code: Optional[str] = None
    suggestions: Optional[List[str]] = None

# Phase 41: GitHub Repo Integration Models
class AddRepoRequest(BaseModel):
    repo_url: str = Field(..., description="GitHub repository URL (e.g., https://github.com/owner/repo)")
    github_token: Optional[str] = Field(default=None, description="GitHub Personal Access Token for private repos")
    branch: str = Field(default="main", description="Default branch to use")

class LinkRepoRequest(BaseModel):
    repo_id: str = Field(..., description="Repository ID to link")
    branch: str = Field(default="main", description="Branch to work on")

class GitOperationRequest(BaseModel):
    repo_id: str = Field(..., description="Repository ID")
    operation: str = Field(..., description="Git operation: clone, pull, commit, push, create_branch, create_pr")
    message: Optional[str] = Field(default=None, description="Commit message or PR title")
    branch: Optional[str] = Field(default=None, description="Branch name for operations")
    body: Optional[str] = Field(default=None, description="PR body/description")
    base_branch: Optional[str] = Field(default="main", description="Base branch for PR")

class CreateGitHubRepoRequest(BaseModel):
    name: str = Field(..., description="Repository name")
    description: str = Field(default="", description="Repository description")
    private: bool = Field(default=False, description="Whether the repository should be private")
    github_token: str = Field(..., description="GitHub Personal Access Token with repo creation permissions")
    auto_init: bool = Field(default=True, description="Initialize with README")

REPOS_DIR = os.environ.get("MINI_DEVIN_REPOS", "/root/mini-devin/repos")

DANGEROUS_COMMANDS = [
    "rm -rf /", "rm -rf /*", "mkfs", "dd if=", ":(){", "fork bomb",
    "chmod -R 777 /", "chown -R", "> /dev/sda", "mv /* /dev/null",
    "wget http", "curl http", "nc -e", "bash -i", "/dev/tcp"
]

ALLOWED_DIRS = ["/tmp", WORKSPACE_DIR, UPLOADS_DIR, REPOS_DIR]

def is_path_allowed(path: str) -> bool:
    abs_path = os.path.abspath(path)
    return any(abs_path.startswith(d) for d in ALLOWED_DIRS)

def is_command_safe(command: str) -> tuple[bool, str]:
    cmd_lower = command.lower()
    for dangerous in DANGEROUS_COMMANDS:
        if dangerous in cmd_lower:
            return False, f"Command blocked: contains '{dangerous}'"
    return True, ""

def execute_terminal(command: str, working_dir: str = "/tmp") -> ToolResult:
    try:
        safe, reason = is_command_safe(command)
        if not safe:
            return ToolResult(tool="terminal", success=False, output="", error=reason, error_code="command_blocked", suggestions=get_error_suggestions("command_blocked"))
        
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=60, cwd=working_dir,
            env={**os.environ, "PATH": "/usr/local/bin:/usr/bin:/bin"}
        )
        output = result.stdout[:10000] if result.stdout else ""
        if result.returncode != 0:
            error = result.stderr[:2000] if result.stderr else "Command failed with no error output"
            return ToolResult(tool="terminal", success=False, output=output, error=error, error_code="command_failed", suggestions=["Check command syntax", "Verify file paths exist"])
        return ToolResult(tool="terminal", success=True, output=output)
    except subprocess.TimeoutExpired:
        return ToolResult(tool="terminal", success=False, output="", error="Command timed out (60s limit)", error_code="command_timeout", suggestions=get_error_suggestions("command_timeout"))
    except FileNotFoundError:
        return ToolResult(tool="terminal", success=False, output="", error=f"Working directory not found: {working_dir}", error_code="file_not_found", suggestions=get_error_suggestions("file_not_found"))
    except Exception as e:
        return ToolResult(tool="terminal", success=False, output="", error=str(e), error_code="unknown_error", suggestions=["Check command syntax", "Try a simpler command"])

def execute_file_read(path: str) -> ToolResult:
    try:
        if not is_path_allowed(path):
            return ToolResult(tool="file_read", success=False, output="", error=f"Access denied. Allowed: {ALLOWED_DIRS}", error_code="permission_denied", suggestions=get_error_suggestions("permission_denied"))
        if not os.path.exists(path):
            return ToolResult(tool="file_read", success=False, output="", error=f"File not found: {path}", error_code="file_not_found", suggestions=get_error_suggestions("file_not_found"))
        with open(path, 'r') as f:
            content = f.read(100000)
        return ToolResult(tool="file_read", success=True, output=content)
    except UnicodeDecodeError:
        return ToolResult(tool="file_read", success=False, output="", error="File is not a text file (binary content)", error_code="binary_file", suggestions=["Use a different tool for binary files"])
    except Exception as e:
        return ToolResult(tool="file_read", success=False, output="", error=str(e), error_code="unknown_error")

def execute_file_write(path: str, content: str) -> ToolResult:
    try:
        if not is_path_allowed(path):
            return ToolResult(tool="file_write", success=False, output="", error=f"Access denied. Allowed: {ALLOWED_DIRS}", error_code="permission_denied", suggestions=get_error_suggestions("permission_denied"))
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)
        return ToolResult(tool="file_write", success=True, output=f"Written {len(content)} bytes to {path}")
    except Exception as e:
        return ToolResult(tool="file_write", success=False, output="", error=str(e), error_code="unknown_error")

def execute_list_files(path: str = "/tmp") -> ToolResult:
    try:
        if not is_path_allowed(path):
            return ToolResult(tool="list_files", success=False, output="", error=f"Access denied. Allowed: {ALLOWED_DIRS}", error_code="permission_denied", suggestions=get_error_suggestions("permission_denied"))
        if not os.path.exists(path):
            return ToolResult(tool="list_files", success=False, output="", error=f"Directory not found: {path}", error_code="file_not_found", suggestions=get_error_suggestions("file_not_found"))
        entries = []
        for entry in os.scandir(path):
            entry_type = "dir" if entry.is_dir() else "file"
            size = entry.stat().st_size if entry.is_file() else 0
            entries.append(f"{entry_type}\t{size}\t{entry.name}")
        return ToolResult(tool="list_files", success=True, output="\n".join(entries[:200]) if entries else "Directory is empty")
    except Exception as e:
        return ToolResult(tool="list_files", success=False, output="", error=str(e), error_code="unknown_error")

def execute_git(command: str, working_dir: str = "/tmp") -> ToolResult:
    try:
        allowed_git_commands = ["status", "log", "diff", "branch", "show", "ls-files", "rev-parse", "config --get"]
        cmd_parts = command.strip().split()
        if not cmd_parts:
            return ToolResult(tool="git", success=False, output="", error="No git command provided", error_code="invalid_input")
        
        git_cmd = cmd_parts[0] if len(cmd_parts) == 1 else " ".join(cmd_parts[:2])
        if not any(git_cmd.startswith(allowed) for allowed in allowed_git_commands):
            return ToolResult(tool="git", success=False, output="", error=f"Git command not allowed. Allowed: {allowed_git_commands}", error_code="command_blocked", suggestions=["Use read-only git commands"])
        
        full_command = f"git {command}"
        result = subprocess.run(full_command, shell=True, capture_output=True, text=True, timeout=30, cwd=working_dir)
        if result.returncode != 0:
            return ToolResult(tool="git", success=False, output="", error=result.stderr[:1000], error_code="command_failed", suggestions=["Check if directory is a git repository"])
        return ToolResult(tool="git", success=True, output=result.stdout[:5000])
    except subprocess.TimeoutExpired:
        return ToolResult(tool="git", success=False, output="", error="Git command timed out", error_code="command_timeout")
    except Exception as e:
        return ToolResult(tool="git", success=False, output="", error=str(e), error_code="unknown_error")

def execute_code_analysis(path: str, analysis_type: str = "structure") -> ToolResult:
    try:
        if not is_path_allowed(path):
            return ToolResult(tool="code_analysis", success=False, output="", error=f"Access denied. Allowed: {ALLOWED_DIRS}", error_code="permission_denied")
        if not os.path.exists(path):
            return ToolResult(tool="code_analysis", success=False, output="", error=f"File not found: {path}", error_code="file_not_found", suggestions=get_error_suggestions("file_not_found"))
        
        with open(path, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        if analysis_type == "structure":
            functions = re.findall(r'(?:def|function|func)\s+(\w+)', content)
            classes = re.findall(r'(?:class)\s+(\w+)', content)
            imports = re.findall(r'(?:import|from|require|include)\s+[\w.]+', content)
            result = f"File: {path}\nLines: {len(lines)}\nClasses: {', '.join(classes) if classes else 'None'}\nFunctions: {', '.join(functions) if functions else 'None'}\nImports: {len(imports)} found\n"
            return ToolResult(tool="code_analysis", success=True, output=result)
        elif analysis_type == "complexity":
            indent_levels = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
            max_indent = max(indent_levels) if indent_levels else 0
            avg_indent = sum(indent_levels) / len(indent_levels) if indent_levels else 0
            result = f"File: {path}\nTotal lines: {len(lines)}\nNon-empty lines: {len([l for l in lines if l.strip()])}\nMax nesting depth: {max_indent // 4}\nAverage indentation: {avg_indent:.1f} spaces\n"
            return ToolResult(tool="code_analysis", success=True, output=result)
        else:
            return ToolResult(tool="code_analysis", success=False, output="", error=f"Unknown analysis type: {analysis_type}. Use 'structure' or 'complexity'", error_code="invalid_input")
    except Exception as e:
        return ToolResult(tool="code_analysis", success=False, output="", error=str(e), error_code="unknown_error")

def execute_search_files(path: str, pattern: str, file_pattern: str = "*") -> ToolResult:
    try:
        if not is_path_allowed(path):
            return ToolResult(tool="search_files", success=False, output="", error=f"Access denied. Allowed: {ALLOWED_DIRS}", error_code="permission_denied")
        if not os.path.exists(path):
            return ToolResult(tool="search_files", success=False, output="", error=f"Directory not found: {path}", error_code="file_not_found")
        
        import fnmatch
        results = []
        for root, dirs, files in os.walk(path):
            for filename in fnmatch.filter(files, file_pattern):
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, 'r') as f:
                        for i, line in enumerate(f, 1):
                            if pattern.lower() in line.lower():
                                results.append(f"{filepath}:{i}: {line.strip()[:100]}")
                                if len(results) >= 50:
                                    break
                except:
                    pass
                if len(results) >= 50:
                    break
            if len(results) >= 50:
                break
        
        if results:
            return ToolResult(tool="search_files", success=True, output="\n".join(results))
        else:
            return ToolResult(tool="search_files", success=True, output=f"No matches found for '{pattern}'")
    except Exception as e:
        return ToolResult(tool="search_files", success=False, output="", error=str(e), error_code="unknown_error")

def execute_memory_store(key: str, value: str, session_id: str) -> ToolResult:
    try:
        conn = get_db()
        c = conn.cursor()
        memory_id = str(uuid.uuid4())[:8]
        now = datetime.utcnow().isoformat()
        c.execute("INSERT OR REPLACE INTO agent_memory (memory_id, session_id, memory_type, content, created_at, last_accessed, access_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (memory_id, session_id, "user_stored", json.dumps({"key": key, "value": value}), now, now, 1))
        conn.commit()
        conn.close()
        return ToolResult(tool="memory_store", success=True, output=f"Stored memory: {key}")
    except Exception as e:
        return ToolResult(tool="memory_store", success=False, output="", error=str(e), error_code="unknown_error")

def execute_memory_recall(key: str, session_id: str) -> ToolResult:
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT content FROM agent_memory WHERE session_id=? AND content LIKE ?", (session_id, f'%"key": "{key}"%'))
        row = c.fetchone()
        if row:
            data = json.loads(row[0])
            c.execute("UPDATE agent_memory SET last_accessed=?, access_count=access_count+1 WHERE session_id=? AND content LIKE ?",
                      (datetime.utcnow().isoformat(), session_id, f'%"key": "{key}"%'))
            conn.commit()
            conn.close()
            return ToolResult(tool="memory_recall", success=True, output=data.get("value", ""))
        conn.close()
        return ToolResult(tool="memory_recall", success=False, output="", error=f"No memory found for key: {key}", error_code="not_found")
    except Exception as e:
        return ToolResult(tool="memory_recall", success=False, output="", error=str(e), error_code="unknown_error")

def execute_create_github_repo(name: str, description: str, private: bool, github_token: str, auto_init: bool = True) -> ToolResult:
    """Execute GitHub repo creation via API."""
    import httpx
    
    if not name or not name.strip():
        return ToolResult(tool="create_github_repo", success=False, output="", error="Repository name is required", error_code="invalid_input")
    
    repo_name = name.strip()
    if not all(c.isalnum() or c in '-_.' for c in repo_name):
        return ToolResult(tool="create_github_repo", success=False, output="", error="Repository name can only contain alphanumeric characters, hyphens, underscores, and periods", error_code="invalid_input")
    
    if not github_token:
        return ToolResult(tool="create_github_repo", success=False, output="", error="GitHub token is required. Please provide a Personal Access Token with 'repo' scope.", error_code="missing_token")
    
    github_api_url = "https://api.github.com/user/repos"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    payload = {
        "name": repo_name,
        "description": description or "",
        "private": private,
        "auto_init": auto_init
    }
    
    try:
        with httpx.Client() as client:
            response = client.post(github_api_url, headers=headers, json=payload, timeout=30.0)
            
            if response.status_code == 201:
                repo_data = response.json()
                owner = repo_data["owner"]["login"]
                repo_url = repo_data["clone_url"]
                html_url = repo_data["html_url"]
                default_branch = repo_data.get("default_branch", "main")
                
                # Auto-add to database
                repo_id = str(uuid.uuid4())[:8]
                local_path = os.path.join(REPOS_DIR, f"{owner}_{repo_name}_{repo_id}")
                now = datetime.utcnow().isoformat()
                
                conn = get_db()
                c = conn.cursor()
                c.execute("""INSERT INTO github_repos 
                             (repo_id, repo_url, repo_name, owner, github_token, local_path, default_branch, created_at, status)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                          (repo_id, repo_url, repo_name, owner, github_token, local_path, default_branch, now, "pending"))
                conn.commit()
                conn.close()
                
                result = {
                    "repo_id": repo_id,
                    "repo_url": repo_url,
                    "html_url": html_url,
                    "repo_name": repo_name,
                    "owner": owner,
                    "private": repo_data["private"],
                    "default_branch": default_branch,
                    "local_path": local_path
                }
                return ToolResult(tool="create_github_repo", success=True, output=json.dumps(result, indent=2))
            elif response.status_code == 401:
                return ToolResult(tool="create_github_repo", success=False, output="", error="Invalid GitHub token. Make sure your token has 'repo' scope.", error_code="auth_error")
            elif response.status_code == 422:
                error_data = response.json()
                errors = error_data.get("errors", [])
                if errors and errors[0].get("message", "").startswith("name already exists"):
                    return ToolResult(tool="create_github_repo", success=False, output="", error=f"Repository '{repo_name}' already exists in your GitHub account", error_code="already_exists")
                return ToolResult(tool="create_github_repo", success=False, output="", error=f"GitHub API error: {error_data.get('message', 'Unknown error')}", error_code="api_error")
            else:
                return ToolResult(tool="create_github_repo", success=False, output="", error=f"GitHub API error: {response.text}", error_code="api_error")
    except httpx.TimeoutException:
        return ToolResult(tool="create_github_repo", success=False, output="", error="GitHub API request timed out", error_code="timeout")
    except Exception as e:
        return ToolResult(tool="create_github_repo", success=False, output="", error=f"Failed to create repo: {str(e)}", error_code="unknown_error")

# ============================================
# GitHub API Tools for Agent (Features 1-7)
# ============================================

def get_github_token_for_session(session_id: str) -> Optional[str]:
    """Get GitHub token from session's linked repo."""
    conn = None
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute("""SELECT gr.github_token FROM github_repos gr
                     JOIN session_repos sr ON gr.repo_id = sr.repo_id
                     WHERE sr.session_id = ?""", (session_id,))
        row = c.fetchone()
        if row and row[0]:
            return row[0]
        return None
    except sqlite3.Error as e:
        print(f"Database error in get_github_token_for_session: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error in get_github_token_for_session: {e}")
        return None
    finally:
        if conn:
            conn.close()

def get_repo_info_for_session(session_id: str) -> Optional[dict]:
    """Get repo info from session's linked repo."""
    conn = None
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute("""SELECT gr.repo_id, gr.owner, gr.repo_name, gr.github_token, gr.local_path, gr.default_branch
                     FROM github_repos gr
                     JOIN session_repos sr ON gr.repo_id = sr.repo_id
                     WHERE sr.session_id = ?""", (session_id,))
        row = c.fetchone()
        if row:
            return {
                "repo_id": row[0],
                "owner": row[1],
                "repo_name": row[2],
                "github_token": row[3],
                "local_path": row[4],
                "default_branch": row[5] or "main"
            }
        return None
    except sqlite3.Error as e:
        print(f"Database error in get_repo_info_for_session: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error in get_repo_info_for_session: {e}")
        return None
    finally:
        if conn:
            conn.close()

def get_github_error_suggestions(status_code: int, error_message: str) -> List[str]:
    """Get actionable suggestions based on GitHub API error."""
    suggestions = []
    error_lower = error_message.lower()
    
    if status_code == 401:
        suggestions.append("Check if your GitHub token is valid and not expired")
        suggestions.append("Generate a new token at github.com/settings/tokens")
    elif status_code == 403:
        if "rate limit" in error_lower:
            suggestions.append("GitHub API rate limit exceeded. Wait a few minutes and try again")
            suggestions.append("Consider using a token with higher rate limits")
        elif "abuse" in error_lower:
            suggestions.append("GitHub abuse detection triggered. Wait and retry with smaller requests")
        else:
            suggestions.append("Check if your token has the required permissions/scopes")
            suggestions.append("For Actions: enable 'actions:read' scope")
            suggestions.append("For PRs/Issues: enable 'repo' scope")
            suggestions.append("For fine-grained tokens: grant access to this specific repository")
    elif status_code == 404:
        suggestions.append("Check if the repository exists and is accessible")
        suggestions.append("Verify the owner/repo name is correct")
        suggestions.append("For private repos: ensure your token has access")
    elif status_code == 422:
        suggestions.append("Check the request parameters are valid")
        if "branch" in error_lower:
            suggestions.append("Verify the branch name exists")
        if "already exists" in error_lower:
            suggestions.append("The resource already exists")
    
    return suggestions

def execute_github_api(method: str, endpoint: str, token: str, data: Optional[dict] = None, max_retries: int = 3) -> tuple[bool, dict]:
    """Execute GitHub API request with retry logic and better error handling."""
    import httpx
    import time
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    
    last_error = None
    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=30.0) as client:
                url = f"https://api.github.com{endpoint}"
                if method == "GET":
                    response = client.get(url, headers=headers)
                elif method == "POST":
                    response = client.post(url, headers=headers, json=data or {})
                elif method == "PATCH":
                    response = client.patch(url, headers=headers, json=data or {})
                elif method == "PUT":
                    response = client.put(url, headers=headers, json=data or {})
                elif method == "DELETE":
                    response = client.delete(url, headers=headers)
                else:
                    return False, {"error": f"Unknown method: {method}"}
                
                if response.status_code in [200, 201, 204]:
                    if response.status_code == 204:
                        return True, {"message": "Success"}
                    return True, response.json()
                elif response.status_code == 429 or (response.status_code == 403 and "rate limit" in response.text.lower()):
                    retry_after = int(response.headers.get("Retry-After", 60))
                    if attempt < max_retries - 1:
                        time.sleep(min(retry_after, 30))
                        continue
                elif response.status_code >= 500 and attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                
                error_data = response.json() if response.text else {"message": "Unknown error"}
                error_msg = error_data.get("message", str(error_data))
                suggestions = get_github_error_suggestions(response.status_code, error_msg)
                return False, {
                    "error": error_msg, 
                    "status_code": response.status_code,
                    "suggestions": suggestions
                }
        except httpx.TimeoutException:
            last_error = "GitHub API request timed out"
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
    
    return False, {"error": last_error or "Request failed after retries"}

# Feature 1: PR Creation Tool
def execute_create_pr(owner: str, repo: str, title: str, head: str, base: str, body: str, token: str) -> ToolResult:
    """Create a pull request on GitHub."""
    if not token:
        return ToolResult(tool="create_pr", success=False, output="", error="GitHub token required", error_code="missing_token")
    
    data = {
        "title": title,
        "head": head,
        "base": base,
        "body": body or ""
    }
    
    success, result = execute_github_api("POST", f"/repos/{owner}/{repo}/pulls", token, data)
    
    if success:
        pr_data = {
            "pr_number": result.get("number"),
            "html_url": result.get("html_url"),
            "state": result.get("state"),
            "title": result.get("title"),
            "head": result.get("head", {}).get("ref"),
            "base": result.get("base", {}).get("ref")
        }
        return ToolResult(tool="create_pr", success=True, output=json.dumps(pr_data, indent=2))
    else:
        return ToolResult(tool="create_pr", success=False, output="", error=result.get("error", "Failed to create PR"), error_code="api_error")

# Feature 2: PR Review/Merge Tools
def execute_list_prs(owner: str, repo: str, state: str, token: str, page: int = 1, per_page: int = 30) -> ToolResult:
    """List pull requests on GitHub with pagination support."""
    if not token:
        return ToolResult(tool="list_prs", success=False, output="", error="GitHub token required", error_code="missing_token")
    
    per_page = min(per_page, 100)
    success, result = execute_github_api("GET", f"/repos/{owner}/{repo}/pulls?state={state}&per_page={per_page}&page={page}", token)
    
    if success:
        prs = []
        for pr in result:
            prs.append({
                "number": pr.get("number"),
                "title": pr.get("title"),
                "state": pr.get("state"),
                "user": pr.get("user", {}).get("login"),
                "head": pr.get("head", {}).get("ref"),
                "base": pr.get("base", {}).get("ref"),
                "html_url": pr.get("html_url"),
                "created_at": pr.get("created_at"),
                "mergeable": pr.get("mergeable")
            })
        output = {
            "prs": prs,
            "page": page,
            "per_page": per_page,
            "count": len(prs),
            "has_more": len(prs) == per_page
        }
        return ToolResult(tool="list_prs", success=True, output=json.dumps(output, indent=2))
    else:
        return ToolResult(tool="list_prs", success=False, output="", error=result.get("error", "Failed to list PRs"), error_code="api_error", suggestions=result.get("suggestions", []))

def execute_view_pr(owner: str, repo: str, pr_number: int, token: str) -> ToolResult:
    """View a specific pull request."""
    if not token:
        return ToolResult(tool="view_pr", success=False, output="", error="GitHub token required", error_code="missing_token")
    
    success, result = execute_github_api("GET", f"/repos/{owner}/{repo}/pulls/{pr_number}", token)
    
    if success:
        pr_data = {
            "number": result.get("number"),
            "title": result.get("title"),
            "body": result.get("body"),
            "state": result.get("state"),
            "user": result.get("user", {}).get("login"),
            "head": result.get("head", {}).get("ref"),
            "base": result.get("base", {}).get("ref"),
            "html_url": result.get("html_url"),
            "created_at": result.get("created_at"),
            "updated_at": result.get("updated_at"),
            "mergeable": result.get("mergeable"),
            "mergeable_state": result.get("mergeable_state"),
            "additions": result.get("additions"),
            "deletions": result.get("deletions"),
            "changed_files": result.get("changed_files"),
            "commits": result.get("commits"),
            "comments": result.get("comments"),
            "review_comments": result.get("review_comments")
        }
        return ToolResult(tool="view_pr", success=True, output=json.dumps(pr_data, indent=2))
    else:
        return ToolResult(tool="view_pr", success=False, output="", error=result.get("error", "Failed to view PR"), error_code="api_error")

def execute_merge_pr(owner: str, repo: str, pr_number: int, merge_method: str, token: str) -> ToolResult:
    """Merge a pull request."""
    if not token:
        return ToolResult(tool="merge_pr", success=False, output="", error="GitHub token required", error_code="missing_token")
    
    if merge_method not in ["merge", "squash", "rebase"]:
        merge_method = "merge"
    
    data = {"merge_method": merge_method}
    
    success, result = execute_github_api("PUT", f"/repos/{owner}/{repo}/pulls/{pr_number}/merge", token, data)
    
    if success:
        return ToolResult(tool="merge_pr", success=True, output=f"PR #{pr_number} merged successfully. SHA: {result.get('sha', 'N/A')}")
    else:
        return ToolResult(tool="merge_pr", success=False, output="", error=result.get("error", "Failed to merge PR"), error_code="api_error")

# Feature 3: Issue Management
def execute_list_issues(owner: str, repo: str, state: str, token: str, page: int = 1, per_page: int = 30) -> ToolResult:
    """List issues on GitHub with pagination support."""
    if not token:
        return ToolResult(tool="list_issues", success=False, output="", error="GitHub token required", error_code="missing_token")
    
    per_page = min(per_page, 100)
    success, result = execute_github_api("GET", f"/repos/{owner}/{repo}/issues?state={state}&per_page={per_page}&page={page}", token)
    
    if success:
        issues = []
        for issue in result:
            if "pull_request" not in issue:
                issues.append({
                    "number": issue.get("number"),
                    "title": issue.get("title"),
                    "state": issue.get("state"),
                    "user": issue.get("user", {}).get("login"),
                    "labels": [l.get("name") for l in issue.get("labels", [])],
                    "html_url": issue.get("html_url"),
                    "created_at": issue.get("created_at"),
                    "comments": issue.get("comments")
                })
        output = {
            "issues": issues,
            "page": page,
            "per_page": per_page,
            "count": len(issues),
            "has_more": len(result) == per_page
        }
        return ToolResult(tool="list_issues", success=True, output=json.dumps(output, indent=2))
    else:
        return ToolResult(tool="list_issues", success=False, output="", error=result.get("error", "Failed to list issues"), error_code="api_error", suggestions=result.get("suggestions", []))

def execute_create_issue(owner: str, repo: str, title: str, body: str, labels: List[str], token: str) -> ToolResult:
    """Create an issue on GitHub."""
    if not token:
        return ToolResult(tool="create_issue", success=False, output="", error="GitHub token required", error_code="missing_token")
    
    data = {
        "title": title,
        "body": body or ""
    }
    if labels:
        data["labels"] = labels
    
    success, result = execute_github_api("POST", f"/repos/{owner}/{repo}/issues", token, data)
    
    if success:
        issue_data = {
            "number": result.get("number"),
            "title": result.get("title"),
            "html_url": result.get("html_url"),
            "state": result.get("state")
        }
        return ToolResult(tool="create_issue", success=True, output=json.dumps(issue_data, indent=2))
    else:
        return ToolResult(tool="create_issue", success=False, output="", error=result.get("error", "Failed to create issue"), error_code="api_error")

def execute_close_issue(owner: str, repo: str, issue_number: int, token: str) -> ToolResult:
    """Close an issue on GitHub."""
    if not token:
        return ToolResult(tool="close_issue", success=False, output="", error="GitHub token required", error_code="missing_token")
    
    data = {"state": "closed"}
    
    success, result = execute_github_api("PATCH", f"/repos/{owner}/{repo}/issues/{issue_number}", token, data)
    
    if success:
        return ToolResult(tool="close_issue", success=True, output=f"Issue #{issue_number} closed successfully")
    else:
        return ToolResult(tool="close_issue", success=False, output="", error=result.get("error", "Failed to close issue"), error_code="api_error")

# Feature 4: Branch Management
def execute_list_branches(owner: str, repo: str, token: str, page: int = 1, per_page: int = 30) -> ToolResult:
    """List branches on GitHub with pagination support."""
    if not token:
        return ToolResult(tool="list_branches", success=False, output="", error="GitHub token required", error_code="missing_token")
    
    per_page = min(per_page, 100)
    success, result = execute_github_api("GET", f"/repos/{owner}/{repo}/branches?per_page={per_page}&page={page}", token)
    
    if success:
        branches = []
        for branch in result:
            branches.append({
                "name": branch.get("name"),
                "protected": branch.get("protected"),
                "sha": branch.get("commit", {}).get("sha", "")[:7]
            })
        output = {
            "branches": branches,
            "page": page,
            "per_page": per_page,
            "count": len(branches),
            "has_more": len(branches) == per_page
        }
        return ToolResult(tool="list_branches", success=True, output=json.dumps(output, indent=2))
    else:
        return ToolResult(tool="list_branches", success=False, output="", error=result.get("error", "Failed to list branches"), error_code="api_error", suggestions=result.get("suggestions", []))

def execute_delete_branch(owner: str, repo: str, branch: str, token: str, default_branch: str = "main") -> ToolResult:
    """Delete a branch on GitHub with proper safeguards."""
    if not token:
        return ToolResult(tool="delete_branch", success=False, output="", error="GitHub token required", error_code="missing_token")
    
    protected_branches = ["main", "master", default_branch]
    if branch in protected_branches:
        return ToolResult(
            tool="delete_branch", 
            success=False, 
            output="", 
            error=f"Cannot delete protected branch '{branch}'. Protected branches: {', '.join(set(protected_branches))}", 
            error_code="protected_branch",
            suggestions=["Use a different branch name", "Check if this is the default branch for the repo"]
        )
    
    success, result = execute_github_api("DELETE", f"/repos/{owner}/{repo}/git/refs/heads/{branch}", token)
    
    if success:
        return ToolResult(tool="delete_branch", success=True, output=f"Branch '{branch}' deleted successfully")
    else:
        return ToolResult(tool="delete_branch", success=False, output="", error=result.get("error", "Failed to delete branch"), error_code="api_error", suggestions=result.get("suggestions", []))

# Feature 5: GitHub Actions Integration
def execute_ci_status(owner: str, repo: str, ref: str, token: str) -> ToolResult:
    """Get CI status for a ref (branch/commit)."""
    if not token:
        return ToolResult(tool="ci_status", success=False, output="", error="GitHub token required", error_code="missing_token")
    
    # Get check runs
    success, result = execute_github_api("GET", f"/repos/{owner}/{repo}/commits/{ref}/check-runs", token)
    
    if success:
        check_runs = []
        for run in result.get("check_runs", []):
            check_runs.append({
                "name": run.get("name"),
                "status": run.get("status"),
                "conclusion": run.get("conclusion"),
                "started_at": run.get("started_at"),
                "completed_at": run.get("completed_at"),
                "html_url": run.get("html_url")
            })
        
        # Also get workflow runs
        success2, result2 = execute_github_api("GET", f"/repos/{owner}/{repo}/actions/runs?head_sha={ref}&per_page=10", token)
        
        workflow_runs = []
        if success2:
            for run in result2.get("workflow_runs", []):
                workflow_runs.append({
                    "id": run.get("id"),
                    "name": run.get("name"),
                    "status": run.get("status"),
                    "conclusion": run.get("conclusion"),
                    "html_url": run.get("html_url"),
                    "created_at": run.get("created_at")
                })
        
        output = {
            "ref": ref,
            "check_runs": check_runs,
            "workflow_runs": workflow_runs,
            "total_checks": len(check_runs),
            "total_workflows": len(workflow_runs)
        }
        return ToolResult(tool="ci_status", success=True, output=json.dumps(output, indent=2))
    else:
        return ToolResult(tool="ci_status", success=False, output="", error=result.get("error", "Failed to get CI status"), error_code="api_error")

def execute_view_workflow_logs(owner: str, repo: str, run_id: int, token: str) -> ToolResult:
    """View workflow run logs."""
    if not token:
        return ToolResult(tool="view_workflow_logs", success=False, output="", error="GitHub token required", error_code="missing_token")
    
    # Get workflow run details
    success, result = execute_github_api("GET", f"/repos/{owner}/{repo}/actions/runs/{run_id}", token)
    
    if success:
        run_data = {
            "id": result.get("id"),
            "name": result.get("name"),
            "status": result.get("status"),
            "conclusion": result.get("conclusion"),
            "html_url": result.get("html_url"),
            "logs_url": result.get("logs_url"),
            "created_at": result.get("created_at"),
            "updated_at": result.get("updated_at"),
            "head_branch": result.get("head_branch"),
            "head_sha": result.get("head_sha")[:7] if result.get("head_sha") else None
        }
        
        # Get jobs for this run
        success2, result2 = execute_github_api("GET", f"/repos/{owner}/{repo}/actions/runs/{run_id}/jobs", token)
        
        if success2:
            jobs = []
            for job in result2.get("jobs", []):
                steps = []
                for step in job.get("steps", []):
                    steps.append({
                        "name": step.get("name"),
                        "status": step.get("status"),
                        "conclusion": step.get("conclusion"),
                        "number": step.get("number")
                    })
                jobs.append({
                    "id": job.get("id"),
                    "name": job.get("name"),
                    "status": job.get("status"),
                    "conclusion": job.get("conclusion"),
                    "steps": steps
                })
            run_data["jobs"] = jobs
        
        return ToolResult(tool="view_workflow_logs", success=True, output=json.dumps(run_data, indent=2))
    else:
        return ToolResult(tool="view_workflow_logs", success=False, output="", error=result.get("error", "Failed to get workflow logs"), error_code="api_error")

# Feature 6: Code Review Comments
def execute_add_pr_comment(owner: str, repo: str, pr_number: int, body: str, token: str) -> ToolResult:
    """Add a comment to a pull request."""
    if not token:
        return ToolResult(tool="add_pr_comment", success=False, output="", error="GitHub token required", error_code="missing_token")
    
    data = {"body": body}
    
    success, result = execute_github_api("POST", f"/repos/{owner}/{repo}/issues/{pr_number}/comments", token, data)
    
    if success:
        comment_data = {
            "id": result.get("id"),
            "html_url": result.get("html_url"),
            "body": result.get("body")[:200] + "..." if len(result.get("body", "")) > 200 else result.get("body")
        }
        return ToolResult(tool="add_pr_comment", success=True, output=json.dumps(comment_data, indent=2))
    else:
        return ToolResult(tool="add_pr_comment", success=False, output="", error=result.get("error", "Failed to add comment"), error_code="api_error")

def execute_add_review_comment(owner: str, repo: str, pr_number: int, body: str, commit_id: str, path: str, line: int, token: str, side: str = "RIGHT") -> ToolResult:
    """Add an inline review comment to a pull request.
    
    Args:
        side: Which side of the diff to comment on. 'LEFT' for old code, 'RIGHT' for new code (default).
    """
    if not token:
        return ToolResult(tool="add_review_comment", success=False, output="", error="GitHub token required", error_code="missing_token")
    
    if side not in ["LEFT", "RIGHT"]:
        side = "RIGHT"
    
    data = {
        "body": body,
        "commit_id": commit_id,
        "path": path,
        "line": line,
        "side": side
    }
    
    success, result = execute_github_api("POST", f"/repos/{owner}/{repo}/pulls/{pr_number}/comments", token, data)
    
    if success:
        comment_data = {
            "id": result.get("id"),
            "html_url": result.get("html_url"),
            "path": result.get("path"),
            "line": result.get("line"),
            "side": result.get("side")
        }
        return ToolResult(tool="add_review_comment", success=True, output=json.dumps(comment_data, indent=2))
    else:
        return ToolResult(tool="add_review_comment", success=False, output="", error=result.get("error", "Failed to add review comment"), error_code="api_error", suggestions=result.get("suggestions", []))

def execute_list_pr_comments(owner: str, repo: str, pr_number: int, token: str) -> ToolResult:
    """List comments on a pull request."""
    if not token:
        return ToolResult(tool="list_pr_comments", success=False, output="", error="GitHub token required", error_code="missing_token")
    
    # Get issue comments (general comments)
    success, result = execute_github_api("GET", f"/repos/{owner}/{repo}/issues/{pr_number}/comments", token)
    
    comments = []
    if success:
        for comment in result:
            comments.append({
                "id": comment.get("id"),
                "type": "general",
                "user": comment.get("user", {}).get("login"),
                "body": comment.get("body")[:200] + "..." if len(comment.get("body", "")) > 200 else comment.get("body"),
                "created_at": comment.get("created_at"),
                "html_url": comment.get("html_url")
            })
    
    # Get review comments (inline comments)
    success2, result2 = execute_github_api("GET", f"/repos/{owner}/{repo}/pulls/{pr_number}/comments", token)
    
    if success2:
        for comment in result2:
            comments.append({
                "id": comment.get("id"),
                "type": "review",
                "user": comment.get("user", {}).get("login"),
                "body": comment.get("body")[:200] + "..." if len(comment.get("body", "")) > 200 else comment.get("body"),
                "path": comment.get("path"),
                "line": comment.get("line"),
                "created_at": comment.get("created_at"),
                "html_url": comment.get("html_url")
            })
    
    return ToolResult(tool="list_pr_comments", success=True, output=json.dumps(comments, indent=2))

# Feature 7: Better Error Handling - Helper function
def execute_github_tool_with_session(tool_name: str, tool_func, session_id: str, **kwargs) -> ToolResult:
    """Execute a GitHub tool with automatic token retrieval from session."""
    repo_info = get_repo_info_for_session(session_id)
    
    if not repo_info:
        return ToolResult(
            tool=tool_name, 
            success=False, 
            output="", 
            error="No GitHub repository linked to this session. Use the web interface to add and link a repo first.",
            error_code="no_repo_linked",
            suggestions=["Add a GitHub repo via the web interface", "Link the repo to your session", "Provide repo details manually"]
        )
    
    if not repo_info.get("github_token"):
        return ToolResult(
            tool=tool_name,
            success=False,
            output="",
            error="No GitHub token configured for this repository. Add a token when adding the repo.",
            error_code="missing_token",
            suggestions=["Update the repo with a GitHub Personal Access Token", "Create a new token at github.com/settings/tokens"]
        )
    
    # Add repo info to kwargs
    kwargs["owner"] = repo_info["owner"]
    kwargs["repo"] = repo_info["repo_name"]
    kwargs["token"] = repo_info["github_token"]
    
    return tool_func(**kwargs)

def get_llm_client(provider: str):
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            import openai
            return openai.OpenAI(api_key=api_key)
    elif provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            try:
                import anthropic
                return anthropic.Anthropic(api_key=api_key)
            except ImportError:
                return None
    elif provider == "ollama":
        ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        return {"type": "ollama", "url": ollama_url}
    return None

async def call_llm(provider: str, model: str, messages: list, max_tokens: int = 2000) -> str:
    client = get_llm_client(provider)
    if not client:
        raise APIError(f"LLM provider '{provider}' not configured", "api_not_configured", 500)
    
    try:
        if provider == "openai":
            response = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, temperature=0.7)
            return response.choices[0].message.content
        elif provider == "anthropic":
            system_msg = ""
            user_msgs = []
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                else:
                    user_msgs.append(msg)
            response = client.messages.create(model=model, max_tokens=max_tokens, system=system_msg, messages=user_msgs)
            return response.content[0].text
        elif provider == "ollama":
            import httpx
            async with httpx.AsyncClient(timeout=120.0) as http_client:
                response = await http_client.post(f"{client['url']}/api/chat", json={"model": model, "messages": messages, "stream": False})
                if response.status_code == 200:
                    return response.json().get("message", {}).get("content", "")
                else:
                    raise APIError(f"Ollama error: {response.text}", "ollama_error", response.status_code)
        raise APIError(f"Unknown provider: {provider}", "unknown_provider", 400)
    except Exception as e:
        if isinstance(e, APIError):
            raise
        raise APIError(f"LLM call failed: {str(e)}", "api_error", 500, {"original_error": str(e)})

def get_session_memory(session_id: str) -> str:
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT content FROM agent_memory WHERE session_id=? ORDER BY last_accessed DESC LIMIT 10", (session_id,))
        rows = c.fetchall()
        conn.close()
        if rows:
            memories = []
            for row in rows:
                try:
                    data = json.loads(row[0])
                    memories.append(f"- {data.get('key', 'unknown')}: {data.get('value', '')}")
                except:
                    pass
            if memories:
                return "\n\n## Your Memory (from previous interactions)\n" + "\n".join(memories)
        return ""
    except:
        return ""

SYSTEM_PROMPT = """You are Mini-Devin, an autonomous AI software engineer. You MUST use tools to accomplish tasks - do not just describe what you would do.

## CRITICAL: You MUST output tool calls as JSON blocks

When you need to perform an action, you MUST output a JSON block like this:

```json
{"tool": "terminal", "command": "python3 --version"}
```

## Available Tools

### Basic Tools

1. **terminal** - Run shell commands (including git commit, push, etc.)
   ```json
   {"tool": "terminal", "command": "ls -la"}
   ```
   Use terminal for ALL git write operations: git add, git commit, git push, git checkout -b, etc.

2. **file_write** - Create or overwrite a file
   ```json
   {"tool": "file_write", "path": "/tmp/hello.py", "content": "print('Hello!')"}
   ```

3. **file_read** - Read file contents
   ```json
   {"tool": "file_read", "path": "/tmp/hello.py"}
   ```

4. **list_files** - List directory contents
   ```json
   {"tool": "list_files", "path": "/tmp"}
   ```

5. **git** - Run git commands (read-only: status, log, diff, branch, show)
   ```json
   {"tool": "git", "command": "status"}
   ```

6. **code_analysis** - Analyze code structure or complexity
   ```json
   {"tool": "code_analysis", "path": "/tmp/script.py", "analysis_type": "structure"}
   ```

7. **search_files** - Search for text in files
   ```json
   {"tool": "search_files", "path": "/tmp", "pattern": "TODO", "file_pattern": "*.py"}
   ```

8. **memory_store** - Store information for later recall
   ```json
   {"tool": "memory_store", "key": "user_preference", "value": "prefers Python"}
   ```

9. **memory_recall** - Recall stored information
   ```json
   {"tool": "memory_recall", "key": "user_preference"}
   ```

### GitHub Tools (automatically uses linked repo's token)

10. **create_github_repo** - Create a new GitHub repository
    ```json
    {"tool": "create_github_repo", "name": "my-repo", "description": "My project", "private": false, "github_token": "ghp_xxx"}
    ```

11. **create_pr** - Create a pull request
    ```json
    {"tool": "create_pr", "title": "Add feature", "head": "feature-branch", "base": "main", "body": "Description of changes"}
    ```

12. **list_prs** - List pull requests
    ```json
    {"tool": "list_prs", "state": "open"}
    ```
    state can be: "open", "closed", "all"

13. **view_pr** - View a specific pull request
    ```json
    {"tool": "view_pr", "pr_number": 1}
    ```

14. **merge_pr** - Merge a pull request
    ```json
    {"tool": "merge_pr", "pr_number": 1, "merge_method": "merge"}
    ```
    merge_method can be: "merge", "squash", "rebase"

15. **list_issues** - List issues
    ```json
    {"tool": "list_issues", "state": "open"}
    ```

16. **create_issue** - Create an issue
    ```json
    {"tool": "create_issue", "title": "Bug report", "body": "Description", "labels": ["bug"]}
    ```

17. **close_issue** - Close an issue
    ```json
    {"tool": "close_issue", "issue_number": 1}
    ```

18. **list_branches** - List branches on GitHub
    ```json
    {"tool": "list_branches"}
    ```

19. **delete_branch** - Delete a branch on GitHub
    ```json
    {"tool": "delete_branch", "branch": "feature-branch"}
    ```

20. **ci_status** - Get CI/GitHub Actions status
    ```json
    {"tool": "ci_status", "ref": "main"}
    ```
    ref can be a branch name or commit SHA

21. **view_workflow_logs** - View GitHub Actions workflow logs
    ```json
    {"tool": "view_workflow_logs", "run_id": 12345}
    ```

22. **add_pr_comment** - Add a comment to a PR
    ```json
    {"tool": "add_pr_comment", "pr_number": 1, "body": "LGTM!"}
    ```

23. **add_review_comment** - Add an inline review comment
    ```json
    {"tool": "add_review_comment", "pr_number": 1, "body": "Consider refactoring", "commit_id": "abc123", "path": "src/main.py", "line": 42}
    ```

24. **list_pr_comments** - List all comments on a PR
    ```json
    {"tool": "list_pr_comments", "pr_number": 1}
    ```

## Working Directory
You are working in a git repository. The remote 'origin' is ALREADY configured with authentication.
- Use `git remote -v` to see the remote URL
- You can directly push with `git push origin <branch>` - NO need to add remote again
- Create branches with `git checkout -b <branch-name>`
- Commit with `git add . && git commit -m "message"`
- Push with `git push -u origin <branch-name>`

## Git Workflow for PRs
1. First check current status: `git status` and `git remote -v`
2. Create a new branch: `git checkout -b feature/my-feature`
3. Make changes using file_write
4. Stage and commit: `git add . && git commit -m "Add feature"`
5. Push to remote: `git push -u origin feature/my-feature`
6. Create PR using: `{"tool": "create_pr", "title": "My Feature", "head": "feature/my-feature", "base": "main", "body": "Description"}`
7. Check CI status: `{"tool": "ci_status", "ref": "feature/my-feature"}`
8. Merge when ready: `{"tool": "merge_pr", "pr_number": 1}`

## Error Handling
If a tool fails, you'll receive an error message with suggestions. Use these to fix the issue and try again.

## Instructions
1. ALWAYS use tools - never just describe what you would do
2. After each tool execution, you'll see the results
3. If a tool fails, read the error and suggestions, then try a different approach
4. Continue using tools until the task is complete
5. For git operations, ALWAYS use terminal tool (not the read-only git tool)
6. The git remote is ALREADY configured - do NOT try to add a new remote
7. Use GitHub tools (create_pr, merge_pr, etc.) for GitHub API operations
8. Provide a summary when done

REMEMBER: Always output the JSON tool block, never just describe what you would do!"""

def parse_tool_calls(response: str) -> list:
    tools = []
    pattern = r'```json\s*(\{[^`]+\})\s*```'
    matches = re.findall(pattern, response, re.DOTALL)
    for match in matches:
        try:
            tool_call = json.loads(match.strip())
            if "tool" in tool_call:
                tools.append(tool_call)
        except json.JSONDecodeError:
            pass
    
    if not tools:
        pattern2 = r'\{[^{}]*"tool"\s*:\s*"[^"]+\"[^{}]*\}'
        matches2 = re.findall(pattern2, response)
        for match in matches2:
            try:
                tool_call = json.loads(match)
                if "tool" in tool_call:
                    tools.append(tool_call)
            except:
                pass
    
    return tools

def execute_tool(tool_call: dict, session_id: str = "", default_working_dir: str = "/tmp") -> ToolResult:
    tool = tool_call.get("tool")
    try:
        if tool == "terminal":
            return execute_terminal(tool_call.get("command", ""), tool_call.get("working_dir", default_working_dir))
        elif tool == "file_read":
            return execute_file_read(tool_call.get("path", ""))
        elif tool == "file_write":
            return execute_file_write(tool_call.get("path", ""), tool_call.get("content", ""))
        elif tool == "list_files":
            return execute_list_files(tool_call.get("path", "/tmp"))
        elif tool == "git":
            return execute_git(tool_call.get("command", ""), tool_call.get("working_dir", "/tmp"))
        elif tool == "code_analysis":
            return execute_code_analysis(tool_call.get("path", ""), tool_call.get("analysis_type", "structure"))
        elif tool == "search_files":
            return execute_search_files(tool_call.get("path", "/tmp"), tool_call.get("pattern", ""), tool_call.get("file_pattern", "*"))
        elif tool == "memory_store":
            return execute_memory_store(tool_call.get("key", ""), tool_call.get("value", ""), session_id)
        elif tool == "memory_recall":
            return execute_memory_recall(tool_call.get("key", ""), session_id)
        elif tool == "create_github_repo":
            return execute_create_github_repo(
                name=tool_call.get("name", ""),
                description=tool_call.get("description", ""),
                private=tool_call.get("private", False),
                github_token=tool_call.get("github_token", ""),
                auto_init=tool_call.get("auto_init", True)
            )
        # Feature 1: PR Creation
        elif tool == "create_pr":
            repo_info = get_repo_info_for_session(session_id)
            if not repo_info:
                return ToolResult(tool="create_pr", success=False, output="", error="No repo linked to session", error_code="no_repo_linked", suggestions=["Link a GitHub repo to this session first"])
            return execute_create_pr(
                owner=tool_call.get("owner", repo_info.get("owner", "")),
                repo=tool_call.get("repo", repo_info.get("repo_name", "")),
                title=tool_call.get("title", ""),
                head=tool_call.get("head", ""),
                base=tool_call.get("base", repo_info.get("default_branch", "main")),
                body=tool_call.get("body", ""),
                token=tool_call.get("token", repo_info.get("github_token", ""))
            )
        # Feature 2: PR Review/Merge
        elif tool == "list_prs":
            repo_info = get_repo_info_for_session(session_id)
            if not repo_info:
                return ToolResult(tool="list_prs", success=False, output="", error="No repo linked to session", error_code="no_repo_linked")
            return execute_list_prs(
                owner=tool_call.get("owner", repo_info.get("owner", "")),
                repo=tool_call.get("repo", repo_info.get("repo_name", "")),
                state=tool_call.get("state", "open"),
                token=repo_info.get("github_token", ""),
                page=tool_call.get("page", 1),
                per_page=tool_call.get("per_page", 30)
            )
        elif tool == "view_pr":
            repo_info = get_repo_info_for_session(session_id)
            if not repo_info:
                return ToolResult(tool="view_pr", success=False, output="", error="No repo linked to session", error_code="no_repo_linked")
            return execute_view_pr(
                owner=tool_call.get("owner", repo_info.get("owner", "")),
                repo=tool_call.get("repo", repo_info.get("repo_name", "")),
                pr_number=tool_call.get("pr_number", 0),
                token=repo_info.get("github_token", "")
            )
        elif tool == "merge_pr":
            repo_info = get_repo_info_for_session(session_id)
            if not repo_info:
                return ToolResult(tool="merge_pr", success=False, output="", error="No repo linked to session", error_code="no_repo_linked")
            return execute_merge_pr(
                owner=tool_call.get("owner", repo_info.get("owner", "")),
                repo=tool_call.get("repo", repo_info.get("repo_name", "")),
                pr_number=tool_call.get("pr_number", 0),
                merge_method=tool_call.get("merge_method", "merge"),
                token=repo_info.get("github_token", "")
            )
        # Feature 3: Issue Management
        elif tool == "list_issues":
            repo_info = get_repo_info_for_session(session_id)
            if not repo_info:
                return ToolResult(tool="list_issues", success=False, output="", error="No repo linked to session", error_code="no_repo_linked")
            return execute_list_issues(
                owner=tool_call.get("owner", repo_info.get("owner", "")),
                repo=tool_call.get("repo", repo_info.get("repo_name", "")),
                state=tool_call.get("state", "open"),
                token=repo_info.get("github_token", ""),
                page=tool_call.get("page", 1),
                per_page=tool_call.get("per_page", 30)
            )
        elif tool == "create_issue":
            repo_info = get_repo_info_for_session(session_id)
            if not repo_info:
                return ToolResult(tool="create_issue", success=False, output="", error="No repo linked to session", error_code="no_repo_linked")
            return execute_create_issue(
                owner=tool_call.get("owner", repo_info.get("owner", "")),
                repo=tool_call.get("repo", repo_info.get("repo_name", "")),
                title=tool_call.get("title", ""),
                body=tool_call.get("body", ""),
                labels=tool_call.get("labels", []),
                token=repo_info.get("github_token", "")
            )
        elif tool == "close_issue":
            repo_info = get_repo_info_for_session(session_id)
            if not repo_info:
                return ToolResult(tool="close_issue", success=False, output="", error="No repo linked to session", error_code="no_repo_linked")
            return execute_close_issue(
                owner=tool_call.get("owner", repo_info.get("owner", "")),
                repo=tool_call.get("repo", repo_info.get("repo_name", "")),
                issue_number=tool_call.get("issue_number", 0),
                token=repo_info.get("github_token", "")
            )
        # Feature 4: Branch Management
        elif tool == "list_branches":
            repo_info = get_repo_info_for_session(session_id)
            if not repo_info:
                return ToolResult(tool="list_branches", success=False, output="", error="No repo linked to session", error_code="no_repo_linked")
            return execute_list_branches(
                owner=tool_call.get("owner", repo_info.get("owner", "")),
                repo=tool_call.get("repo", repo_info.get("repo_name", "")),
                token=repo_info.get("github_token", ""),
                page=tool_call.get("page", 1),
                per_page=tool_call.get("per_page", 30)
            )
        elif tool == "delete_branch":
            repo_info = get_repo_info_for_session(session_id)
            if not repo_info:
                return ToolResult(tool="delete_branch", success=False, output="", error="No repo linked to session", error_code="no_repo_linked")
            return execute_delete_branch(
                owner=tool_call.get("owner", repo_info.get("owner", "")),
                repo=tool_call.get("repo", repo_info.get("repo_name", "")),
                branch=tool_call.get("branch", ""),
                token=repo_info.get("github_token", ""),
                default_branch=repo_info.get("default_branch", "main")
            )
        # Feature 5: GitHub Actions/CI
        elif tool == "ci_status":
            repo_info = get_repo_info_for_session(session_id)
            if not repo_info:
                return ToolResult(tool="ci_status", success=False, output="", error="No repo linked to session", error_code="no_repo_linked")
            return execute_ci_status(
                owner=tool_call.get("owner", repo_info.get("owner", "")),
                repo=tool_call.get("repo", repo_info.get("repo_name", "")),
                ref=tool_call.get("ref", "main"),
                token=repo_info.get("github_token", "")
            )
        elif tool == "view_workflow_logs":
            repo_info = get_repo_info_for_session(session_id)
            if not repo_info:
                return ToolResult(tool="view_workflow_logs", success=False, output="", error="No repo linked to session", error_code="no_repo_linked")
            return execute_view_workflow_logs(
                owner=tool_call.get("owner", repo_info.get("owner", "")),
                repo=tool_call.get("repo", repo_info.get("repo_name", "")),
                run_id=tool_call.get("run_id", 0),
                token=repo_info.get("github_token", "")
            )
        # Feature 6: PR Comments
        elif tool == "add_pr_comment":
            repo_info = get_repo_info_for_session(session_id)
            if not repo_info:
                return ToolResult(tool="add_pr_comment", success=False, output="", error="No repo linked to session", error_code="no_repo_linked")
            return execute_add_pr_comment(
                owner=tool_call.get("owner", repo_info.get("owner", "")),
                repo=tool_call.get("repo", repo_info.get("repo_name", "")),
                pr_number=tool_call.get("pr_number", 0),
                body=tool_call.get("body", ""),
                token=repo_info.get("github_token", "")
            )
        elif tool == "add_review_comment":
            repo_info = get_repo_info_for_session(session_id)
            if not repo_info:
                return ToolResult(tool="add_review_comment", success=False, output="", error="No repo linked to session", error_code="no_repo_linked")
            return execute_add_review_comment(
                owner=tool_call.get("owner", repo_info.get("owner", "")),
                repo=tool_call.get("repo", repo_info.get("repo_name", "")),
                pr_number=tool_call.get("pr_number", 0),
                body=tool_call.get("body", ""),
                commit_id=tool_call.get("commit_id", ""),
                path=tool_call.get("path", ""),
                line=tool_call.get("line", 0),
                token=repo_info.get("github_token", ""),
                side=tool_call.get("side", "RIGHT")
            )
        elif tool == "list_pr_comments":
            repo_info = get_repo_info_for_session(session_id)
            if not repo_info:
                return ToolResult(tool="list_pr_comments", success=False, output="", error="No repo linked to session", error_code="no_repo_linked")
            return execute_list_pr_comments(
                owner=tool_call.get("owner", repo_info.get("owner", "")),
                repo=tool_call.get("repo", repo_info.get("repo_name", "")),
                pr_number=tool_call.get("pr_number", 0),
                token=repo_info.get("github_token", "")
            )
        else:
            return ToolResult(tool=str(tool), success=False, output="", error=f"Unknown tool: {tool}", error_code="unknown_tool", suggestions=get_error_suggestions("unknown_tool"))
    except Exception as e:
        return ToolResult(tool=str(tool), success=False, output="", error=f"Tool execution error: {str(e)}", error_code="execution_error", suggestions=["Check tool parameters", "Try again"])

async def broadcast_to_session(session_id: str, message: dict):
    if session_id in active_websockets:
        dead_connections = []
        for ws in active_websockets[session_id]:
            try:
                await ws.send_json(message)
            except:
                dead_connections.append(ws)
        for ws in dead_connections:
            active_websockets[session_id].remove(ws)

async def execute_agent_task(session_id: str, task_id: str, description: str, model: str, provider: str = "openai", max_iterations: int = 10):
    conn = get_db()
    c = conn.cursor()
    
    # Get session's working directory
    c.execute("SELECT working_dir FROM sessions WHERE session_id=?", (session_id,))
    session_row = c.fetchone()
    working_dir = session_row[0] if session_row and session_row[0] else "/tmp"
    
    llm_client = get_llm_client(provider)
    if not llm_client:
        error_msg = f"LLM provider '{provider}' not configured"
        error_code = "api_not_configured"
        suggestions = get_error_suggestions("api_error")
        c.execute("UPDATE tasks SET status=?, result=?, error_message=?, error_code=? WHERE task_id=?", ("failed", error_msg, error_msg, error_code, task_id))
        c.execute("INSERT INTO task_outputs (task_id, output_type, content, created_at) VALUES (?, ?, ?, ?)", (task_id, "error", json.dumps({"message": error_msg, "code": error_code, "suggestions": suggestions}), datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()
        await broadcast_to_session(session_id, {"type": "task_failed", "task_id": task_id, "error": error_msg, "error_code": error_code, "suggestions": suggestions})
        return
    
    c.execute("UPDATE tasks SET status=?, started_at=? WHERE task_id=?", ("running", datetime.utcnow().isoformat(), task_id))
    conn.commit()
    
    await broadcast_to_session(session_id, {"type": "task_started", "task_id": task_id})
    
    c.execute("INSERT INTO task_outputs (task_id, output_type, content, created_at) VALUES (?, ?, ?, ?)", (task_id, "thinking", f"Analyzing task: {description}", datetime.utcnow().isoformat()))
    conn.commit()
    await broadcast_to_session(session_id, {"type": "thinking", "task_id": task_id, "content": f"Analyzing task: {description}"})
    
    memory_context = get_session_memory(session_id)
    system_prompt = SYSTEM_PROMPT + memory_context
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": description}]
    iteration = 0
    final_response = ""
    
    try:
        while iteration < max_iterations:
            iteration += 1
            
            c.execute("UPDATE sessions SET iteration=? WHERE session_id=?", (iteration, session_id))
            conn.commit()
            await broadcast_to_session(session_id, {"type": "iteration", "task_id": task_id, "iteration": iteration, "max": max_iterations})
            
            try:
                agent_response = await call_llm(provider, model, messages, max_tokens=2000)
            except APIError as e:
                c.execute("UPDATE tasks SET status=?, result=?, error_message=?, error_code=? WHERE task_id=?", ("failed", e.message, e.message, e.code, task_id))
                c.execute("INSERT INTO task_outputs (task_id, output_type, content, created_at) VALUES (?, ?, ?, ?)", (task_id, "error", json.dumps({"message": e.message, "code": e.code, "suggestions": get_error_suggestions(e.code)}), datetime.utcnow().isoformat()))
                conn.commit()
                await broadcast_to_session(session_id, {"type": "task_failed", "task_id": task_id, "error": e.message, "error_code": e.code, "suggestions": get_error_suggestions(e.code)})
                conn.close()
                return
            
            final_response = agent_response
            
            c.execute("INSERT INTO task_outputs (task_id, output_type, content, created_at) VALUES (?, ?, ?, ?)", (task_id, "response", agent_response, datetime.utcnow().isoformat()))
            conn.commit()
            await broadcast_to_session(session_id, {"type": "response", "task_id": task_id, "content": agent_response, "iteration": iteration})
            
            tool_calls = parse_tool_calls(agent_response)
            
            if not tool_calls:
                break
            
            tool_results = []
            for tool_call in tool_calls:
                await broadcast_to_session(session_id, {"type": "tool_started", "task_id": task_id, "tool": tool_call})
                
                result = execute_tool(tool_call, session_id, working_dir)
                tool_results.append(result)
                
                tool_output = {"tool": result.tool, "success": result.success, "output": result.output[:5000], "error": result.error, "error_code": result.error_code, "suggestions": result.suggestions}
                c.execute("INSERT INTO task_outputs (task_id, output_type, content, created_at) VALUES (?, ?, ?, ?)", (task_id, "tool", json.dumps(tool_output), datetime.utcnow().isoformat()))
                conn.commit()
                await broadcast_to_session(session_id, {"type": "tool_result", "task_id": task_id, "result": tool_output})
            
            tool_output_text = "\n\n".join([
                f"**Tool: {r.tool}**\nSuccess: {r.success}\n```\n{r.output}\n```\n{f'Error: {r.error}' if r.error else ''}{f' (Suggestions: {r.suggestions})' if r.suggestions else ''}"
                for r in tool_results
            ])
            
            messages.append({"role": "assistant", "content": agent_response})
            messages.append({"role": "user", "content": f"Tool execution results:\n\n{tool_output_text}\n\nContinue with the task. If you need to use more tools, output the JSON block. If the task is complete, provide a summary."})
        
        c.execute("UPDATE tasks SET status=?, completed_at=?, result=? WHERE task_id=?", ("completed", datetime.utcnow().isoformat(), final_response, task_id))
        conn.commit()
        await broadcast_to_session(session_id, {"type": "task_completed", "task_id": task_id, "result": final_response})
        
    except Exception as e:
        error_msg = str(e)
        error_code = "unknown_error"
        tb = traceback.format_exc()
        c.execute("UPDATE tasks SET status=?, result=?, error_message=?, error_code=? WHERE task_id=?", ("failed", f"Error: {error_msg}", error_msg, error_code, task_id))
        c.execute("INSERT INTO task_outputs (task_id, output_type, content, created_at) VALUES (?, ?, ?, ?)", (task_id, "error", json.dumps({"message": error_msg, "code": error_code, "traceback": tb[:1000]}), datetime.utcnow().isoformat()))
        conn.commit()
        await broadcast_to_session(session_id, {"type": "task_failed", "task_id": task_id, "error": error_msg, "error_code": error_code})
    
    conn.close()

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Phase 45: Use structured logging for startup
    logger.info("Starting Mini-Devin API v5.4", extra={"version": "5.4.0"})
    logger.info("Features: Tool Execution, SQLite Storage, WebSocket, Multi-Model, File Upload, Agent Memory, Export, Background Task Queue, Monitoring")
    
    init_db()
    logger.info(f"SQLite database initialized", extra={"db_path": DB_PATH})
    logger.info(f"Directories configured", extra={
        "workspace": WORKSPACE_DIR,
        "uploads": UPLOADS_DIR,
        "memory": MEMORY_DIR
    })
    
    # Phase 45: Log monitoring configuration
    logger.info("Monitoring configured", extra={
        "log_level": LOG_LEVEL,
        "log_format": LOG_FORMAT,
        "sentry_enabled": bool(SENTRY_DSN),
        "metrics_enabled": ENABLE_METRICS
    })
    
    # Phase 44: Start background task queue
    queue = get_task_queue()
    await queue.start()
    logger.info("Background task queue started", extra={
        "workers": TASK_QUEUE_MAX_WORKERS,
        "max_retries": TASK_QUEUE_MAX_RETRIES,
        "retry_delay": TASK_QUEUE_RETRY_DELAY,
        "retry_backoff": TASK_QUEUE_RETRY_BACKOFF
    })
    
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    ollama_url = os.environ.get("OLLAMA_URL")
    
    providers = []
    if openai_key:
        providers.append("openai")
        logger.info("OpenAI API key configured")
    if anthropic_key:
        providers.append("anthropic")
        logger.info("Anthropic API key configured")
    if ollama_url:
        providers.append("ollama")
        logger.info(f"Ollama URL configured", extra={"ollama_url": ollama_url})
    
    if not providers:
        logger.warning("No LLM providers configured - set OPENAI_API_KEY, ANTHROPIC_API_KEY, or OLLAMA_URL")
    else:
        logger.info(f"Available providers: {', '.join(providers)}", extra={"providers": providers})
    
    yield
    
    # Phase 44: Stop background task queue gracefully
    logger.info("Stopping background task queue...")
    await queue.stop()
    logger.info("Shutting down Mini-Devin API...")

app = FastAPI(title="Mini-Devin API", version="5.4.0", description="Autonomous AI Software Engineer with Multi-Model Support, File Upload, Agent Memory, Export, and Monitoring", lifespan=lifespan)

# Phase 45: Add request logging middleware (before CORS)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def get_configured_providers():
    providers = []
    if os.environ.get("OPENAI_API_KEY"):
        providers.append({"id": "openai", "name": "OpenAI", "models": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]})
    if os.environ.get("ANTHROPIC_API_KEY"):
        providers.append({"id": "anthropic", "name": "Anthropic", "models": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"]})
    if os.environ.get("OLLAMA_URL"):
        providers.append({"id": "ollama", "name": "Ollama (Local)", "models": ["llama3", "codellama", "mistral"]})
    return providers

@app.get("/")
async def root():
    providers = get_configured_providers()
    return {
        "name": "Mini-Devin API",
        "version": "5.0.0",
        "status": "running",
        "mode": "full-agent" if providers else "limited",
        "llm_configured": bool(providers),
        "providers": [p["id"] for p in providers],
        "features": ["tool_execution", "persistent_storage", "websocket", "multi_model", "file_upload", "agent_memory", "export", "error_handling"],
        "tools": ["terminal", "file_read", "file_write", "list_files", "git", "code_analysis", "search_files", "memory_store", "memory_recall"],
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "5.0.0"}

@app.get("/api/health")
async def api_health():
    providers = get_configured_providers()
    return {"status": "healthy", "mode": "full-agent" if providers else "limited", "llm_configured": bool(providers), "providers": [p["id"] for p in providers], "version": "5.1.0"}

# ============================================================================
# Phase 42: Authentication Endpoints
# ============================================================================

@app.post("/api/auth/register")
async def register(request: Request, data: RegisterRequest):
    """Register a new user account."""
    # Rate limit by IP
    client_ip = get_client_ip(request)
    allowed, rate_info = check_rate_limit(f"register:{client_ip}")
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Too many registration attempts. Try again in {rate_info['retry_after']} seconds.",
            headers={"Retry-After": str(rate_info["retry_after"])}
        )
    
    conn = get_db()
    try:
        c = conn.cursor()
        
        # Check if username exists
        c.execute("SELECT user_id FROM users WHERE username = ?", (data.username,))
        if c.fetchone():
            raise HTTPException(status_code=400, detail="Username already exists")
        
        # Check if email exists (if provided)
        if data.email:
            c.execute("SELECT user_id FROM users WHERE email = ?", (data.email,))
            if c.fetchone():
                raise HTTPException(status_code=400, detail="Email already registered")
        
        # Create user
        user_id = str(uuid.uuid4())
        password_hash = hash_password(data.password)
        now = datetime.utcnow().isoformat()
        
        c.execute("""INSERT INTO users (user_id, username, email, password_hash, created_at, role)
                     VALUES (?, ?, ?, ?, ?, ?)""",
                  (user_id, data.username, data.email, password_hash, now, "user"))
        conn.commit()
        
        # Create token
        token_data = create_jwt_token(user_id, data.username, "user")
        
        # Store session
        c.execute("""INSERT INTO user_sessions (token_id, user_id, created_at, expires_at, last_activity, ip_address, user_agent)
                     VALUES (?, ?, ?, ?, ?, ?, ?)""",
                  (token_data["token_id"], user_id, now, token_data["expires_at"], now,
                   client_ip, request.headers.get("User-Agent", "unknown")))
        conn.commit()
        
        return {
            "success": True,
            "message": "Registration successful",
            "user": {"user_id": user_id, "username": data.username, "role": "user"},
            "token": token_data["token"],
            "expires_at": token_data["expires_at"],
            "expires_in": token_data["expires_in"]
        }
    finally:
        conn.close()

@app.post("/api/auth/login")
async def login(request: Request, data: LoginRequest):
    """Login and get JWT token."""
    # Rate limit by IP
    client_ip = get_client_ip(request)
    allowed, rate_info = check_rate_limit(f"login:{client_ip}")
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Too many login attempts. Try again in {rate_info['retry_after']} seconds.",
            headers={"Retry-After": str(rate_info["retry_after"])}
        )
    
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("SELECT user_id, username, password_hash, role, is_active FROM users WHERE username = ?",
                  (data.username,))
        user = c.fetchone()
        
        if not user:
            raise HTTPException(status_code=401, detail="Invalid username or password")
        
        if not user["is_active"]:
            raise HTTPException(status_code=403, detail="Account is disabled")
        
        if not verify_password(data.password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid username or password")
        
        # Update last login
        now = datetime.utcnow().isoformat()
        c.execute("UPDATE users SET last_login = ? WHERE user_id = ?", (now, user["user_id"]))
        
        # Create token
        token_data = create_jwt_token(user["user_id"], user["username"], user["role"])
        
        # Store session
        c.execute("""INSERT INTO user_sessions (token_id, user_id, created_at, expires_at, last_activity, ip_address, user_agent)
                     VALUES (?, ?, ?, ?, ?, ?, ?)""",
                  (token_data["token_id"], user["user_id"], now, token_data["expires_at"], now,
                   client_ip, request.headers.get("User-Agent", "unknown")))
        conn.commit()
        
        return {
            "success": True,
            "message": "Login successful",
            "user": {"user_id": user["user_id"], "username": user["username"], "role": user["role"]},
            "token": token_data["token"],
            "expires_at": token_data["expires_at"],
            "expires_in": token_data["expires_in"]
        }
    finally:
        conn.close()

@app.post("/api/auth/logout")
async def logout(user: dict = Depends(require_auth)):
    """Logout and revoke current token."""
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("UPDATE user_sessions SET is_revoked = 1 WHERE token_id = ?", (user.get("token_id"),))
        conn.commit()
        return {"success": True, "message": "Logged out successfully"}
    finally:
        conn.close()

@app.post("/api/auth/logout-all")
async def logout_all(user: dict = Depends(require_auth)):
    """Logout from all devices by revoking all tokens."""
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("UPDATE user_sessions SET is_revoked = 1 WHERE user_id = ?", (user.get("user_id"),))
        conn.commit()
        return {"success": True, "message": "Logged out from all devices"}
    finally:
        conn.close()

@app.post("/api/auth/refresh")
async def refresh_token(request: Request, user: dict = Depends(require_auth)):
    """Refresh JWT token before it expires."""
    client_ip = get_client_ip(request)
    
    conn = get_db()
    try:
        c = conn.cursor()
        
        # Revoke old token
        c.execute("UPDATE user_sessions SET is_revoked = 1 WHERE token_id = ?", (user.get("token_id"),))
        
        # Create new token
        token_data = create_jwt_token(user["user_id"], user["username"], user.get("role", "user"))
        
        # Store new session
        now = datetime.utcnow().isoformat()
        c.execute("""INSERT INTO user_sessions (token_id, user_id, created_at, expires_at, last_activity, ip_address, user_agent)
                     VALUES (?, ?, ?, ?, ?, ?, ?)""",
                  (token_data["token_id"], user["user_id"], now, token_data["expires_at"], now,
                   client_ip, request.headers.get("User-Agent", "unknown")))
        conn.commit()
        
        return {
            "success": True,
            "message": "Token refreshed",
            "token": token_data["token"],
            "expires_at": token_data["expires_at"],
            "expires_in": token_data["expires_in"]
        }
    finally:
        conn.close()

@app.get("/api/auth/me")
async def get_current_user_info(user: dict = Depends(require_auth)):
    """Get current user information."""
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("SELECT user_id, username, email, created_at, last_login, role FROM users WHERE user_id = ?",
                  (user["user_id"],))
        user_data = c.fetchone()
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "user_id": user_data["user_id"],
            "username": user_data["username"],
            "email": user_data["email"],
            "created_at": user_data["created_at"],
            "last_login": user_data["last_login"],
            "role": user_data["role"]
        }
    finally:
        conn.close()

@app.post("/api/auth/change-password")
async def change_password(data: ChangePasswordRequest, user: dict = Depends(require_auth)):
    """Change user password."""
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("SELECT password_hash FROM users WHERE user_id = ?", (user["user_id"],))
        row = c.fetchone()
        
        if not row or not verify_password(data.current_password, row["password_hash"]):
            raise HTTPException(status_code=401, detail="Current password is incorrect")
        
        # Update password
        new_hash = hash_password(data.new_password)
        c.execute("UPDATE users SET password_hash = ? WHERE user_id = ?", (new_hash, user["user_id"]))
        
        # Revoke all tokens except current
        c.execute("UPDATE user_sessions SET is_revoked = 1 WHERE user_id = ? AND token_id != ?",
                  (user["user_id"], user.get("token_id")))
        conn.commit()
        
        return {"success": True, "message": "Password changed successfully"}
    finally:
        conn.close()

@app.get("/api/auth/sessions")
async def list_user_sessions(user: dict = Depends(require_auth)):
    """List all active sessions for current user."""
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("""SELECT token_id, created_at, expires_at, last_activity, ip_address, user_agent
                     FROM user_sessions WHERE user_id = ? AND is_revoked = 0
                     ORDER BY last_activity DESC""", (user["user_id"],))
        sessions = []
        for row in c.fetchall():
            sessions.append({
                "token_id": row["token_id"][:8] + "...",  # Partial ID for security
                "created_at": row["created_at"],
                "expires_at": row["expires_at"],
                "last_activity": row["last_activity"],
                "ip_address": row["ip_address"],
                "user_agent": row["user_agent"],
                "is_current": row["token_id"] == user.get("token_id")
            })
        return {"sessions": sessions, "count": len(sessions)}
    finally:
        conn.close()

@app.get("/api/rate-limit")
async def get_rate_limit_status(request: Request):
    """Get current rate limit status for the client."""
    client_ip = get_client_ip(request)
    _, rate_info = check_rate_limit(f"api:{client_ip}")
    # Don't count this request
    rate_limit_storage[f"api:{client_ip}"].pop()
    rate_info["remaining"] += 1
    return rate_info

@app.get("/api/db-pool")
async def get_db_pool_status():
    """Get database connection pool statistics.
    
    Returns pool configuration and current usage metrics.
    Useful for monitoring database connection health.
    """
    return get_db_pool_stats()

# ============================================================================
# Phase 45: Logging & Monitoring Endpoints
# ============================================================================

@app.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    """Export Prometheus metrics in text format.
    
    Returns all collected metrics in Prometheus exposition format.
    Scrape this endpoint with Prometheus for monitoring.
    """
    if not metrics:
        return PlainTextResponse("# Metrics disabled\n", status_code=200)
    
    # Add system metrics
    queue = get_task_queue()
    queue_stats = queue.get_stats()
    metrics.set_gauge("task_queue_pending", queue_stats["queue_size"])
    metrics.set_gauge("task_queue_active", queue_stats["active_tasks"])
    metrics.set_gauge("task_queue_completed", queue_stats["completed_tasks"])
    metrics.set_gauge("task_queue_failed", queue_stats["failed_tasks"])
    metrics.set_gauge("task_queue_workers", queue_stats["max_workers"])
    
    # Add DB pool metrics
    pool_stats = get_db_pool_stats()
    metrics.set_gauge("db_pool_size", pool_stats["pool_size"])
    metrics.set_gauge("db_pool_checked_out", pool_stats["checked_out"])
    metrics.set_gauge("db_pool_overflow", pool_stats["overflow"])
    
    return PlainTextResponse(metrics.export(), media_type="text/plain")

@app.get("/api/logs/config")
async def get_logging_config():
    """Get current logging configuration.
    
    Returns log level, format, and monitoring status.
    """
    return {
        "log_level": LOG_LEVEL,
        "log_format": LOG_FORMAT,
        "sentry_enabled": bool(SENTRY_DSN),
        "metrics_enabled": ENABLE_METRICS,
        "environment": os.environ.get("ENVIRONMENT", "production"),
        "version": "5.4.0"
    }

@app.get("/api/metrics/summary")
async def get_metrics_summary():
    """Get a JSON summary of key metrics.
    
    Returns aggregated metrics in JSON format for dashboards.
    """
    if not metrics:
        return {"enabled": False, "message": "Metrics disabled"}
    
    # Calculate request stats
    total_requests = sum(v for k, v in metrics.counters.items() if k.startswith("http_requests_total"))
    error_count = sum(v for k, v in metrics.counters.items() if "errors_total" in k)
    
    # Calculate latency stats
    all_latencies = []
    for k, v in metrics.histograms.items():
        if "http_request_duration" in k:
            all_latencies.extend(v)
    
    avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0
    p95_latency = sorted(all_latencies)[int(len(all_latencies) * 0.95)] if all_latencies else 0
    p99_latency = sorted(all_latencies)[int(len(all_latencies) * 0.99)] if all_latencies else 0
    
    # Get queue stats
    queue = get_task_queue()
    queue_stats = queue.get_stats()
    
    return {
        "enabled": True,
        "requests": {
            "total": total_requests,
            "errors": error_count,
            "error_rate": error_count / total_requests if total_requests > 0 else 0
        },
        "latency": {
            "avg_ms": round(avg_latency * 1000, 2),
            "p95_ms": round(p95_latency * 1000, 2),
            "p99_ms": round(p99_latency * 1000, 2),
            "samples": len(all_latencies)
        },
        "task_queue": {
            "pending": queue_stats["queue_size"],
            "active": queue_stats["active_tasks"],
            "completed": queue_stats["completed_tasks"],
            "failed": queue_stats["failed_tasks"]
        },
        "db_pool": get_db_pool_stats()
    }

# ============================================================================
# Phase 44: Task Queue Management Endpoints
# ============================================================================

@app.get("/api/task-queue")
async def get_task_queue_status():
    """Get background task queue statistics and configuration.
    
    Returns queue status, worker count, pending/active/completed/failed task counts,
    and retry configuration.
    """
    queue = get_task_queue()
    return queue.get_stats()

@app.get("/api/task-queue/task/{task_id}")
async def get_queued_task_info(task_id: str):
    """Get detailed information about a specific task in the queue.
    
    Returns task status, retry count, errors, and timing information.
    """
    queue = get_task_queue()
    info = queue.get_task_info(task_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Task not found in queue")
    return info

@app.post("/api/task-queue/task/{task_id}/cancel")
async def cancel_queued_task(task_id: str):
    """Cancel a pending or running task.
    
    Marks the task as cancelled. Running tasks will be stopped at the next checkpoint.
    """
    queue = get_task_queue()
    success = await queue.cancel(task_id)
    if success:
        return {"status": "cancelled", "task_id": task_id}
    raise HTTPException(status_code=400, detail="Could not cancel task")

@app.post("/api/task-queue/task/{task_id}/retry")
async def retry_failed_task(task_id: str):
    """Manually retry a failed task.
    
    Re-queues a failed task for execution. Only works for tasks that have failed.
    """
    queue = get_task_queue()
    success = await queue.retry(task_id)
    if success:
        return {"status": "requeued", "task_id": task_id}
    raise HTTPException(status_code=400, detail="Task not found in failed tasks or cannot be retried")

@app.get("/api/task-queue/failed")
async def list_failed_tasks():
    """List all failed tasks that can be retried.
    
    Returns details about each failed task including error messages and retry counts.
    """
    queue = get_task_queue()
    return {
        "count": len(queue.failed_tasks),
        "tasks": list(queue.failed_tasks.items())
    }

@app.post("/api/task-queue/retry-all")
async def retry_all_failed_tasks():
    """Retry all failed tasks.
    
    Re-queues all failed tasks for execution.
    """
    queue = get_task_queue()
    retried = []
    for task_id in list(queue.failed_tasks.keys()):
        if await queue.retry(task_id):
            retried.append(task_id)
    return {"status": "requeued", "count": len(retried), "task_ids": retried}

@app.get("/api/status")
async def get_status():
    providers = get_configured_providers()
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM sessions")
    session_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM tasks WHERE status='completed'")
    completed_tasks = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM uploaded_files")
    file_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM agent_memory")
    memory_count = c.fetchone()[0]
    conn.close()
    return {
        "status": "running",
        "mode": "full-agent" if providers else "limited",
        "version": "5.0.0",
        "active_sessions": session_count,
        "completed_tasks": completed_tasks,
        "uploaded_files": file_count,
        "memory_entries": memory_count,
        "llm_configured": bool(providers),
        "providers": [p["id"] for p in providers],
        "features": ["tool_execution", "persistent_storage", "websocket", "multi_model", "file_upload", "agent_memory", "export", "error_handling"]
    }

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    if session_id not in active_websockets:
        active_websockets[session_id] = []
    active_websockets[session_id].append(websocket)
    
    try:
        await websocket.send_json({"type": "connected", "session_id": session_id})
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except:
                pass
    except WebSocketDisconnect:
        if session_id in active_websockets:
            active_websockets[session_id].remove(websocket)
            if not active_websockets[session_id]:
                del active_websockets[session_id]

@app.get("/api/sessions")
async def list_sessions():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM sessions ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    sessions = []
    for row in rows:
        sessions.append({
            "session_id": row[0], "created_at": row[1], "status": row[2],
            "working_directory": row[3], "model": row[4], "max_iterations": row[5],
            "current_task": row[6], "iteration": row[7], "total_tasks": row[8],
            "llm_enabled": bool(os.environ.get("OPENAI_API_KEY"))
        })
    return {"sessions": sessions, "total": len(sessions)}

@app.post("/api/sessions")
async def create_session(request: CreateSessionRequest):
    providers = get_configured_providers()
    session_id = str(uuid.uuid4())[:8]
    created_at = datetime.utcnow().isoformat()
    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT INTO sessions (session_id, created_at, status, working_directory, model, provider, max_iterations, iteration, total_tasks) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
              (session_id, created_at, "active", request.working_directory, request.model, request.provider, request.max_iterations, 0, 0))
    conn.commit()
    conn.close()
    return {"session_id": session_id, "created_at": created_at, "status": "active", "provider": request.provider, "model": request.model, "llm_enabled": bool(providers)}

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": row[0], "created_at": row[1], "status": row[2],
        "working_directory": row[3], "model": row[4], "max_iterations": row[5],
        "current_task": row[6], "iteration": row[7], "total_tasks": row[8],
        "llm_enabled": bool(os.environ.get("OPENAI_API_KEY"))
    }

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM task_outputs WHERE task_id IN (SELECT task_id FROM tasks WHERE session_id=?)", (session_id,))
    c.execute("DELETE FROM tasks WHERE session_id=?", (session_id,))
    c.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted", "session_id": session_id}

@app.post("/api/sessions/{session_id}/tasks")
async def create_task(session_id: str, request: CreateTaskRequest):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT model, provider, max_iterations FROM sessions WHERE session_id=?", (session_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Session not found")
    model, provider, max_iterations = row[0], row[1] or "openai", row[2]
    providers = get_configured_providers()
    task_id = str(uuid.uuid4())[:8]
    created_at = datetime.utcnow().isoformat()
    c.execute("INSERT INTO tasks (task_id, session_id, description, status, created_at) VALUES (?, ?, ?, ?, ?)",
              (task_id, session_id, request.description, "queued", created_at))
    c.execute("UPDATE sessions SET total_tasks = total_tasks + 1, current_task = ? WHERE session_id = ?", (task_id, session_id))
    conn.commit()
    conn.close()
    if providers:
        asyncio.create_task(execute_agent_task(session_id, task_id, request.description, model, provider, max_iterations))
    return {"task_id": task_id, "session_id": session_id, "description": request.description, "status": "queued", "created_at": created_at, "provider": provider, "model": model}

@app.get("/api/sessions/{session_id}/tasks")
async def list_tasks(session_id: str):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM tasks WHERE session_id=? ORDER BY created_at DESC", (session_id,))
    rows = c.fetchall()
    conn.close()
    tasks = []
    for row in rows:
        tasks.append({
            "task_id": row[0], "session_id": row[1], "description": row[2],
            "status": row[3], "created_at": row[4], "started_at": row[5],
            "completed_at": row[6], "result": row[7]
        })
    return {"tasks": tasks, "total": len(tasks)}

@app.get("/api/sessions/{session_id}/tasks/{task_id}")
async def get_task(session_id: str, task_id: str):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM tasks WHERE task_id=? AND session_id=?", (task_id, session_id))
    row = c.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "task_id": row[0], "session_id": row[1], "description": row[2],
        "status": row[3], "created_at": row[4], "started_at": row[5],
        "completed_at": row[6], "result": row[7]
    }

@app.get("/api/sessions/{session_id}/tasks/{task_id}/output")
async def get_task_output(session_id: str, task_id: str):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT status, result FROM tasks WHERE task_id=? AND session_id=?", (task_id, session_id))
    task_row = c.fetchone()
    if not task_row:
        conn.close()
        raise HTTPException(status_code=404, detail="Task not found")
    c.execute("SELECT output_type, content FROM task_outputs WHERE task_id=? ORDER BY created_at", (task_id,))
    output_rows = c.fetchall()
    conn.close()
    outputs = [{"type": row[0], "content": row[1]} for row in output_rows]
    return {"task_id": task_id, "status": task_row[0], "outputs": outputs, "result": task_row[1]}

@app.get("/api/tools")
async def list_tools():
    return {
        "tools": [
            {"name": "terminal", "description": "Run shell commands", "params": ["command", "working_dir"]},
            {"name": "file_read", "description": "Read file contents", "params": ["path"]},
            {"name": "file_write", "description": "Write content to file", "params": ["path", "content"]},
            {"name": "list_files", "description": "List directory contents", "params": ["path"]},
            {"name": "git", "description": "Run git commands (read-only)", "params": ["command", "working_dir"]},
            {"name": "code_analysis", "description": "Analyze code structure/complexity", "params": ["path", "analysis_type"]},
            {"name": "search_files", "description": "Search for text in files", "params": ["path", "pattern", "file_pattern"]},
            {"name": "memory_store", "description": "Store information for later recall", "params": ["key", "value"]},
            {"name": "memory_recall", "description": "Recall stored information", "params": ["key"]},
            {"name": "create_github_repo", "description": "Create a new GitHub repository", "params": ["name", "description", "private", "github_token", "auto_init"]}
        ]
    }

@app.get("/api/skills")
async def list_skills():
    return {"skills": [
        {"id": "write-script", "name": "Write Script", "description": "Write and run a script", "tags": ["python", "scripting"], "version": "1.0.0", "is_custom": False},
        {"id": "analyze-code", "name": "Analyze Code", "description": "Analyze code structure and complexity", "tags": ["analysis"], "version": "1.0.0", "is_custom": False},
        {"id": "search-codebase", "name": "Search Codebase", "description": "Search for patterns in code", "tags": ["search"], "version": "1.0.0", "is_custom": False},
        {"id": "git-status", "name": "Git Status", "description": "Check git repository status", "tags": ["git"], "version": "1.0.0", "is_custom": False},
        {"id": "file-operations", "name": "File Operations", "description": "Read, write, and list files", "tags": ["files"], "version": "1.0.0", "is_custom": False}
    ]}

@app.get("/api/skills/tags")
async def list_skill_tags():
    return {"tags": ["python", "scripting", "analysis", "search", "git", "files"]}

@app.get("/api/providers")
async def list_providers():
    return {"providers": get_configured_providers()}

@app.post("/api/sessions/{session_id}/files")
async def upload_file(session_id: str, file: UploadFile = File(...)):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT session_id FROM sessions WHERE session_id=?", (session_id,))
    if not c.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Session not found")
    
    file_id = str(uuid.uuid4())[:8]
    session_upload_dir = os.path.join(UPLOADS_DIR, session_id)
    os.makedirs(session_upload_dir, exist_ok=True)
    
    file_ext = os.path.splitext(file.filename)[1] if file.filename else ""
    stored_filename = f"{file_id}{file_ext}"
    stored_path = os.path.join(session_upload_dir, stored_filename)
    
    content = await file.read()
    with open(stored_path, "wb") as f:
        f.write(content)
    
    c.execute("INSERT INTO uploaded_files (file_id, session_id, original_name, stored_path, file_size, mime_type, uploaded_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (file_id, session_id, file.filename, stored_path, len(content), file.content_type, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()
    
    await broadcast_to_session(session_id, {"type": "file_uploaded", "file_id": file_id, "filename": file.filename, "size": len(content)})
    return {"file_id": file_id, "filename": file.filename, "size": len(content), "mime_type": file.content_type}

@app.get("/api/sessions/{session_id}/files")
async def list_files_endpoint(session_id: str):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT file_id, original_name, file_size, mime_type, uploaded_at FROM uploaded_files WHERE session_id=?", (session_id,))
    rows = c.fetchall()
    conn.close()
    files = [{"file_id": r[0], "filename": r[1], "size": r[2], "mime_type": r[3], "uploaded_at": r[4]} for r in rows]
    return {"files": files, "total": len(files)}

@app.get("/api/sessions/{session_id}/files/{file_id}")
async def download_file(session_id: str, file_id: str):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT stored_path, original_name, mime_type FROM uploaded_files WHERE file_id=? AND session_id=?", (file_id, session_id))
    row = c.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(row[0], filename=row[1], media_type=row[2])

@app.delete("/api/sessions/{session_id}/files/{file_id}")
async def delete_file(session_id: str, file_id: str):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT stored_path FROM uploaded_files WHERE file_id=? AND session_id=?", (file_id, session_id))
    row = c.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="File not found")
    try:
        os.remove(row[0])
    except:
        pass
    c.execute("DELETE FROM uploaded_files WHERE file_id=?", (file_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted", "file_id": file_id}

@app.get("/api/sessions/{session_id}/export")
async def export_session(session_id: str, format: str = "json"):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,))
    session_row = c.fetchone()
    if not session_row:
        conn.close()
        raise HTTPException(status_code=404, detail="Session not found")
    
    c.execute("SELECT * FROM tasks WHERE session_id=? ORDER BY created_at", (session_id,))
    task_rows = c.fetchall()
    
    tasks_data = []
    for task in task_rows:
        c.execute("SELECT output_type, content, created_at FROM task_outputs WHERE task_id=? ORDER BY created_at", (task[0],))
        outputs = [{"type": o[0], "content": o[1], "created_at": o[2]} for o in c.fetchall()]
        tasks_data.append({
            "task_id": task[0], "description": task[2], "status": task[3],
            "created_at": task[4], "started_at": task[5], "completed_at": task[6],
            "result": task[7], "outputs": outputs
        })
    
    c.execute("SELECT memory_id, memory_type, content, created_at FROM agent_memory WHERE session_id=?", (session_id,))
    memories = [{"memory_id": m[0], "type": m[1], "content": m[2], "created_at": m[3]} for m in c.fetchall()]
    
    c.execute("SELECT file_id, original_name, file_size, uploaded_at FROM uploaded_files WHERE session_id=?", (session_id,))
    files = [{"file_id": f[0], "filename": f[1], "size": f[2], "uploaded_at": f[3]} for f in c.fetchall()]
    conn.close()
    
    export_data = {
        "session_id": session_id,
        "created_at": session_row[1],
        "model": session_row[4],
        "provider": session_row[5] if len(session_row) > 5 else "openai",
        "tasks": tasks_data,
        "memories": memories,
        "files": files,
        "exported_at": datetime.utcnow().isoformat()
    }
    
    if format == "json":
        return JSONResponse(content=export_data)
    elif format == "markdown":
        md = f"# Session Export: {session_id}\n\n"
        md += f"**Created:** {session_row[1]}\n**Model:** {session_row[4]}\n\n"
        md += "## Tasks\n\n"
        for task in tasks_data:
            md += f"### Task: {task['description'][:50]}...\n"
            md += f"- Status: {task['status']}\n- Created: {task['created_at']}\n\n"
            md += "#### Outputs\n"
            for output in task['outputs']:
                md += f"**{output['type']}:**\n```\n{output['content'][:500]}\n```\n\n"
        if memories:
            md += "## Memories\n\n"
            for mem in memories:
                md += f"- {mem['content']}\n"
        return JSONResponse(content={"format": "markdown", "content": md})
    elif format == "txt":
        txt = f"Session Export: {session_id}\n{'='*50}\n\n"
        txt += f"Created: {session_row[1]}\nModel: {session_row[4]}\n\n"
        txt += "TASKS\n" + "-"*50 + "\n\n"
        for task in tasks_data:
            txt += f"Task: {task['description']}\nStatus: {task['status']}\n\n"
            for output in task['outputs']:
                txt += f"[{output['type']}]\n{output['content'][:500]}\n\n"
        return JSONResponse(content={"format": "txt", "content": txt})
    else:
        raise HTTPException(status_code=400, detail="Invalid format. Use: json, markdown, or txt")

@app.get("/api/sessions/{session_id}/memory")
async def get_session_memory_endpoint(session_id: str):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT memory_id, memory_type, content, created_at, last_accessed, access_count FROM agent_memory WHERE session_id=? ORDER BY last_accessed DESC", (session_id,))
    rows = c.fetchall()
    conn.close()
    memories = []
    for r in rows:
        try:
            content = json.loads(r[2])
        except:
            content = {"raw": r[2]}
        memories.append({"memory_id": r[0], "type": r[1], "content": content, "created_at": r[3], "last_accessed": r[4], "access_count": r[5]})
    return {"memories": memories, "total": len(memories)}

@app.post("/api/sessions/{session_id}/memory")
async def store_memory_endpoint(session_id: str, entry: MemoryEntry):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT session_id FROM sessions WHERE session_id=?", (session_id,))
    if not c.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Session not found")
    
    memory_id = str(uuid.uuid4())[:8]
    now = datetime.utcnow().isoformat()
    content = json.dumps({"key": entry.key, "value": entry.value})
    c.execute("INSERT INTO agent_memory (memory_id, session_id, memory_type, content, created_at, last_accessed, access_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (memory_id, session_id, entry.memory_type, content, now, now, 1))
    conn.commit()
    conn.close()
    
    await broadcast_to_session(session_id, {"type": "memory_stored", "memory_id": memory_id, "key": entry.key})
    return {"memory_id": memory_id, "key": entry.key, "stored_at": now}

@app.delete("/api/sessions/{session_id}/memory/{memory_id}")
async def delete_memory(session_id: str, memory_id: str):
    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM agent_memory WHERE memory_id=? AND session_id=?", (memory_id, session_id))
    conn.commit()
    conn.close()
    return {"status": "deleted", "memory_id": memory_id}

# ============================================
# Phase 41: GitHub Repo Integration Endpoints
# ============================================

def parse_github_url(url: str) -> tuple[str, str]:
    """Parse GitHub URL to extract owner and repo name."""
    import re
    patterns = [
        r'github\.com[:/]([^/]+)/([^/\.]+)',
        r'github\.com/([^/]+)/([^/]+?)(?:\.git)?$'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1), match.group(2).replace('.git', '')
    raise ValueError(f"Invalid GitHub URL: {url}")

def get_repo_clone_url(repo_url: str, token: Optional[str] = None) -> str:
    """Get clone URL with token for authentication.
    
    Uses x-access-token as username with the PAT as password for proper
    GitHub authentication. This format works with GitHub PATs and avoids
    the 'could not read Username' error when terminal prompts are disabled.
    """
    if token and token.strip():
        # Use x-access-token:TOKEN@ format for GitHub PAT authentication
        # This is the recommended format that works without interactive prompts
        if repo_url.startswith("https://"):
            return repo_url.replace("https://", f"https://x-access-token:{token}@")
    return repo_url

def execute_git_command(command: str, working_dir: str, timeout: int = 120) -> tuple[bool, str]:
    """Execute a git command and return success status and output."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=working_dir,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        )
        if result.returncode != 0:
            return False, result.stderr or result.stdout
        return True, result.stdout
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)

@app.get("/api/repos")
async def list_repos():
    """List all connected GitHub repositories."""
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT repo_id, repo_url, repo_name, owner, default_branch, local_path, created_at, last_synced, status FROM github_repos ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    
    repos = []
    for row in rows:
        repos.append({
            "repo_id": row[0],
            "repo_url": row[1],
            "repo_name": row[2],
            "owner": row[3],
            "default_branch": row[4],
            "local_path": row[5],
            "created_at": row[6],
            "last_synced": row[7],
            "status": row[8],
            "has_token": bool(row[1])  # Don't expose actual token
        })
    return {"repos": repos}

@app.post("/api/repos")
async def add_repo(request: AddRepoRequest):
    """Add a new GitHub repository."""
    try:
        owner, repo_name = parse_github_url(request.repo_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    repo_id = str(uuid.uuid4())[:8]
    local_path = os.path.join(REPOS_DIR, f"{owner}_{repo_name}_{repo_id}")
    now = datetime.utcnow().isoformat()
    
    # Normalize repo URL
    normalized_url = f"https://github.com/{owner}/{repo_name}.git"
    
    conn = get_db()
    c = conn.cursor()
    c.execute("""INSERT INTO github_repos 
                 (repo_id, repo_url, repo_name, owner, github_token, local_path, default_branch, created_at, status)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
              (repo_id, normalized_url, repo_name, owner, request.github_token, local_path, request.branch, now, "pending"))
    conn.commit()
    conn.close()
    
    return {
        "repo_id": repo_id,
        "repo_url": normalized_url,
        "repo_name": repo_name,
        "owner": owner,
        "local_path": local_path,
        "status": "pending",
        "message": "Repository added. Use POST /api/repos/{repo_id}/clone to clone it."
    }

@app.post("/api/repos/create-github")
async def create_github_repo(request: CreateGitHubRepoRequest):
    """Create a new GitHub repository using the GitHub API.
    
    This endpoint allows the agent to create new repositories on GitHub.
    Requires a GitHub Personal Access Token with 'repo' scope.
    """
    import httpx
    
    # Validate repo name (GitHub naming rules)
    if not request.name or not request.name.strip():
        raise HTTPException(status_code=400, detail="Repository name is required")
    
    repo_name = request.name.strip()
    if not all(c.isalnum() or c in '-_.' for c in repo_name):
        raise HTTPException(status_code=400, detail="Repository name can only contain alphanumeric characters, hyphens, underscores, and periods")
    
    # Create repo via GitHub API
    github_api_url = "https://api.github.com/user/repos"
    headers = {
        "Authorization": f"Bearer {request.github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    payload = {
        "name": repo_name,
        "description": request.description,
        "private": request.private,
        "auto_init": request.auto_init
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(github_api_url, headers=headers, json=payload, timeout=30.0)
            
            if response.status_code == 201:
                repo_data = response.json()
                
                # Auto-add to our database
                owner = repo_data["owner"]["login"]
                repo_url = repo_data["clone_url"]
                repo_id = str(uuid.uuid4())[:8]
                local_path = os.path.join(REPOS_DIR, f"{owner}_{repo_name}_{repo_id}")
                now = datetime.utcnow().isoformat()
                default_branch = repo_data.get("default_branch", "main")
                
                conn = get_db()
                c = conn.cursor()
                c.execute("""INSERT INTO github_repos 
                             (repo_id, repo_url, repo_name, owner, github_token, local_path, default_branch, created_at, status)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                          (repo_id, repo_url, repo_name, owner, request.github_token, local_path, default_branch, now, "pending"))
                conn.commit()
                conn.close()
                
                return {
                    "success": True,
                    "repo_id": repo_id,
                    "repo_url": repo_url,
                    "html_url": repo_data["html_url"],
                    "repo_name": repo_name,
                    "owner": owner,
                    "private": repo_data["private"],
                    "default_branch": default_branch,
                    "local_path": local_path,
                    "message": f"Repository '{repo_name}' created successfully on GitHub. Use POST /api/repos/{repo_id}/clone to clone it locally."
                }
            elif response.status_code == 401:
                raise HTTPException(status_code=401, detail="Invalid GitHub token. Make sure your token has 'repo' scope.")
            elif response.status_code == 422:
                error_data = response.json()
                errors = error_data.get("errors", [])
                if errors and errors[0].get("message", "").startswith("name already exists"):
                    raise HTTPException(status_code=422, detail=f"Repository '{repo_name}' already exists in your GitHub account")
                raise HTTPException(status_code=422, detail=f"GitHub API error: {error_data.get('message', 'Unknown error')}")
            else:
                raise HTTPException(status_code=response.status_code, detail=f"GitHub API error: {response.text}")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="GitHub API request timed out")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Failed to connect to GitHub API: {str(e)}")

@app.get("/api/repos/{repo_id}")
async def get_repo(repo_id: str):
    """Get repository details."""
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT repo_id, repo_url, repo_name, owner, default_branch, local_path, created_at, last_synced, status FROM github_repos WHERE repo_id=?", (repo_id,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Repository not found")
    
    # Check if local path exists and get git status
    local_status = None
    current_branch = None
    if row[5] and os.path.exists(row[5]):
        success, output = execute_git_command("git status --porcelain", row[5])
        if success:
            local_status = "clean" if not output.strip() else "modified"
        success, branch_output = execute_git_command("git branch --show-current", row[5])
        if success:
            current_branch = branch_output.strip()
    
    return {
        "repo_id": row[0],
        "repo_url": row[1],
        "repo_name": row[2],
        "owner": row[3],
        "default_branch": row[4],
        "local_path": row[5],
        "created_at": row[6],
        "last_synced": row[7],
        "status": row[8],
        "local_status": local_status,
        "current_branch": current_branch
    }

@app.delete("/api/repos/{repo_id}")
async def delete_repo(repo_id: str):
    """Delete a repository and its local clone."""
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT local_path FROM github_repos WHERE repo_id=?", (repo_id,))
    row = c.fetchone()
    
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Repository not found")
    
    local_path = row[0]
    
    # Delete from database
    c.execute("DELETE FROM session_repos WHERE repo_id=?", (repo_id,))
    c.execute("DELETE FROM github_repos WHERE repo_id=?", (repo_id,))
    conn.commit()
    conn.close()
    
    # Delete local clone if exists
    if local_path and os.path.exists(local_path):
        import shutil
        shutil.rmtree(local_path, ignore_errors=True)
    
    return {"status": "deleted", "repo_id": repo_id}

@app.post("/api/repos/{repo_id}/clone")
async def clone_repo(repo_id: str):
    """Clone a repository to local storage."""
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT repo_url, github_token, local_path, default_branch FROM github_repos WHERE repo_id=?", (repo_id,))
    row = c.fetchone()
    
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Repository not found")
    
    repo_url, token, local_path, default_branch = row
    
    # Check if already cloned
    if os.path.exists(local_path):
        conn.close()
        return {"status": "already_cloned", "local_path": local_path}
    
    # Clone the repository
    clone_url = get_repo_clone_url(repo_url, token)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    success, output = execute_git_command(f"git clone --branch {default_branch} {clone_url} {local_path}", "/tmp", timeout=300)
    
    if not success:
        c.execute("UPDATE github_repos SET status=? WHERE repo_id=?", ("clone_failed", repo_id))
        conn.commit()
        conn.close()
        raise HTTPException(status_code=500, detail=f"Clone failed: {output}")
    
    # Update status
    now = datetime.utcnow().isoformat()
    c.execute("UPDATE github_repos SET status=?, last_synced=? WHERE repo_id=?", ("cloned", now, repo_id))
    conn.commit()
    conn.close()
    
    return {"status": "cloned", "local_path": local_path, "message": "Repository cloned successfully"}

@app.post("/api/repos/{repo_id}/pull")
async def pull_repo(repo_id: str):
    """Pull latest changes from remote."""
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT local_path, github_token, repo_url FROM github_repos WHERE repo_id=?", (repo_id,))
    row = c.fetchone()
    
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Repository not found")
    
    local_path, token, repo_url = row
    
    if not os.path.exists(local_path):
        conn.close()
        raise HTTPException(status_code=400, detail="Repository not cloned yet")
    
    # Set remote URL with token if needed
    if token:
        clone_url = get_repo_clone_url(repo_url, token)
        execute_git_command(f"git remote set-url origin {clone_url}", local_path)
    
    success, output = execute_git_command("git pull", local_path)
    
    if not success:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Pull failed: {output}")
    
    now = datetime.utcnow().isoformat()
    c.execute("UPDATE github_repos SET last_synced=? WHERE repo_id=?", (now, repo_id))
    conn.commit()
    conn.close()
    
    return {"status": "pulled", "output": output}

@app.post("/api/repos/{repo_id}/branch")
async def create_branch(repo_id: str, branch_name: str):
    """Create a new branch."""
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT local_path FROM github_repos WHERE repo_id=?", (repo_id,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Repository not found")
    
    local_path = row[0]
    if not os.path.exists(local_path):
        raise HTTPException(status_code=400, detail="Repository not cloned yet")
    
    success, output = execute_git_command(f"git checkout -b {branch_name}", local_path)
    
    if not success:
        raise HTTPException(status_code=500, detail=f"Branch creation failed: {output}")
    
    return {"status": "created", "branch": branch_name}

@app.post("/api/repos/{repo_id}/checkout")
async def checkout_branch(repo_id: str, branch_name: str):
    """Checkout an existing branch."""
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT local_path FROM github_repos WHERE repo_id=?", (repo_id,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Repository not found")
    
    local_path = row[0]
    if not os.path.exists(local_path):
        raise HTTPException(status_code=400, detail="Repository not cloned yet")
    
    success, output = execute_git_command(f"git checkout {branch_name}", local_path)
    
    if not success:
        raise HTTPException(status_code=500, detail=f"Checkout failed: {output}")
    
    return {"status": "checked_out", "branch": branch_name}

@app.post("/api/repos/{repo_id}/commit")
async def commit_changes(repo_id: str, message: str):
    """Commit all changes."""
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT local_path FROM github_repos WHERE repo_id=?", (repo_id,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Repository not found")
    
    local_path = row[0]
    if not os.path.exists(local_path):
        raise HTTPException(status_code=400, detail="Repository not cloned yet")
    
    # Stage all changes
    execute_git_command("git add -A", local_path)
    
    # Commit
    success, output = execute_git_command(f'git commit -m "{message}"', local_path)
    
    if not success:
        if "nothing to commit" in output:
            return {"status": "no_changes", "message": "Nothing to commit"}
        raise HTTPException(status_code=500, detail=f"Commit failed: {output}")
    
    return {"status": "committed", "message": message, "output": output}

@app.post("/api/repos/{repo_id}/push")
async def push_changes(repo_id: str, branch: Optional[str] = None):
    """Push changes to remote."""
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT local_path, github_token, repo_url FROM github_repos WHERE repo_id=?", (repo_id,))
    row = c.fetchone()
    
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Repository not found")
    
    local_path, token, repo_url = row
    conn.close()
    
    if not os.path.exists(local_path):
        raise HTTPException(status_code=400, detail="Repository not cloned yet")
    
    # Set remote URL with token if needed
    if token:
        clone_url = get_repo_clone_url(repo_url, token)
        execute_git_command(f"git remote set-url origin {clone_url}", local_path)
    
    # Get current branch if not specified
    if not branch:
        success, branch = execute_git_command("git branch --show-current", local_path)
        if success:
            branch = branch.strip()
        else:
            branch = "main"
    
    success, output = execute_git_command(f"git push -u origin {branch}", local_path)
    
    if not success:
        raise HTTPException(status_code=500, detail=f"Push failed: {output}")
    
    return {"status": "pushed", "branch": branch, "output": output}

@app.post("/api/repos/{repo_id}/pr")
async def create_pull_request(repo_id: str, title: str, body: str = "", base_branch: str = "main", head_branch: Optional[str] = None):
    """Create a pull request on GitHub."""
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT local_path, github_token, owner, repo_name FROM github_repos WHERE repo_id=?", (repo_id,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Repository not found")
    
    local_path, token, owner, repo_name = row
    
    if not token:
        raise HTTPException(status_code=400, detail="GitHub token required to create PR")
    
    if not os.path.exists(local_path):
        raise HTTPException(status_code=400, detail="Repository not cloned yet")
    
    # Get current branch if not specified
    if not head_branch:
        success, head_branch = execute_git_command("git branch --show-current", local_path)
        if success:
            head_branch = head_branch.strip()
        else:
            raise HTTPException(status_code=500, detail="Could not determine current branch")
    
    # Create PR using GitHub API
    import urllib.request
    
    pr_data = json.dumps({
        "title": title,
        "body": body,
        "head": head_branch,
        "base": base_branch
    }).encode('utf-8')
    
    req = urllib.request.Request(
        f"https://api.github.com/repos/{owner}/{repo_name}/pulls",
        data=pr_data,
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json"
        },
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            pr_response = json.loads(response.read().decode('utf-8'))
            return {
                "status": "created",
                "pr_number": pr_response.get("number"),
                "pr_url": pr_response.get("html_url"),
                "title": title
            }
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        raise HTTPException(status_code=e.code, detail=f"GitHub API error: {error_body}")

@app.post("/api/sessions/{session_id}/repos")
async def link_repo_to_session(session_id: str, request: LinkRepoRequest):
    """Link a repository to a session."""
    conn = get_db()
    c = conn.cursor()
    
    # Verify session exists
    c.execute("SELECT session_id FROM sessions WHERE session_id=?", (session_id,))
    if not c.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Verify repo exists
    c.execute("SELECT repo_id, local_path FROM github_repos WHERE repo_id=?", (request.repo_id,))
    repo_row = c.fetchone()
    if not repo_row:
        conn.close()
        raise HTTPException(status_code=404, detail="Repository not found")
    
    now = datetime.utcnow().isoformat()
    c.execute("""INSERT OR REPLACE INTO session_repos (session_id, repo_id, branch, linked_at)
                 VALUES (?, ?, ?, ?)""", (session_id, request.repo_id, request.branch, now))
    
    # Update session working directory to repo path
    c.execute("UPDATE sessions SET working_directory=? WHERE session_id=?", (repo_row[1], session_id))
    
    conn.commit()
    conn.close()
    
    return {"status": "linked", "session_id": session_id, "repo_id": request.repo_id, "branch": request.branch}

@app.get("/api/sessions/{session_id}/repos")
async def get_session_repos(session_id: str):
    """Get repositories linked to a session."""
    conn = get_db()
    c = conn.cursor()
    c.execute("""SELECT r.repo_id, r.repo_url, r.repo_name, r.owner, r.local_path, sr.branch, sr.linked_at
                 FROM session_repos sr
                 JOIN github_repos r ON sr.repo_id = r.repo_id
                 WHERE sr.session_id=?""", (session_id,))
    rows = c.fetchall()
    conn.close()
    
    repos = []
    for row in rows:
        repos.append({
            "repo_id": row[0],
            "repo_url": row[1],
            "repo_name": row[2],
            "owner": row[3],
            "local_path": row[4],
            "branch": row[5],
            "linked_at": row[6]
        })
    
    return {"repos": repos}
