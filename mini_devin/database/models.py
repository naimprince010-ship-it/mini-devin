"""SQLAlchemy models for Mini-Devin persistence."""

from datetime import datetime, timezone
from typing import Optional
from enum import Enum as PyEnum

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    Text,
    DateTime,
    ForeignKey,
    Enum,
    JSON,
)
from sqlalchemy.orm import relationship, DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


class UserModel(Base):
    """Database model for users."""
    __tablename__ = "users"

    id = Column(String(36), primary_key=True)
    email = Column(String(256), unique=True, nullable=False, index=True)
    username = Column(String(128), unique=True, nullable=False, index=True)
    hashed_password = Column(String(256), nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    is_admin = Column(Boolean, nullable=False, default=False)
    
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    
    sessions = relationship("SessionModel", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("APIKeyModel", back_populates="user", cascade="all, delete-orphan")

    def to_dict(self) -> dict:
        """Convert to dictionary (excludes password)."""
        return {
            "user_id": self.id,
            "email": self.email,
            "username": self.username,
            "is_active": self.is_active,
            "is_admin": self.is_admin,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
        }


class APIKeyModel(Base):
    """Database model for API keys."""
    __tablename__ = "api_keys"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(128), nullable=False)
    key_hash = Column(String(256), nullable=False, unique=True)
    key_prefix = Column(String(8), nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    user = relationship("UserModel", back_populates="api_keys")

    def to_dict(self) -> dict:
        """Convert to dictionary (excludes key hash)."""
        return {
            "api_key_id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "key_prefix": self.key_prefix,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


class SessionStatus(str, PyEnum):
    """Session status enum."""
    IDLE = "idle"
    RUNNING = "running"
    ERROR = "error"
    TERMINATED = "terminated"


class TaskStatus(str, PyEnum):
    """Task status enum."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SessionModel(Base):
    """Database model for agent sessions."""
    __tablename__ = "sessions"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    working_directory = Column(String(1024), nullable=False, default=".")
    model = Column(String(128), nullable=False, default="gpt-4o")
    max_iterations = Column(Integer, nullable=False, default=50)
    status = Column(Enum(SessionStatus), nullable=False, default=SessionStatus.IDLE)
    iteration = Column(Integer, nullable=False, default=0)
    
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    user = relationship("UserModel", back_populates="sessions")
    tasks = relationship("TaskModel", back_populates="session", cascade="all, delete-orphan")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "session_id": self.id,
            "working_directory": self.working_directory,
            "model": self.model,
            "max_iterations": self.max_iterations,
            "status": self.status.value if self.status else "idle",
            "iteration": self.iteration,
            "total_tasks": len(self.tasks) if self.tasks else 0,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class TaskModel(Base):
    """Database model for tasks."""
    __tablename__ = "tasks"

    id = Column(String(36), primary_key=True)
    session_id = Column(String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    description = Column(Text, nullable=False)
    acceptance_criteria = Column(JSON, nullable=True, default=list)
    status = Column(Enum(TaskStatus), nullable=False, default=TaskStatus.PENDING)
    iteration = Column(Integer, nullable=False, default=0)
    error_message = Column(Text, nullable=True)
    
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    session = relationship("SessionModel", back_populates="tasks")
    result = relationship("TaskResultModel", back_populates="task", uselist=False, cascade="all, delete-orphan")
    artifacts = relationship("ArtifactModel", back_populates="task", cascade="all, delete-orphan")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "task_id": self.id,
            "session_id": self.session_id,
            "description": self.description,
            "acceptance_criteria": self.acceptance_criteria or [],
            "status": self.status.value if self.status else "pending",
            "iteration": self.iteration,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class TaskResultModel(Base):
    """Database model for task results."""
    __tablename__ = "task_results"

    id = Column(String(36), primary_key=True)
    task_id = Column(String(36), ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False, unique=True)
    success = Column(Boolean, nullable=False, default=False)
    summary = Column(Text, nullable=True)
    files_modified = Column(JSON, nullable=True, default=list)
    commands_executed = Column(JSON, nullable=True, default=list)
    verification_passed = Column(Boolean, nullable=True)
    total_iterations = Column(Integer, nullable=False, default=0)
    total_tool_calls = Column(Integer, nullable=False, default=0)
    duration_seconds = Column(Float, nullable=True)
    
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    
    task = relationship("TaskModel", back_populates="result")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "result_id": self.id,
            "task_id": self.task_id,
            "success": self.success,
            "summary": self.summary,
            "files_modified": self.files_modified or [],
            "commands_executed": self.commands_executed or [],
            "verification_passed": self.verification_passed,
            "total_iterations": self.total_iterations,
            "total_tool_calls": self.total_tool_calls,
            "duration_seconds": self.duration_seconds,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ArtifactModel(Base):
    """Database model for task artifacts."""
    __tablename__ = "artifacts"

    id = Column(String(36), primary_key=True)
    task_id = Column(String(36), ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(256), nullable=False)
    artifact_type = Column(String(64), nullable=False)
    file_path = Column(String(1024), nullable=True)
    content = Column(Text, nullable=True)
    size_bytes = Column(Integer, nullable=True)
    
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    
    task = relationship("TaskModel", back_populates="artifacts")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "artifact_id": self.id,
            "task_id": self.task_id,
            "name": self.name,
            "type": self.artifact_type,
            "file_path": self.file_path,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class EvaluationRunModel(Base):
    """Database model for evaluation runs."""
    __tablename__ = "evaluation_runs"

    id = Column(String(36), primary_key=True)
    benchmark_id = Column(String(128), nullable=False)
    benchmark_name = Column(String(256), nullable=True)
    model = Column(String(128), nullable=False)
    
    total_tasks = Column(Integer, nullable=False, default=0)
    passed_tasks = Column(Integer, nullable=False, default=0)
    failed_tasks = Column(Integer, nullable=False, default=0)
    pass_rate = Column(Float, nullable=True)
    
    results = Column(JSON, nullable=True, default=dict)
    config = Column(JSON, nullable=True, default=dict)
    
    started_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime(timezone=True), nullable=True)
    duration_seconds = Column(Float, nullable=True)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "run_id": self.id,
            "benchmark_id": self.benchmark_id,
            "benchmark_name": self.benchmark_name,
            "model": self.model,
            "total_tasks": self.total_tasks,
            "passed_tasks": self.passed_tasks,
            "failed_tasks": self.failed_tasks,
            "pass_rate": self.pass_rate,
            "results": self.results or {},
            "config": self.config or {},
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
        }
