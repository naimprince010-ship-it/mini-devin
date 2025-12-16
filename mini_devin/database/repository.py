"""Repository classes for database operations."""

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .models import (
    SessionModel,
    TaskModel,
    TaskResultModel,
    ArtifactModel,
    EvaluationRunModel,
    SessionStatus,
    TaskStatus,
)


class SessionRepository:
    """Repository for session database operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        working_directory: str = ".",
        model: str = "gpt-4o",
        max_iterations: int = 50,
    ) -> SessionModel:
        """Create a new session."""
        session_id = str(uuid.uuid4())[:8]
        db_session = SessionModel(
            id=session_id,
            working_directory=working_directory,
            model=model,
            max_iterations=max_iterations,
            status=SessionStatus.IDLE,
            iteration=0,
        )
        self.session.add(db_session)
        await self.session.flush()
        return db_session

    async def get(self, session_id: str) -> Optional[SessionModel]:
        """Get a session by ID."""
        result = await self.session.execute(
            select(SessionModel)
            .where(SessionModel.id == session_id)
            .options(selectinload(SessionModel.tasks))
        )
        return result.scalar_one_or_none()

    async def list_all(self) -> list[SessionModel]:
        """List all sessions."""
        result = await self.session.execute(
            select(SessionModel)
            .options(selectinload(SessionModel.tasks))
            .order_by(SessionModel.created_at.desc())
        )
        return list(result.scalars().all())

    async def update_status(
        self,
        session_id: str,
        status: SessionStatus,
        iteration: Optional[int] = None,
    ) -> bool:
        """Update session status."""
        values = {
            "status": status,
            "updated_at": datetime.now(timezone.utc),
        }
        if iteration is not None:
            values["iteration"] = iteration
        
        result = await self.session.execute(
            update(SessionModel)
            .where(SessionModel.id == session_id)
            .values(**values)
        )
        return result.rowcount > 0

    async def delete(self, session_id: str) -> bool:
        """Delete a session."""
        result = await self.session.execute(
            delete(SessionModel).where(SessionModel.id == session_id)
        )
        return result.rowcount > 0

    async def count_active(self) -> int:
        """Count active sessions."""
        result = await self.session.execute(
            select(SessionModel).where(
                SessionModel.status.in_([SessionStatus.IDLE, SessionStatus.RUNNING])
            )
        )
        return len(result.scalars().all())


class TaskRepository:
    """Repository for task database operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        session_id: str,
        description: str,
        acceptance_criteria: Optional[list[str]] = None,
    ) -> TaskModel:
        """Create a new task."""
        task_id = str(uuid.uuid4())[:8]
        task = TaskModel(
            id=task_id,
            session_id=session_id,
            description=description,
            acceptance_criteria=acceptance_criteria or [],
            status=TaskStatus.PENDING,
            iteration=0,
        )
        self.session.add(task)
        await self.session.flush()
        return task

    async def get(self, task_id: str) -> Optional[TaskModel]:
        """Get a task by ID."""
        result = await self.session.execute(
            select(TaskModel)
            .where(TaskModel.id == task_id)
            .options(
                selectinload(TaskModel.result),
                selectinload(TaskModel.artifacts),
            )
        )
        return result.scalar_one_or_none()

    async def list_by_session(self, session_id: str) -> list[TaskModel]:
        """List all tasks for a session."""
        result = await self.session.execute(
            select(TaskModel)
            .where(TaskModel.session_id == session_id)
            .options(selectinload(TaskModel.result))
            .order_by(TaskModel.created_at.desc())
        )
        return list(result.scalars().all())

    async def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        iteration: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> bool:
        """Update task status."""
        values = {"status": status}
        if iteration is not None:
            values["iteration"] = iteration
        if error_message is not None:
            values["error_message"] = error_message
        
        if status == TaskStatus.RUNNING:
            values["started_at"] = datetime.now(timezone.utc)
        elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            values["completed_at"] = datetime.now(timezone.utc)
        
        result = await self.session.execute(
            update(TaskModel)
            .where(TaskModel.id == task_id)
            .values(**values)
        )
        return result.rowcount > 0

    async def create_result(
        self,
        task_id: str,
        success: bool,
        summary: Optional[str] = None,
        files_modified: Optional[list[str]] = None,
        commands_executed: Optional[list[str]] = None,
        verification_passed: Optional[bool] = None,
        total_iterations: int = 0,
        total_tool_calls: int = 0,
        duration_seconds: Optional[float] = None,
    ) -> TaskResultModel:
        """Create a task result."""
        result_id = str(uuid.uuid4())[:8]
        result = TaskResultModel(
            id=result_id,
            task_id=task_id,
            success=success,
            summary=summary,
            files_modified=files_modified or [],
            commands_executed=commands_executed or [],
            verification_passed=verification_passed,
            total_iterations=total_iterations,
            total_tool_calls=total_tool_calls,
            duration_seconds=duration_seconds,
        )
        self.session.add(result)
        await self.session.flush()
        return result

    async def get_result(self, task_id: str) -> Optional[TaskResultModel]:
        """Get task result."""
        result = await self.session.execute(
            select(TaskResultModel).where(TaskResultModel.task_id == task_id)
        )
        return result.scalar_one_or_none()

    async def count_completed(self) -> int:
        """Count completed tasks."""
        result = await self.session.execute(
            select(TaskModel).where(TaskModel.status == TaskStatus.COMPLETED)
        )
        return len(result.scalars().all())


class ArtifactRepository:
    """Repository for artifact database operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        task_id: str,
        name: str,
        artifact_type: str,
        file_path: Optional[str] = None,
        content: Optional[str] = None,
        size_bytes: Optional[int] = None,
    ) -> ArtifactModel:
        """Create a new artifact."""
        artifact_id = str(uuid.uuid4())[:8]
        artifact = ArtifactModel(
            id=artifact_id,
            task_id=task_id,
            name=name,
            artifact_type=artifact_type,
            file_path=file_path,
            content=content,
            size_bytes=size_bytes,
        )
        self.session.add(artifact)
        await self.session.flush()
        return artifact

    async def list_by_task(self, task_id: str) -> list[ArtifactModel]:
        """List all artifacts for a task."""
        result = await self.session.execute(
            select(ArtifactModel)
            .where(ArtifactModel.task_id == task_id)
            .order_by(ArtifactModel.created_at.desc())
        )
        return list(result.scalars().all())

    async def get(self, artifact_id: str) -> Optional[ArtifactModel]:
        """Get an artifact by ID."""
        result = await self.session.execute(
            select(ArtifactModel).where(ArtifactModel.id == artifact_id)
        )
        return result.scalar_one_or_none()

    async def get_by_name(self, task_id: str, name: str) -> Optional[ArtifactModel]:
        """Get an artifact by task ID and name."""
        result = await self.session.execute(
            select(ArtifactModel).where(
                ArtifactModel.task_id == task_id,
                ArtifactModel.name == name,
            )
        )
        return result.scalar_one_or_none()


class EvaluationRepository:
    """Repository for evaluation run database operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        benchmark_id: str,
        model: str,
        benchmark_name: Optional[str] = None,
        config: Optional[dict] = None,
    ) -> EvaluationRunModel:
        """Create a new evaluation run."""
        run_id = str(uuid.uuid4())[:8]
        run = EvaluationRunModel(
            id=run_id,
            benchmark_id=benchmark_id,
            benchmark_name=benchmark_name,
            model=model,
            config=config or {},
        )
        self.session.add(run)
        await self.session.flush()
        return run

    async def get(self, run_id: str) -> Optional[EvaluationRunModel]:
        """Get an evaluation run by ID."""
        result = await self.session.execute(
            select(EvaluationRunModel).where(EvaluationRunModel.id == run_id)
        )
        return result.scalar_one_or_none()

    async def list_all(self) -> list[EvaluationRunModel]:
        """List all evaluation runs."""
        result = await self.session.execute(
            select(EvaluationRunModel).order_by(EvaluationRunModel.started_at.desc())
        )
        return list(result.scalars().all())

    async def list_by_benchmark(self, benchmark_id: str) -> list[EvaluationRunModel]:
        """List evaluation runs for a benchmark."""
        result = await self.session.execute(
            select(EvaluationRunModel)
            .where(EvaluationRunModel.benchmark_id == benchmark_id)
            .order_by(EvaluationRunModel.started_at.desc())
        )
        return list(result.scalars().all())

    async def update_results(
        self,
        run_id: str,
        total_tasks: int,
        passed_tasks: int,
        failed_tasks: int,
        results: dict,
        duration_seconds: Optional[float] = None,
    ) -> bool:
        """Update evaluation run results."""
        pass_rate = passed_tasks / total_tasks if total_tasks > 0 else 0.0
        
        result = await self.session.execute(
            update(EvaluationRunModel)
            .where(EvaluationRunModel.id == run_id)
            .values(
                total_tasks=total_tasks,
                passed_tasks=passed_tasks,
                failed_tasks=failed_tasks,
                pass_rate=pass_rate,
                results=results,
                completed_at=datetime.now(timezone.utc),
                duration_seconds=duration_seconds,
            )
        )
        return result.rowcount > 0
