"""
Hierarchical Planner — Long-term Project Execution

Decomposes a high-level project goal into milestones,
then executes each milestone as an isolated sub-agent task,
passing results forward and persisting state between sessions.

Features:
  - Project plan stored to disk → survives restarts
  - Each milestone gets its own agent task + context window
  - Results from completed milestones injected into the next
  - Can resume a partially-completed plan after days/weeks
  - Milestones can be retried independently if they fail
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class MilestoneStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    SKIPPED   = "skipped"


@dataclass
class MilestoneSpec:
    """A single milestone in a hierarchical project plan."""
    id: str
    index: int
    name: str
    description: str
    acceptance_criteria: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)   # ids of blocking milestones
    estimated_hours: float = 0.0
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MilestoneResult:
    """Execution result for one milestone."""
    milestone_id: str
    status: MilestoneStatus
    summary: str = ""
    artifacts: List[str] = field(default_factory=list)   # file paths produced
    task_id: Optional[str] = None
    session_id: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value if isinstance(self.status, MilestoneStatus) else str(self.status)
        return d


@dataclass
class ProjectPlan:
    """A full hierarchical project plan with execution state."""
    id: str
    project_id: str
    goal: str
    milestones: List[MilestoneSpec] = field(default_factory=list)
    results: Dict[str, MilestoneResult] = field(default_factory=dict)
    status: str = "pending"     # pending | running | completed | failed | paused
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    working_dir: str = "."

    # ── serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "goal": self.goal,
            "milestones": [m.to_dict() for m in self.milestones],
            "results": {k: v.to_dict() for k, v in self.results.items()},
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "working_dir": self.working_dir,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectPlan":
        milestones = [MilestoneSpec(**m) for m in data.get("milestones", [])]
        results: Dict[str, MilestoneResult] = {}
        for k, v in data.get("results", {}).items():
            if not isinstance(v, dict):
                continue
            vr = dict(v)
            st = vr.get("status")
            if isinstance(st, str):
                try:
                    vr["status"] = MilestoneStatus(st)
                except ValueError:
                    vr["status"] = MilestoneStatus.PENDING
            results[k] = MilestoneResult(**vr)
        return cls(
            id=data["id"],
            project_id=data["project_id"],
            goal=data["goal"],
            milestones=milestones,
            results=results,
            status=data.get("status", "pending"),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=data.get("updated_at", datetime.now(timezone.utc).isoformat()),
            working_dir=data.get("working_dir", "."),
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    def get_milestone(self, mid: str) -> Optional[MilestoneSpec]:
        return next((m for m in self.milestones if m.id == mid), None)

    def pending_milestones(self) -> List[MilestoneSpec]:
        return [
            m for m in self.milestones
            if self.results.get(m.id, MilestoneResult(m.id, MilestoneStatus.PENDING)).status
            == MilestoneStatus.PENDING
        ]

    def next_runnable(self) -> Optional[MilestoneSpec]:
        """Return the next milestone whose dependencies are all complete."""
        completed_ids = {
            mid for mid, r in self.results.items()
            if r.status == MilestoneStatus.COMPLETED
        }
        for m in self.milestones:
            r = self.results.get(m.id)
            if r and r.status != MilestoneStatus.PENDING:
                continue
            if all(dep in completed_ids for dep in m.depends_on):
                return m
        return None

    def progress(self) -> Dict[str, Any]:
        total = len(self.milestones)
        done = sum(
            1 for r in self.results.values()
            if r.status == MilestoneStatus.COMPLETED
        )
        failed = sum(
            1 for r in self.results.values()
            if r.status == MilestoneStatus.FAILED
        )
        return {
            "total": total,
            "completed": done,
            "failed": failed,
            "pending": total - done - failed,
            "percent": round(done / total * 100) if total else 0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# LLM-based plan generation
# ─────────────────────────────────────────────────────────────────────────────

PLANNER_PROMPT = """\
You are a senior software architect. Break the following project goal into 3–8 concrete milestones.

Rules:
- Each milestone must be independently testable and deployable
- Milestones must be ordered by dependency (foundational first)
- Each milestone should take 1–5 days of work
- Be specific: name the files, APIs, or features produced
- Include acceptance criteria for each milestone

Return JSON only:
{
  "milestones": [
    {
      "name": "...",
      "description": "...",
      "acceptance_criteria": ["...", "..."],
      "depends_on_indexes": [],
      "estimated_hours": 8,
      "tags": ["backend", "api"]
    }
  ]
}
"""


async def generate_plan_with_llm(
    goal: str,
    project_context: str = "",
    llm_client: Any = None,
) -> List[Dict[str, Any]]:
    """Use LLM to generate milestone specs for a project goal."""
    import re

    try:
        if llm_client is None:
            from ..core.llm_client import create_llm_client

            llm_client = create_llm_client()

        prompt = f"Project goal: {goal}"
        if project_context:
            prompt += f"\n\nProject context:\n{project_context}"

        llm_client.set_system_prompt(PLANNER_PROMPT)
        response = await llm_client.chat(prompt)

        match = re.search(r"\{[\s\S]*\}", response or "")
        if not match:
            return []
        data = json.loads(match.group(0))
        return data.get("milestones", []) or []
    except Exception as e:
        print(f"[planner] generate_plan_with_llm failed: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Plan store (disk)
# ─────────────────────────────────────────────────────────────────────────────

class PlanStore:
    """Persists project plans to disk."""

    def __init__(self, plans_dir: str = "project_plans"):
        self._base = Path(plans_dir)
        self._base.mkdir(parents=True, exist_ok=True)
        self._plans: Dict[str, ProjectPlan] = {}
        self._load_all()

    def _path(self, plan_id: str) -> Path:
        return self._base / f"{plan_id}.json"

    def _load_all(self) -> None:
        for f in self._base.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                plan = ProjectPlan.from_dict(data)
                self._plans[plan.id] = plan
            except Exception:
                pass

    def save(self, plan: ProjectPlan) -> None:
        plan.updated_at = datetime.now(timezone.utc).isoformat()
        self._plans[plan.id] = plan
        self._path(plan.id).write_text(
            json.dumps(plan.to_dict(), indent=2), encoding="utf-8"
        )

    def get(self, plan_id: str) -> Optional[ProjectPlan]:
        return self._plans.get(plan_id)

    def list_for_project(self, project_id: str) -> List[ProjectPlan]:
        return [p for p in self._plans.values() if p.project_id == project_id]

    def list_all(self) -> List[ProjectPlan]:
        return list(self._plans.values())

    def delete(self, plan_id: str) -> bool:
        p = self._path(plan_id)
        if p.exists():
            p.unlink()
        return bool(self._plans.pop(plan_id, None))


# ─────────────────────────────────────────────────────────────────────────────
# Hierarchical executor
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalPlanner:
    """
    Creates and executes multi-milestone project plans.

    Usage:
        planner = HierarchicalPlanner()

        # 1. Create a plan (LLM decomposes the goal)
        plan = await planner.create_plan(
            project_id="my-saas",
            goal="Build a SaaS app with auth, dashboard, and billing",
        )

        # 2. Execute the plan (sub-agent per milestone)
        await planner.execute_plan(plan.id, session_manager, connection_manager)
    """

    def __init__(self, plans_dir: str = "project_plans"):
        self.store = PlanStore(plans_dir)
        self._running_plans: Dict[str, asyncio.Task] = {}

    # ── Plan creation ─────────────────────────────────────────────────────────

    async def create_plan(
        self,
        project_id: str,
        goal: str,
        milestones: Optional[List[Dict[str, Any]]] = None,
        working_dir: str = ".",
        project_context: str = "",
        llm_client: Any = None,
    ) -> ProjectPlan:
        """
        Create a hierarchical project plan.

        If `milestones` is provided, use those directly.
        Otherwise, call the LLM to decompose the goal.
        """
        if milestones is None:
            raw = await generate_plan_with_llm(goal, project_context, llm_client)
        else:
            raw = milestones

        if not raw:
            raise ValueError(
                "No milestones returned. Usually: missing/invalid LLM API key, model error, or the "
                "model did not return valid JSON with a 'milestones' array. Check .env OPENAI_API_KEY "
                "and try again with a shorter goal."
            )

        specs: List[MilestoneSpec] = []
        id_map: Dict[int, str] = {}
        for i, m in enumerate(raw):
            mid = str(uuid.uuid4())[:8]
            id_map[i] = mid
            dep_ids = [
                id_map[j]
                for j in m.get("depends_on_indexes", [])
                if j in id_map
            ]
            specs.append(MilestoneSpec(
                id=mid,
                index=i,
                name=m.get("name", f"Milestone {i + 1}"),
                description=m.get("description", ""),
                acceptance_criteria=m.get("acceptance_criteria", []),
                depends_on=dep_ids,
                estimated_hours=float(m.get("estimated_hours", 8)),
                tags=m.get("tags", []),
            ))

        plan = ProjectPlan(
            id=str(uuid.uuid4())[:12],
            project_id=project_id,
            goal=goal,
            milestones=specs,
            working_dir=working_dir,
        )
        self.store.save(plan)
        return plan

    # ── Execution ─────────────────────────────────────────────────────────────

    async def execute_plan(
        self,
        plan_id: str,
        session_manager: Any,
        connection_manager: Any,
        on_milestone_start: Optional[Callable] = None,
        on_milestone_done: Optional[Callable] = None,
    ) -> ProjectPlan:
        """
        Execute remaining milestones sequentially as sub-agent tasks.
        Can be called again to resume a partially-completed plan.
        """
        plan = self.store.get(plan_id)
        if not plan:
            raise ValueError(f"Plan '{plan_id}' not found")

        plan.status = "running"
        self.store.save(plan)

        while True:
            ms = plan.next_runnable()
            if ms is None:
                break

            # Build task prompt with previous milestone context
            context = self._build_milestone_context(plan, ms)
            task_description = self._milestone_to_task(ms, context)

            # Mark running
            plan.results[ms.id] = MilestoneResult(
                milestone_id=ms.id,
                status=MilestoneStatus.RUNNING,
                started_at=datetime.now(timezone.utc).isoformat(),
            )
            self.store.save(plan)

            if on_milestone_start:
                on_milestone_start(ms)

            # Spawn sub-agent task
            try:
                session = await session_manager.create_session()
                task = await session_manager.create_task(
                    session_id=session.session_id,
                    description=task_description,
                    connection_manager=connection_manager,
                )
                # Run and wait for completion
                await session_manager.run_task(
                    session_id=session.session_id,
                    task_id=task.task_id,
                    connection_manager=connection_manager,
                )
                # Get final task state
                final_task = await session_manager.get_task(task.task_id)
                success = final_task and final_task.status.value == "completed"

                plan.results[ms.id] = MilestoneResult(
                    milestone_id=ms.id,
                    status=MilestoneStatus.COMPLETED if success else MilestoneStatus.FAILED,
                    summary=final_task.result_summary if final_task and hasattr(final_task, "result_summary") else "",
                    task_id=task.task_id,
                    session_id=session.session_id,
                    started_at=plan.results[ms.id].started_at,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    error=None if success else "Task did not reach COMPLETED status",
                )
            except Exception as exc:
                plan.results[ms.id] = MilestoneResult(
                    milestone_id=ms.id,
                    status=MilestoneStatus.FAILED,
                    started_at=plan.results[ms.id].started_at if ms.id in plan.results else None,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    error=str(exc),
                )

            self.store.save(plan)

            if on_milestone_done:
                on_milestone_done(ms, plan.results[ms.id])

            # Stop on first failure (configurable)
            if plan.results[ms.id].status == MilestoneStatus.FAILED:
                plan.status = "failed"
                self.store.save(plan)
                break

        else:
            prog = plan.progress()
            plan.status = "completed" if prog["failed"] == 0 else "failed"
            self.store.save(plan)

        return plan

    # ── Context building ──────────────────────────────────────────────────────

    def _build_milestone_context(self, plan: ProjectPlan, ms: MilestoneSpec) -> str:
        lines = [
            f"Overall project goal: {plan.goal}",
            f"Current milestone ({ms.index + 1}/{len(plan.milestones)}): {ms.name}",
        ]
        completed = [
            (plan.get_milestone(mid), plan.results[mid])
            for mid in ms.depends_on
            if mid in plan.results and plan.results[mid].status == MilestoneStatus.COMPLETED
        ]
        if completed:
            lines.append("\nCompleted dependencies:")
            for dep_ms, dep_result in completed:
                if dep_ms:
                    lines.append(
                        f"  ✅ {dep_ms.name}: {dep_result.summary or 'completed'}"
                    )
        return "\n".join(lines)

    def _milestone_to_task(self, ms: MilestoneSpec, context: str) -> str:
        criteria = "\n".join(f"- {c}" for c in ms.acceptance_criteria) or "- Complete the milestone"
        return (
            f"{context}\n\n"
            f"## Your Task: {ms.name}\n\n"
            f"{ms.description}\n\n"
            f"## Acceptance Criteria\n{criteria}"
        )

    # ── Query helpers ─────────────────────────────────────────────────────────

    def get_plan(self, plan_id: str) -> Optional[ProjectPlan]:
        return self.store.get(plan_id)

    def list_plans(self, project_id: Optional[str] = None) -> List[ProjectPlan]:
        if project_id:
            return self.store.list_for_project(project_id)
        return self.store.list_all()

    def delete_plan(self, plan_id: str) -> bool:
        return self.store.delete(plan_id)

    def retry_milestone(self, plan_id: str, milestone_id: str) -> bool:
        """Reset a failed milestone to PENDING so it will be retried."""
        plan = self.store.get(plan_id)
        if not plan:
            return False
        if milestone_id in plan.results:
            plan.results[milestone_id].status = MilestoneStatus.PENDING
            plan.results[milestone_id].error = None
            plan.status = "pending"
            self.store.save(plan)
            return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Singletons
# ─────────────────────────────────────────────────────────────────────────────

_planner: Optional[HierarchicalPlanner] = None


def default_plans_dir() -> str:
    base = os.environ.get("MINI_DEVIN_DATA", "data")
    return str(Path(base) / "project_plans")


def get_planner(plans_dir: str | None = None) -> HierarchicalPlanner:
    global _planner
    if _planner is None:
        _planner = HierarchicalPlanner(plans_dir or default_plans_dir())
    return _planner
