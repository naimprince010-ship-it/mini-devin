"""Tests for hierarchical project planner hardening."""

from types import SimpleNamespace

import pytest

from mini_devin.integrations.hierarchical_planner import (
    HierarchicalPlanner,
    MilestoneStatus,
    deterministic_milestones_for_goal,
)


def test_deterministic_ecommerce_fallback_has_buyer_flow_milestones():
    milestones = deterministic_milestones_for_goal("Build an ecommerce website")

    names = " ".join(m["name"] for m in milestones).lower()
    criteria = " ".join(" ".join(m["acceptance_criteria"]) for m in milestones).lower()
    assert "storefront" in names
    assert "cart" in names
    assert "checkout" in names
    assert "preview" in criteria


@pytest.mark.asyncio
async def test_create_plan_falls_back_when_llm_planner_fails(tmp_path):
    class BrokenPlannerLLM:
        def set_system_prompt(self, prompt):
            pass

        async def chat(self, prompt):
            raise RuntimeError("rate limit")

    planner = HierarchicalPlanner(plans_dir=str(tmp_path / "plans"))

    plan = await planner.create_plan(
        project_id="p1",
        goal="Build an ecommerce website",
        working_dir=str(tmp_path / "workspace"),
        llm_client=BrokenPlannerLLM(),
    )

    assert plan.milestones
    assert plan.working_dir.endswith("workspace")
    assert any("checkout" in m.name.lower() for m in plan.milestones)


@pytest.mark.asyncio
async def test_execute_plan_uses_plan_working_dir_and_finishes(tmp_path):
    planner = HierarchicalPlanner(plans_dir=str(tmp_path / "plans"))
    plan = await planner.create_plan(
        project_id="p1",
        goal="Small task",
        working_dir=str(tmp_path / "workspace"),
        milestones=[
            {
                "name": "One",
                "description": "Do one thing",
                "acceptance_criteria": ["Done"],
                "depends_on_indexes": [],
            }
        ],
    )

    class FakeSessionManager:
        def __init__(self):
            self.created_working_dirs = []

        async def create_session(self, working_directory="."):
            self.created_working_dirs.append(working_directory)
            return SimpleNamespace(session_id="s1")

        async def create_task(self, session_id, description, connection_manager):
            return SimpleNamespace(task_id="t1")

        async def run_task(self, session_id, task_id, connection_manager):
            return None

        async def get_task(self, session_id, task_id):
            return SimpleNamespace(
                status=SimpleNamespace(value="completed"),
                result=SimpleNamespace(summary="milestone complete"),
            )

    fake = FakeSessionManager()
    executed = await planner.execute_plan(plan.id, fake, connection_manager=SimpleNamespace())

    assert fake.created_working_dirs == [str(tmp_path / "workspace")]
    assert executed.status == "completed"
    result = next(iter(executed.results.values()))
    assert result.status == MilestoneStatus.COMPLETED
    assert result.summary == "milestone complete"
