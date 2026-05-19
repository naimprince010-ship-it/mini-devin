"""Tests for workspace PLAN.md generation."""

from pathlib import Path

from mini_devin.orchestrator.planner import Planner


def test_ecommerce_goal_gets_milestone_plan(tmp_path: Path):
    plan_path = Planner.sync_plan_file(
        tmp_path,
        "Build me an ecommerce website with products, cart, checkout, and admin.",
    )

    plan = plan_path.read_text(encoding="utf-8")
    assert "Design the ecommerce architecture" in plan
    assert "product listing" in plan
    assert "cart" in plan
    assert "checkout" in plan
    assert "Open/verify the user-facing flow" in plan


def test_generic_goal_still_gets_implementation_and_verification_steps(tmp_path: Path):
    plan_path = Planner.sync_plan_file(tmp_path, "Fix the failing test.")

    plan = plan_path.read_text(encoding="utf-8")
    assert "Inspect repository layout" in plan
    assert "Implement changes" in plan
    assert "Verify with tests" in plan
