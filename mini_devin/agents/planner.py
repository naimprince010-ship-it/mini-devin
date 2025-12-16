"""
Planner Agent for Mini-Devin

This module implements a specialized planning agent that creates detailed
execution plans before the main agent starts working. It decomposes complex
tasks into milestones and steps, validates plans for feasibility, and
supports plan refinement and replanning.
"""

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from ..core.llm_client import LLMClient, create_llm_client
from ..schemas.state import (
    Milestone,
    PlanState,
    PlanStep,
    StepStatus,
    TaskState,
    TaskType,
)


class PlanQuality(str, Enum):
    """Quality assessment of a plan."""
    
    EXCELLENT = "excellent"
    """Plan is comprehensive, well-structured, and ready for execution."""
    
    GOOD = "good"
    """Plan is solid but could use minor improvements."""
    
    ACCEPTABLE = "acceptable"
    """Plan is workable but has some gaps."""
    
    NEEDS_WORK = "needs_work"
    """Plan has significant issues that should be addressed."""
    
    POOR = "poor"
    """Plan is inadequate and needs major revision."""


class PlanningStrategy(str, Enum):
    """Strategies for creating plans."""
    
    WATERFALL = "waterfall"
    """Sequential steps, each depending on the previous."""
    
    ITERATIVE = "iterative"
    """Cycles of implementation and verification."""
    
    EXPLORATORY = "exploratory"
    """Start with exploration, then plan based on findings."""
    
    TEST_DRIVEN = "test_driven"
    """Write tests first, then implement to pass them."""
    
    MINIMAL = "minimal"
    """Smallest possible plan to achieve the goal."""


@dataclass
class PlanValidationResult:
    """Result of validating a plan."""
    
    is_valid: bool
    quality: PlanQuality
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    estimated_steps: int = 0
    estimated_complexity: str = "medium"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "quality": self.quality.value,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "estimated_steps": self.estimated_steps,
            "estimated_complexity": self.estimated_complexity,
        }


@dataclass
class TaskAnalysis:
    """Analysis of a task before planning."""
    
    task_type: TaskType
    complexity: str
    estimated_steps: int
    key_challenges: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    suggested_strategy: PlanningStrategy = PlanningStrategy.ITERATIVE
    prerequisites: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_type": self.task_type.value,
            "complexity": self.complexity,
            "estimated_steps": self.estimated_steps,
            "key_challenges": self.key_challenges,
            "required_tools": self.required_tools,
            "suggested_strategy": self.suggested_strategy.value,
            "prerequisites": self.prerequisites,
            "risks": self.risks,
        }


@dataclass
class PlanningResult:
    """Result of the planning process."""
    
    success: bool
    plan: PlanState | None
    analysis: TaskAnalysis | None
    validation: PlanValidationResult | None
    reasoning: str = ""
    alternatives: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "plan": self.plan.model_dump() if self.plan else None,
            "analysis": self.analysis.to_dict() if self.analysis else None,
            "validation": self.validation.to_dict() if self.validation else None,
            "reasoning": self.reasoning,
            "alternatives": self.alternatives,
        }


PLANNER_SYSTEM_PROMPT = """You are a planning agent for Mini-Devin, an autonomous AI software engineer. Your job is to analyze tasks and create detailed, actionable execution plans.

## Your Responsibilities
1. Analyze the task to understand what needs to be done
2. Identify the type of task (bug fix, feature, refactor, etc.)
3. Break down complex tasks into milestones and steps
4. Ensure each step has clear success criteria
5. Consider dependencies between steps
6. Identify potential risks and challenges
7. Suggest the best strategy for the task

## Planning Guidelines
- Start with exploration/understanding before making changes
- Include verification steps after each significant change
- Keep steps small and focused (one logical action per step)
- Always include rollback points for risky operations
- Consider edge cases and error handling
- Include testing as part of the plan

## Output Format
Provide your plan as a JSON object with the following structure:
{
    "analysis": {
        "task_type": "bug_fix|feature|refactor|code_review|documentation|testing|exploration|general",
        "complexity": "simple|medium|complex|very_complex",
        "estimated_steps": number,
        "key_challenges": ["challenge1", "challenge2"],
        "required_tools": ["terminal", "editor", "browser_search"],
        "suggested_strategy": "waterfall|iterative|exploratory|test_driven|minimal",
        "prerequisites": ["prerequisite1"],
        "risks": ["risk1", "risk2"]
    },
    "milestones": [
        {
            "name": "Milestone Name",
            "description": "What this milestone achieves",
            "verification": "How to verify completion"
        }
    ],
    "steps": [
        {
            "description": "What this step does",
            "expected_outcome": "How we know it succeeded",
            "milestone": "Milestone Name",
            "depends_on": []
        }
    ],
    "reasoning": "Explanation of why this plan was chosen"
}"""


class PlannerAgent:
    """
    A planning agent that creates detailed execution plans.
    
    The planner analyzes tasks and creates structured plans with:
    - Task analysis (type, complexity, challenges)
    - Milestones (major checkpoints)
    - Steps (individual actions)
    - Dependencies and verification criteria
    """
    
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        default_strategy: PlanningStrategy = PlanningStrategy.ITERATIVE,
        max_steps: int = 50,
        include_verification_steps: bool = True,
    ):
        """
        Initialize the planner agent.
        
        Args:
            llm_client: LLM client for intelligent planning
            default_strategy: Default planning strategy to use
            max_steps: Maximum number of steps in a plan
            include_verification_steps: Whether to auto-add verification steps
        """
        self.llm = llm_client or create_llm_client()
        self.default_strategy = default_strategy
        self.max_steps = max_steps
        self.include_verification_steps = include_verification_steps
        
        self.llm.set_system_prompt(PLANNER_SYSTEM_PROMPT)
    
    async def create_plan(
        self,
        task: TaskState,
        context: str | None = None,
        strategy: PlanningStrategy | None = None,
    ) -> PlanningResult:
        """
        Create an execution plan for a task.
        
        Args:
            task: The task to plan for
            context: Optional context about the codebase
            strategy: Optional strategy override
            
        Returns:
            PlanningResult with the plan and analysis
        """
        prompt = self._build_planning_prompt(task, context, strategy)
        
        response = await self.llm.chat(prompt)
        
        result = self._parse_planning_response(response, task)
        
        if result.plan:
            result.validation = self.validate_plan(result.plan, task)
            
            if self.include_verification_steps:
                result.plan = self._add_verification_steps(result.plan)
        
        return result
    
    async def analyze_task(
        self,
        task: TaskState,
        context: str | None = None,
    ) -> TaskAnalysis:
        """
        Analyze a task without creating a full plan.
        
        Args:
            task: The task to analyze
            context: Optional context about the codebase
            
        Returns:
            TaskAnalysis with task characteristics
        """
        prompt = f"""Analyze the following task and provide insights:

Task: {task.goal.description}

Acceptance Criteria:
{chr(10).join(f'- {c}' for c in task.goal.acceptance_criteria) if task.goal.acceptance_criteria else '- Complete the task successfully'}

{f'Context: {context}' if context else ''}

Provide your analysis as JSON with:
- task_type: The type of task
- complexity: simple/medium/complex/very_complex
- estimated_steps: Number of steps needed
- key_challenges: Main challenges to overcome
- required_tools: Tools needed (terminal, editor, browser_search, etc.)
- suggested_strategy: Best approach
- prerequisites: What needs to be done first
- risks: Potential risks"""
        
        response = await self.llm.chat(prompt)
        
        return self._parse_analysis_response(response)
    
    async def refine_plan(
        self,
        plan: PlanState,
        feedback: str,
        task: TaskState,
    ) -> PlanningResult:
        """
        Refine an existing plan based on feedback.
        
        Args:
            plan: The current plan
            feedback: Feedback on what to improve
            task: The original task
            
        Returns:
            PlanningResult with the refined plan
        """
        prompt = f"""Refine the following plan based on the feedback:

Original Task: {task.goal.description}

Current Plan:
{self._format_plan_for_prompt(plan)}

Feedback: {feedback}

Provide an improved plan as JSON with the same structure."""
        
        response = await self.llm.chat(prompt)
        
        result = self._parse_planning_response(response, task)
        
        if result.plan:
            result.plan.replan_count = plan.replan_count + 1
            result.plan.last_replanned_at = datetime.utcnow()
            result.validation = self.validate_plan(result.plan, task)
        
        return result
    
    async def replan_from_failure(
        self,
        plan: PlanState,
        failed_step: PlanStep,
        error: str,
        task: TaskState,
    ) -> PlanningResult:
        """
        Create a new plan after a step failure.
        
        Args:
            plan: The current plan
            failed_step: The step that failed
            error: The error message
            task: The original task
            
        Returns:
            PlanningResult with a recovery plan
        """
        prompt = f"""A step in the plan failed. Create a recovery plan:

Original Task: {task.goal.description}

Failed Step: {failed_step.description}
Error: {error}

Completed Steps:
{self._format_completed_steps(plan)}

Create a new plan that:
1. Addresses the failure
2. Completes the remaining work
3. Avoids the same error

Provide the recovery plan as JSON."""
        
        response = await self.llm.chat(prompt)
        
        result = self._parse_planning_response(response, task)
        
        if result.plan:
            result.plan.replan_count = plan.replan_count + 1
            result.plan.last_replanned_at = datetime.utcnow()
            result.reasoning = f"Recovery plan after failure: {error}"
        
        return result
    
    def validate_plan(
        self,
        plan: PlanState,
        task: TaskState,
    ) -> PlanValidationResult:
        """
        Validate a plan for completeness and feasibility.
        
        Args:
            plan: The plan to validate
            task: The task the plan is for
            
        Returns:
            PlanValidationResult with validation results
        """
        issues = []
        suggestions = []
        
        if not plan.steps:
            issues.append("Plan has no steps")
        
        if len(plan.steps) > self.max_steps:
            issues.append(f"Plan has too many steps ({len(plan.steps)} > {self.max_steps})")
        
        if not plan.milestones:
            suggestions.append("Consider adding milestones for better progress tracking")
        
        for step in plan.steps:
            if not step.description:
                issues.append(f"Step {step.step_id} has no description")
            if not step.expected_outcome:
                suggestions.append(f"Step {step.step_id} should have an expected outcome")
        
        step_ids = {s.step_id for s in plan.steps}
        for step in plan.steps:
            for dep in step.depends_on:
                if dep not in step_ids:
                    issues.append(f"Step {step.step_id} depends on non-existent step {dep}")
        
        has_exploration = any("explore" in s.description.lower() or "understand" in s.description.lower() for s in plan.steps)
        if not has_exploration:
            suggestions.append("Consider adding an exploration step at the beginning")
        
        has_verification = any("verify" in s.description.lower() or "test" in s.description.lower() for s in plan.steps)
        if not has_verification:
            suggestions.append("Consider adding verification steps")
        
        if issues:
            quality = PlanQuality.POOR if len(issues) > 2 else PlanQuality.NEEDS_WORK
        elif len(suggestions) > 3:
            quality = PlanQuality.ACCEPTABLE
        elif suggestions:
            quality = PlanQuality.GOOD
        else:
            quality = PlanQuality.EXCELLENT
        
        return PlanValidationResult(
            is_valid=len(issues) == 0,
            quality=quality,
            issues=issues,
            suggestions=suggestions,
            estimated_steps=len(plan.steps),
            estimated_complexity=self._estimate_complexity(plan),
        )
    
    def create_minimal_plan(
        self,
        task: TaskState,
    ) -> PlanState:
        """
        Create a minimal plan without LLM (for simple tasks).
        
        Args:
            task: The task to plan for
            
        Returns:
            A minimal PlanState
        """
        plan_id = str(uuid.uuid4())
        
        steps = [
            PlanStep(
                step_id=f"{plan_id}-1",
                description="Understand the task and explore relevant code",
                expected_outcome="Have a clear understanding of what needs to be done",
            ),
            PlanStep(
                step_id=f"{plan_id}-2",
                description="Implement the required changes",
                expected_outcome="Changes are made to the codebase",
                depends_on=[f"{plan_id}-1"],
            ),
            PlanStep(
                step_id=f"{plan_id}-3",
                description="Verify the changes work correctly",
                expected_outcome="Tests pass and changes are verified",
                depends_on=[f"{plan_id}-2"],
            ),
        ]
        
        milestones = [
            Milestone(
                milestone_id=f"{plan_id}-m1",
                name="Task Complete",
                description="All changes implemented and verified",
                steps=[s.step_id for s in steps],
                verification="All acceptance criteria met",
            ),
        ]
        
        return PlanState(
            plan_id=plan_id,
            task_id=task.task_id,
            milestones=milestones,
            steps=steps,
            initial_analysis=f"Minimal plan for: {task.goal.description}",
        )
    
    def get_next_step(self, plan: PlanState) -> PlanStep | None:
        """
        Get the next step to execute in the plan.
        
        Args:
            plan: The current plan
            
        Returns:
            The next PlanStep to execute, or None if done
        """
        completed_ids = {s.step_id for s in plan.steps if s.status == StepStatus.COMPLETED}
        
        for step in plan.steps:
            if step.status == StepStatus.PENDING:
                deps_met = all(dep in completed_ids for dep in step.depends_on)
                if deps_met:
                    return step
        
        return None
    
    def mark_step_complete(
        self,
        plan: PlanState,
        step_id: str,
        result: str | None = None,
    ) -> PlanState:
        """
        Mark a step as completed.
        
        Args:
            plan: The current plan
            step_id: ID of the step to mark complete
            result: Optional result description
            
        Returns:
            Updated PlanState
        """
        for step in plan.steps:
            if step.step_id == step_id:
                step.status = StepStatus.COMPLETED
                step.completed_at = datetime.utcnow()
                step.result = result
                break
        
        self._update_milestone_status(plan)
        
        return plan
    
    def mark_step_failed(
        self,
        plan: PlanState,
        step_id: str,
        error: str,
    ) -> PlanState:
        """
        Mark a step as failed.
        
        Args:
            plan: The current plan
            step_id: ID of the step to mark failed
            error: Error message
            
        Returns:
            Updated PlanState
        """
        for step in plan.steps:
            if step.step_id == step_id:
                step.status = StepStatus.FAILED
                step.completed_at = datetime.utcnow()
                step.error = error
                break
        
        return plan
    
    def get_plan_progress(self, plan: PlanState) -> dict[str, Any]:
        """
        Get progress statistics for a plan.
        
        Args:
            plan: The plan to analyze
            
        Returns:
            Dictionary with progress statistics
        """
        total = len(plan.steps)
        completed = sum(1 for s in plan.steps if s.status == StepStatus.COMPLETED)
        failed = sum(1 for s in plan.steps if s.status == StepStatus.FAILED)
        in_progress = sum(1 for s in plan.steps if s.status == StepStatus.IN_PROGRESS)
        pending = sum(1 for s in plan.steps if s.status == StepStatus.PENDING)
        
        milestones_completed = sum(1 for m in plan.milestones if m.completed)
        
        return {
            "total_steps": total,
            "completed_steps": completed,
            "failed_steps": failed,
            "in_progress_steps": in_progress,
            "pending_steps": pending,
            "progress_percent": (completed / total * 100) if total > 0 else 0,
            "milestones_total": len(plan.milestones),
            "milestones_completed": milestones_completed,
            "replan_count": plan.replan_count,
        }
    
    def _build_planning_prompt(
        self,
        task: TaskState,
        context: str | None,
        strategy: PlanningStrategy | None,
    ) -> str:
        """Build the prompt for plan creation."""
        parts = []
        
        parts.append("Create a detailed execution plan for the following task:")
        parts.append("")
        parts.append(f"**Task:** {task.goal.description}")
        parts.append("")
        
        if task.goal.acceptance_criteria:
            parts.append("**Acceptance Criteria:**")
            for criterion in task.goal.acceptance_criteria:
                parts.append(f"- {criterion}")
            parts.append("")
        
        if context:
            parts.append(f"**Context:** {context}")
            parts.append("")
        
        strategy_to_use = strategy or self.default_strategy
        parts.append(f"**Suggested Strategy:** {strategy_to_use.value}")
        parts.append("")
        
        parts.append("**Constraints:**")
        parts.append(f"- Maximum steps: {self.max_steps}")
        parts.append(f"- Include verification: {self.include_verification_steps}")
        parts.append("")
        
        parts.append("Provide your plan as JSON.")
        
        return "\n".join(parts)
    
    def _parse_planning_response(
        self,
        response: str,
        task: TaskState,
    ) -> PlanningResult:
        """Parse the LLM response into a PlanningResult."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = {}
        except json.JSONDecodeError:
            data = {}
        
        analysis = None
        if "analysis" in data:
            analysis = self._parse_analysis_from_dict(data["analysis"])
        
        plan_id = str(uuid.uuid4())
        
        steps = []
        milestone_step_map: dict[str, list[str]] = {}
        
        for i, step_data in enumerate(data.get("steps", [])):
            step_id = f"{plan_id}-{i+1}"
            step = PlanStep(
                step_id=step_id,
                description=step_data.get("description", ""),
                expected_outcome=step_data.get("expected_outcome", ""),
                depends_on=[f"{plan_id}-{int(d.split('-')[-1]) if '-' in str(d) else d}" 
                           for d in step_data.get("depends_on", []) if d],
            )
            steps.append(step)
            
            milestone_name = step_data.get("milestone", "Default")
            if milestone_name not in milestone_step_map:
                milestone_step_map[milestone_name] = []
            milestone_step_map[milestone_name].append(step_id)
        
        milestones = []
        for i, milestone_data in enumerate(data.get("milestones", [])):
            milestone_name = milestone_data.get("name", f"Milestone {i+1}")
            milestone = Milestone(
                milestone_id=f"{plan_id}-m{i+1}",
                name=milestone_name,
                description=milestone_data.get("description", ""),
                steps=milestone_step_map.get(milestone_name, []),
                verification=milestone_data.get("verification", ""),
            )
            milestones.append(milestone)
        
        if not milestones and steps:
            milestones = [
                Milestone(
                    milestone_id=f"{plan_id}-m1",
                    name="Task Complete",
                    description="Complete all steps",
                    steps=[s.step_id for s in steps],
                    verification="All steps completed successfully",
                )
            ]
        
        plan = PlanState(
            plan_id=plan_id,
            task_id=task.task_id,
            milestones=milestones,
            steps=steps,
            initial_analysis=data.get("reasoning", ""),
        ) if steps else None
        
        return PlanningResult(
            success=plan is not None and len(steps) > 0,
            plan=plan,
            analysis=analysis,
            validation=None,
            reasoning=data.get("reasoning", ""),
            alternatives=data.get("alternatives", []),
        )
    
    def _parse_analysis_response(self, response: str) -> TaskAnalysis:
        """Parse the LLM response into a TaskAnalysis."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = {}
        except json.JSONDecodeError:
            data = {}
        
        return self._parse_analysis_from_dict(data)
    
    def _parse_analysis_from_dict(self, data: dict) -> TaskAnalysis:
        """Parse analysis from a dictionary."""
        try:
            task_type = TaskType(data.get("task_type", "general"))
        except ValueError:
            task_type = TaskType.GENERAL
        
        try:
            strategy = PlanningStrategy(data.get("suggested_strategy", "iterative"))
        except ValueError:
            strategy = PlanningStrategy.ITERATIVE
        
        return TaskAnalysis(
            task_type=task_type,
            complexity=data.get("complexity", "medium"),
            estimated_steps=data.get("estimated_steps", 5),
            key_challenges=data.get("key_challenges", []),
            required_tools=data.get("required_tools", ["terminal", "editor"]),
            suggested_strategy=strategy,
            prerequisites=data.get("prerequisites", []),
            risks=data.get("risks", []),
        )
    
    def _format_plan_for_prompt(self, plan: PlanState) -> str:
        """Format a plan for inclusion in a prompt."""
        lines = []
        
        lines.append("Milestones:")
        for m in plan.milestones:
            status = "DONE" if m.completed else "PENDING"
            lines.append(f"  [{status}] {m.name}: {m.description}")
        
        lines.append("\nSteps:")
        for s in plan.steps:
            lines.append(f"  [{s.status.value}] {s.step_id}: {s.description}")
            if s.result:
                lines.append(f"    Result: {s.result}")
            if s.error:
                lines.append(f"    Error: {s.error}")
        
        return "\n".join(lines)
    
    def _format_completed_steps(self, plan: PlanState) -> str:
        """Format completed steps for a prompt."""
        completed = [s for s in plan.steps if s.status == StepStatus.COMPLETED]
        if not completed:
            return "No steps completed yet."
        
        lines = []
        for s in completed:
            lines.append(f"- {s.description}")
            if s.result:
                lines.append(f"  Result: {s.result}")
        
        return "\n".join(lines)
    
    def _add_verification_steps(self, plan: PlanState) -> PlanState:
        """Add verification steps after implementation steps."""
        new_steps = []
        verification_count = 0
        
        for step in plan.steps:
            new_steps.append(step)
            
            is_implementation = any(word in step.description.lower() for word in [
                "implement", "create", "add", "modify", "change", "update", "fix", "write"
            ])
            
            has_following_verification = False
            step_idx = plan.steps.index(step)
            if step_idx < len(plan.steps) - 1:
                next_step = plan.steps[step_idx + 1]
                has_following_verification = any(word in next_step.description.lower() for word in [
                    "verify", "test", "check", "validate", "confirm"
                ])
            
            if is_implementation and not has_following_verification:
                verification_count += 1
                verify_step = PlanStep(
                    step_id=f"{plan.plan_id}-v{verification_count}",
                    description=f"Verify: {step.description}",
                    expected_outcome="Changes work as expected",
                    depends_on=[step.step_id],
                )
                new_steps.append(verify_step)
        
        plan.steps = new_steps
        return plan
    
    def _update_milestone_status(self, plan: PlanState) -> None:
        """Update milestone completion status based on step status."""
        completed_step_ids = {s.step_id for s in plan.steps if s.status == StepStatus.COMPLETED}
        
        for milestone in plan.milestones:
            if milestone.steps:
                milestone.completed = all(sid in completed_step_ids for sid in milestone.steps)
    
    def _estimate_complexity(self, plan: PlanState) -> str:
        """Estimate the complexity of a plan."""
        num_steps = len(plan.steps)
        num_milestones = len(plan.milestones)
        
        has_dependencies = any(s.depends_on for s in plan.steps)
        
        if num_steps <= 3 and num_milestones <= 1:
            return "simple"
        elif num_steps <= 10 and num_milestones <= 3:
            return "medium"
        elif num_steps <= 20 or (has_dependencies and num_steps > 10):
            return "complex"
        else:
            return "very_complex"


def create_planner_agent(
    default_strategy: PlanningStrategy = PlanningStrategy.ITERATIVE,
    max_steps: int = 50,
    include_verification_steps: bool = True,
) -> PlannerAgent:
    """
    Create a planner agent with default configuration.
    
    Args:
        default_strategy: Default planning strategy to use
        max_steps: Maximum number of steps in a plan
        include_verification_steps: Whether to auto-add verification steps
        
    Returns:
        Configured PlannerAgent instance
    """
    return PlannerAgent(
        default_strategy=default_strategy,
        max_steps=max_steps,
        include_verification_steps=include_verification_steps,
    )
