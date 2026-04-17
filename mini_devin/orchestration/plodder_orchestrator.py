"""
Plodder Orchestrator — OpenHands-style Manager (supervisor) + Worker sessions.

Flow: PlanEvent → dynamic DAG execution with parallel ready batching.
On WorkerObservation failure: SupervisorRoutingPlanner pivot (self-healing plan update),
DAG mutation (skip dependents, inject replacement_subtasks), bounded by ``max_replans`` per task id.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

from .protocol import PlanEvent, SupervisorAction, WorkerObservation
from .task_scheduler import (
    SchedulableUnit,
    auto_skip_when_dep_skipped,
    transitive_descendants,
)
from .worker_runtime import WorkerRuntime


@dataclass
class OrchestratorRunResult:
    plan_event: PlanEvent
    observations: dict[str, WorkerObservation] = field(default_factory=dict)
    actions: list[SupervisorAction] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    pivot_events: list[dict[str, Any]] = field(default_factory=list)
    aborted: bool = False
    abort_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan_event.to_json(),
            "observations": {k: v.to_json() for k, v in self.observations.items()},
            "actions": [a.to_json() for a in self.actions],
            "errors": self.errors,
            "pivot_events": self.pivot_events,
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
        }


def _prior_context(observations: dict[str, WorkerObservation]) -> str:
    if not observations:
        return ""
    lines: list[str] = []
    for sid, ob in observations.items():
        st = ob.result.get("status") or ("completed" if ob.success else "failed")
        lines.append(f"- [{sid}] {st}: {(ob.result.get('summary') or '')[:1200]}")
    return "Completed sub-tasks so far:\n" + "\n".join(lines)


def _parse_replacement_units(
    raw_list: list[Any],
    *,
    completed_ids: set[str],
) -> list[SchedulableUnit]:
    """Build schedulable units from supervisor pivot output; deps must land in completed ∪ new ids."""
    rows: list[tuple[str, str, list[str], tuple[str, ...]]] = []
    for s in raw_list:
        if not isinstance(s, dict):
            continue
        sid = str(s.get("id", "")).strip()
        goal = str(s.get("goal", "")).strip()
        if not sid or not goal:
            continue
        ac = s.get("acceptance_criteria") or []
        if not isinstance(ac, list):
            ac = [str(ac)]
        else:
            ac = [str(x) for x in ac]
        deps = s.get("depends_on") or []
        if not isinstance(deps, list):
            deps = []
        rows.append((sid, goal, ac, tuple(str(x) for x in deps)))

    new_ids = {r[0] for r in rows}
    allow = completed_ids | new_ids
    out: list[SchedulableUnit] = []
    for sid, goal, ac, deps in rows:
        filtered = tuple(d for d in deps if d in allow)
        dropped = [d for d in deps if d not in allow]
        if dropped:
            print(f"[Orchestrator] pivot deps dropped (unknown ids): {dropped}")
        out.append(SchedulableUnit(id=sid, goal=goal, acceptance_criteria=ac, depends_on=filtered))
    return out


class PlodderOrchestrator:
    """
    Supervisor loop: routing + JSON-RPC style dispatch; workers run ``Agent.run``.

    Self-healing: failed observations trigger :meth:`SupervisorRoutingPlanner.pivot_after_failure`,
    in-memory DAG updates (skip descendants of failed node, append replacement subtasks).
    """

    def __init__(
        self,
        session_manager: Any,
        supervisor: Any | None = None,
        *,
        max_retries_per_subtask: int = 2,
        max_replans: int = 3,
        playbook_tags: Sequence[str] | None = None,
    ):
        from ..agents.planner import SupervisorRoutingPlanner
        from ..skills.playbook import playbook_tags_from_env

        self._session_manager = session_manager
        self._supervisor = supervisor or SupervisorRoutingPlanner()
        self._max_retries = max(0, int(max_retries_per_subtask))
        self._max_replans = max(0, int(max_replans))
        self._playbook_tags: list[str] = (
            [str(t).strip() for t in playbook_tags if str(t).strip()]
            if playbook_tags is not None
            else playbook_tags_from_env()
        )

    def _parse_subtasks(self, raw: dict[str, Any]) -> list[SchedulableUnit]:
        out: list[SchedulableUnit] = []
        for s in raw.get("subtasks") or []:
            sid = str(s.get("id", "")).strip()
            if not sid:
                continue
            goal = str(s.get("goal", "")).strip()
            if not goal:
                continue
            ac = s.get("acceptance_criteria") or []
            if not isinstance(ac, list):
                ac = [str(ac)]
            else:
                ac = [str(x) for x in ac]
            deps = s.get("depends_on") or []
            if not isinstance(deps, list):
                deps = []
            dep_t = tuple(str(x) for x in deps)
            out.append(
                SchedulableUnit(
                    id=sid,
                    goal=goal,
                    acceptance_criteria=ac,
                    depends_on=dep_t,
                )
            )
        if not out:
            raise ValueError("Supervisor returned no valid subtasks")
        return out

    async def _execute_worker_unit(
        self,
        unit: SchedulableUnit,
        *,
        prior_obs: dict[str, WorkerObservation],
        parent: Any,
        workspace: str,
        connection_manager: Any | None,
        actions: list[SupervisorAction],
    ) -> WorkerObservation:
        act = SupervisorAction.dispatch(
            WorkerRuntime.new_action_id(),
            unit.id,
            unit.goal,
            unit.acceptance_criteria,
        )
        actions.append(act)
        prior = dict(prior_obs)
        last_err: str | None = None
        worker_session_id = ""
        for attempt in range(self._max_retries + 1):
            desc = self._worker_task_prompt(unit, prior, attempt, last_err, workspace=workspace)
            try:
                worker_session = await WorkerRuntime.create_worker_session(
                    self._session_manager,
                    shared_workspace=workspace,
                    model=getattr(parent, "model", None),
                    max_iterations=getattr(parent, "max_iterations", None),
                )
                worker_session_id = worker_session.session_id
                task = await self._session_manager.create_task(
                    session_id=worker_session.session_id,
                    description=desc,
                    connection_manager=connection_manager,
                )
                await self._session_manager.run_task(
                    session_id=worker_session.session_id,
                    task_id=task.task_id,
                    connection_manager=connection_manager,
                )
                final = await self._session_manager.get_task(worker_session.session_id, task.task_id)
                success, summary, err = self._classify_task_outcome(final)
                obs = WorkerObservation.from_task_outcome(
                    act.id,
                    unit.id,
                    success=success,
                    summary=summary,
                    worker_session_id=worker_session.session_id,
                    task_id=task.task_id,
                    error=err,
                )
                if success or attempt >= self._max_retries:
                    return obs
                last_err = err or "worker failed"
            except Exception as e:  # noqa: BLE001
                last_err = str(e)
                if attempt >= self._max_retries:
                    return WorkerObservation.from_task_outcome(
                        act.id,
                        unit.id,
                        success=False,
                        summary="",
                        worker_session_id=worker_session_id or "?",
                        task_id="",
                        error=last_err,
                    )
        raise RuntimeError("unreachable worker loop")

    async def run_goal(
        self,
        parent_session_id: str,
        goal: str,
        *,
        connection_manager: Any | None = None,
    ) -> OrchestratorRunResult:
        parent = await self._session_manager.get_session(parent_session_id)
        if not parent:
            raise ValueError(f"Session not found: {parent_session_id}")
        workspace = parent.working_directory

        raw_plan = await self._supervisor.decompose_to_subtasks(
            goal,
            workspace_hint=workspace,
        )
        units: list[SchedulableUnit] = self._parse_subtasks(raw_plan)
        plan_event = PlanEvent(
            type="plan.created",
            goal=goal,
            workspace=workspace,
            subtasks=[{"id": u.id, "goal": u.goal, "depends_on": list(u.depends_on)} for u in units],
            reasoning=str(raw_plan.get("reasoning", "")),
        )

        observations: dict[str, WorkerObservation] = {}
        actions: list[SupervisorAction] = []
        errors: list[str] = []
        pivot_events: list[dict[str, Any]] = []
        done: set[str] = set()
        skipped: set[str] = set()
        pivot_counts: dict[str, int] = defaultdict(int)
        aborted = False
        abort_reason: str | None = None

        while True:
            active = [u for u in units if u.id not in done and u.id not in skipped]
            if not active:
                break

            ready = [u for u in active if all(d in done for d in u.depends_on)]
            if not ready:
                msg = "Deadlock: no runnable subtasks (dependencies blocked or invalid DAG)."
                errors.append(msg)
                aborted = True
                abort_reason = msg
                break

            results = await asyncio.gather(
                *[
                    self._execute_worker_unit(
                        u,
                        prior_obs=observations,
                        parent=parent,
                        workspace=workspace,
                        connection_manager=connection_manager,
                        actions=actions,
                    )
                    for u in ready
                ]
            )

            pairs = list(zip(ready, results))
            for unit, obs in pairs:
                observations[unit.id] = obs
            for unit, obs in pairs:
                if obs.success:
                    done.add(unit.id)

            for unit, obs in pairs:
                if obs.success:
                    continue

                if pivot_counts[unit.id] >= self._max_replans:
                    aborted = True
                    abort_reason = (
                        f"Stopped: sub-task {unit.id} exceeded max_replans={self._max_replans} "
                        "without a successful pivot/worker outcome."
                    )
                    errors.append(abort_reason)
                    pivot_events.append(
                        {
                            "event": "pivot_blocked",
                            "failed_subtask_id": unit.id,
                            "observation": obs.to_json(),
                            "limit": self._max_replans,
                        }
                    )
                    break

                pivot_counts[unit.id] += 1
                completed_ids = sorted(done)
                remaining_summary = json.dumps(
                    [{"id": u.id, "depends_on": list(u.depends_on)} for u in units if u.id not in done],
                    indent=2,
                )
                try:
                    pivot_data = await self._supervisor.pivot_after_failure(
                        failed_subtask_id=unit.id,
                        observation=obs.to_json(),
                        overall_goal=goal,
                        completed_subtask_ids=completed_ids,
                        remaining_plan_summary=remaining_summary,
                        workspace_hint=workspace,
                    )
                except Exception as exc:  # noqa: BLE001
                    aborted = True
                    abort_reason = f"Pivot LLM/plan failed for {unit.id}: {exc}"
                    errors.append(abort_reason)
                    pivot_events.append(
                        {"event": "pivot_error", "failed_subtask_id": unit.id, "error": str(exc)}
                    )
                    break

                plan_event.type = "plan.updated"
                plan_event.reasoning = str(pivot_data.get("reasoning", plan_event.reasoning))
                pivoted_meta = pivot_data.get("pivoted") or []
                if isinstance(pivoted_meta, list):
                    plan_event.pivoted.extend([x for x in pivoted_meta if isinstance(x, dict)])

                skip_new = {unit.id} | transitive_descendants(units, unit.id)
                skipped |= skip_new
                auto_skip_when_dep_skipped(units, skipped)
                units = [u for u in units if u.id not in skipped]

                replacement_raw = pivot_data.get("replacement_subtasks") or []
                new_units = _parse_replacement_units(
                    replacement_raw if isinstance(replacement_raw, list) else [],
                    completed_ids=set(done),
                )
                if not new_units:
                    errors.append(
                        f"pivot for {unit.id} returned no replacement_subtasks; branch abandoned."
                    )
                existing = {u.id for u in units}
                for nu in new_units:
                    if nu.id in existing:
                        errors.append(f"Skipping duplicate replacement id {nu.id}")
                        continue
                    units.append(nu)
                    existing.add(nu.id)

                plan_event.subtasks = [
                    {"id": u.id, "goal": u.goal, "depends_on": list(u.depends_on)} for u in units
                ]
                pivot_events.append(
                    {
                        "event": "plan_pivot",
                        "failed_subtask_id": unit.id,
                        "observation_status": obs.result.get("status", "failed"),
                        "pivot": pivot_data,
                        "skipped_ids": sorted(skip_new),
                        "added_ids": [u.id for u in new_units],
                    }
                )
                # Recompute ready set next outer iteration (DAG changed).
                break

            if aborted:
                break

        return OrchestratorRunResult(
            plan_event=plan_event,
            observations=observations,
            actions=actions,
            errors=errors,
            pivot_events=pivot_events,
            aborted=aborted,
            abort_reason=abort_reason,
        )

    def _worker_task_prompt(
        self,
        unit: SchedulableUnit,
        prior_obs: dict[str, WorkerObservation],
        attempt: int,
        last_error: str | None,
        *,
        workspace: str,
    ) -> str:
        from ..skills.playbook import format_playbooks_for_prompt

        base = _prior_context(prior_obs)
        ac = "\n".join(f"- {c}" for c in unit.acceptance_criteria) or "- Complete the sub-goal."
        hdr = f"[Orchestrator sub-task {unit.id}]\n\nPrimary goal:\n{unit.goal}\n\nAcceptance:\n{ac}\n"
        if base:
            hdr += f"\n{base}\n"
        if attempt > 0 and last_error:
            hdr += (
                f"\n## Self-correction (attempt {attempt + 1})\n"
                f"Previous run failed: {last_error}\nAdjust approach and satisfy acceptance criteria.\n"
            )
        if self._playbook_tags and workspace:
            block = format_playbooks_for_prompt(workspace, self._playbook_tags)
            if block:
                hdr += "\n" + block + "\n"
        return hdr

    @staticmethod
    def _classify_task_outcome(final: Any) -> tuple[bool, str, str | None]:
        if final is None:
            return False, "", "missing task"
        st = final.status
        val = st.value if hasattr(st, "value") else str(st)
        ok = val == "completed"
        summary = ""
        err: str | None = None
        if getattr(final, "result", None):
            summary = getattr(final.result, "summary", "") or ""
            if getattr(final.result, "status", "") == "failed":
                ok = False
        if not ok:
            err = getattr(final, "error_message", None) or "task did not complete"
        return ok, summary, err
