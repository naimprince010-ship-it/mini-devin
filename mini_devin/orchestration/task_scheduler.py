"""
Parallel / sequential execution for sub-tasks with ``depends_on`` (DAG batches).

Independent tasks in the same topological layer run under ``asyncio.gather``.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar

import asyncio

T = TypeVar("T")


@dataclass(frozen=True)
class SchedulableUnit:
    id: str
    goal: str
    acceptance_criteria: list[str]
    depends_on: tuple[str, ...]


def topological_layers(nodes: list[SchedulableUnit]) -> list[list[SchedulableUnit]]:
    """
    Return layers of nodes; layer i contains all nodes whose deps are in layers 0..i-1.

    Raises ValueError on cycle or unknown dependency ids.
    """
    by_id = {n.id: n for n in nodes}
    if len(by_id) != len(nodes):
        raise ValueError("Duplicate subtask id")

    blocked_by: dict[str, set[str]] = defaultdict(set)
    pending_deps: dict[str, set[str]] = {}
    for n in nodes:
        deps = set(n.depends_on)
        unknown = deps - by_id.keys()
        if unknown:
            raise ValueError(f"Subtask {n.id} depends on unknown ids: {unknown}")
        pending_deps[n.id] = set(deps)
        for d in deps:
            blocked_by[d].add(n.id)

    remaining = set(by_id.keys())
    layers: list[list[SchedulableUnit]] = []

    while remaining:
        ready = [by_id[i] for i in remaining if not pending_deps[i]]
        if not ready:
            raise ValueError("Cycle or unsatisfiable depends_on in sub-task graph")
        layers.append(ready)
        for n in ready:
            remaining.remove(n.id)
        for n in ready:
            for child in blocked_by[n.id]:
                if child in pending_deps:
                    pending_deps[child].discard(n.id)

    return layers


async def run_layer(
    items: list[SchedulableUnit],
    runner: Callable[[SchedulableUnit], Awaitable[T]],
) -> list[T]:
    return await asyncio.gather(*[runner(x) for x in items])


async def run_dag(
    nodes: list[SchedulableUnit],
    runner: Callable[[SchedulableUnit], Awaitable[T]],
) -> list[T]:
    """Run all nodes respecting dependencies; parallel within each layer."""
    layers = topological_layers(nodes)
    out: list[T] = []
    for layer in layers:
        out.extend(await run_layer(layer, runner))
    return out


def transitive_descendants(nodes: list[SchedulableUnit], root_id: str) -> set[str]:
    """All task ids that (transitively) depend on ``root_id`` (excluding ``root_id``)."""
    children: dict[str, set[str]] = defaultdict(set)
    for n in nodes:
        for d in n.depends_on:
            children[d].add(n.id)
    out: set[str] = set()
    dq: deque[str] = deque([root_id])
    seen = {root_id}
    while dq:
        cur = dq.popleft()
        for ch in children.get(cur, ()):
            if ch not in seen:
                seen.add(ch)
                out.add(ch)
                dq.append(ch)
    return out


def auto_skip_when_dep_skipped(units: list[SchedulableUnit], skipped: set[str]) -> None:
    """In-place: extend ``skipped`` for any unit that transitively depends on a skipped id."""
    while True:
        added = False
        for u in units:
            if u.id in skipped:
                continue
            if any(d in skipped for d in u.depends_on):
                skipped.add(u.id)
                added = True
        if not added:
            break
