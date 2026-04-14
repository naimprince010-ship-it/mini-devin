"""End-to-end backbone tests: EventStream + LocalRuntime + AgentStateStore."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from mini_devin.backbone import (
    AgentStateStore,
    CmdRunAction,
    CommandOutputObservation,
    EventStream,
)
from mini_devin.backbone.event_models import is_observation_event
from mini_devin.backbone.local_runtime import LocalRuntime
from mini_devin.core.tool_interface import ToolRegistry
from mini_devin.tools.terminal import create_terminal_tool


async def _wait_observation(
    q: asyncio.Queue,
    *,
    source_action_id: str,
    timeout: float = 15.0,
):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise TimeoutError("no matching observation")
        ev = await asyncio.wait_for(q.get(), timeout=remaining)
        if is_observation_event(ev) and getattr(ev, "source_action_id", None) == source_action_id:
            return ev


@pytest.mark.asyncio
async def test_cmd_run_action_local_runtime_state_store(tmp_path: Path) -> None:
    """Publish CmdRunAction → LocalRuntime → observation persisted on AgentStateStore."""
    wd = str(tmp_path)
    registry = ToolRegistry()
    registry.register(create_terminal_tool(working_directory=wd))

    store = AgentStateStore()
    stream = EventStream(store=store)
    runtime = LocalRuntime(registry)
    listener = stream.register_subscriber()

    await runtime.start(stream)
    try:
        action = CmdRunAction(command="echo PL_BACKBONE_OK", cwd=".", timeout_sec=30)
        await stream.publish(action)

        obs = await _wait_observation(listener, source_action_id=action.id)
        assert isinstance(obs, CommandOutputObservation)
        assert "PL_BACKBONE_OK" in obs.stdout or "PL_BACKBONE_OK" in (obs.stdout + obs.stderr)
        assert obs.exit_code == 0

        snap = await store.snapshot()
        kinds = [e.type for e in snap]  # type: ignore[union-attr]
        assert "CmdRunAction" in kinds
        assert "CommandOutputObservation" in kinds
    finally:
        stream.unregister_subscriber(listener)
        await runtime.stop()


@pytest.mark.asyncio
async def test_agent_controller_dispatch_and_wait_with_local_runtime(tmp_path: Path) -> None:
    """AgentController + LocalRuntime round-trip (same stack as run_simple Local path)."""
    from mini_devin.backbone.controller import AgentController

    wd = str(tmp_path)
    registry = ToolRegistry()
    registry.register(create_terminal_tool(working_directory=wd))

    store = AgentStateStore()
    stream = EventStream(store=store)
    runtime = LocalRuntime(registry)
    ctrl = AgentController(stream, store, runtime)
    await ctrl.start()
    try:
        out = await ctrl.dispatch_and_wait(
            "terminal",
            {"command": "echo CTRL_OK", "working_directory": ".", "timeout_seconds": 30},
        )
        assert "CTRL_OK" in out
    finally:
        await ctrl.stop()


@pytest.mark.asyncio
async def test_docker_runtime_generic_terminal_maps_to_shell(tmp_path: Path) -> None:
    """DockerRuntime accepts GenericToolAction(terminal) when docker is available."""
    import shutil

    if not shutil.which("docker"):
        pytest.skip("docker CLI not on PATH")

    from mini_devin.backbone.docker_runtime import DockerRuntime
    from mini_devin.backbone.event_models import GenericToolAction

    wd = str(tmp_path)
    store = AgentStateStore()
    stream = EventStream(store=store)
    runtime = DockerRuntime(wd)
    q = stream.register_subscriber()
    await runtime.start(stream)
    try:
        act = GenericToolAction(
            tool_name="terminal",
            arguments={"command": "echo DOCKER_GENERIC", "working_directory": ".", "timeout_seconds": 30},
        )
        await stream.publish(act)
        obs = await _wait_observation(q, source_action_id=act.id)
        from mini_devin.backbone.event_models import CommandOutputObservation

        assert isinstance(obs, CommandOutputObservation)
        assert "DOCKER_GENERIC" in obs.stdout or "DOCKER_GENERIC" in (obs.stdout + obs.stderr)
    finally:
        stream.unregister_subscriber(q)
        await runtime.stop()
