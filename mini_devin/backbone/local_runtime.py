"""Execute actions on the host using the existing :class:`~mini_devin.core.tool_interface.ToolRegistry`.

This is the **registry execution** path: ``GenericToolAction`` maps to
``registry.get(name).execute(...)``. It mirrors the *host* ``tool.execute`` branch
of :meth:`mini_devin.orchestrator.agent.Agent._execute_tool` but **without** the
agent’s process-sandbox / Docker-sandbox wrappers (those stay in
:class:`~mini_devin.backbone.controller.AgentHostRuntime` when using the backbone
from :meth:`~mini_devin.orchestrator.agent.Agent.run_simple`).
"""

from __future__ import annotations

import json
from typing import Any

from mini_devin.backbone.base_runtime import BaseRuntime
from mini_devin.backbone.event_models import (
    AgentEvent,
    CmdRunAction,
    CommandOutputObservation,
    ErrorObservation,
    FileWriteAction,
    FileWriteObservation,
    GenericToolAction,
    GenericToolObservation,
)
from mini_devin.backbone.event_stream import EventStream
from mini_devin.core.tool_interface import ToolExecutionError, ToolRegistry, get_global_registry
from mini_devin.schemas.tools import EditorAction, TerminalInput, ToolStatus, WriteFileInput


class LocalRuntime(BaseRuntime):
    """
    Default runtime: maps typed actions onto ``terminal`` / ``editor`` / other tools.

    Swap for :class:`~mini_devin.backbone.docker_runtime.DockerRuntime` by starting
    a different runtime against the same :class:`~mini_devin.backbone.event_stream.EventStream`.
    """

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        super().__init__()
        self._registry = registry or get_global_registry()

    async def handle_action(self, stream: EventStream, event: AgentEvent) -> None:
        if isinstance(event, CmdRunAction):
            await self._run_terminal(stream, event)
        elif isinstance(event, FileWriteAction):
            await self._run_write_file(stream, event)
        elif isinstance(event, GenericToolAction):
            await self._run_generic(stream, event)
        else:
            await stream.publish(
                ErrorObservation(
                    causation_id=event.id,
                    source_action_id=event.id,
                    message=f"Unsupported action type: {event.type}",  # type: ignore[union-attr]
                    error_type="unsupported_action",
                )
            )

    async def _run_terminal(self, stream: EventStream, action: CmdRunAction) -> None:
        tool = self._registry.get("terminal")
        if tool is None:
            await stream.publish(
                ErrorObservation(
                    causation_id=action.id,
                    source_action_id=action.id,
                    message="No 'terminal' tool registered",
                    error_type="missing_tool",
                )
            )
            return
        timeout = min(action.timeout_sec or 30, 300)
        payload: dict[str, Any] = TerminalInput(
            command=action.command,
            working_directory=action.cwd or ".",
            timeout_seconds=timeout,
        ).model_dump()
        try:
            out = await tool.execute(payload)
        except ToolExecutionError as e:
            await stream.publish(
                ErrorObservation(
                    causation_id=action.id,
                    source_action_id=action.id,
                    message=str(e),
                    error_type="tool_execution",
                )
            )
            return
        await stream.publish(
            CommandOutputObservation(
                causation_id=action.id,
                source_action_id=action.id,
                stdout=getattr(out, "stdout", "") or "",
                stderr=getattr(out, "stderr", "") or "",
                exit_code=int(getattr(out, "exit_code", -1)),
            )
        )

    async def _run_write_file(self, stream: EventStream, action: FileWriteAction) -> None:
        tool = self._registry.get("editor")
        if tool is None:
            await stream.publish(
                ErrorObservation(
                    causation_id=action.id,
                    source_action_id=action.id,
                    message="No 'editor' tool registered",
                    error_type="missing_tool",
                )
            )
            return
        payload = WriteFileInput(
            action=EditorAction.WRITE_FILE,
            path=action.path,
            content=action.content,
        ).model_dump()
        try:
            out = await tool.execute(payload)
        except ToolExecutionError as e:
            await stream.publish(
                ErrorObservation(
                    causation_id=action.id,
                    source_action_id=action.id,
                    message=str(e),
                    error_type="tool_execution",
                )
            )
            return
        st = getattr(out, "status", None)
        ok = st == ToolStatus.SUCCESS
        await stream.publish(
            FileWriteObservation(
                causation_id=action.id,
                source_action_id=action.id,
                path=getattr(out, "path", action.path),
                bytes_written=int(getattr(out, "bytes_written", 0)),
                success=ok,
            )
        )

    async def _run_generic(self, stream: EventStream, action: GenericToolAction) -> None:
        tool = self._registry.get(action.tool_name)
        if tool is None:
            await stream.publish(
                ErrorObservation(
                    causation_id=action.id,
                    source_action_id=action.id,
                    message=f"Unknown tool: {action.tool_name}",
                    error_type="missing_tool",
                )
            )
            return
        try:
            out = await tool.execute(dict(action.arguments))
        except ToolExecutionError as e:
            await stream.publish(
                ErrorObservation(
                    causation_id=action.id,
                    source_action_id=action.id,
                    message=str(e),
                    error_type="tool_execution",
                )
            )
            return
        body = out.model_dump_json() if hasattr(out, "model_dump_json") else json.dumps(out.model_dump())
        ok = getattr(out, "status", None) == ToolStatus.SUCCESS
        await stream.publish(
            GenericToolObservation(
                causation_id=action.id,
                source_action_id=action.id,
                tool_name=action.tool_name,
                content=body,
                success=ok,
            )
        )
