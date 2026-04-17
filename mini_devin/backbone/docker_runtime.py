"""Isolated command execution via Docker (stub-friendly, opt-in).

For **full** Plodder tool parity (editor, browser, GitHub, …), keep using
:class:`~mini_devin.backbone.local_runtime.LocalRuntime` or
:class:`~mini_devin.backbone.controller.AgentHostRuntime` on the host; use
Docker here for **CmdRunAction** isolation or extend this class with
``docker exec`` patterns (OpenHands: Runtime implements Action → Observation).
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

from mini_devin.backbone.base_runtime import BaseRuntime
from mini_devin.backbone.event_models import (
    AgentEvent,
    CmdRunAction,
    CommandOutputObservation,
    ErrorObservation,
    FileWriteAction,
    GenericToolAction,
)
from mini_devin.backbone.event_stream import EventStream


class DockerRuntime(BaseRuntime):
    """
    Runs :class:`CmdRunAction` inside ``docker run`` with a workspace bind-mount.

    - Image: ``PLODDER_DOCKER_RUNTIME_IMAGE`` (default ``python:3.11-slim``).
    - Mount: host ``workspace_root`` → container ``/workspace``.

    **Scope:** ``CmdRunAction`` and ``GenericToolAction`` with ``tool_name="terminal"``
    run via ``docker run``. Other ``GenericToolAction`` / ``FileWriteAction`` emit
    :class:`ErrorObservation` until extended (e.g. ``docker exec`` + sidecar sync).
    """

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        container_workdir: str = "/workspace",
        image: str | None = None,
    ) -> None:
        super().__init__()
        self._workspace_root = str(Path(workspace_root).resolve())
        self._container_workdir = container_workdir
        self._image = image or os.environ.get("PLODDER_DOCKER_RUNTIME_IMAGE", "python:3.11-slim")

    async def handle_action(self, stream: EventStream, event: AgentEvent) -> None:
        if isinstance(event, CmdRunAction):
            await self._docker_shell(stream, event)
        elif isinstance(event, GenericToolAction) and event.tool_name == "terminal":
            cmd = str(event.arguments.get("command", "") or "").strip()
            if not cmd:
                await stream.publish(
                    ErrorObservation(
                        causation_id=event.id,
                        source_action_id=event.id,
                        message="DockerRuntime: terminal action requires non-empty command",
                        error_type="validation",
                    )
                )
                return
            raw_to = event.arguments.get("timeout_seconds", 30)
            try:
                to = int(raw_to)
            except (TypeError, ValueError):
                to = 30
            to = max(1, min(3600, to))
            wd = event.arguments.get("working_directory")
            inner = CmdRunAction(
                id=event.id,
                ts=event.ts,
                causation_id=event.causation_id,
                command=cmd,
                cwd=str(wd) if wd else None,
                timeout_sec=to,
            )
            await self._docker_shell(stream, inner)
        elif isinstance(event, (FileWriteAction, GenericToolAction)):
            await stream.publish(
                ErrorObservation(
                    causation_id=event.id,
                    source_action_id=event.id,
                    message=(
                        "DockerRuntime: FileWriteAction and non-terminal GenericToolAction "
                        "are not implemented in-container yet. Use LocalRuntime / AgentHostRuntime."
                    ),
                    error_type="unsupported_action",
                )
            )
        else:
            await stream.publish(
                ErrorObservation(
                    causation_id=event.id,
                    source_action_id=event.id,
                    message=f"Unsupported action: {type(event).__name__}",
                    error_type="unsupported_action",
                )
            )

    async def _docker_shell(self, stream: EventStream, action: CmdRunAction) -> None:
        if not shutil.which("docker"):
            await stream.publish(
                ErrorObservation(
                    causation_id=action.id,
                    source_action_id=action.id,
                    message="docker CLI not found on PATH",
                    error_type="docker_missing",
                )
            )
            return

        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{self._workspace_root}:{self._container_workdir}",
            "-w",
            self._container_workdir,
            self._image,
            "sh",
            "-c",
            action.command,
        ]
        timeout = float(min(action.timeout_sec or 300, 3600))
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await stream.publish(
                ErrorObservation(
                    causation_id=action.id,
                    source_action_id=action.id,
                    message=f"docker run timed out after {timeout}s",
                    error_type="timeout",
                )
            )
            return
        except OSError as e:
            await stream.publish(
                ErrorObservation(
                    causation_id=action.id,
                    source_action_id=action.id,
                    message=str(e),
                    error_type="os_error",
                )
            )
            return

        stdout = stdout_b.decode(errors="replace")
        stderr = stderr_b.decode(errors="replace")
        code = int(proc.returncode or 0)
        await stream.publish(
            CommandOutputObservation(
                causation_id=action.id,
                source_action_id=action.id,
                stdout=stdout,
                stderr=stderr,
                exit_code=code,
            )
        )
