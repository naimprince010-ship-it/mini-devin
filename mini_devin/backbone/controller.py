"""
AgentController — OpenHands-style orchestration over :class:`EventStream`.

OpenHands pattern (simplified): the controller / policy issues **Actions**; a
:class:`~mini_devin.backbone.base_runtime.BaseRuntime` consumes them and emits
**Observations** on the same stream; the controller blocks until the matching
observation for that action id arrives, then continues (next LLM turn, etc.).
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from mini_devin.backbone.base_runtime import BaseRuntime
from mini_devin.backbone.event_models import (
    AgentEvent,
    GenericToolAction,
    GenericToolObservation,
    is_action_event,
    is_observation_event,
)
from mini_devin.backbone.event_stream import EventStream
from mini_devin.backbone.state_store import AgentStateStore, format_observation_as_tool_result


def observation_to_llm_text(ev: AgentEvent) -> str:
    """Flatten an observation event to a string for tool results."""
    if isinstance(ev, GenericToolObservation):
        return ev.content
    return format_observation_as_tool_result(ev)


class AgentController:
    """
    Owns the stream + store + runtime lifecycle; exposes ``dispatch_and_wait``.

    Typical wiring::

        store = AgentStateStore()
        stream = EventStream(store=store)
        runtime = LocalRuntime(agent.registry)  # or AgentHostRuntime(agent)
        ctrl = AgentController(stream, store, runtime)
        await ctrl.start()
        text = await ctrl.dispatch_and_wait(\"terminal\", {\"command\": \"echo hi\"})
        await ctrl.stop()
    """

    def __init__(
        self,
        stream: EventStream,
        state_store: AgentStateStore,
        runtime: BaseRuntime,
    ) -> None:
        self.stream = stream
        self.store = state_store
        self.runtime = runtime
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        await self.runtime.start(self.stream)
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        await self.runtime.stop()
        self._started = False

    async def dispatch_and_wait(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> str:
        """
        Publish a :class:`GenericToolAction` and block until its observation arrives.

        The runtime must publish an observation whose ``source_action_id`` equals
        the action's ``id``.
        """
        if not self._started:
            await self.start()
        to = timeout or float(os.environ.get("PLODDER_BACKBONE_TOOL_TIMEOUT", "600"))
        action = GenericToolAction(tool_name=tool_name, arguments=dict(arguments))
        # Subscribe before publish so a fast runtime cannot miss delivery of the observation.
        q = self.stream.register_subscriber()
        try:
            await self.stream.publish(action)
            deadline = asyncio.get_event_loop().time() + to
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Backbone: no observation for action {action.id} within {to}s",
                    )
                ev = await asyncio.wait_for(q.get(), timeout=remaining)
                if is_action_event(ev):
                    continue
                if is_observation_event(ev) and getattr(ev, "source_action_id", None) == action.id:
                    if ev.type == "ErrorObservation":  # type: ignore[union-attr]
                        return observation_to_llm_text(ev)
                    return observation_to_llm_text(ev)
        finally:
            self.stream.unregister_subscriber(q)


class AgentHostRuntime(BaseRuntime):
    """
    Runtime that executes tools by delegating to :meth:`Agent._execute_tool` with
    ``_inline_backbone=True`` (full parity: sandbox, session JSONL, artifacts).

    Used when stream dispatch must not lose Agent-specific behaviour.
    """

    def __init__(self, agent: Any) -> None:
        super().__init__()
        self._agent = agent

    async def handle_action(self, stream: EventStream, event: AgentEvent) -> None:
        if not isinstance(event, GenericToolAction):
            from mini_devin.backbone.event_models import ErrorObservation

            await stream.publish(
                ErrorObservation(
                    causation_id=event.id,
                    source_action_id=event.id,
                    message="AgentHostRuntime only handles GenericToolAction",
                    error_type="unsupported_action",
                )
            )
            return

        text = await self._agent._execute_tool(
            event.tool_name,
            event.arguments,
            thought=None,
            activity_source="backbone_host",
            _inline_backbone=True,
        )
        await stream.publish(
            GenericToolObservation(
                causation_id=event.id,
                source_action_id=event.id,
                tool_name=event.tool_name,
                content=text,
                success=not str(text).startswith("Error:"),
            )
        )
