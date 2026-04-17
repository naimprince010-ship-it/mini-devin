"""Abstract runtime: consumes Action events, emits Observation events."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from mini_devin.backbone.event_models import AgentEvent, is_action_event
from mini_devin.backbone.event_stream import EventStream


class BaseRuntime(ABC):
    """
    Pluggable execution environment.

    Subclasses run a background task that reads from a subscriber queue on the
    :class:`EventStream`, execute **actions** only, and ``publish`` matching
    observations. Policy (LLM / controller) never shells out directly.
    """

    def __init__(self) -> None:
        self._task: asyncio.Task[None] | None = None
        self._queue: asyncio.Queue[AgentEvent] | None = None
        self._stream: EventStream | None = None

    @abstractmethod
    async def handle_action(self, stream: EventStream, event: AgentEvent) -> None:
        """Execute one action and publish observation(s) to ``stream``."""

    async def _consume_loop(self, stream: EventStream) -> None:
        assert self._queue is not None
        q = self._queue
        while True:
            ev = await q.get()
            if is_action_event(ev):
                await self.handle_action(stream, ev)

    async def start(self, stream: EventStream) -> None:
        if self._task is not None:
            return
        self._stream = stream
        self._queue = stream.register_subscriber()
        self._task = asyncio.create_task(self._consume_loop(stream), name=f"{type(self).__name__}-consume")

    async def stop(self) -> None:
        if self._stream is not None and self._queue is not None:
            self._stream.unregister_subscriber(self._queue)
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._queue = None
        self._stream = None
