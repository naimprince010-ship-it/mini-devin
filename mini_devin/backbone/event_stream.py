"""Central pub/sub stream: all Actions and Observations flow through here."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from mini_devin.backbone.event_models import AgentEvent
from mini_devin.backbone.state_store import AgentStateStore


class EventStream:
    """
    Backbone bus: publish typed events; runtimes and UIs subscribe via queues.

    Every publish is appended to the bound :class:`AgentStateStore` first, then
    fan-out to subscriber queues (OpenHands-style EventStream, simplified).
    """

    def __init__(
        self,
        store: AgentStateStore | None = None,
        *,
        on_publish: Callable[[AgentEvent], Awaitable[None]] | None = None,
    ) -> None:
        self._store = store or AgentStateStore()
        self._on_publish = on_publish
        self._queues: list[asyncio.Queue[AgentEvent]] = []
        self._meta_lock = asyncio.Lock()

    @property
    def store(self) -> AgentStateStore:
        return self._store

    def register_subscriber(self) -> asyncio.Queue[AgentEvent]:
        """Return a new queue that receives every subsequent event (fan-out)."""
        q: asyncio.Queue[AgentEvent] = asyncio.Queue()
        self._queues.append(q)
        return q

    def unregister_subscriber(self, q: asyncio.Queue[AgentEvent]) -> None:
        try:
            self._queues.remove(q)
        except ValueError:
            pass

    async def publish(self, event: AgentEvent) -> None:
        await self._store.append(event)
        if self._on_publish is not None:
            await self._on_publish(event)
        async with self._meta_lock:
            targets = list(self._queues)
        for q in targets:
            await q.put(event)
