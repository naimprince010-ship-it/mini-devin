"""
Plodder SDK — programmatic API (OpenHands-style composable library surface).

Use this when embedding Plodder in scripts, CI, or another Python service.
The web UI and CLI remain separate entrypoints; they share the same agent core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mini_devin.orchestrator.agent import Agent, create_agent

__all__ = [
    "Agent",
    "PlodderClient",
    "create_agent",
]


@dataclass
class PlodderClient:
    """
    Thin async client over :func:`create_agent` for one-liner task runs.

    Example::

        client = PlodderClient(working_directory=\"/path/to/repo\")
        summary = await client.run_task(\"Add a health check endpoint\")
    """

    model: str = "gpt-4o"
    api_key: str | None = None
    working_directory: str | None = None
    verbose: bool = False
    callbacks: dict[str, Any] | None = None
    _agent: Agent | None = None

    async def connect(self) -> Agent:
        """Build the underlying :class:`~mini_devin.orchestrator.agent.Agent`."""
        self._agent = await create_agent(
            model=self.model,
            api_key=self.api_key,
            working_directory=self.working_directory,
            verbose=self.verbose,
            callbacks=self.callbacks,
        )
        return self._agent

    async def run_task(self, instruction: str) -> str:
        """Run a single natural-language task and return the agent summary text."""
        if self._agent is None:
            await self.connect()
        assert self._agent is not None
        return await self._agent.run_simple(instruction)

    @property
    def agent(self) -> Agent:
        """Direct access to the agent after :meth:`connect` or :meth:`run_task`."""
        if self._agent is None:
            raise RuntimeError("Agent not initialized; await connect() or run_task() first.")
        return self._agent
