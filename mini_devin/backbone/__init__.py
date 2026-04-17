"""
Modular event-driven backbone (OpenHands-style).

This package is **additive**: the legacy :mod:`mini_devin.orchestrator.agent` loop
remains authoritative until you bridge it here.

Suggested layout (implemented under ``mini_devin/backbone/``)::

    backbone/
        event_models.py    # Pydantic Actions / Observations
        event_stream.py    # Central bus
        state_store.py     # Append-only history + reconstruction helpers
        base_runtime.py    # ABC for Local vs Docker
        local_runtime.py   # Host ToolRegistry execution
        docker_runtime.py  # docker run for CmdRunAction (extend for editor)

Next steps for migration:

1. Introduce ``AgentController`` that only publishes Actions and subscribes for Observations.
2. Replace direct ``_execute_tool`` calls with ``await stream.publish(GenericToolAction(...))``.
3. Map ``orchestrator/standard_events.py`` :class:`AgentStreamEvent` rows to these Pydantic events for persistence parity.
"""

from mini_devin.backbone.base_runtime import BaseRuntime
from mini_devin.backbone.controller import AgentController, AgentHostRuntime, observation_to_llm_text
from mini_devin.backbone.docker_runtime import DockerRuntime
from mini_devin.backbone.event_models import (
    Action,
    AgentEvent,
    AgentMessageEvent,
    CmdRunAction,
    CommandOutputObservation,
    ErrorObservation,
    FileWriteAction,
    FileWriteObservation,
    GenericToolAction,
    GenericToolObservation,
    Observation,
    is_action_event,
    is_observation_event,
    parse_agent_event,
)
from mini_devin.backbone.event_stream import EventStream
from mini_devin.backbone.local_runtime import LocalRuntime
from mini_devin.backbone.state_store import AgentStateStore, format_observation_as_tool_result

__all__ = [
    "Action",
    "AgentController",
    "AgentEvent",
    "AgentHostRuntime",
    "AgentMessageEvent",
    "AgentStateStore",
    "BaseRuntime",
    "CmdRunAction",
    "CommandOutputObservation",
    "DockerRuntime",
    "ErrorObservation",
    "EventStream",
    "format_observation_as_tool_result",
    "FileWriteAction",
    "FileWriteObservation",
    "GenericToolAction",
    "GenericToolObservation",
    "LocalRuntime",
    "Observation",
    "observation_to_llm_text",
    "is_action_event",
    "is_observation_event",
    "parse_agent_event",
]
