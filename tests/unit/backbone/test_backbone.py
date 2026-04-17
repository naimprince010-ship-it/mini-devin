"""Unit tests for mini_devin.backbone."""

import asyncio

import pytest

from mini_devin.backbone import (
    AgentMessageEvent,
    AgentStateStore,
    CmdRunAction,
    EventStream,
    parse_agent_event,
)


@pytest.mark.asyncio
async def test_event_stream_persists_and_fanout() -> None:
    store = AgentStateStore()
    stream = EventStream(store=store)
    q = stream.register_subscriber()

    msg = AgentMessageEvent(role="user", content="hello")
    await stream.publish(msg)

    assert await store.count() == 1
    got = await asyncio.wait_for(q.get(), timeout=2.0)
    assert got.id == msg.id
    assert got.type == "AgentMessageEvent"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_parse_agent_event_roundtrip() -> None:
    cmd = CmdRunAction(command="echo hi", cwd=".")
    data = cmd.model_dump(mode="json")
    parsed = parse_agent_event(data)
    assert isinstance(parsed, CmdRunAction)
    assert parsed.command == "echo hi"


@pytest.mark.asyncio
async def test_reconstruct_tail() -> None:
    store = AgentStateStore()
    await store.append(AgentMessageEvent(role="user", content="u"))
    await store.append(AgentMessageEvent(role="assistant", content="a"))
    msgs = await store.reconstruct_openai_style_tail(tail=10)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
