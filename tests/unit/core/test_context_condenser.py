"""Unit tests for context condenser (OpenHands-style middle summarization)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from mini_devin.core.context_condenser import (
    condense_chat_messages,
    find_suffix_start_for_last_n_tool_observations,
    split_goal_prefix,
)


def test_split_goal_prefix_system_and_first_user():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "goal"},
        {"role": "assistant", "content": "hi"},
    ]
    prefix, end = split_goal_prefix(messages)
    assert len(prefix) == 2
    assert prefix[1]["content"] == "goal"
    assert end == 2


def test_find_suffix_start_links_assistant_to_tool():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "g"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_a", "name": "terminal", "content": "out1"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_b",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_b", "name": "terminal", "content": "out2"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_c",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_c", "name": "terminal", "content": "out3"},
    ]
    k = find_suffix_start_for_last_n_tool_observations(messages, last_n=3)
    assert k == 2
    suffix = messages[k:]
    assert len([m for m in suffix if m.get("role") == "tool"]) == 3


@pytest.fixture
def low_condense_thresholds(monkeypatch):
    monkeypatch.setenv("LLM_CONDENSE_MIN_MESSAGES", "4")
    monkeypatch.setenv("LLM_CONDENSE_MIN_CHARS", "50")


def test_condense_skips_without_tools(low_condense_thresholds):
    summarizer = MagicMock()
    summarizer.completion_ephemeral = AsyncMock(return_value="X")
    summarizer.config.max_tokens = 8192

    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "goal"},
        {"role": "assistant", "content": "hello"},
    ]

    async def _run():
        return await condense_chat_messages(messages, summarizer=summarizer)

    out = asyncio.run(_run())
    assert out == messages
    summarizer.completion_ephemeral.assert_not_called()


def test_condense_inserts_summary_block(low_condense_thresholds):
    summarizer = MagicMock()
    summarizer.completion_ephemeral = AsyncMock(return_value="MIDDLE_SUMMARY")
    summarizer.config.max_tokens = 8192

    filler = "x" * 200
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "goal"},
        {"role": "assistant", "content": filler},
        {"role": "user", "content": "follow-up instruction"},
        {"role": "assistant", "content": "more reasoning"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "t1",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "t1", "name": "terminal", "content": "o1"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "t2",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "t2", "name": "terminal", "content": "o2"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "t3",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "t3", "name": "terminal", "content": "o3"},
    ]

    async def _run():
        return await condense_chat_messages(messages, summarizer=summarizer)

    out = asyncio.run(_run())
    assert summarizer.completion_ephemeral.await_count == 1
    condensed = [m for m in out if m.get("role") == "user" and "Condensed prior context" in (m.get("content") or "")]
    assert len(condensed) == 1
    assert "MIDDLE_SUMMARY" in condensed[0]["content"]
    assert out[0]["role"] == "system"
    assert out[1]["content"] == "goal"
    assert out[-1]["content"] == "o3"


def test_find_suffix_start_none_when_no_assistant_for_tool_call_id():
    """Do not slice a suffix that begins with an orphan ``tool`` (would break OpenAI ordering)."""
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "goal"},
        {"role": "assistant", "content": "no tools here"},
        {"role": "tool", "tool_call_id": "orphan_id", "name": "terminal", "content": "out"},
    ]
    assert find_suffix_start_for_last_n_tool_observations(messages, last_n=1) is None


def test_condense_skips_when_suffix_would_start_with_incomplete_tool_batch(low_condense_thresholds):
    """Corrupt transcript: assistant issued two tool_calls but only one tool row — never condense."""
    summarizer = MagicMock()
    summarizer.completion_ephemeral = AsyncMock(return_value="SHOULD_NOT")
    summarizer.config.max_tokens = 8192
    filler = "z" * 200
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "goal"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "bad_a",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                },
                {
                    "id": "bad_b",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "bad_a", "name": "terminal", "content": "missing bad_b"},
        {"role": "user", "content": filler},
        {"role": "assistant", "content": "mid"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "ok_c",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "ok_c", "name": "terminal", "content": "done"},
    ]

    async def _run():
        return await condense_chat_messages(messages, summarizer=summarizer)

    out = asyncio.run(_run())
    assert out is messages
    summarizer.completion_ephemeral.assert_not_called()
