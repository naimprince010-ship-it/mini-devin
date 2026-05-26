"""Tests for Plodder's autonomous coding-agent operating policy."""

from mini_devin.orchestrator.agent import SYSTEM_PROMPT, _runtime_context_block


def test_system_prompt_contains_autonomous_coding_policy():
    prompt = SYSTEM_PROMPT.lower()

    assert "task vs question" in prompt
    assert "plan before coding" in prompt
    assert "inspect repository structure" in prompt
    assert "use terminal carefully" in prompt
    assert "minimize diffs" in prompt
    assert "risky edits need reasoning" in prompt
    assert "verify after modifications" in prompt
    assert "retry intelligently" in prompt
    assert "follow existing style" in prompt


def test_task_context_reinforces_agent_policy():
    context = _runtime_context_block("workspace").lower()

    assert "tests" in context
    assert "unit tests you write" in context
