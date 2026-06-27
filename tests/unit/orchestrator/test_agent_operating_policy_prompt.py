"""Tests for Plodder's autonomous coding-agent operating policy."""

from mini_devin.orchestrator.agent import Agent, SYSTEM_PROMPT, _runtime_context_block


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


def test_task_prompt_compactor_reduces_large_structured_prompt(monkeypatch):
    monkeypatch.setenv("PLODDER_TASK_PROMPT_COMPACT", "true")
    monkeypatch.setenv("PLODDER_TASK_PROMPT_COMPACT_MAX_CHARS", "500")

    large_prompt = "\n".join(
        [
            "You are an autonomous software engineering agent. Improve this project safely.",
            "Requirements:",
            "1. Analyze repository structure.",
            "2. Identify top 5 issues.",
            "3. Create prioritized plan.",
            "4. Implement highest-impact task.",
            "5. Add tests and run lint/format/tests.",
            "Success Criteria: all tests pass and no lint warnings.",
            "Extra details: " + ("very long context " * 120),
        ]
    )

    compact = Agent._compact_task_description_for_prompt(large_prompt)

    assert "Key constraints:" in compact
    assert "Analyze repository structure" in compact
    assert "auto-compacted for token efficiency" in compact
    assert len(compact) < len(large_prompt)


def test_task_prompt_compactor_can_be_disabled(monkeypatch):
    monkeypatch.setenv("PLODDER_TASK_PROMPT_COMPACT", "false")
    raw = "Implement a feature and run tests."
    assert Agent._compact_task_description_for_prompt(raw) == raw


def test_acceptance_criteria_compactor_caps_items_and_length(monkeypatch):
    monkeypatch.setenv("PLODDER_TASK_CRITERIA_MAX_ITEMS", "2")
    monkeypatch.setenv("PLODDER_TASK_CRITERION_MAX_CHARS", "20")
    criteria = [
        "First criterion with a lot of extra words to exceed limit",
        "Second criterion also quite long for truncation",
        "Third criterion should be omitted",
    ]

    out = Agent._compact_acceptance_criteria(criteria)
    assert len(out) == 2
    assert out[0].endswith("…")
    assert out[1].endswith("…")
