import json
from pathlib import Path

from plodder.memory.learned_patterns import append_learned_pattern, load_learned_patterns_for_prompt
from plodder.memory.session_memory import EpisodeMemory


def test_episode_memory_condense(tmp_path: Path) -> None:
    m = EpisodeMemory(tmp_path)
    for i in range(12):
        m.append("thought", {"text": f"t{i}"}, round_idx=i)
    txt = m.get_condensed_context(condense_after=10, keep_full=5)
    assert "Short History Summary" in txt
    assert "Recent events" in txt


def test_learned_patterns_roundtrip(tmp_path: Path) -> None:
    append_learned_pattern(tmp_path, "- Avoid bare `except`")
    s = load_learned_patterns_for_prompt(tmp_path)
    assert "Avoid bare" in s


def test_duplicate_warning(tmp_path: Path) -> None:
    m = EpisodeMemory(tmp_path)
    for _ in range(3):
        m.append("action", {"tool": "sandbox_shell", "args": {"argv": ["echo", "x"]}}, round_idx=0)
    ctx = m.get_condensed_context()
    assert "Anti-loop" in ctx or "echo" in ctx
