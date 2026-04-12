"""Tests for global corpus, golden context, specialization, workspace bootstrap."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_specialization_suffix_known():
    from mini_devin.agents.specialized_prompts import get_specialization_system_suffix

    assert "Clean code" in get_specialization_system_suffix("clean_code")
    assert "Performance" in get_specialization_system_suffix("performance")
    assert get_specialization_system_suffix("default") == ""


def test_golden_context_from_tmp(tmp_path: Path):
    from mini_devin.learning.golden_context import format_golden_context_for_prompt

    p = tmp_path / "g.jsonl"
    p.write_text(
        json.dumps(
            {
                "title": "t1",
                "messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    block = format_golden_context_for_prompt(p, max_records=5, max_chars=2000)
    assert "Golden examples" in block
    assert "hello" in block


def test_global_corpus_block_empty_query():
    from mini_devin.memory.global_corpus import format_global_corpus_block

    assert format_global_corpus_block("", limit=3) == ""


def test_workspace_bootstrap_detects_git(tmp_path: Path):
    from mini_devin.orchestrator.workspace_bootstrap import run_workspace_bootstrap

    (tmp_path / ".git").mkdir()
    (tmp_path / "README.md").write_text("# X", encoding="utf-8")

    class Dummy:
        def __init__(self):
            self.working_directory = str(tmp_path)
            self._mem: list[tuple[str, str, str]] = []

        def index_workspace(self, force: bool = False):
            pass

        def add_to_memory(self, content: str, item_type: str = "context", priority: str = "medium"):
            self._mem.append((item_type, priority, content[:40]))

    d = Dummy()
    stats = run_workspace_bootstrap(d)
    assert stats.get("already_git_checkout") is True
    assert any(m[0] == "constraint" for m in d._mem)


def test_workspace_bootstrap_dummy(tmp_path: Path):
    from mini_devin.orchestrator.workspace_bootstrap import run_workspace_bootstrap

    (tmp_path / "README.md").write_text("# Demo\n\nHello repo.", encoding="utf-8")

    class Dummy:
        def __init__(self):
            self.working_directory = str(tmp_path)
            self._mem: list[str] = []

        def index_workspace(self, force: bool = False):
            self.indexed = True

        def add_to_memory(self, content: str, item_type: str = "context", priority: str = "medium"):
            self._mem.append(content[:120])

    d = Dummy()
    stats = run_workspace_bootstrap(d)
    assert stats.get("readme_chars", 0) > 0
    assert d._mem
