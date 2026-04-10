"""Tests for Tree-sitter AST symbol extraction."""

import importlib.util

import pytest

from mini_devin.memory.tree_sitter_symbols import (
    extract_symbols_ast,
    rag_chunks_from_symbols,
)


PY_SAMPLE = '''\
class Greeter:
    """Doc."""
    def hello(self, name: str) -> str:
        return f"hi {name}"

def top():
    pass
'''


def test_extract_python_symbols():
    syms = extract_symbols_ast("sample.py", PY_SAMPLE, "python")
    names = {(s.kind, s.name) for s in syms}
    assert ("class", "Greeter") in names
    assert ("method", "hello") in names
    assert ("function", "top") in names
    hello = next(s for s in syms if s.name == "hello")
    assert hello.parent == "Greeter"


TS_SAMPLE = """export class Box {
  size(): number { return 1; }
}
function outer() { return 0; }
"""


def test_extract_typescript_symbols():
    syms = extract_symbols_ast("sample.ts", TS_SAMPLE, "typescript")
    kinds = {(s.kind, s.name) for s in syms}
    assert ("class", "Box") in kinds
    assert ("method", "size") in kinds
    assert ("function", "outer") in kinds


def test_rag_chunks_metadata():
    syms = extract_symbols_ast("a.py", PY_SAMPLE, "python")
    chunks = rag_chunks_from_symbols("a.py", syms)
    assert chunks
    text, meta = chunks[0]
    assert "ast_symbol" == meta.get("chunk_type")
    assert "file_path" in meta
    assert len(text) > 5


@pytest.mark.skipif(
    importlib.util.find_spec("tree_sitter_python") is None,
    reason="tree-sitter-python not installed",
)
def test_skips_gracefully_empty_file():
    assert extract_symbols_ast("x.py", "", "python") == []
