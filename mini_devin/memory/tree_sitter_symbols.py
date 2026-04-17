"""
AST-based symbol extraction using Tree-sitter (Python, TypeScript, TSX).

Used by SymbolIndex for structural retrieval and optional RAG chunks.
Falls back silently if grammars are not installed.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Callable, Optional

# tree_sitter.Language / Parser
try:
    from tree_sitter import Language as TSLanguage
    from tree_sitter import Parser as TSParser
except ImportError:  # pragma: no cover
    TSLanguage = None  # type: ignore[misc, assignment]
    TSParser = None  # type: ignore[misc, assignment]


@dataclass
class ExtractedSymbol:
    """Intermediate symbol from Tree-sitter walk (mapped to Symbol in symbol_index)."""

    name: str
    kind: str  # "class" | "function" | "method"
    start_line: int
    end_line: int
    parent: Optional[str]
    signature: str
    language: str
    body_preview: str = ""


def _has_grammars() -> bool:
    if TSParser is None or importlib.util.find_spec("tree_sitter") is None:
        return False
    return importlib.util.find_spec("tree_sitter_python") is not None or importlib.util.find_spec(
        "tree_sitter_typescript"
    ) is not None


def _load_python_parser() -> Optional[Callable[[], object]]:
    try:
        import tree_sitter_python as tspython

        lang = TSLanguage(tspython.language())
        return lambda: TSParser(lang)
    except Exception:
        return None


def _load_ts_parser(tsx: bool) -> Optional[Callable[[], object]]:
    try:
        import tree_sitter_typescript as tsts

        cap = tsts.language_tsx() if tsx else tsts.language_typescript()
        lang = TSLanguage(cap)
        return lambda: TSParser(lang)
    except Exception:
        return None


def _node_text(src: bytes, start_byte: int, end_byte: int) -> str:
    return src[start_byte:end_byte].decode("utf-8", errors="replace")


def _first_identifier(node, src: bytes) -> Optional[str]:
    for i in range(node.child_count):
        c = node.child(i)
        if c.type == "identifier":
            return _node_text(src, c.start_byte, c.end_byte)
    return None


def _first_type_identifier(node, src: bytes) -> Optional[str]:
    for i in range(node.child_count):
        c = node.child(i)
        if c.type in ("type_identifier", "identifier"):
            return _node_text(src, c.start_byte, c.end_byte)
    return None


def _find_child(node, type_name: str):
    for i in range(node.child_count):
        c = node.child(i)
        if c.type == type_name:
            return c
    return None


def _signature_snippet(src: bytes, node, max_len: int = 200) -> str:
    raw = _node_text(src, node.start_byte, node.end_byte)
    raw = " ".join(raw.split())
    return raw[:max_len] + ("…" if len(raw) > max_len else "")


def _body_preview(src: bytes, block_node, max_lines: int = 40) -> str:
    if block_node is None:
        return ""
    text = _node_text(src, block_node.start_byte, block_node.end_byte)
    lines = text.splitlines()
    if len(lines) > max_lines:
        return "\n".join(lines[:max_lines]) + "\n…"
    return "\n".join(lines)


def _extract_python(src: bytes, parser_factory: Callable[[], object]) -> list[ExtractedSymbol]:
    parser = parser_factory()
    tree = parser.parse(src)
    out: list[ExtractedSymbol] = []

    def visit(node, parent_class: Optional[str], parent_fn: Optional[str]) -> None:
        if node.type == "decorated_definition":
            for i in range(node.child_count):
                ch = node.child(i)
                if ch.type in ("class_definition", "function_definition"):
                    visit(ch, parent_class, parent_fn)
            return

        if node.type == "class_definition":
            name = _first_identifier(node, src)
            if not name:
                return
            block = _find_child(node, "block")
            sig = _signature_snippet(src, node)
            preview = _body_preview(src, block) if block else ""
            out.append(
                ExtractedSymbol(
                    name=name,
                    kind="class",
                    start_line=node.start_point.row + 1,
                    end_line=node.end_point.row + 1,
                    parent=parent_class,
                    signature=sig,
                    language="python",
                    body_preview=preview,
                )
            )
            if block:
                for i in range(block.child_count):
                    visit(block.child(i), name, None)
            return

        if node.type == "function_definition":
            name = _first_identifier(node, src)
            if not name:
                return
            block = _find_child(node, "block")
            kind = "method" if parent_class else "function"
            par = parent_class if parent_class else parent_fn
            sig = _signature_snippet(src, node)
            preview = _body_preview(src, block) if block else ""
            out.append(
                ExtractedSymbol(
                    name=name,
                    kind=kind,
                    start_line=node.start_point.row + 1,
                    end_line=node.end_point.row + 1,
                    parent=par,
                    signature=sig,
                    language="python",
                    body_preview=preview,
                )
            )
            if block:
                for i in range(block.child_count):
                    visit(block.child(i), parent_class, name)
            return

        for i in range(node.child_count):
            visit(node.child(i), parent_class, parent_fn)

    visit(tree.root_node, None, None)
    return out


def _extract_typescript(src: bytes, parser_factory: Callable[[], object], lang_label: str) -> list[ExtractedSymbol]:
    parser = parser_factory()
    tree = parser.parse(src)
    out: list[ExtractedSymbol] = []

    def visit(node, parent_class: Optional[str]) -> None:
        if node.type == "class_declaration":
            name = _first_type_identifier(node, src)
            if not name:
                return
            body = _find_child(node, "class_body")
            sig = _signature_snippet(src, node)
            preview = _body_preview(src, body) if body else ""
            out.append(
                ExtractedSymbol(
                    name=name,
                    kind="class",
                    start_line=node.start_point.row + 1,
                    end_line=node.end_point.row + 1,
                    parent=parent_class,
                    signature=sig,
                    language=lang_label,
                    body_preview=preview,
                )
            )
            if body:
                for i in range(body.child_count):
                    ch = body.child(i)
                    if ch.type == "method_definition":
                        pname = None
                        for j in range(ch.child_count):
                            cc = ch.child(j)
                            if cc.type == "property_identifier":
                                pname = _node_text(src, cc.start_byte, cc.end_byte)
                                break
                        if pname:
                            mb = _find_child(ch, "statement_block")
                            sig_m = _signature_snippet(src, ch)
                            prv = _body_preview(src, mb) if mb else ""
                            out.append(
                                ExtractedSymbol(
                                    name=pname,
                                    kind="method",
                                    start_line=ch.start_point.row + 1,
                                    end_line=ch.end_point.row + 1,
                                    parent=name,
                                    signature=sig_m,
                                    language=lang_label,
                                    body_preview=prv,
                                )
                            )
                    else:
                        visit(ch, name)
            return

        if node.type == "function_declaration":
            fname = _first_identifier(node, src)
            if not fname:
                return
            block = _find_child(node, "statement_block")
            sig = _signature_snippet(src, node)
            preview = _body_preview(src, block) if block else ""
            out.append(
                ExtractedSymbol(
                    name=fname,
                    kind="function",
                    start_line=node.start_point.row + 1,
                    end_line=node.end_point.row + 1,
                    parent=parent_class,
                    signature=sig,
                    language=lang_label,
                    body_preview=preview,
                )
            )
            if block:
                for i in range(block.child_count):
                    visit(block.child(i), parent_class)
            return

        for i in range(node.child_count):
            visit(node.child(i), parent_class)

    visit(tree.root_node, None)
    return out


def extract_symbols_ast(rel_path: str, content: str, language: str) -> list[ExtractedSymbol]:
    """
    Parse file content and return extracted symbols. Empty list if unsupported or parse fails.
    """
    if TSParser is None or not content.strip():
        return []
    src = content.encode("utf-8")

    try:
        if rel_path.endswith(".tsx"):
            factory = _load_ts_parser(tsx=True)
            if not factory:
                return []
            return _extract_typescript(src, factory, "tsx")
        if language == "python":
            factory = _load_python_parser()
            if not factory:
                return []
            return _extract_python(src, factory)
        if language == "typescript":
            factory = _load_ts_parser(tsx=False)
            if not factory:
                return []
            return _extract_typescript(src, factory, "typescript")
        if language == "javascript":
            is_tsx = rel_path.endswith(".tsx") or rel_path.endswith(".jsx")
            factory = _load_ts_parser(tsx=is_tsx)
            if not factory:
                return []
            label = "tsx" if rel_path.endswith(".tsx") else "javascript"
            return _extract_typescript(src, factory, label)
    except Exception:
        return []


def rag_chunks_from_symbols(rel_path: str, symbols: list[ExtractedSymbol]) -> list[tuple[str, dict]]:
    """
    Build (text, metadata) pairs for vector store — one short doc per symbol for semantic hit quality.
    """
    chunks: list[tuple[str, dict]] = []
    for s in symbols:
        qn = f"{s.parent}.{s.name}" if s.parent else s.name
        header = f"[{s.kind}] {qn} ({rel_path}:{s.start_line})"
        body = f"{s.signature}\n{s.body_preview}".strip()
        text = f"{header}\n{body}".strip()
        if len(text) < 8:
            continue
        chunks.append(
            (
                text,
                {
                    "file_path": rel_path,
                    "chunk_type": "ast_symbol",
                    "symbol": qn,
                    "kind": s.kind,
                    "start_line": s.start_line,
                    "language": s.language,
                },
            )
        )
    return chunks
