"""
Collect editor diagnostics (squiggles) for Python and TypeScript/JavaScript.

Pyright / basedpyright when on PATH or ``PYRIGHT_CMD``; else ``ast`` syntax-only for Python.
``tsc`` via ``npx`` for .ts/.tsx when available.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DiagnosticItem:
    """1-based lines/columns for Monaco ``IMarkerData``."""

    line: int
    start_column: int
    end_line: int
    end_column: int
    message: str
    severity: str  # error | warning | information | hint

    def to_dict(self) -> dict[str, Any]:
        return {
            "line": self.line,
            "startColumn": self.start_column,
            "endLine": self.end_line,
            "endColumn": self.end_column,
            "message": self.message,
            "severity": self.severity,
        }


def _pyright_command() -> list[str] | None:
    custom = os.environ.get("PYRIGHT_CMD", "").strip()
    if custom:
        return custom.split()
    for exe in ("basedpyright", "pyright"):
        p = shutil.which(exe)
        if p:
            return [p]
    return None


def _python_syntax_diagnostics(path: str, source: str) -> tuple[list[DiagnosticItem], str]:
    try:
        ast.parse(source, filename=path)
    except SyntaxError as e:
        if e.lineno is None:
            return (
                [DiagnosticItem(1, 1, 1, 2, str(e), "error")],
                "syntax",
            )
        col = e.offset or 1
        line = e.lineno
        # end column rough
        return (
            [
                DiagnosticItem(
                    line,
                    col,
                    line,
                    col + 1,
                    e.msg or "SyntaxError",
                    "error",
                )
            ],
            "syntax",
        )
    return [], "syntax"


def _pyright_diagnostics(workspace: str, rel_path: str, abs_path: str) -> tuple[list[DiagnosticItem], str]:
    cmd_base = _pyright_command()
    if not cmd_base:
        with open(abs_path, encoding="utf-8", errors="replace") as f:
            return _python_syntax_diagnostics(rel_path, f.read())

    try:
        proc = subprocess.run(
            [*cmd_base, abs_path, "--outputjson"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=int(os.environ.get("PYRIGHT_TIMEOUT", "90")),
            env={**os.environ, "PYTHONUTF8": "1"},
        )
        raw = proc.stdout.strip() or proc.stderr.strip()
        if not raw:
            with open(abs_path, encoding="utf-8", errors="replace") as f:
                return _python_syntax_diagnostics(rel_path, f.read())
        data = json.loads(raw)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        with open(abs_path, encoding="utf-8", errors="replace") as f:
            return _python_syntax_diagnostics(rel_path, f.read())

    def _same_file(a: str, b: str) -> bool:
        try:
            return os.path.samefile(a, b)
        except OSError:
            return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))

    out: list[DiagnosticItem] = []
    for d in data.get("generalDiagnostics", []):
        fpath = d.get("file", "")
        if not _same_file(fpath, abs_path):
            continue
        sev = d.get("severity", "error")
        if sev not in ("error", "warning", "information"):
            sev = "error"
        rng = d.get("range") or {}
        start = rng.get("start") or {}
        end = rng.get("end") or {}
        # Pyright JSON is 0-based lines
        sl = int(start.get("line", 0)) + 1
        sc = int(start.get("character", 0)) + 1
        el = int(end.get("line", start.get("line", 0))) + 1
        ec = int(end.get("character", start.get("character", 0))) + 1
        msg = d.get("message", "diagnostic")
        out.append(DiagnosticItem(sl, sc, el, ec, msg, sev))
    return out, "pyright"


def _tsc_diagnostics(abs_path: str) -> tuple[list[DiagnosticItem], str]:
    npx = shutil.which("npx")
    if not npx:
        return [], "none"
    # Single-file check; may miss project types but gives parse + basic errors
    try:
        proc = subprocess.run(
            [npx, "--yes", "typescript", "tsc", "--noEmit", "--pretty", "false", abs_path],
            capture_output=True,
            text=True,
            timeout=int(os.environ.get("TSC_TIMEOUT", "120")),
            cwd=os.path.dirname(abs_path) or ".",
        )
        text = proc.stderr or proc.stdout or ""
    except (subprocess.TimeoutExpired, OSError):
        return [], "none"

    # file.ts(10,5): error TS2304: message
    pat = re.compile(
        r"^(?P<file>.+?)\((?P<line>\d+),(?P<col>\d+)\):\s*(?P<kind>error|warning)\s+TS\d+:\s*(?P<msg>.+)$"
    )
    out: list[DiagnosticItem] = []
    def _same_file(a: str, b: str) -> bool:
        try:
            return os.path.samefile(a, b)
        except OSError:
            return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))

    for line in text.splitlines():
        m = pat.match(line.strip())
        if not m:
            continue
        if not _same_file(m.group("file"), abs_path):
            continue
        ln = int(m.group("line"))
        col = int(m.group("col"))
        sev = "error" if m.group("kind") == "error" else "warning"
        msg = m.group("msg").strip()
        out.append(DiagnosticItem(ln, col, ln, col + 1, msg, sev))
    return out, "tsc"


def collect_diagnostics(
    workspace_abs: str,
    rel_path: str,
    content: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """
    Return (list of Monaco-style diagnostic dicts, source label).

    If ``content`` is set, it is written to a temp copy of the file for checking
    (does not modify the workspace file).
    """
    base = os.path.abspath(workspace_abs)
    target = os.path.abspath(os.path.join(base, rel_path))
    if not target.startswith(base):
        return [], "none"
    ext = Path(rel_path).suffix.lower()

    if ext == ".py":
        if content is not None:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".py",
                delete=False,
            ) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            try:
                items, src = _pyright_diagnostics(base, rel_path, tmp_path)
                if not items and src == "pyright":
                    items, src = _python_syntax_diagnostics(rel_path, content)
                return [i.to_dict() for i in items], src
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        if not os.path.isfile(target):
            return [], "none"
        items, src = _pyright_diagnostics(base, rel_path, target)
        if not items and src == "pyright":
            with open(target, encoding="utf-8", errors="replace") as f:
                items, src = _python_syntax_diagnostics(rel_path, f.read())
        return [i.to_dict() for i in items], src

    if ext in (".ts", ".tsx", ".js", ".jsx"):
        src_text = content
        if src_text is None:
            if not os.path.isfile(target):
                return [], "none"
            with open(target, encoding="utf-8", errors="replace") as f:
                src_text = f.read()
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=ext,
            delete=False,
        ) as tmp:
            tmp.write(src_text)
            tmp_path = tmp.name
        try:
            items, src = _tsc_diagnostics(tmp_path)
            return [i.to_dict() for i in items], src
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return [], "none"
