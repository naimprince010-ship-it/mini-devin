"""
LanceDB-backed semantic index of workspace source files (local RAG for ``search_codebase``).
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

_EMB_DIM = 256

_SKIP_DIRS = frozenset(
    {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".plodder",
        "dist",
        "build",
        ".tox",
    }
)

_TEXT_EXT = frozenset(
    {
        ".py",
        ".pyi",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".mjs",
        ".cjs",
        ".go",
        ".rs",
        ".java",
        ".kt",
        ".kts",
        ".cs",
        ".php",
        ".rb",
        ".swift",
        ".scala",
        ".sql",
        ".sh",
        ".yaml",
        ".yml",
        ".toml",
        ".json",
        ".md",
        ".html",
        ".css",
        ".vue",
        ".svelte",
    }
)

_TABLE = "code_chunks"


def _hash_bow_embed(text: str, dim: int = _EMB_DIM) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    for tok in re.findall(r"[\w\.#\+\-/@]+", text.lower()):
        h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16) % dim
        v[h] += 1.0
    n = float(np.linalg.norm(v)) or 1.0
    v /= n
    return v


def _chunk_text(text: str, *, max_chars: int = 1800) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


class WorkspaceCodeIndex:
    """
    Per-workspace Lance table under ``.plodder/workspace_code.lance/``.

    Deterministic hash embeddings (no remote embedding API).
    """

    def __init__(self, workspace_root: str | Path) -> None:
        self.root = Path(workspace_root).resolve()
        self._persist_dir = self.root / ".plodder" / "workspace_code.lance"
        self._db: Any = None
        self._table: Any = None

    def _connect(self) -> None:
        if self._db is not None:
            return
        import lancedb

        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self._persist_dir))

    def _open_table(self) -> bool:
        self._connect()
        assert self._db is not None
        try:
            self._table = self._db.open_table(_TABLE)
            return True
        except Exception:
            self._table = None
            return False

    def index_workspace(
        self,
        *,
        max_files: int = 400,
        max_file_bytes: int = 256_000,
    ) -> dict[str, Any]:
        """Walk workspace and rebuild the Lance table from text-like source files."""
        self._connect()
        assert self._db is not None
        rows: list[dict[str, Any]] = []
        seen = 0
        for dirpath, dirnames, filenames in os.walk(self.root, topdown=True):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            if any(p in _SKIP_DIRS for p in Path(dirpath).parts):
                continue
            for name in filenames:
                if seen >= max_files:
                    break
                p = Path(dirpath) / name
                try:
                    rel = p.relative_to(self.root).as_posix()
                except ValueError:
                    continue
                if rel.startswith(".plodder/"):
                    continue
                suf = p.suffix.lower()
                if suf not in _TEXT_EXT and "." in name:
                    continue
                try:
                    if p.stat().st_size > max_file_bytes:
                        continue
                except OSError:
                    continue
                try:
                    raw = p.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                seen += 1
                for i, chunk in enumerate(_chunk_text(raw)):
                    body = f"FILE: {rel}\nCHUNK: {i}\n\n{chunk}"
                    rows.append(
                        {
                            "vector": _hash_bow_embed(body).tolist(),
                            "text": body[:8000],
                            "path": rel,
                            "chunk_index": i,
                        }
                    )

        if not rows:
            return {"indexed_chunks": 0, "indexed_files": 0, "message": "no text files found"}

        try:
            self._db.drop_table(_TABLE)
        except Exception:
            pass
        self._table = self._db.create_table(_TABLE, rows)
        n_files = len({r["path"] for r in rows})
        return {"indexed_chunks": len(rows), "indexed_files": n_files}

    def search(self, query: str, *, top_k: int = 8) -> list[dict[str, Any]]:
        """Vector search over indexed chunks."""
        if not self._open_table() or self._table is None:
            return []
        qv = _hash_bow_embed(query or "")
        try:
            hits = self._table.search(qv.tolist()).limit(max(1, min(int(top_k), 50))).to_list()
        except Exception:
            return []
        out: list[dict[str, Any]] = []
        for row in hits:
            out.append(
                {
                    "path": str(row.get("path", "")),
                    "chunk_index": int(row.get("chunk_index", 0)),
                    "text": str(row.get("text", ""))[:3000],
                    "_distance": row.get("_distance"),
                }
            )
        return out
