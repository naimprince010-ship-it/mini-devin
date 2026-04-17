"""
Session-scoped workspace: list, read, write, delete under a single root.

All paths are relative to ``root``; path-traversal is rejected.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


_DEFAULT_SNAPSHOT_IGNORE = frozenset(
    {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache", ".pytest_cache", "dist", "build"}
)


@dataclass
class SessionWorkspace:
    """Mutable project tree on disk for one autonomous session."""

    root: Path
    max_read_bytes: int = 1_500_000
    max_list_entries: int = 500
    #: Incremental cache for default-parameter ``snapshot_text_files`` (Docker tar upload fast path).
    _snapshot_cache: dict[str, str] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _safe_rel(self, rel: str) -> Path:
        rel = rel.strip().replace("\\", "/").lstrip("/")
        if ".." in Path(rel).parts:
            raise ValueError("path must not contain '..'")
        candidate = (self.root / rel).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as e:
            raise ValueError("path escapes workspace root") from e
        return candidate

    def list_dir(self, rel: str = ".") -> list[dict[str, str]]:
        """Return ``name``, ``kind`` (``file`` | ``dir``), ``path`` (posix rel to root)."""
        base = self._safe_rel(rel or ".")
        if not base.exists():
            return []
        if base.is_file():
            return [{"name": base.name, "kind": "file", "path": self._to_rel(base)}]
        out: list[dict[str, str]] = []
        for name in sorted(os.listdir(base)):
            if len(out) >= self.max_list_entries:
                out.append({"name": "…", "kind": "dir", "path": "(truncated)"})
                break
            p = base / name
            kind = "dir" if p.is_dir() else "file"
            out.append({"name": name, "kind": kind, "path": self._to_rel(p)})
        return out

    def read_file(self, rel: str) -> str:
        path = self._safe_rel(rel)
        if not path.is_file():
            raise FileNotFoundError(rel)
        size = path.stat().st_size
        if size > self.max_read_bytes:
            raise OSError(f"file too large ({size} bytes > {self.max_read_bytes})")
        return path.read_text(encoding="utf-8", errors="replace")

    def write_file(self, rel: str, content: str) -> None:
        path = self._safe_rel(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
        self._bump_snapshot_cache_after_write(self._to_rel(path), content)

    def delete_path(self, rel: str) -> None:
        path = self._safe_rel(rel)
        if path.is_dir():
            import shutil

            shutil.rmtree(path, ignore_errors=False)
            self._prune_snapshot_cache_for_delete_dir(self._to_rel(path))
        elif path.is_file() or path.is_symlink():
            self._snapshot_cache_pop_rel(self._to_rel(path))
            path.unlink(missing_ok=True)  # type: ignore[arg-type]
        else:
            raise FileNotFoundError(rel)

    def exists(self, rel: str) -> bool:
        return self._safe_rel(rel).exists()

    def _to_rel(self, absolute: Path) -> str:
        return str(absolute.relative_to(self.root)).replace("\\", "/")

    def _snapshot_cache_pop_rel(self, rel_posix: str) -> None:
        if self._snapshot_cache is None:
            return
        rel_posix = rel_posix.replace("\\", "/")
        self._snapshot_cache.pop(rel_posix, None)

    def _prune_snapshot_cache_for_delete_dir(self, rel_posix: str) -> None:
        if self._snapshot_cache is None:
            return
        rel_posix = rel_posix.replace("\\", "/")
        prefix = rel_posix + "/"
        for k in list(self._snapshot_cache):
            if k == rel_posix or k.startswith(prefix):
                self._snapshot_cache.pop(k, None)

    def _bump_snapshot_cache_after_write(self, rel_posix: str, content: str) -> None:
        if self._snapshot_cache is None:
            return
        rel_posix = rel_posix.replace("\\", "/")
        raw = content.encode("utf-8")
        if len(raw) > 500_000 or b"\x00" in raw[:8192]:
            self._snapshot_cache.pop(rel_posix, None)
            return
        self._snapshot_cache[rel_posix] = content

    def _walk_text_snapshot(
        self,
        ignore: frozenset[str],
        max_files: int,
        max_file_bytes: int,
    ) -> dict[str, str]:
        out: dict[str, str] = {}

        def walk(cur: Path) -> None:
            if len(out) >= max_files:
                return
            try:
                names = sorted(os.listdir(cur))
            except OSError:
                return
            for name in names:
                if len(out) >= max_files:
                    return
                p = cur / name
                rel = self._to_rel(p)
                if p.is_dir():
                    if name in ignore:
                        continue
                    walk(p)
                elif p.is_file():
                    try:
                        if p.stat().st_size > max_file_bytes:
                            continue
                    except OSError:
                        continue
                    try:
                        data = p.read_bytes()
                    except OSError:
                        continue
                    if b"\x00" in data[:8192]:
                        continue
                    try:
                        text = data.decode("utf-8")
                    except UnicodeDecodeError:
                        continue
                    out[rel.replace("\\", "/")] = text

        walk(self.root)
        return out

    def snapshot_text_files(
        self,
        *,
        ignore_dir_names: frozenset[str] | None = None,
        max_files: int = 200,
        max_file_bytes: int = 500_000,
        force_refresh: bool = False,
    ) -> dict[str, str]:
        """
        Walk under ``root`` and return ``relative_posix_path -> utf-8 text``.

        Skips binary-looking files and large files.

        With **default** ignore/max limits, results are **cached** and updated incrementally on
        ``write_file`` / ``delete_path`` so repeated ``sandbox_shell`` / ``sandbox_run`` calls avoid
        re-walking the tree (faster Docker ``put_archive`` sync).
        """
        ignore = ignore_dir_names if ignore_dir_names is not None else _DEFAULT_SNAPSHOT_IGNORE
        default_mode = ignore_dir_names is None and max_files == 200 and max_file_bytes == 500_000
        if force_refresh:
            self._snapshot_cache = None
        if default_mode and self._snapshot_cache is not None:
            return dict(self._snapshot_cache)
        built = self._walk_text_snapshot(ignore, max_files, max_file_bytes)
        if default_mode:
            self._snapshot_cache = built
        return dict(built)
