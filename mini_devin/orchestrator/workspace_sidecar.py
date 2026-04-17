"""
Background workspace file-tree watcher + cached listing (OpenHands-style sidecar).

The agent can read :meth:`WorkspaceSidecar.get_snapshot_text` without running ``ls``/``find``
on every turn. Updates debounce after filesystem changes.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Callable

_SKIP_DIRNAMES = frozenset(
    {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".plodder",
        ".idea",
        ".vscode",
    }
)


def _walk_workspace_paths(root: Path, *, max_files: int = 8000) -> list[str]:
    out: list[str] = []
    if not root.is_dir():
        return out
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRNAMES]
        if any(p in _SKIP_DIRNAMES for p in Path(dirpath).parts):
            continue
        for name in sorted(filenames):
            if count >= max_files:
                return out
            p = Path(dirpath) / name
            try:
                rel = p.relative_to(root).as_posix()
            except ValueError:
                continue
            out.append(rel)
            count += 1
    return sorted(out)


class WorkspaceSidecar:
    """
    Thread-safe snapshot of workspace paths. Starts a watchdog observer when available.
    Falls back to manual :meth:`refresh` only if watchdog import fails.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        debounce_s: float = 0.85,
        on_refresh: Callable[[], None] | None = None,
    ) -> None:
        self._root = Path(root).resolve()
        self._debounce_s = max(0.2, debounce_s)
        self._on_refresh = on_refresh
        self._lock = threading.Lock()
        self._paths: list[str] = []
        self._dirty = True
        self._gen = 0
        self._observer = None
        self._debounce_timer: threading.Timer | None = None

    @property
    def root(self) -> Path:
        return self._root

    def start(self) -> None:
        self.refresh(blocking=True)
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer

            side = self

            class _H(FileSystemEventHandler):
                def on_any_event(self, event):  # type: ignore[no-untyped-def]
                    side._mark_dirty()

            self._observer = Observer()
            if self._root.is_dir():
                self._observer.schedule(_H(), str(self._root), recursive=True)
                self._observer.start()
        except Exception:
            self._observer = None

    def stop(self) -> None:
        if self._debounce_timer:
            self._debounce_timer.cancel()
            self._debounce_timer = None
        if self._observer:
            try:
                self._observer.stop()
                self._observer.join(timeout=2.0)
            except Exception:
                pass
            self._observer = None

    def _mark_dirty(self) -> None:
        with self._lock:
            self._dirty = True
            self._gen += 1
            gen = self._gen
        if self._debounce_timer:
            self._debounce_timer.cancel()

        def _delayed() -> None:
            with self._lock:
                if self._gen != gen:
                    return
            self.refresh(blocking=True)

        self._debounce_timer = threading.Timer(self._debounce_s, _delayed)
        self._debounce_timer.daemon = True
        self._debounce_timer.start()

    def refresh(self, *, blocking: bool = False) -> None:
        def _do() -> None:
            paths = _walk_workspace_paths(self._root)
            with self._lock:
                self._paths = paths
                self._dirty = False
            if self._on_refresh:
                try:
                    self._on_refresh()
                except Exception:
                    pass
            try:
                snap_dir = self._root / ".plodder"
                snap_dir.mkdir(parents=True, exist_ok=True)
                text = self.get_snapshot_text(max_lines=12000, max_chars=900_000)
                (snap_dir / "workspace_tree_snapshot.txt").write_text(
                    text, encoding="utf-8", errors="replace"
                )
            except OSError:
                pass

        if blocking:
            _do()
        else:
            threading.Thread(target=_do, daemon=True).start()

    def get_paths(self) -> list[str]:
        with self._lock:
            return list(self._paths)

    def get_snapshot_text(self, *, max_lines: int = 4000, max_chars: int = 48_000) -> str:
        paths = self.get_paths()
        lines = [f"# Workspace tree ({self._root.as_posix()})", f"# files: {len(paths)}", ""]
        for i, p in enumerate(paths):
            if i >= max_lines:
                lines.append(f"... ({len(paths) - max_lines} more paths truncated)")
                break
            lines.append(p)
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n…(truncated)…"
        return text

    def is_dirty(self) -> bool:
        with self._lock:
            return self._dirty
