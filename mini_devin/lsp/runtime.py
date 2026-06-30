"""Session-scoped persistent LSP runtime.

Provides a long-lived language-server manager per session/workspace, keeps a
small diagnostics cache, and falls back to request-time diagnostics when no
server is available.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .diagnostics import collect_diagnostics
from .manager import LSPManager, LSPManagerConfig
from .types import Diagnostic


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _monaco_severity(diag: Diagnostic) -> str:
    sev = int(getattr(diag, "severity", 1) or 1)
    if sev == 1:
        return "error"
    if sev == 2:
        return "warning"
    if sev == 3:
        return "information"
    return "hint"


def _diag_to_monaco(diag: Diagnostic) -> dict[str, Any]:
    start = getattr(diag, "range", None).start if getattr(diag, "range", None) else None
    end = getattr(diag, "range", None).end if getattr(diag, "range", None) else None
    start_line = (getattr(start, "line", 0) or 0) + 1
    start_col = (getattr(start, "character", 0) or 0) + 1
    end_line = (getattr(end, "line", getattr(start, "line", 0)) or getattr(start, "line", 0) or 0) + 1
    end_col = (getattr(end, "character", getattr(start, "character", 0)) or getattr(start, "character", 0) or 0) + 1
    return {
        "line": start_line,
        "startColumn": start_col,
        "endLine": end_line,
        "endColumn": max(start_col + 1, end_col),
        "message": str(getattr(diag, "message", "diagnostic") or "diagnostic"),
        "severity": _monaco_severity(diag),
    }


@dataclass
class DiagnosticSnapshot:
    path: str
    diagnostics: list[dict[str, Any]]
    source: str
    updated_at: str


class SessionLspRuntime:
    """Persistent LSP state for one session/workspace."""

    def __init__(self, session_id: str, workspace_path: str):
        self.session_id = session_id
        self.workspace_path = os.path.abspath(workspace_path)
        self._manager = LSPManager(
            LSPManagerConfig(
                workspace_path=self.workspace_path,
                auto_start_servers=False,
            )
        )
        self._lock = asyncio.Lock()
        self._started = False
        self._start_errors: list[str] = []
        self._cache: dict[str, DiagnosticSnapshot] = {}

    def _inside_workspace(self, absolute_path: str) -> bool:
        ws = os.path.normcase(os.path.abspath(self.workspace_path))
        p = os.path.normcase(os.path.abspath(absolute_path))
        return p == ws or p.startswith(ws + os.sep)

    def _resolve_rel(self, rel_path: str) -> str:
        rel = (rel_path or "").replace("\\", "/").lstrip("/")
        abs_target = os.path.abspath(os.path.join(self.workspace_path, rel))
        if not self._inside_workspace(abs_target):
            raise ValueError("Path escapes workspace")
        return rel

    async def start(self, languages: list[str] | None = None) -> dict[str, Any]:
        async with self._lock:
            if self._started:
                return {"started": True, "languages": self._manager.active_languages, "errors": list(self._start_errors)}
            results = await self._manager.start(languages)
            self._started = True
            failed = [lang for lang, ok in results.items() if not ok]
            if failed:
                self._start_errors.append(f"Failed language servers: {', '.join(failed)}")
            return {
                "started": True,
                "results": results,
                "languages": self._manager.active_languages,
                "errors": list(self._start_errors),
            }

    async def stop(self) -> dict[str, Any]:
        async with self._lock:
            await self._manager.stop()
            self._started = False
            return {"stopped": True}

    def status(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "workspace": self.workspace_path,
            "started": self._started,
            "active_languages": self._manager.active_languages,
            "cache_entries": len(self._cache),
            "errors": list(self._start_errors),
        }

    def get_cached(self, rel_path: str | None = None) -> dict[str, Any]:
        if rel_path:
            key = rel_path.replace("\\", "/")
            snap = self._cache.get(key)
            if not snap:
                return {}
            return {
                "path": snap.path,
                "diagnostics": snap.diagnostics,
                "source": snap.source,
                "updated_at": snap.updated_at,
            }
        return {
            "items": [
                {
                    "path": s.path,
                    "diagnostics": s.diagnostics,
                    "source": s.source,
                    "updated_at": s.updated_at,
                }
                for s in self._cache.values()
            ]
        }

    async def refresh_file(self, rel_path: str, content: str | None = None) -> dict[str, Any]:
        rel = self._resolve_rel(rel_path)
        absolute_path = os.path.abspath(os.path.join(self.workspace_path, rel))

        diagnostics: list[dict[str, Any]] = []
        source = "none"

        if self._started:
            try:
                client = await self._manager.ensure_server_for_file(absolute_path)
                if client:
                    await self._manager.open_document(absolute_path, text=content)
                    if content is not None:
                        await self._manager.update_document(absolute_path, content)
                    await asyncio.sleep(float(os.environ.get("PLODDER_LSP_DIAGNOSTIC_SETTLE_MS", "50")) / 1000.0)
                    runtime_diags = self._manager.get_diagnostics(absolute_path)
                    diagnostics = [_diag_to_monaco(d) for d in runtime_diags]
                    source = "lsp"
            except Exception as e:
                self._start_errors.append(f"refresh_file runtime error: {e}")

        if not diagnostics:
            fallback_items, fallback_source = collect_diagnostics(self.workspace_path, rel, content)
            diagnostics = fallback_items
            source = fallback_source

        snap = DiagnosticSnapshot(
            path=rel,
            diagnostics=diagnostics,
            source=source,
            updated_at=_utc_now_iso(),
        )
        self._cache[rel] = snap
        return {
            "path": rel,
            "diagnostics": diagnostics,
            "source": source,
            "updated_at": snap.updated_at,
            "runtime_active": self._started,
            "active_languages": self._manager.active_languages,
        }


_RUNTIME_LOCK = asyncio.Lock()
_RUNTIMES: dict[str, SessionLspRuntime] = {}


def _lsp_auto_start_enabled() -> bool:
    raw = (os.environ.get("PLODDER_LSP_AUTO_START") or "true").strip().lower()
    return raw in ("1", "true", "yes", "on")


async def get_or_create_session_runtime(
    session_id: str,
    workspace_path: str,
    *,
    auto_start: bool | None = None,
) -> SessionLspRuntime:
    async with _RUNTIME_LOCK:
        runtime = _RUNTIMES.get(session_id)
        if runtime is None:
            runtime = SessionLspRuntime(session_id=session_id, workspace_path=workspace_path)
            _RUNTIMES[session_id] = runtime
    if auto_start is None:
        auto_start = _lsp_auto_start_enabled()
    if auto_start and not runtime.status().get("started"):
        await runtime.start()
    return runtime


async def stop_session_runtime(session_id: str) -> None:
    async with _RUNTIME_LOCK:
        runtime = _RUNTIMES.get(session_id)
    if runtime is not None:
        await runtime.stop()


async def remove_session_runtime(session_id: str) -> None:
    async with _RUNTIME_LOCK:
        runtime = _RUNTIMES.pop(session_id, None)
    if runtime is not None:
        await runtime.stop()


async def notify_workspace_file_changed(workspace_path: str, file_path: str) -> None:
    """Notify all runtimes attached to the same workspace that a file changed."""
    ws = os.path.normcase(os.path.abspath(workspace_path))
    target = os.path.normcase(os.path.abspath(file_path))
    rel_path = None
    if target.startswith(ws + os.sep):
        rel_path = os.path.relpath(target, ws).replace("\\", "/")
    elif target == ws:
        return
    if rel_path is None:
        return

    async with _RUNTIME_LOCK:
        runtimes = list(_RUNTIMES.values())
    candidates = [r for r in runtimes if os.path.normcase(os.path.abspath(r.workspace_path)) == ws]
    for runtime in candidates:
        if runtime.status().get("started"):
            try:
                await runtime.refresh_file(rel_path, content=None)
            except Exception:
                continue


async def list_lsp_runtime_status() -> list[dict[str, Any]]:
    async with _RUNTIME_LOCK:
        runtimes = list(_RUNTIMES.values())
    return [r.status() for r in runtimes]
