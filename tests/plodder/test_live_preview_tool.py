"""Plodder ``live_preview`` tool (session-scoped port registration for UI iframe)."""

import asyncio
import tempfile
from pathlib import Path

from plodder.orchestration.session_driver import UnifiedSessionDriver
from plodder.workspace.session_workspace import SessionWorkspace


def test_live_preview_errors_without_session_id() -> None:
    async def _go() -> dict:
        root = Path(tempfile.mkdtemp(prefix="plodder_lp_"))
        ws = SessionWorkspace(root)

        async def _llm(_: list) -> str:
            return ""

        d = UnifiedSessionDriver(llm=_llm, workspace=ws, sandbox=None, session_id=None, max_rounds=1)
        return await d._tool_live_preview_async({"action": "probe"})

    r = asyncio.run(_go())
    assert r.get("tool") == "live_preview"
    assert r.get("ok") is False
    assert "session_id" in (r.get("error") or "").lower()


def test_live_preview_resolves_session_from_context_binding() -> None:
    """SessionManager binds API session_id via contextvar for Plodder tools spawned in-process."""

    async def _go() -> dict:
        from plodder.orchestration.session_driver import plodder_session_id_binding

        root = Path(tempfile.mkdtemp(prefix="plodder_lp_ctx_"))
        ws = SessionWorkspace(root)

        async def _llm(_: list) -> str:
            return ""

        with plodder_session_id_binding("ctx-binding-session"):
            d = UnifiedSessionDriver(llm=_llm, workspace=ws, sandbox=None, session_id=None, max_rounds=1)
            return await d._tool_live_preview_async({"action": "probe", "ports": [65532]})

    r = asyncio.run(_go())
    assert r.get("tool") == "live_preview"
    assert r.get("ok") is True


def test_live_preview_probe_ok_with_session_id() -> None:
    async def _go() -> dict:
        root = Path(tempfile.mkdtemp(prefix="plodder_lp_"))
        ws = SessionWorkspace(root)

        async def _llm(_: list) -> str:
            return ""

        d = UnifiedSessionDriver(
            llm=_llm,
            workspace=ws,
            sandbox=None,
            session_id="pytest-plodder-live-preview",
            max_rounds=1,
        )
        return await d._tool_live_preview_async({"action": "probe", "ports": [65533]})

    r = asyncio.run(_go())
    assert r.get("tool") == "live_preview"
    assert r.get("ok") is True
    assert r.get("action") == "probe"
    assert isinstance(r.get("listening_ports"), list)
    assert "allowed_ports" in r
    assert "what_live_preview_is_for" in r
