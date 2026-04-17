"""Bridge token lifecycle (no WebSocket)."""

import asyncio

import pytest

from mini_devin.bridge.manager import BridgeManager

pytestmark = pytest.mark.eval


def test_bridge_token_roundtrip():
    async def _run() -> None:
        bm = BridgeManager()
        t = await bm.create_token("sess-bridge-1", ttl_seconds=900)
        sid = await bm.consume_token(t)
        assert sid == "sess-bridge-1"
        assert await bm.consume_token(t) is None

    asyncio.run(_run())


def test_bridge_expired_token_returns_none():
    async def _run() -> None:
        bm = BridgeManager()
        t = await bm.create_token("sess-expired", ttl_seconds=-3600)
        assert await bm.consume_token(t) is None

    asyncio.run(_run())
