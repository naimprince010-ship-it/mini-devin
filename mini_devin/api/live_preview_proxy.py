"""Reverse-proxy HTTP from the API to ``127.0.0.1:{registered_port}`` (Vite / dev servers)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Awaitable, Callable

import httpx
from fastapi import HTTPException, Request
from starlette.responses import Response, StreamingResponse

_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


def _sanitize_proxy_subpath(path: str) -> str:
    p = (path or "").lstrip("/")
    if ".." in p or p.startswith("//"):
        raise HTTPException(status_code=400, detail="Invalid proxy path")
    return p


def _filter_response_headers(resp: httpx.Response) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in resp.headers.multi_items():
        lk = k.lower()
        if lk in _HOP_BY_HOP:
            continue
        if lk == "content-length" and resp.headers.get("transfer-encoding"):
            continue
        out[k] = v
    return out


async def proxy_live_preview(
    session_id: str,
    subpath: str,
    request: Request,
    *,
    get_port: Callable[[str], Awaitable[int | None]],
) -> Response:
    port = await get_port(session_id)
    if not port:
        raise HTTPException(
            status_code=404,
            detail="No live preview port. Run live_preview tool (probe or set_active_port) after starting the dev server.",
        )
    rel = _sanitize_proxy_subpath(subpath)
    upstream = f"http://127.0.0.1:{port}/{rel}" if rel else f"http://127.0.0.1:{port}/"

    headers: dict[str, str] = {}
    for k, v in request.headers.items():
        lk = k.lower()
        if lk in _HOP_BY_HOP or lk == "host":
            continue
        headers[k] = v
    headers["Host"] = f"127.0.0.1:{port}"

    body = await request.body()
    timeout = httpx.Timeout(120.0, connect=10.0)
    client = httpx.AsyncClient(timeout=timeout, follow_redirects=False)
    try:
        req = client.build_request(
            request.method,
            upstream,
            headers=headers,
            content=body if body else None,
            params=list(request.query_params.multi_items()),
        )
        resp = await client.send(req, stream=True)
    except httpx.RequestError as e:
        await client.aclose()
        raise HTTPException(status_code=502, detail=f"Upstream dev server unreachable: {e}") from e

    out_h = _filter_response_headers(resp)

    async def body_iter() -> AsyncIterator[bytes]:
        try:
            async for chunk in resp.aiter_raw():
                yield chunk
        finally:
            await resp.aclose()
            await client.aclose()

    ct = resp.headers.get("content-type")
    return StreamingResponse(
        body_iter(),
        status_code=resp.status_code,
        headers=out_h,
        media_type=ct.split(";")[0].strip() if ct else None,
    )
