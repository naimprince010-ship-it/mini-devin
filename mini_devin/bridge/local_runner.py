"""
Run on your laptop: connects to the cloud Mini-Devin session WebSocket and executes
terminal commands locally (see .env.example: BRIDGE_ISSUE_SECRET).

    poetry run python -m mini_devin.bridge.local_runner --api-base https://host/app/api --session-id <uuid>

Or: poetry run python scripts/local_bridge.py ...
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import websockets


def _ws_url_from_api_base(api_base: str, token: str) -> str:
    u = urllib.parse.urlparse(api_base.rstrip("/"))
    scheme = "wss" if u.scheme == "https" else "ws"
    path = u.path.rstrip("/") + "/bridge/ws"
    q = urllib.parse.urlencode({"token": token})
    return urllib.parse.urlunparse((scheme, u.netloc, path, "", q, ""))


def _issue_token(api_base: str, session_id: str, secret: str) -> str:
    base = api_base.rstrip("/")
    url = f"{base}/sessions/{urllib.parse.quote(session_id, safe='')}/bridge/token"
    req = urllib.request.Request(
        url,
        method="POST",
        headers={"X-MiniDevin-Bridge-Secret": secret},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise SystemExit(f"Token request failed HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise SystemExit(f"Token request failed: {e}") from e
    token = body.get("token")
    if not token:
        raise SystemExit("Token response missing 'token' field")
    return str(token)


def _cwd_allowed(workspace: Path, requested: str) -> str:
    root = workspace.resolve()
    try:
        target = Path(requested).expanduser().resolve()
    except OSError:
        return str(root)
    if target == root or root in target.parents:
        return str(target)
    return str(root)


async def _run_bridge(ws_url: str, workspace: Path) -> None:
    async with websockets.connect(ws_url) as ws:
        print(f"[local-bridge] connected — workspace {workspace.resolve()}")
        async for message in ws:
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                continue
            if data.get("type") != "exec_request":
                continue
            rid = data.get("id")
            if not isinstance(rid, str):
                continue
            cmd = data.get("command")
            if not isinstance(cmd, str):
                await ws.send(
                    json.dumps(
                        {
                            "type": "exec_response",
                            "id": rid,
                            "exit_code": -1,
                            "stdout": "",
                            "stderr": "invalid exec_request: missing command",
                        }
                    )
                )
                continue
            cwd = _cwd_allowed(workspace, str(data.get("cwd") or workspace))
            timeout = float(data.get("timeout_seconds") or 120)
            env = os.environ.copy()
            raw_env = data.get("env")
            if isinstance(raw_env, dict):
                for k, v in raw_env.items():
                    env[str(k)] = str(v)

            proc = await asyncio.create_subprocess_shell(
                cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            try:
                out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                resp = {
                    "type": "exec_response",
                    "id": rid,
                    "exit_code": -1,
                    "stdout": "",
                    "stderr": "local-bridge: command timed out",
                }
            else:
                resp = {
                    "type": "exec_response",
                    "id": rid,
                    "exit_code": int(proc.returncode if proc.returncode is not None else 0),
                    "stdout": out_b.decode("utf-8", errors="replace"),
                    "stderr": err_b.decode("utf-8", errors="replace"),
                }
            await ws.send(json.dumps(resp))


def main() -> None:
    p = argparse.ArgumentParser(description="Mini-Devin local terminal bridge (run commands on this machine).")
    p.add_argument(
        "--api-base",
        help="HTTP API base, e.g. https://your-app.ondigitalocean.app/app/api",
    )
    p.add_argument("--session-id", help="Mini-Devin session UUID from the dashboard")
    p.add_argument(
        "--secret",
        default=os.getenv("MINIDEVIN_BRIDGE_SECRET") or os.getenv("BRIDGE_ISSUE_SECRET"),
        help="Same value as server BRIDGE_ISSUE_SECRET (or set MINIDEVIN_BRIDGE_SECRET)",
    )
    p.add_argument(
        "--token",
        help="Skip HTTP mint; use a token you already obtained (one-time use)",
    )
    p.add_argument(
        "--ws-url",
        help="Full WebSocket URL including token=… (overrides --api-base token build)",
    )
    p.add_argument(
        "-w",
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Root directory; agent cwd is clamped under this path",
    )
    args = p.parse_args()

    ws_url: str | None = args.ws_url
    if ws_url:
        pass
    elif args.token and args.api_base:
        ws_url = _ws_url_from_api_base(args.api_base, args.token)
    elif args.api_base and args.session_id and args.secret:
        token = _issue_token(args.api_base, args.session_id, args.secret)
        ws_url = _ws_url_from_api_base(args.api_base, token)
    else:
        p.error("Need --ws-url, or (--token and --api-base), or (--api-base, --session-id, and --secret)")

    asyncio.run(_run_bridge(ws_url, args.workspace))


if __name__ == "__main__":
    main()
