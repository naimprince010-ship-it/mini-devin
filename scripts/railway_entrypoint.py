"""
Railway / Docker entry: optional Postgres migrations in background, then exec uvicorn as PID 1.

Uses Python (not shell) so Windows CRLF in repo checkouts cannot break the container CMD.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import traceback

APP = "mini_devin.api:app"


def _listen_port() -> str:
    """Match ``scripts/bootstrap._resolve_listen_port`` (Railway ``PORT`` quirks)."""
    raw = (os.environ.get("PORT") or "").strip()
    if not raw:
        return "8000"
    first = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n", 1)[0].strip()
    for tok in first.replace(",", " ").split():
        if tok.isdigit():
            return tok
    print(f"[railway-entrypoint] invalid PORT={raw!r}; using 8000", flush=True)
    return "8000"


def _maybe_alembic_background() -> None:
    if os.getenv("SKIP_ALEMBIC_UPGRADE", "").strip().lower() in ("1", "true", "yes", "on"):
        print("[railway-entrypoint] SKIP_ALEMBIC_UPGRADE set", flush=True)
        return
    dbr = (os.getenv("DATABASE_URL") or "").strip().lower()
    if not dbr or "postgres" not in dbr:
        print("[railway-entrypoint] skip alembic (no postgres DATABASE_URL)", flush=True)
        return

    def _run() -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = "." + os.pathsep + env.get("PYTHONPATH", "").strip(os.pathsep)
        print("[railway-entrypoint] alembic upgrade head (background)", flush=True)
        try:
            r = subprocess.run(
                [sys.executable, "-m", "alembic", "upgrade", "head"],
                cwd="/app",
                env=env,
                capture_output=True,
                text=True,
                timeout=float(os.getenv("ALEMBIC_UPGRADE_TIMEOUT", "300")),
            )
            if r.stdout:
                print(r.stdout.rstrip(), flush=True)
            if r.stderr:
                print(r.stderr.rstrip(), flush=True)
            if r.returncode != 0:
                print(f"[railway-entrypoint] alembic failed exit={r.returncode}", flush=True)
            else:
                print("[railway-entrypoint] alembic: ok", flush=True)
        except Exception as e:
            print(f"[railway-entrypoint] alembic error: {e}", flush=True)
            traceback.print_exc()

    threading.Thread(target=_run, daemon=True, name="alembic").start()


def main() -> None:
    os.chdir("/app")
    pyp = os.environ.get("PYTHONPATH", "").strip()
    os.environ["PYTHONPATH"] = "." + (os.pathsep + pyp if pyp else "")
    port = _listen_port()
    os.environ["PORT"] = port
    print(f"[railway-entrypoint] cwd={os.getcwd()} PORT={port!r}", flush=True)

    _maybe_alembic_background()

    # Railway terminates TLS at the edge; forwarded headers help clients see correct scheme/host.
    args = [
        sys.executable,
        "-m",
        "uvicorn",
        APP,
        "--host",
        "0.0.0.0",
        "--port",
        port,
        "--log-level",
        "info",
        "--proxy-headers",
        "--forwarded-allow-ips",
        "*",
    ]
    print(f"[railway-entrypoint] exec uvicorn port={port}", flush=True)
    os.execvpe(sys.executable, args, os.environ)


if __name__ == "__main__":
    main()
