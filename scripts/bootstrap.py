"""
Bootstrap Script for Plodder

This script acts as a watchdog for recursive self-development.
It launches the Plodder API and monitors for a restart signal.
If the agent modifies core logic and wants to apply changes, it can 
touch a `.restart_flag` file to trigger a restart.
"""

import os
import subprocess
import time
import sys
from datetime import datetime
from pathlib import Path

RESTART_FLAG = Path(".restart_flag")


def _run_alembic_upgrade_if_applicable() -> None:
    """Apply DB migrations before uvicorn (Railway/Postgres). Skipped for SQLite or unset URL."""
    if os.getenv("SKIP_ALEMBIC_UPGRADE", "").strip().lower() in ("1", "true", "yes", "on"):
        print("[Bootstrap] SKIP_ALEMBIC_UPGRADE set; skipping alembic upgrade head.")
        return
    raw = (os.getenv("DATABASE_URL") or "").strip()
    if not raw:
        print("[Bootstrap] DATABASE_URL unset; skipping alembic (SQLite / init_db create_all path).")
        return
    if "sqlite" in raw.lower():
        print("[Bootstrap] SQLite DATABASE_URL; skipping alembic upgrade head.")
        return

    print("[Bootstrap] Running database migrations: alembic upgrade head")
    env = os.environ.copy()
    env["PYTHONPATH"] = f".{os.pathsep}{env.get('PYTHONPATH', '')}"
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=os.getcwd(),
            env=env,
            capture_output=True,
            text=True,
            timeout=float(os.getenv("ALEMBIC_UPGRADE_TIMEOUT", "300")),
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "alembic upgrade head timed out (set ALEMBIC_UPGRADE_TIMEOUT seconds or SKIP_ALEMBIC_UPGRADE=1)."
        ) from None

    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.stderr:
        print(proc.stderr.rstrip())
    if proc.returncode != 0:
        raise RuntimeError(
            f"alembic upgrade head failed (exit {proc.returncode}). "
            "Fix migrations or set SKIP_ALEMBIC_UPGRADE=1 to start without migrating (not recommended)."
        )
    print("[Bootstrap] alembic upgrade head completed successfully.")


def _resolve_listen_port() -> str:
    """Pick a single numeric port from PORT (Railway injects this).

    Duplicate ``PORT`` rows or copy-paste mistakes can produce multiline / comma-separated
    values; ``str.isdigit()`` then fails and we used to fall back to 8000 while the edge
    still targets 8080 → 502.
    """
    raw = (os.getenv("PORT") or "").strip()
    if not raw:
        return "8000"
    first_line = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n", 1)[0].strip()
    parts = [p.strip() for p in first_line.replace(",", " ").split() if p.strip()]
    for token in parts:
        if token.isdigit():
            if raw != token and os.getenv("RAILWAY_ENVIRONMENT_ID"):
                print(
                    "[Bootstrap] Warning: PORT env had extra characters; "
                    f"using first integer token {token!r} from raw {raw!r}. "
                    "Remove duplicate/custom PORT variables in Railway."
                )
            return token
    print(f"[Bootstrap] Invalid PORT={raw!r}, falling back to 8000")
    return "8000"


def run_server():
    """Run the Plodder server and restart if flag is detected."""
    print(f"[Bootstrap] Initializing Plodder Watchdog at {datetime.now().isoformat()}...")
    print(f"[Bootstrap] Working Directory: {os.getcwd()}")
    print(f"[Bootstrap] Python Executable: {sys.executable}")
    
    # Clean up stale restart flag at startup
    if RESTART_FLAG.exists():
        print("[Bootstrap] Removing stale restart flag.")
        RESTART_FLAG.unlink()

    # Configuration
    api_module = "mini_devin.api.app"
    host = "0.0.0.0"
    port = _resolve_listen_port()

    # Ensure current directory is in PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = f".{os.pathsep}{env.get('PYTHONPATH', '')}"
    # Child must see the same port Railway expects (some platforms pass PORT only to PID 1).
    env["PORT"] = port

    if os.getenv("RAILWAY_ENVIRONMENT_ID"):
        port_keys = sorted(k for k in os.environ if "PORT" in k.upper())
        print(f"[Bootstrap] Railway env keys containing PORT: {port_keys}")
        print(
            "[Bootstrap] Railway: Public Networking → domain → Target port must equal this "
            f"listen port ({port}) or the edge returns 502."
        )

    _run_alembic_upgrade_if_applicable()

    while True:
        # Start server with uvicorn
        # We try to use uvicorn directly as a command first, fallback to python -m
        print(
            f"[Bootstrap] [{datetime.now().isoformat()}] Starting server on {host}:{port} "
            f"(resolved listen port {port!r}, raw PORT env was {os.environ.get('PORT', '')!r})..."
        )
        try:
            process = subprocess.Popen(
                ["uvicorn", f"{api_module}:app", "--host", host, "--port", port, "--log-level", "debug"],
                env=env,
                stdout=None, # Inherit to show in platform logs
                stderr=None
            )
        except Exception as e:
            print(f"[Bootstrap] Failed to start with 'uvicorn' command: {e}. Trying 'python -m uvicorn'...")
            process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", f"{api_module}:app", "--host", host, "--port", port, "--log-level", "debug"],
                env=env,
                stdout=None,
                stderr=None
            )
        
        print(f"[Bootstrap] Plodder process started with PID: {process.pid}")
        
        last_ping = time.time()
        while process.poll() is None:
            # Simple heartbeat in logs
            if time.time() - last_ping > 60:
                print(f"[Bootstrap] [{datetime.now().isoformat()}] Watchdog heartbeat: PID {process.pid} is alive.")
                last_ping = time.time()

            # Check for restart signal
            if RESTART_FLAG.exists():
                print("[Bootstrap] Restart signal detected. Terminating process...")
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    print("[Bootstrap] Process didn't terminate in time. Killing...")
                    process.kill()
                
                RESTART_FLAG.unlink() # Delete the flag
                print("[Bootstrap] Restarting shortly...")
                break # Break inner loop to restart
            
            time.sleep(2)
        
        # If process exited naturally
        ret_code = process.poll()
        if ret_code is not None and not RESTART_FLAG.exists():
            print(f"[Bootstrap] Plodder exited with code {ret_code}")
            if ret_code != 0:
                wait_time = 5
                print(f"[Bootstrap] Unexpected exit. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("[Bootstrap] Server shut down gracefully. Exiting watchdog.")
                break

if __name__ == "__main__":
    try:
        run_server()
    except KeyboardInterrupt:
        print("\n[Bootstrap] Watchdog stopped by user.")
    except Exception as e:
        print(f"[Bootstrap] Fatal error: {e}")
        import traceback
        traceback.print_exc()
