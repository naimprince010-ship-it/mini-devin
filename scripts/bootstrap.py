"""
Bootstrap Script for Mini-Devin

This script acts as a watchdog for recursive self-development.
It launches the Mini-Devin API and monitors for a restart signal.
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

def run_server():
    """Run the Mini-Devin server and restart if flag is detected."""
    print(f"[Bootstrap] Initializing Mini-Devin Watchdog at {datetime.now().isoformat()}...")
    print(f"[Bootstrap] Working Directory: {os.getcwd()}")
    print(f"[Bootstrap] Python Executable: {sys.executable}")
    
    # Clean up stale restart flag at startup
    if RESTART_FLAG.exists():
        print("[Bootstrap] Removing stale restart flag.")
        RESTART_FLAG.unlink()

    # Configuration
    api_module = "mini_devin.api.app"
    host = "0.0.0.0"
    # Railway sets PORT; empty string must fall back or uvicorn gets --port "" and fails → 502.
    _port_raw = (os.getenv("PORT") or "").strip()
    port = _port_raw if _port_raw else "8000"
    if not port.isdigit():
        print(f"[Bootstrap] Invalid PORT={_port_raw!r}, falling back to 8000")
        port = "8000"

    # Ensure current directory is in PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = f".{os.pathsep}{env.get('PYTHONPATH', '')}"
    # Child must see the same port Railway expects (some platforms pass PORT only to PID 1).
    env["PORT"] = port

    if os.getenv("RAILWAY_ENVIRONMENT_ID"):
        print(
            "[Bootstrap] Railway: Public Networking → domain → Target port must equal this "
            f"listen port ({port}) or the edge returns 502."
        )

    while True:
        # Start server with uvicorn
        # We try to use uvicorn directly as a command first, fallback to python -m
        print(
            f"[Bootstrap] [{datetime.now().isoformat()}] Starting server on {host}:{port} "
            f"(PORT env was {os.environ.get('PORT', '')!r})..."
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
        
        print(f"[Bootstrap] Mini-Devin process started with PID: {process.pid}")
        
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
            print(f"[Bootstrap] Mini-Devin exited with code {ret_code}")
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
