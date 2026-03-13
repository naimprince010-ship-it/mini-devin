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
    port = os.getenv("PORT", "8000")
    
    # Ensure current directory is in PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = f".{os.pathsep}{env.get('PYTHONPATH', '')}"
    
    while True:
        # Start server with uvicorn
        # We try to use uvicorn directly as a command first, fallback to python -m
        print(f"[Bootstrap] Starting server on {host}:{port}...")
        try:
            process = subprocess.Popen(
                ["uvicorn", f"{api_module}:app", "--host", host, "--port", port],
                env=env
            )
        except FileNotFoundError:
            print("[Bootstrap] 'uvicorn' command not found in PATH, trying 'python -m uvicorn'...")
            process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", f"{api_module}:app", "--host", host, "--port", port],
                env=env
            )
        
        print(f"[Bootstrap] Mini-Devin running with PID: {process.pid}")
        
        while process.poll() is None:
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
