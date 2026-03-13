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
from pathlib import Path

RESTART_FLAG = Path(".restart_flag")

def run_server():
    """Run the Mini-Devin server and restart if flag is detected."""
    print("🚀 [Bootstrap] Initializing Mini-Devin Watchdog...")
    
    # Clean up stale restart flag at startup
    if RESTART_FLAG.exists():
        print("🗑️ [Bootstrap] Removing stale restart flag.")
        RESTART_FLAG.unlink()

    # Configuration
    python_exe = sys.executable
    api_module = "mini_devin.api.app"
    host = "0.0.0.0"
    port = os.getenv("PORT", "8000")
    
    while True:
        # Start server with uvicorn directly via subprocess
        # This ensures we have control over the process lifecycle
        print(f"📡 [Bootstrap] Starting server on {host}:{port}...")
        process = subprocess.Popen([
            python_exe, "-m", "uvicorn", 
            f"{api_module}:app", 
            "--host", host, 
            "--port", port
        ])
        print(f"✅ [Bootstrap] Mini-Devin running with PID: {process.pid}")
        
        while process.poll() is None:
            # Check for restart signal
            if RESTART_FLAG.exists():
                print("🔄 [Bootstrap] Restart signal detected. Terminating process...")
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    print("⚠️ [Bootstrap] Process didn't terminate in time. Killing...")
                    process.kill()
                
                RESTART_FLAG.unlink() # Delete the flag
                print("🔁 [Bootstrap] Restarting shortly...")
                break # Break inner loop to restart
            
            time.sleep(2)
        
        # If process exited naturally
        ret_code = process.poll()
        if ret_code is not None and not RESTART_FLAG.exists():
            print(f"⚠️ [Bootstrap] Mini-Devin exited with code {ret_code}")
            if ret_code != 0:
                wait_time = 5
                print(f"⏳ [Bootstrap] Unexpected exit. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("✨ [Bootstrap] Server shut down gracefully. Exiting watchdog.")
                break

if __name__ == "__main__":
    try:
        run_server()
    except KeyboardInterrupt:
        print("\n🛑 [Bootstrap] Watchdog stopped by user.")
    except Exception as e:
        print(f"💥 [Bootstrap] Fatal error: {e}")
        import traceback
        traceback.print_exc()
