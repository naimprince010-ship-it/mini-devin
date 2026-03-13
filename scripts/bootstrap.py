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
    print("🚀 [Bootstrap] Starting Mini-Devin...")
    
    # Path to python (adjust if needed or use sys.executable)
    # Using sys.executable ensures it uses the same environment
    python_exe = sys.executable
    api_module = "mini_devin.api.app"
    
    while True:
        process = subprocess.Popen([python_exe, "-m", api_module])
        print(f"✅ [Bootstrap] Mini-Devin running with PID: {process.pid}")
        
        while process.poll() is None:
            # Check for restart flag
            if RESTART_FLAG.exists():
                print("🔄 [Bootstrap] Restart signal detected. Restarting Mini-Devin...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                
                RESTART_FLAG.unlink() # Delete the flag
                break # Break inner loop to restart
            
            time.sleep(2)
        
        if process.returncode is not None and not RESTART_FLAG.exists():
            print(f"⚠️ [Bootstrap] Mini-Devin exited with code {process.returncode}")
            if process.returncode != 0:
                print("Re-trying in 5 seconds...")
                time.sleep(5)
            else:
                print("Exiting bootstrap.")
                break

if __name__ == "__main__":
    try:
        run_server()
    except KeyboardInterrupt:
        print("\n🛑 [Bootstrap] Stopped.")
