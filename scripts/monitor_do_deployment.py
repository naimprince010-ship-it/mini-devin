import requests
import os
import json
import time

TOKEN = os.getenv("DIGITALOCEAN_TOKEN")
APP_ID = os.getenv("DIGITALOCEAN_APP_ID", "50952664-2dc5-42d1-b4b1-4b967782a432")

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

def monitor_deployment():
    print(f"Monitoring App: {APP_ID}")
    
    last_id = None
    
    while True:
        try:
            url = f"https://api.digitalocean.com/v2/apps/{APP_ID}/deployments"
            r = requests.get(url, headers=headers)
            if r.status_code != 200:
                print(f"Error: {r.status_code}")
                time.sleep(10)
                continue
                
            deployments = r.json().get("deployments", [])
            if not deployments:
                print("No deployments.")
                time.sleep(10)
                continue
                
            latest = deployments[0]
            dep_id = latest["id"]
            phase = latest["phase"]
            
            if dep_id != last_id:
                print(f"\nNew Deployment Detected: {dep_id}")
                last_id = dep_id
                
            print(f"Status: {phase} (Updated at {latest.get('updated_at')})")
            
            if phase in ["ACTIVE", "ERROR", "CANCELED"]:
                print(f"Final Phase Reached: {phase}")
                if phase == "ERROR":
                   print("Check logs for details.")
                break
                
            time.sleep(15)
        except KeyboardInterrupt:
            print("Stopped.")
            break
        except Exception as e:
            print(f"Exception: {e}")
            time.sleep(10)

if __name__ == "__main__":
    monitor_deployment()
