import requests
import json
import sys
import os

TOKEN = os.getenv("DIGITALOCEAN_TOKEN")
APP_ID = os.getenv("DIGITALOCEAN_APP_ID", "50952664-2dc5-42d1-b4b1-4b967782a432")
# DO App Platform component slug (rename in DO when you rename the service)
DO_COMPONENT = os.getenv("DIGITALOCEAN_COMPONENT_SLUG", "plodder")

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

def get_deployment_details(dep_id):
    # 1. Get deployment info
    url = f"https://api.digitalocean.com/v2/apps/{APP_ID}/deployments/{dep_id}"
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        print(f"Error fetching deployment {dep_id}: {r.status_code} - {r.text}")
        return
    
    dep = r.json()["deployment"]
    print(f"Deployment ID: {dep['id']}")
    print(f"Phase: {dep['phase']}")
    print(f"Created At: {dep.get('created_at')}")
    
    # 2. Get build logs
    print("\n--- Build Logs ---")
    url = f"https://api.digitalocean.com/v2/apps/{APP_ID}/deployments/{dep_id}/components/{DO_COMPONENT}/logs?type=BUILD"
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        print(r.json().get("live_url", "No live URL found"))
    else:
        print(f"Error fetching build logs: {r.status_code}")

    # 3. Get runtime logs
    print("\n--- Runtime Logs ---")
    url = f"https://api.digitalocean.com/v2/apps/{APP_ID}/deployments/{dep_id}/components/{DO_COMPONENT}/logs?type=RUN"
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        print(r.json().get("live_url", "No live URL found"))
    else:
        print(f"Error fetching runtime logs: {r.status_code}")

if __name__ == "__main__":
    dep_id = sys.argv[1] if len(sys.argv) > 1 else None
    if not dep_id:
        # Get latest failed deployment
        url = f"https://api.digitalocean.com/v2/apps/{APP_ID}/deployments"
        r = requests.get(url, headers=headers)
        deployments = r.json()["deployments"]
        for d in deployments:
            if d["phase"] == "ERROR":
                dep_id = d["id"]
                break
    
    if dep_id:
        get_deployment_details(dep_id)
    else:
        print("No failed deployment found.")
