import os
import requests

TOKEN = os.environ.get("DIGITALOCEAN_TOKEN", "")
APP_ID = os.environ.get("DIGITALOCEAN_APP_ID", "")

def list_deployments():
    url = f"https://api.digitalocean.com/v2/apps/{APP_ID}/deployments"
    headers = {"Authorization": f"Bearer {TOKEN}"}
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        print(f"Error: {r.status_code}")
        print(r.text)
        return
    
    data = r.json()
    for d in data.get("deployments", [])[:5]:
        print(f"ID: {d['id']}")
        print(f"Cause: {d['cause']}")
        print(f"Phase: {d['phase']}")
        print(f"Created: {d['created_at']}")
        print("-" * 20)

if __name__ == "__main__":
    list_deployments()
