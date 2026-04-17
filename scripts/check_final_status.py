import requests
import json

import os

TOKEN = os.getenv("DIGITALOCEAN_TOKEN")
APP_ID = os.getenv("DIGITALOCEAN_APP_ID", "")

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

def check_status():
    url = f"https://api.digitalocean.com/v2/apps/{APP_ID}/deployments"
    r = requests.get(url, headers=headers)
    deployments = r.json().get("deployments", [])
    if deployments:
        latest = deployments[0]
        print(f"Deployment ID: {latest['id']}, Phase: {latest['phase']}, Commit: {latest.get('cause')}")
    else:
        print("No deployments found.")

if __name__ == "__main__":
    check_status()
