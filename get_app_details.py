import os
import requests

TOKEN = os.environ.get("DIGITALOCEAN_TOKEN", "")
APP_ID = os.environ.get("DIGITALOCEAN_APP_ID", "")

def get_app_details():
    url = f"https://api.digitalocean.com/v2/apps/{APP_ID}"
    headers = {"Authorization": f"Bearer {TOKEN}"}
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        print(f"Error: {r.status_code}")
        return
    
    app = r.json().get('app', {})
    print("Services:")
    for s in app.get('spec', {}).get('services', []):
        print(f"  - {s['name']}")
    
    print("\nStatic Sites:")
    for s in app.get('spec', {}).get('static_sites', []):
        print(f"  - {s['name']}")

if __name__ == "__main__":
    get_app_details()
