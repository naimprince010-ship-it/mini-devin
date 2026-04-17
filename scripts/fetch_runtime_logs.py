import requests
import json

TOKEN = os.getenv("DIGITALOCEAN_TOKEN")
APP_ID = os.getenv("DIGITALOCEAN_APP_ID", "50952664-2dc5-42d1-b4b1-4b967782a432")

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

def get_runtime_logs():
    # 1. Get latest deployment
    url = f"https://api.digitalocean.com/v2/apps/{APP_ID}/deployments"
    r = requests.get(url, headers=headers)
    latest = r.json()["deployments"][0]
    dep_id = latest["id"]
    print(f"Latest Deployment ID: {dep_id}, Phase: {latest['phase']}")
    
    # 2. Get component names
    url = f"https://api.digitalocean.com/v2/apps/{APP_ID}"
    r = requests.get(url, headers=headers)
    app = r.json()["app"]
    components = []
    components += [s["name"] for s in app["spec"].get("services", [])]
    components += [s["name"] for s in app["spec"].get("static_sites", [])]
    components += [s["name"] for s in app["spec"].get("workers", [])]
    components += [s["name"] for s in app["spec"].get("jobs", [])]
    
    for component in components:
        print(f"\n--- Runtime Logs for {component} ---")
        # type=RUN for runtime logs
        log_url = f"https://api.digitalocean.com/v2/apps/{APP_ID}/deployments/{dep_id}/components/{component}/logs?type=RUN"
        r = requests.get(log_url, headers=headers)
        if r.status_code == 200:
            print(json.dumps(r.json(), indent=2))
        else:
            print(f"Error: {r.status_code} - {r.text}")

if __name__ == "__main__":
    get_runtime_logs()
