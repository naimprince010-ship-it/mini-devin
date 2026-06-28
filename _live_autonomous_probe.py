import json
import time
import urllib.request
from collections import Counter

BASE = "http://127.0.0.1:8000"

PROMPT = """You are an autonomous software engineering agent.

Your task is to improve this project without breaking existing functionality.

Requirements:
1. Analyze the repository structure.
2. Identify the top 5 most important issues or missing features.
3. Create a prioritized implementation plan.
4. Select the highest-impact task.
5. Create a new git branch for the work.
6. Implement the feature following the project's existing coding style.
7. Add or update unit tests.
8. Run formatting, linting, and all relevant tests.
9. If tests fail, debug and fix the problems automatically.
10. Commit the changes with a conventional commit message.
11. Generate a markdown report including:
   - What was changed
   - Files modified
   - Test results
   - Remaining issues
   - Risk assessment
12. Do not modify unrelated files.
13. Stop immediately if you encounter destructive operations or missing permissions, and explain why.

Success Criteria:
- All tests pass.
- No new warnings or lint errors.
- Code follows project conventions.
- The repository remains buildable."""


def req(method: str, path: str, payload=None, timeout=40):
    url = BASE + path
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    r = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


health = req("GET", "/health")
print("health", health)

session = req("POST", "/api/sessions", {})
sid = session["session_id"]
print("session", sid)

task = req(
    "POST",
    f"/api/sessions/{sid}/tasks",
    {
        "description": PROMPT,
        "task_type": "feature",
        "max_iterations": 14,
    },
)
tid = task["task_id"]
print("task", tid)

final_status = None
for i in range(18):
    time.sleep(5)
    t = req("GET", f"/api/sessions/{sid}/tasks/{tid}")
    status = t.get("status")
    it = t.get("iteration_count")
    print(f"tick={i} status={status} iter={it}")
    if status in {"completed", "failed", "cancelled", "blocked"}:
        final_status = status
        break

feed = req("GET", f"/api/sessions/{sid}/activity-feed?limit=140")
events = list(feed.get("events") or [])
print("events", len(events))

kind_counts = Counter(e.get("type") for e in events)
tool_counts = Counter((e.get("tool") or "-") for e in events if e.get("tool"))
print("type_counts", dict(kind_counts))
print("tool_counts", dict(tool_counts))

print("last_events")
for e in events[-12:]:
    print(
        e.get("ts"),
        "|",
        e.get("type"),
        "|",
        e.get("tool") or "-",
        "|",
        e.get("action") or "-",
    )

print("final_status", final_status)
