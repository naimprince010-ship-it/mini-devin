#!/usr/bin/env bash
# Probe the new session memory endpoint end-to-end.
set -e
BASE=http://localhost:8081/api
echo "=== sessions ==="
SID=$(curl -s "$BASE/sessions" | python3 -c '
import sys, json
d = json.load(sys.stdin)
rows = d if isinstance(d, list) else d.get("sessions", [])
for s in rows:
    print(s.get("session_id"), "|", s.get("workspace_id"), "|", (s.get("title") or "")[:40])
print("FIRST=" + (rows[0].get("session_id") if rows else ""))
' | tee /dev/stderr | sed -n 's/^FIRST=//p')

echo "=== chosen session: $SID ==="
echo "--- GET memory ---"
curl -s -w "\nHTTP %{http_code}\n" "$BASE/sessions/$SID/memory"
echo "--- POST memory ---"
curl -s -w "\nHTTP %{http_code}\n" -X POST "$BASE/sessions/$SID/memory" \
  -H 'Content-Type: application/json' \
  -d '{"key":"probe_note","value":"endpoint smoke test entry"}'
echo "--- GET memory again ---"
curl -s -w "\nHTTP %{http_code}\n" "$BASE/sessions/$SID/memory"
