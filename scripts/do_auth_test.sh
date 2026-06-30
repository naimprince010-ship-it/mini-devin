#!/usr/bin/env bash
set -euo pipefail
echo '{"model":"openai/llama-4-maverick","messages":[{"role":"user","content":"hi"}],"max_tokens":5}' > /tmp/p.json
code=$(curl -s -o /tmp/r.txt -w "%{http_code}" -X POST https://inference.do-ai.run/v1/chat/completions \
  -H "Authorization: Bearer ${OPENAI_API_KEY}" \
  -H "Content-Type: application/json" \
  --data @/tmp/p.json)
echo "HTTP=$code"
head -c 400 /tmp/r.txt
echo
