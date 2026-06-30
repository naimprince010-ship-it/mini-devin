#!/usr/bin/env bash
cd /var/tmp/agent-workspace 2>/dev/null || cd /tmp
echo "KEY_PREFIX=${OPENAI_API_KEY:0:10}" > out.txt
echo "BASE=$OPENAI_API_BASE" >> out.txt
echo '{"model":"openai/llama-4-maverick","messages":[{"role":"user","content":"hi"}],"max_tokens":5}' > q.json
code=$(curl -s -o resp.txt -w "%{http_code}" -X POST "$OPENAI_API_BASE/chat/completions" \
  -H "Authorization: Bearer ${OPENAI_API_KEY}" -H "Content-Type: application/json" --data @q.json)
echo "HTTP=$code" >> out.txt
echo "BODY:" >> out.txt
head -c 500 resp.txt >> out.txt
cat out.txt
