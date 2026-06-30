#!/usr/bin/env bash
{
  echo "KEY_PREFIX=${OPENAI_API_KEY:0:10}"
  echo "BASE=$OPENAI_API_BASE"
  echo '{"model":"openai/llama-4-maverick","messages":[{"role":"user","content":"hi"}],"max_tokens":5}' > /tmp/p.json
  code=$(curl -s -o /tmp/r.txt -w "%{http_code}" -X POST "$OPENAI_API_BASE/chat/completions" \
    -H "Authorization: Bearer ${OPENAI_API_KEY}" -H "Content-Type: application/json" --data @/tmp/p.json)
  echo "HTTP=$code"
  echo "BODY:"; head -c 500 /tmp/r.txt
} > /tmp/out.txt 2>&1
