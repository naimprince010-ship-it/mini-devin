# Benchmark Lessons for Chat

- Source run: a5eba15a-87a
- Benchmark: humaneval
- Model: openai/gpt-5.3-codex
- Pass rate: 0.0%

## Reliability Rules
- Return only valid Python code; never include markdown fences.
- Define exact function names expected by tests.
- Match exact output shape and deterministic ordering for assertion checks.
- Prefer efficient implementations to avoid timeout failures.
