# Golden examples (synthetic / curated)

`examples.jsonl` — one JSON object per line. Supported shapes:

1. **Chat-style** (recommended):

```json
{"title": "Small fix pattern", "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

2. **Instruction / output**:

```json
{"instruction": "Add a health check endpoint", "output": "1. Open routes... 2. Add GET /health returning 200..."}
```

Set `GOLDEN_DATA_PATH` to override the default path. Lines starting with `#` are ignored.
