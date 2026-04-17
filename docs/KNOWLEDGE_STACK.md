# Knowledge stack (RAG, golden data, specialization, repo bootstrap)

Mini-Devin supports four complementary layers **without** weight training:

1. **Global corpus RAG** — Optional second vector index built from many repos (`scripts/ingest_global_corpus.py`). See [GLOBAL_CORPUS.md](./GLOBAL_CORPUS.md).
2. **Golden / synthetic JSONL** — Curated few-shot examples in `data/golden/examples.jsonl` (or `GOLDEN_DATA_PATH`).
3. **Specialized system prompt** — `AGENT_SPECIALIZATION=clean_code|performance|security` appends focused rules.
4. **Workspace bootstrap** — On each task start: index the workspace and load `README*` into **working memory** so the model sees repo context early.

Runtime toggles live in `.env` (see `.env.example`).
