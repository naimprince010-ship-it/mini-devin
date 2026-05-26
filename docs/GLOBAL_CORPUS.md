# Global corpus (optional RAG)

Mini-Devin can search a **separate** persisted vector index built from many Git repositories. This is **not** model training: it is retrieval-augmented context at prompt time.

## Legal and compliance

- Index only repositories and revisions **you have the right** to copy/embed for your use case (license, employer policy, customer contracts).
- Public GitHub does **not** automatically mean “free for any ML/RAG product”; read each license (MIT, Apache-2.0, GPL implications, etc.).
- This feature is **opt-in** (`GLOBAL_RAG_ENABLED`). You are responsible for the manifest contents and retention policy.

## Build the index

1. Copy `data/global_corpus/manifest.example.json` and edit URLs (start with a **small** list while testing).
2. From the `mini-devin` repo root:

```bash
python scripts/ingest_global_corpus.py --manifest data/global_corpus/my_manifest.json --max-repos 10
```

Requires `git` on PATH. Optional: set `MINI_DEVIN_GLOBAL_CORPUS_PATH` to store the index elsewhere.

## Use at runtime

In `.env`:

```env
GLOBAL_RAG_ENABLED=true
# Optional: OpenAI embeddings for the corpus (otherwise simple local embeddings)
# GLOBAL_RAG_OPENAI_EMBEDDINGS=true
```

The agent injects top matching **snippets** into the chat as reference only; the **current workspace** remains authoritative for edits.

## Seed hosted Project Memory from top GitHub repos

For Plodder's hosted project-memory path, use the resumable top-repo seeder. It
creates one project per repository and calls `/api/projects/{project_id}/ingest-repo`,
so each repo gets the enhanced digest: manifests, dependency map, symbol map,
import edges, selected code chunks, and inventory.

Start small:

```bash
python scripts/train_from_top_github_repos.py \
  --api-base http://174.138.34.120 \
  --max-repos 10 \
  --dry-run
```

Run the real corpus job:

```bash
python scripts/train_from_top_github_repos.py \
  --api-base http://174.138.34.120 \
  --max-repos 2000 \
  --state data/top_repo_training_state.json
```

Use `GITHUB_TOKEN` to avoid low unauthenticated GitHub API limits. The state file
records completed and failed repos, so rerunning the same command resumes instead
of starting over. The default discovery uses language buckets because GitHub's
repository search API caps deep paging; this gives broad coverage across popular
TypeScript, Python, JavaScript, Go, Rust, Java, C++, C#, PHP, and Ruby projects.

For stricter legal/compliance control, provide an explicit manifest:

```bash
python scripts/train_from_top_github_repos.py \
  --api-base http://174.138.34.120 \
  --repos-file data/global_corpus/approved_repos.txt \
  --max-repos 2000
```
