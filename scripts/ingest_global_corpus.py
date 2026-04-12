#!/usr/bin/env python3
"""
Shallow-clone repos from a manifest and append chunks into the global corpus VectorStore.

Manifest: JSON file — list of objects:
  [{"url": "https://github.com/org/repo", "ref": "main"}, ...]
Or a plain text file with one git URL per line.

Requirements: git on PATH, run from mini-devin repo root with:
  python scripts/ingest_global_corpus.py --manifest data/global_corpus/manifest.example.json

Legal: you must have rights to index the material. See docs/GLOBAL_CORPUS.md.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


TEXT_EXTENSIONS = {".md", ".py", ".ts", ".tsx", ".js", ".jsx", ".rs", ".go", ".toml", ".yaml", ".yml"}


def _load_manifest(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not raw:
        return []
    if raw.startswith("["):
        data = json.loads(raw)
        return [x for x in data if isinstance(x, dict)]
    repos = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        repos.append({"url": line, "ref": "HEAD"})
    return repos


def _git_clone(url: str, ref: str, dest: Path) -> bool:
    cmd = ["git", "clone", "--depth", "1", "--single-branch", url, str(dest)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(f"[ingest] clone failed {url}: {r.stderr or r.stdout}", file=sys.stderr)
        return False
    if ref and ref.upper() != "HEAD":
        r2 = subprocess.run(
            ["git", "-C", str(dest), "fetch", "--depth", "1", "origin", ref],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if r2.returncode == 0:
            subprocess.run(
                ["git", "-C", str(dest), "checkout", "FETCH_HEAD"],
                capture_output=True,
                text=True,
                timeout=120,
            )
    return True


def _iter_source_files(root: Path, max_files: int) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*"):
        if len(out) >= max_files:
            break
        if not p.is_file():
            continue
        if any(part in (".git", "node_modules", "dist", "build", "__pycache__", ".venv") for part in p.parts):
            continue
        if p.suffix.lower() not in TEXT_EXTENSIONS and p.name not in ("Dockerfile",):
            continue
        try:
            if p.stat().st_size > 120_000:
                continue
        except OSError:
            continue
        out.append(p)
    return out


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    p = argparse.ArgumentParser(description="Ingest Git repos into Mini-Devin global corpus")
    p.add_argument("--manifest", type=str, required=True, help="Path to manifest JSON or URL list")
    p.add_argument("--max-repos", type=int, default=20)
    p.add_argument("--max-files-per-repo", type=int, default=200)
    p.add_argument("--chunk-size", type=int, default=1200)
    args = p.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    from mini_devin.memory.vector_store import Document, create_vector_store
    import os

    corpus_path = os.environ.get("MINI_DEVIN_GLOBAL_CORPUS_PATH", "").strip()
    if corpus_path:
        store_path = Path(corpus_path)
    else:
        store_path = Path.home() / ".mini-devin" / "global_corpus" / "vector_store.json"
    store_path.parent.mkdir(parents=True, exist_ok=True)
    use_openai = os.environ.get("GLOBAL_RAG_OPENAI_EMBEDDINGS", "").lower() in ("1", "true", "yes")
    store = create_vector_store(
        persist_path=str(store_path),
        use_openai=use_openai,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    entries = _load_manifest(manifest_path)[: args.max_repos]
    added = 0
    for spec in entries:
        url = str(spec.get("url", "")).strip()
        if not url:
            continue
        ref = str(spec.get("ref", "HEAD")).strip() or "HEAD"
        slug = url.rstrip("/").split("/")[-1].replace(".git", "") or "repo"
        with tempfile.TemporaryDirectory(prefix="mdn-corpus-") as tmp:
            clone_dir = Path(tmp) / slug
            if not _git_clone(url, ref, clone_dir):
                continue
            files = _iter_source_files(clone_dir, args.max_files_per_repo)
            batch: list[Document] = []
            for fp in files:
                try:
                    text = fp.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                rel = str(fp.relative_to(clone_dir)).replace("\\", "/")
                cs = max(400, args.chunk_size)
                for i in range(0, len(text), cs):
                    chunk = text[i : i + cs].strip()
                    if len(chunk) < 80:
                        continue
                    doc = Document.from_text(
                        chunk,
                        {
                            "repo": url,
                            "file_path": rel,
                            "chunk": i // cs,
                            "source": "global_corpus_ingest",
                        },
                    )
                    batch.append(doc)
                    if len(batch) >= 64:
                        store.add_batch(batch)
                        added += len(batch)
                        batch.clear()
            if batch:
                store.add_batch(batch)
                added += len(batch)
        print(f"[ingest] done {url} (+chunks)")

    print(f"[ingest] total new documents (chunks): {added}")
    print(f"[ingest] store: {store_path}")


if __name__ == "__main__":
    main()
