#!/usr/bin/env python3
"""
Index Markdown language / library docs into Plodder's LanceDB vector store.

Uses ``plodder.rag.doc_ingestion.DocumentationStore`` (same table as runtime RAG).
Prefer running ``scripts/prepare_docs.py`` first (Learn X in Y Minutes → ``docs/languages``).

Examples::

    python scripts/prepare_docs.py
    python scripts/ingest_plodder_docs.py --docs-dir ./docs/languages

    # Default: ./docs → ./data/plodder_lance
    python scripts/ingest_plodder_docs.py

    # Large tree of language cheat-sheets
    python scripts/ingest_plodder_docs.py --docs-dir ./docs/languages

    # Custom persist path (e.g. Docker volume mount)
    python scripts/ingest_plodder_docs.py --persist-dir /data/plodder_lance
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="Ingest Markdown into Plodder LanceDB RAG")
    p.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("docs"),
        help="Root folder scanned recursively for *.md (default: ./docs)",
    )
    p.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("data/plodder_lance"),
        help="LanceDB directory (default: ./data/plodder_lance)",
    )
    args = p.parse_args()

    repo_root = Path.cwd()
    docs_dir = (repo_root / args.docs_dir).resolve() if not args.docs_dir.is_absolute() else args.docs_dir
    persist_dir = (repo_root / args.persist_dir).resolve() if not args.persist_dir.is_absolute() else args.persist_dir

    if not docs_dir.is_dir():
        print(f"ERROR: docs dir does not exist: {docs_dir}", file=sys.stderr)
        print("Create it and add .md files (e.g. docs/languages/zig.md).", file=sys.stderr)
        return 1

    from plodder.rag.doc_ingestion import DocumentationStore

    store = DocumentationStore(docs_dir=docs_dir, persist_dir=persist_dir)
    total_chunks = store.index_all()
    print(f"Done. Indexed chunks (rows) total: {total_chunks}")
    print(f"  docs_dir     = {docs_dir}")
    print(f"  persist_dir  = {persist_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
