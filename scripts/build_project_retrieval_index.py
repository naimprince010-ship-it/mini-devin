#!/usr/bin/env python3
"""Build a chunk-level retrieval index from Project Memory repository ingests."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mini_devin.integrations.project_memory import default_project_memory_dir
from mini_devin.integrations.project_retrieval_index import load_project_memory_docs, save_index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Plodder chunk-level project retrieval index")
    parser.add_argument("--memory-dir", default=os.environ.get("PLODDER_PROJECT_MEMORY_DIR") or default_project_memory_dir())
    parser.add_argument("--output", default="/data/project_retrieval_index.json")
    parser.add_argument("--max-section-chars", type=int, default=3500)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    docs = load_project_memory_docs(Path(args.memory_dir), max_section_chars=args.max_section_chars)
    save_index(docs, Path(args.output))
    repos = {doc.repo for doc in docs}
    by_type: dict[str, int] = {}
    for doc in docs:
        by_type[doc.chunk_type] = by_type.get(doc.chunk_type, 0) + 1
    report = {
        "memory_dir": args.memory_dir,
        "output": args.output,
        "documents": len(docs),
        "repos": len(repos),
        "chunk_types": dict(sorted(by_type.items())),
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Built retrieval index: {args.output}")
        print(f"Repos: {report['repos']} | documents: {report['documents']}")
        print("Chunk types:")
        for key, value in report["chunk_types"].items():
            print(f"- {key}: {value}")
    return 0 if docs else 1


if __name__ == "__main__":
    raise SystemExit(main())
