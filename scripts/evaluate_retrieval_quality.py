#!/usr/bin/env python3
"""Evaluate retrieval quality across Plodder project-memory repo ingests."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mini_devin.integrations.project_memory import default_project_memory_dir
from mini_devin.integrations.project_retrieval_index import (
    RetrievalDoc,
    load_index,
    load_project_memory_docs,
    search_docs,
)


DEFAULT_CASES: list[dict[str, Any]] = [
    {
        "id": "react-hooks",
        "query": "React hooks useState useEffect component rendering reconciliation",
        "expected": ["facebook/react"],
    },
    {
        "id": "next-app-router",
        "query": "Next.js app router server components route handlers middleware",
        "expected": ["vercel/next.js"],
    },
    {
        "id": "grpc-protobuf",
        "query": "gRPC protobuf HTTP2 channel server client interceptors",
        "expected": ["grpc/grpc"],
    },
    {
        "id": "django-orm",
        "query": "Django ORM models migrations QuerySet admin middleware",
        "expected": ["django/django"],
    },
    {
        "id": "fastapi-openapi",
        "query": "FastAPI dependency injection pydantic OpenAPI ASGI routes",
        "expected": ["fastapi/fastapi"],
    },
    {
        "id": "kubernetes-controllers",
        "query": "Kubernetes controller kubelet apiserver scheduler CRD operator",
        "expected": ["kubernetes/kubernetes"],
    },
    {
        "id": "rust-swc",
        "query": "Rust TypeScript compiler parser transform bundler minifier",
        "expected": ["swc-project/swc"],
    },
    {
        "id": "tailwind-css",
        "query": "Tailwind CSS utility classes config plugin responsive design",
        "expected": ["tailwindlabs/tailwindcss"],
    },
    {
        "id": "laravel-eloquent",
        "query": "Laravel Eloquent migrations artisan routes controllers blade",
        "expected": ["laravel/framework"],
    },
    {
        "id": "vscode-extension",
        "query": "VS Code editor extension language server workbench marketplace",
        "expected": ["microsoft/vscode"],
    },
]


def load_cases(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return list(DEFAULT_CASES)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("queries file must contain a JSON array")
    return [c for c in raw if isinstance(c, dict)]


def evaluate_case(
    case: dict[str, Any],
    docs: list[RetrievalDoc],
    *,
    top_k: int,
    scorer: str,
) -> dict[str, Any]:
    query = str(case.get("query") or "").strip()
    expected = [str(x) for x in (case.get("expected") or [])]
    ranked = search_docs(docs, query, top_k=len(docs), scorer=scorer)
    hits = ranked[:top_k]

    expected_lower = {x.lower() for x in expected}
    rank = None
    for idx, hit in enumerate(ranked, start=1):
        if hit["repo"].lower() in expected_lower or hit["project_id"].lower() in expected_lower:
            rank = idx
            break

    return {
        "id": case.get("id") or query[:40],
        "query": query,
        "expected": expected,
        "rank": rank,
        "hit_top_k": rank is not None and rank <= top_k,
        "top": [
            {
                "rank": i,
                "repo": hit["repo"],
                "project_id": hit["project_id"],
                "score": hit["score"],
                "title": hit["title"],
                "chunk_type": hit.get("chunk_type"),
            }
            for i, hit in enumerate(hits, start=1)
        ],
    }


def evaluate(
    memory_dir: Path,
    cases: list[dict[str, Any]],
    *,
    top_k: int,
    max_entry_chars: int = 30000,
    scorer: str = "hybrid",
    index_file: Path | None = None,
) -> dict[str, Any]:
    if index_file and index_file.is_file():
        docs = load_index(index_file)
        index_source = str(index_file)
    else:
        docs = load_project_memory_docs(memory_dir, max_section_chars=max_entry_chars)
        index_source = "memory_dir"
    results = [evaluate_case(case, docs, top_k=top_k, scorer=scorer) for case in cases]
    evaluated = len(results)
    top1 = sum(1 for r in results if r["rank"] == 1)
    topk = sum(1 for r in results if r["hit_top_k"])
    mrr = sum((1 / r["rank"]) for r in results if r["rank"]) / evaluated if evaluated else 0.0
    missing = sorted(
        {
            repo
            for case in cases
            for repo in case.get("expected", [])
            if all(doc.repo.lower() != str(repo).lower() for doc in docs)
        }
    )
    return {
        "memory_dir": str(memory_dir),
        "documents": len(docs),
        "max_entry_chars": max_entry_chars,
        "scorer": scorer,
        "index_source": index_source,
        "cases": evaluated,
        "top1": top1,
        f"top{top_k}": topk,
        "top1_rate": round(top1 / evaluated, 3) if evaluated else 0.0,
        f"top{top_k}_rate": round(topk / evaluated, 3) if evaluated else 0.0,
        "mrr": round(mrr, 3),
        "missing_expected_repos": missing,
        "results": results,
    }


def print_report(report: dict[str, Any], *, top_k: int) -> None:
    print(f"Memory dir: {report['memory_dir']}")
    print(f"Documents: {report['documents']}")
    print(
        f"Cases: {report['cases']} | Top-1: {report['top1']} "
        f"({report['top1_rate']:.0%}) | Top-{top_k}: {report[f'top{top_k}']} "
        f"({report[f'top{top_k}_rate']:.0%}) | MRR: {report['mrr']}"
    )
    if report["missing_expected_repos"]:
        print("Missing expected repos: " + ", ".join(report["missing_expected_repos"]))
    print()
    for result in report["results"]:
        status = "PASS" if result["hit_top_k"] else "FAIL"
        rank = result["rank"] or "-"
        print(f"[{status}] {result['id']} expected={result['expected']} rank={rank}")
        for hit in result["top"][:5]:
            print(f"  {hit['rank']}. {hit['repo']} score={hit['score']} title={hit['title']}")
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Plodder project-memory retrieval quality")
    parser.add_argument("--memory-dir", default=os.environ.get("PLODDER_PROJECT_MEMORY_DIR") or default_project_memory_dir())
    parser.add_argument("--queries-file", default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--max-entry-chars",
        type=int,
        default=30000,
        help="Characters per memory entry to score; keeps large repo digests fast.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only")
    parser.add_argument("--output", default="", help="Optional path to write JSON report")
    parser.add_argument("--scorer", choices=["hybrid", "semantic", "lexical"], default="hybrid")
    parser.add_argument("--index-file", default="", help="Optional prebuilt chunk retrieval index JSON")
    args = parser.parse_args(argv)

    cases = load_cases(Path(args.queries_file) if args.queries_file else None)
    report = evaluate(
        Path(args.memory_dir),
        cases,
        top_k=args.top_k,
        max_entry_chars=args.max_entry_chars,
        scorer=args.scorer,
        index_file=Path(args.index_file) if args.index_file else None,
    )
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report, top_k=args.top_k)
    return 0 if report[f"top{args.top_k}_rate"] >= 0.7 else 1


if __name__ == "__main__":
    raise SystemExit(main())
