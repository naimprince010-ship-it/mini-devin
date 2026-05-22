from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "evaluate_retrieval_quality.py"
spec = importlib.util.spec_from_file_location("evaluate_retrieval_quality", SCRIPT_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def _write_project(memory_dir: Path, project_id: str, repo: str, content: str) -> None:
    project_dir = memory_dir / project_id
    project_dir.mkdir(parents=True)
    (project_dir / "project.json").write_text(
        json.dumps({"id": project_id, "name": repo, "description": "", "repo_url": "", "tech_stack": []}),
        encoding="utf-8",
    )
    (project_dir / "entries.json").write_text(
        json.dumps(
            [
                {
                    "id": f"{project_id}-entry",
                    "project_id": project_id,
                    "category": "context",
                    "title": f"Repository ingest: {repo}",
                    "content": content,
                    "tags": [],
                    "importance": 8,
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                }
            ]
        ),
        encoding="utf-8",
    )


def test_retrieval_quality_scores_expected_repo(tmp_path: Path) -> None:
    _write_project(
        tmp_path,
        "gh-framework-react",
        "facebook/react",
        "React hooks useState useEffect component reconciliation JSX rendering.",
    )
    _write_project(
        tmp_path,
        "gh-grpc-grpc",
        "grpc/grpc",
        "gRPC protobuf HTTP2 channels client server interceptors.",
    )

    report = mod.evaluate(
        tmp_path,
        [{"id": "react", "query": "useEffect component rendering hooks", "expected": ["facebook/react"]}],
        top_k=1,
        max_entry_chars=10000,
        scorer="hybrid",
    )

    assert report["documents"] == 4
    assert report["top1"] == 1
    assert report["results"][0]["top"][0]["repo"] == "facebook/react"


def test_missing_expected_repo_is_reported(tmp_path: Path) -> None:
    _write_project(tmp_path, "gh-only-repo", "owner/only", "Some unrelated content.")

    report = mod.evaluate(
        tmp_path,
        [{"id": "missing", "query": "grpc protobuf", "expected": ["grpc/grpc"]}],
        top_k=3,
        max_entry_chars=10000,
        scorer="hybrid",
    )

    assert report["missing_expected_repos"] == ["grpc/grpc"]
    assert report["results"][0]["rank"] is None


def test_chunk_index_finds_file_level_signal(tmp_path: Path) -> None:
    from mini_devin.integrations.project_retrieval_index import load_project_memory_docs, search_docs

    digest = """
    # Repository snapshot

    ## File inventory (3 paths)

    ```
    packages/react/src/ReactHooks.js
    packages/react-dom/src/client/ReactDOMRoot.js
    README.md
    ```
    """
    _write_project(tmp_path, "gh-facebook-react", "facebook/react", digest)

    docs = load_project_memory_docs(tmp_path)
    results = search_docs(docs, "React hooks useState useEffect", top_k=1)

    assert results[0]["repo"] == "facebook/react"
    assert results[0]["chunk_type"] == "file_inventory"
