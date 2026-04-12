"""Repository digest builder for project memory ingest."""

from __future__ import annotations

from pathlib import Path

import pytest

from mini_devin.integrations.repo_ingest import (
    build_repo_digest,
    ingest_allowlist_roots,
    is_path_allowed_for_repo_ingest,
)


def test_build_repo_digest_includes_readme_and_inventory(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Title\n\nHello from repo.", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text("x = 1\n", encoding="utf-8")

    out = build_repo_digest(tmp_path, max_total_chars=80_000)
    assert "README.md" in out["markdown"]
    assert "Hello from repo" in out["markdown"]
    assert "src/mod.py" in out["markdown"]
    assert out["paths_count"] >= 1
    assert isinstance(out["manifest_files_used"], list)


def test_is_path_allowed_respects_extra_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("REPO_INGEST_EXTRA_ROOT", str(tmp_path))
    roots = ingest_allowlist_roots()
    assert str(tmp_path.resolve()) in {str(r) for r in roots}

    repo = tmp_path / "nested" / "repo"
    repo.mkdir(parents=True)
    assert is_path_allowed_for_repo_ingest(repo)
