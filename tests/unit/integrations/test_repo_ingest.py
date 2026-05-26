"""Repository digest builder for project memory ingest."""

from __future__ import annotations

from pathlib import Path

import pytest

from mini_devin.integrations.repo_ingest import (
    build_repo_digest,
    ingest_allowlist_roots,
    is_path_allowed_for_repo_ingest,
)
from mini_devin.integrations.project_memory import default_project_memory_dir


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


def test_build_repo_digest_adds_code_intelligence_sections(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text(
        """
        {
          "name": "sample-app",
          "packageManager": "pnpm@10.0.0",
          "scripts": {"test": "vitest", "build": "next build"},
          "dependencies": {"next": "latest", "zod": "latest"},
          "devDependencies": {"vitest": "latest"}
        }
        """,
        encoding="utf-8",
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "api.ts").write_text(
        "import { z } from 'zod';\n"
        "import { helper } from './helper';\n\n"
        "export function createSurvey(input: unknown) {\n"
        "  return helper(z.object({ name: z.string() }).parse(input));\n"
        "}\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "helper.ts").write_text(
        "export const helper = (value: unknown) => value;\n",
        encoding="utf-8",
    )

    out = build_repo_digest(tmp_path, max_total_chars=80_000)
    markdown = out["markdown"]

    assert "## Dependency and command map" in markdown
    assert "`sample-app`" in markdown
    assert "`test`" in markdown
    assert "## Symbol map" in markdown
    assert "createSurvey" in markdown
    assert "## Import/dependency edges" in markdown
    assert "`src/api.ts` ->" in markdown
    assert "`./helper`" in markdown
    assert "## Selected code chunks" in markdown


def test_build_repo_digest_skips_non_object_package_json(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text('"not an object"', encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "package.json").write_text(
        '{"name":"bad-shapes","dependencies":"not an object","scripts":["test"]}',
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# Odd repo\n", encoding="utf-8")

    out = build_repo_digest(tmp_path, max_total_chars=80_000)

    assert "## Dependency and command map" in out["markdown"]
    assert "`bad-shapes`" in out["markdown"]


def test_is_path_allowed_respects_extra_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("REPO_INGEST_EXTRA_ROOT", str(tmp_path))
    roots = ingest_allowlist_roots()
    assert str(tmp_path.resolve()) in {str(r) for r in roots}

    repo = tmp_path / "nested" / "repo"
    repo.mkdir(parents=True)
    assert is_path_allowed_for_repo_ingest(repo)


def test_project_memory_defaults_to_workspace_volume_parent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("PLODDER_DATA", raising=False)
    monkeypatch.delenv("MINI_DEVIN_DATA", raising=False)
    workspace_root = tmp_path / "data" / "agent-workspace"
    monkeypatch.setenv("PLODDER_AGENT_WORKSPACE_ROOT", str(workspace_root))

    assert default_project_memory_dir() == str(tmp_path / "data" / "project_memory")
