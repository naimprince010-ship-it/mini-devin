"""Top GitHub repository training script helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "train_from_top_github_repos.py"
spec = importlib.util.spec_from_file_location("train_from_top_github_repos", SCRIPT_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def test_project_id_for_repo_is_stable_and_safe() -> None:
    assert mod.project_id_for_repo("formbricks/formbricks") == "gh-formbricks-formbricks"
    assert mod.project_id_for_repo("Some Org/Repo.Name") == "gh-some-org-repo-name"


def test_training_state_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    state = mod.TrainingState()
    state.completed["org/repo"] = {"project_id": "gh-org-repo"}
    state.failed["org/bad"] = {"error": "clone failed"}
    state.save(path)

    loaded = mod.TrainingState.load(path)
    assert loaded.completed["org/repo"]["project_id"] == "gh-org-repo"
    assert loaded.failed["org/bad"]["error"] == "clone failed"


def test_load_repos_file_supports_plain_text(tmp_path: Path) -> None:
    manifest = tmp_path / "repos.txt"
    manifest.write_text(
        "\ufeff# comment\nhttps://github.com/formbricks/formbricks.git\n\nhttps://github.com/pypa/sampleproject\n",
        encoding="utf-8",
    )

    repos = mod.load_repos_file(manifest)

    assert [r.full_name for r in repos] == ["formbricks/formbricks", "pypa/sampleproject"]
    assert repos[0].clone_url.endswith(".git")
