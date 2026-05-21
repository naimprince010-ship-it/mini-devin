from __future__ import annotations

from pathlib import Path

from mini_devin.integrations import swe_bench


def test_fallback_task_uses_valid_local_fixture() -> None:
    tasks = swe_bench.load_tasks(limit=1, use_huggingface=False)

    assert tasks[0].repo == "plodder-fixtures/string-utils"
    assert tasks[0].base_commit.startswith("local:")
    assert tasks[0].fail_to_pass == ["tests/test_string_utils.py::test_contains_word_is_case_insensitive"]


def test_local_fixture_has_real_failing_test(tmp_path: Path) -> None:
    task = swe_bench.load_tasks(limit=1, use_huggingface=False)[0]
    repo_dir = swe_bench._setup_repo(task, str(tmp_path))

    assert repo_dir is not None
    output, results = swe_bench._run_tests(repo_dir, task.fail_to_pass)

    assert results[task.fail_to_pass[0]] is False
    assert "FAILED" in output


def test_collect_patch_includes_tracked_and_untracked_files(tmp_path: Path) -> None:
    task = swe_bench.load_tasks(limit=1, use_huggingface=False)[0]
    repo_dir = Path(swe_bench._setup_repo(task, str(tmp_path)) or "")

    (repo_dir / "string_utils.py").write_text(
        'def contains_word(text: str, word: str) -> bool:\n    return word.lower() in text.lower()\n',
        encoding="utf-8",
    )
    (repo_dir / "notes.txt").write_text("new diagnostic file\n", encoding="utf-8")

    patch = swe_bench._collect_patch(str(repo_dir))

    assert "return word.lower() in text.lower()" in patch
    assert "notes.txt" in patch
    assert "new diagnostic file" in patch


def test_format_agent_log_includes_debug_sections() -> None:
    log = swe_bench._format_agent_log(
        status="failed",
        summary="Task failed",
        commands=["pytest -q"],
        files=["string_utils.py"],
        timeline="[action] terminal: pytest -q\n[observation]: failed",
        error="boom",
    )

    assert "status: failed" in log
    assert "commands_executed" in log
    assert "files_modified" in log
    assert "timeline" in log
    assert "boom" in log
