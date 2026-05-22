from __future__ import annotations

import asyncio
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
    base_ref = swe_bench._current_head(str(repo_dir))

    (repo_dir / "string_utils.py").write_text(
        'def contains_word(text: str, word: str) -> bool:\n    return word.lower() in text.lower()\n',
        encoding="utf-8",
    )
    (repo_dir / "notes.txt").write_text("new diagnostic file\n", encoding="utf-8")
    (repo_dir / "PLAN.md").write_text("runtime plan\n", encoding="utf-8")
    (repo_dir / ".plodder").mkdir(exist_ok=True)
    (repo_dir / ".plodder" / "session_events.jsonl").write_text("runtime event\n", encoding="utf-8")

    patch = swe_bench._collect_patch(str(repo_dir))

    assert "return word.lower() in text.lower()" in patch
    assert "notes.txt" in patch
    assert "new diagnostic file" in patch
    assert "PLAN.md" not in patch
    assert ".plodder" not in patch

    swe_bench._git("add", "string_utils.py", cwd=str(repo_dir))
    swe_bench._git("commit", "-m", "Agent checkpoint", cwd=str(repo_dir))

    patch_after_checkpoint = swe_bench._collect_patch(str(repo_dir), base_ref=base_ref)
    assert "return word.lower() in text.lower()" in patch_after_checkpoint


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


def test_runner_retries_failed_attempt_with_feedback(tmp_path: Path) -> None:
    task = swe_bench.load_tasks(limit=1, use_huggingface=False)[0]
    runner = swe_bench.SWEBenchRunner(workspace_root=str(tmp_path))
    calls: list[str] = []

    async def flaky_agent(task, repo_dir, retry_feedback=""):
        calls.append(retry_feedback)
        if len(calls) == 2:
            Path(repo_dir, "string_utils.py").write_text(
                'def contains_word(text: str, word: str) -> bool:\n'
                '    """Return whether word appears inside text, ignoring case."""\n'
                "    return word.casefold() in text.casefold()\n",
                encoding="utf-8",
            )
        return {
            "patch": "",
            "agent_log": "retry-aware fake agent",
            "workspace": repo_dir,
        }

    result = asyncio.run(
        runner._run_single_task(task, "run-retry", flaky_agent, retry_limit=1)
    )

    assert result.status == swe_bench.BenchmarkStatus.RESOLVED
    assert result.attempt_count == 2
    assert len(result.attempts) == 2
    assert calls[0] == ""
    assert "Previous Test Output" in calls[1]
    assert "return word.casefold() in text.casefold()" in result.patch


def test_runner_times_out_stuck_agent(tmp_path: Path, monkeypatch) -> None:
    task = swe_bench.load_tasks(limit=1, use_huggingface=False)[0]
    runner = swe_bench.SWEBenchRunner(workspace_root=str(tmp_path))
    monkeypatch.setenv("PLODDER_SWEBENCH_AGENT_TIMEOUT_SEC", "1")

    async def stuck_agent(task, repo_dir, retry_feedback=""):
        await asyncio.sleep(5)
        return {"patch": "", "agent_log": "too late"}

    result = asyncio.run(
        runner._run_single_task(task, "run-timeout", stuck_agent, retry_limit=0)
    )

    assert result.status == swe_bench.BenchmarkStatus.UNRESOLVED
    assert "timed out" in result.agent_log
