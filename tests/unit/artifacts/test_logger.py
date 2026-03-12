"""Unit tests for the ArtifactLogger module."""

import json
import tempfile
from pathlib import Path

import pytest

from mini_devin.artifacts.logger import ArtifactLogger, create_artifact_logger


@pytest.fixture
def tmp_dir(tmp_path: Path) -> str:
    """Return a temporary directory path as a string."""
    return str(tmp_path)


@pytest.fixture
def logger(tmp_dir: str) -> ArtifactLogger:
    """Create an ArtifactLogger instance for testing."""
    return create_artifact_logger(
        base_dir=tmp_dir,
        task_id="task-abc-123",
        task_description="Test task description",
    )


class TestArtifactLoggerInit:
    """Tests for ArtifactLogger initialisation."""

    def test_creates_run_directory(self, tmp_dir: str) -> None:
        """A run directory is created under base_dir/task_id on construction."""
        logger = create_artifact_logger(tmp_dir, "my-task", "desc")
        run_dir = Path(logger.get_run_dir())
        assert run_dir.exists()
        assert run_dir.name == "my-task"

    def test_run_json_written_on_init(self, logger: ArtifactLogger) -> None:
        """run.json is written immediately after construction."""
        meta_path = Path(logger.get_run_dir()) / "run.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["task_id"] == "task-abc-123"
        assert meta["task_description"] == "Test task description"
        assert meta["status"] == "running"
        assert meta["model"] == ""
        assert meta["total_tokens"] == 0
        assert meta["iterations"] == 0
        assert meta["commands_executed"] == []
        assert meta["files_modified"] == []
        assert meta["completed_at"] is None

    def test_get_run_dir_returns_correct_path(
        self, tmp_dir: str, logger: ArtifactLogger
    ) -> None:
        """get_run_dir() returns the full path to the task run directory."""
        expected = str(Path(tmp_dir) / "task-abc-123")
        assert logger.get_run_dir() == expected

    def test_nested_base_dir_created(self, tmp_path: Path) -> None:
        """ArtifactLogger creates deeply nested directories if needed."""
        base_dir = str(tmp_path / "a" / "b" / "c")
        logger = create_artifact_logger(base_dir, "task-1", "desc")
        assert Path(logger.get_run_dir()).exists()


class TestSetModel:
    """Tests for ArtifactLogger.set_model."""

    def test_model_persisted_in_run_json(self, logger: ArtifactLogger) -> None:
        """set_model() persists the model name to run.json."""
        logger.set_model("gpt-4o")
        meta = json.loads((Path(logger.get_run_dir()) / "run.json").read_text())
        assert meta["model"] == "gpt-4o"


class TestLogToolCall:
    """Tests for ArtifactLogger.log_tool_call."""

    def test_appends_record_to_jsonl(self, logger: ArtifactLogger) -> None:
        """Each log_tool_call() appends one line to tool_calls.jsonl."""
        logger.log_tool_call(
            call_id="c1",
            tool_name="terminal",
            arguments={"command": "ls"},
            result="file.py",
            success=True,
            duration_ms=42,
        )
        jsonl_path = Path(logger.get_run_dir()) / "tool_calls.jsonl"
        assert jsonl_path.exists()
        lines = jsonl_path.read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["call_id"] == "c1"
        assert record["tool_name"] == "terminal"
        assert record["arguments"] == {"command": "ls"}
        assert record["result"] == "file.py"
        assert record["success"] is True
        assert record["duration_ms"] == 42
        assert "timestamp" in record
        assert "error" not in record

    def test_error_field_included_when_provided(self, logger: ArtifactLogger) -> None:
        """error key is present in the record when an error message is given."""
        logger.log_tool_call(
            call_id="c2",
            tool_name="editor",
            arguments={},
            result="",
            success=False,
            error="file not found",
        )
        jsonl_path = Path(logger.get_run_dir()) / "tool_calls.jsonl"
        record = json.loads(jsonl_path.read_text().strip())
        assert record["error"] == "file not found"

    def test_multiple_calls_produce_multiple_lines(
        self, logger: ArtifactLogger
    ) -> None:
        """Multiple tool calls produce one JSONL line each."""
        for i in range(5):
            logger.log_tool_call(f"c{i}", "terminal", {}, "out", True)
        jsonl_path = Path(logger.get_run_dir()) / "tool_calls.jsonl"
        lines = jsonl_path.read_text().strip().splitlines()
        assert len(lines) == 5

    def test_non_string_result_is_coerced(self, logger: ArtifactLogger) -> None:
        """Non-string results are converted to strings."""
        logger.log_tool_call("c1", "terminal", {}, {"key": "val"}, True)
        jsonl_path = Path(logger.get_run_dir()) / "tool_calls.jsonl"
        record = json.loads(jsonl_path.read_text().strip())
        assert isinstance(record["result"], str)


class TestAddCommandExecuted:
    """Tests for ArtifactLogger.add_command_executed."""

    def test_command_added_to_metadata(self, logger: ArtifactLogger) -> None:
        """add_command_executed() stores commands in run.json."""
        logger.add_command_executed("pytest tests/")
        meta = json.loads((Path(logger.get_run_dir()) / "run.json").read_text())
        assert "pytest tests/" in meta["commands_executed"]

    def test_duplicate_commands_not_repeated(self, logger: ArtifactLogger) -> None:
        """The same command is stored only once."""
        logger.add_command_executed("ls")
        logger.add_command_executed("ls")
        meta = json.loads((Path(logger.get_run_dir()) / "run.json").read_text())
        assert meta["commands_executed"].count("ls") == 1


class TestAddFileModified:
    """Tests for ArtifactLogger.add_file_modified."""

    def test_file_path_added_to_metadata(self, logger: ArtifactLogger) -> None:
        """add_file_modified() stores file paths in run.json."""
        logger.add_file_modified("src/main.py")
        meta = json.loads((Path(logger.get_run_dir()) / "run.json").read_text())
        assert "src/main.py" in meta["files_modified"]

    def test_duplicate_files_not_repeated(self, logger: ArtifactLogger) -> None:
        """The same file path is stored only once."""
        logger.add_file_modified("README.md")
        logger.add_file_modified("README.md")
        meta = json.loads((Path(logger.get_run_dir()) / "run.json").read_text())
        assert meta["files_modified"].count("README.md") == 1


class TestIncrementIteration:
    """Tests for ArtifactLogger.increment_iteration."""

    def test_iteration_counter_increments(self, logger: ArtifactLogger) -> None:
        """increment_iteration() increases the iteration count."""
        logger.increment_iteration()
        logger.increment_iteration()
        logger.increment_iteration()
        meta = json.loads((Path(logger.get_run_dir()) / "run.json").read_text())
        assert meta["iterations"] == 3


class TestUpdateTokens:
    """Tests for ArtifactLogger.update_tokens."""

    def test_token_count_updated(self, logger: ArtifactLogger) -> None:
        """update_tokens() persists the token count to run.json."""
        logger.update_tokens(4096)
        meta = json.loads((Path(logger.get_run_dir()) / "run.json").read_text())
        assert meta["total_tokens"] == 4096

    def test_token_count_overwritten_on_update(
        self, logger: ArtifactLogger
    ) -> None:
        """Calling update_tokens() again replaces the previous value."""
        logger.update_tokens(100)
        logger.update_tokens(9999)
        meta = json.loads((Path(logger.get_run_dir()) / "run.json").read_text())
        assert meta["total_tokens"] == 9999


class TestSetDiff:
    """Tests for ArtifactLogger.set_diff."""

    def test_diff_patch_file_written(self, logger: ArtifactLogger) -> None:
        """set_diff() writes a diff.patch file to the run directory."""
        diff_text = "diff --git a/foo.py b/foo.py\n+new line\n"
        logger.set_diff(diff_text)
        patch_path = Path(logger.get_run_dir()) / "diff.patch"
        assert patch_path.exists()
        assert patch_path.read_text() == diff_text


class TestComplete:
    """Tests for ArtifactLogger.complete."""

    def test_status_and_summary_persisted(self, logger: ArtifactLogger) -> None:
        """complete() writes status and summary to run.json."""
        logger.complete("completed", "All done!")
        meta = json.loads((Path(logger.get_run_dir()) / "run.json").read_text())
        assert meta["status"] == "completed"
        assert meta["summary"] == "All done!"
        assert meta["completed_at"] is not None

    def test_failed_status(self, logger: ArtifactLogger) -> None:
        """complete() accepts 'failed' as a status value."""
        logger.complete("failed", "Something went wrong")
        meta = json.loads((Path(logger.get_run_dir()) / "run.json").read_text())
        assert meta["status"] == "failed"

    def test_blocked_status(self, logger: ArtifactLogger) -> None:
        """complete() accepts 'blocked' as a status value."""
        logger.complete("blocked", "Waiting for user input")
        meta = json.loads((Path(logger.get_run_dir()) / "run.json").read_text())
        assert meta["status"] == "blocked"

    def test_completed_at_set_only_after_complete(
        self, logger: ArtifactLogger
    ) -> None:
        """completed_at is None before complete() and set afterwards."""
        meta_before = json.loads(
            (Path(logger.get_run_dir()) / "run.json").read_text()
        )
        assert meta_before["completed_at"] is None
        logger.complete("completed", "done")
        meta_after = json.loads(
            (Path(logger.get_run_dir()) / "run.json").read_text()
        )
        assert meta_after["completed_at"] is not None


class TestFullWorkflow:
    """Integration-style tests running through a full artifact lifecycle."""

    def test_full_run_lifecycle(self, tmp_dir: str) -> None:
        """Exercise every public method and verify final run.json is correct."""
        logger = create_artifact_logger(tmp_dir, "full-run-1", "Fix the bug")
        logger.set_model("claude-3-5-sonnet")
        logger.log_tool_call("c1", "terminal", {"cmd": "pytest"}, "5 passed", True, 150)
        logger.log_tool_call("c2", "editor", {"action": "write_file"}, "ok", True, 80)
        logger.add_command_executed("pytest tests/")
        logger.add_file_modified("src/bug.py")
        logger.increment_iteration()
        logger.increment_iteration()
        logger.update_tokens(2048)
        logger.set_diff("--- a/src/bug.py\n+++ b/src/bug.py\n")
        logger.complete("completed", "Fixed the bug in src/bug.py")

        run_dir = Path(logger.get_run_dir())
        meta = json.loads((run_dir / "run.json").read_text())
        assert meta["model"] == "claude-3-5-sonnet"
        assert meta["status"] == "completed"
        assert meta["total_tokens"] == 2048
        assert meta["iterations"] == 2
        assert "pytest tests/" in meta["commands_executed"]
        assert "src/bug.py" in meta["files_modified"]
        assert meta["completed_at"] is not None

        jsonl_lines = (run_dir / "tool_calls.jsonl").read_text().strip().splitlines()
        assert len(jsonl_lines) == 2

        assert (run_dir / "diff.patch").read_text().startswith("--- a/src/bug.py")
