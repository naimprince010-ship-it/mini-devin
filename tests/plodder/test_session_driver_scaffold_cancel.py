"""sandbox_shell result shaping when npm/create-vite cancels in headless mode."""

from plodder.orchestration.session_driver import UnifiedSessionDriver, _shell_scaffold_operation_cancelled
from plodder.sandbox.execution_sandbox import SandboxResult


def test_shell_scaffold_operation_cancelled_detects_create_vite() -> None:
    assert _shell_scaffold_operation_cancelled(
        "Operation cancelled\n", "", "sh -c npx create-vite@latest foo --template react-ts"
    )
    assert not _shell_scaffold_operation_cancelled("ok\n", "", "sh -c npx create-vite@latest foo")
    assert not _shell_scaffold_operation_cancelled(
        "Operation cancelled\n", "", "sh -c echo hello"
    )


def test_sandbox_result_dict_marks_soft_cancel_as_not_ok() -> None:
    r = SandboxResult(
        stdout="Operation cancelled\n",
        stderr="",
        exit_code=0,
        timed_out=False,
        container_id=None,
        command="sh -c npx create-vite@latest smoke-ui --template react-ts",
    )
    d = UnifiedSessionDriver._sandbox_result_dict("sandbox_shell", r)
    assert d["ok"] is False
    assert d["exit_code"] == 0


def test_sandbox_result_dict_plain_success_unchanged() -> None:
    r = SandboxResult(
        stdout="done\n",
        stderr="",
        exit_code=0,
        timed_out=False,
        container_id=None,
        command="sh -c npm install",
    )
    d = UnifiedSessionDriver._sandbox_result_dict("sandbox_shell", r)
    assert d["ok"] is True
