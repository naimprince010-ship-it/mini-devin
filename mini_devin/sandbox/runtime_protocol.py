"""
Execution runtime protocol — swap process vs Docker sandboxes behind one surface.

Implementations:
  - :class:`mini_devin.sandbox.process_sandbox.ProcessSandbox` (stateful cwd via ``.mini_devin/_shell``)
  - :class:`mini_devin.sandbox.sandbox.SimpleDockerSandbox`

Callers that depend only on :class:`CommandRuntime` can choose either backend without
changing orchestration logic. Aligns with an **OpenHands-style** “persistent shell
semantics” using a state file (no copied upstream code).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ExecutionResult:
    """Normalized result of a shell invocation (stdout/stderr split + timeout flag)."""

    exit_code: int
    stdout: bytes
    stderr: bytes
    timed_out: bool = False

    @property
    def output(self) -> bytes:
        """Backward-compatible combined body (stdout then stderr), similar to legacy ``exec_run``."""
        if not self.stderr:
            return self.stdout
        if not self.stdout:
            return self.stderr
        return self.stdout + b"\n" + self.stderr


@runtime_checkable
class CommandRuntime(Protocol):
    """
    Minimal runtime contract for “run bash in workspace”.

    ``WORKSPACE`` is the logical path inside the sandbox (e.g. ``/workspace``);
    implementations map it to the host project root.
    """

    WORKSPACE: str

    @property
    def container_id(self) -> str:
        """Opaque id for logs (``process`` or Docker container id)."""
        ...

    def exec_bash(
        self,
        command: str,
        *,
        workdir: str | None = None,
        environment: dict[str, str] | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> tuple[int, bytes] | tuple[int, tuple[bytes, bytes]] | ExecutionResult:
        """
        Run ``command`` via ``bash -lc`` (often wrapped for **stateful cwd**).

        Implementations may return demuxed tuples or a full :class:`ExecutionResult`
        (e.g. on timeout); use :func:`normalize_exec_result` for a single result object.
        """
        ...

    def stop(self, *, remove: bool = True, timeout: int = 10) -> None:
        """Tear down runtime (no-op for process backend)."""
        ...


def normalize_exec_result(
    raw: tuple[int, bytes] | tuple[int, tuple[bytes, bytes]] | ExecutionResult,
) -> ExecutionResult:
    """Flatten docker demux, legacy combined bytes, or pass through :class:`ExecutionResult`."""
    if isinstance(raw, ExecutionResult):
        return raw
    code, body = raw
    if isinstance(body, tuple):
        out, err = body[0] or b"", body[1] or b""
        return ExecutionResult(exit_code=code, stdout=out, stderr=err, timed_out=False)
    return ExecutionResult(exit_code=code, stdout=body or b"", stderr=b"", timed_out=False)


def execution_result_with_timeout(base: ExecutionResult, *, timed_out: bool) -> ExecutionResult:
    """Return a copy with ``timed_out`` set (frozen dataclass)."""
    return ExecutionResult(
        exit_code=base.exit_code,
        stdout=base.stdout,
        stderr=base.stderr,
        timed_out=timed_out,
    )
