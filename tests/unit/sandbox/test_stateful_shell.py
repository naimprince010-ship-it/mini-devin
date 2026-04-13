"""Stateful cwd via .mini_devin/_shell (ProcessSandbox)."""

import os

import pytest

from mini_devin.sandbox.runtime_protocol import normalize_exec_result
from mini_devin.sandbox.process_sandbox import ProcessSandbox


@pytest.mark.skipif(os.name == "nt", reason="ProcessSandbox stateful mode is POSIX-only")
def test_cd_persists_across_invocations(tmp_path):
    sb = ProcessSandbox(str(tmp_path), stateful_shell=True)
    er1 = normalize_exec_result(sb.exec_bash("mkdir -p deep/nested && cd deep/nested && pwd"))
    out1 = er1.stdout.decode("utf-8", errors="replace")
    assert "deep/nested" in out1.replace("\\", "/")

    er2 = normalize_exec_result(sb.exec_bash("pwd"))
    out2 = er2.stdout.decode("utf-8", errors="replace").strip().replace("\\", "/")
    assert out2.endswith("deep/nested")
