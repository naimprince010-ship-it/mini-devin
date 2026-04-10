"""Sandbox factory: Docker vs E2B selection."""

from mini_devin.sandbox.factory import create_execution_sandbox, get_sandbox_backend
from mini_devin.sandbox.docker_sandbox import DockerSandbox


def test_get_sandbox_backend_default(monkeypatch):
    monkeypatch.delenv("SANDBOX_BACKEND", raising=False)
    assert get_sandbox_backend() == "docker"


def test_e2b_without_key_falls_back_to_docker(monkeypatch, tmp_path):
    monkeypatch.setenv("SANDBOX_BACKEND", "e2b")
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    d = tmp_path / "repo"
    d.mkdir()
    sb = create_execution_sandbox(str(d))
    assert isinstance(sb, DockerSandbox)


def test_docker_explicit(monkeypatch, tmp_path):
    monkeypatch.setenv("SANDBOX_BACKEND", "docker")
    d = tmp_path / "r"
    d.mkdir()
    sb = create_execution_sandbox(str(d))
    assert isinstance(sb, DockerSandbox)
    assert sb.backend == "docker"
