"""
Fixed regression checks for Mini-Devin platform contracts.
Run on every PR via CI job `eval` — see scenarios.json for the manifest.
"""

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.eval


def test_is_git_remote_url_detects_common_forms():
    from mini_devin.api.app import _is_git_remote_url

    assert _is_git_remote_url("https://github.com/org/repo")
    assert _is_git_remote_url("HTTPS://GITHUB.COM/org/repo")
    assert _is_git_remote_url("http://git.example.com/r.git")
    assert _is_git_remote_url("git@github.com:org/repo.git")
    assert not _is_git_remote_url("")
    assert not _is_git_remote_url("   ")
    assert not _is_git_remote_url("/var/workspace")
    assert not _is_git_remote_url(r"C:\dev\proj")
    assert not _is_git_remote_url("./relative")


def test_git_clone_url_injects_github_token(monkeypatch):
    from mini_devin.api import app as app_mod

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    assert app_mod._git_clone_url_with_token("https://github.com/a/b") == "https://github.com/a/b"

    monkeypatch.setenv("GITHUB_TOKEN", "sec")
    out = app_mod._git_clone_url_with_token("https://github.com/org/repo.git")
    assert "x-access-token:sec@" in out
    assert out.startswith("https://")


def test_normalize_database_url_for_asyncpg():
    from mini_devin.database.config import _normalize_async_database_url

    u = _normalize_async_database_url("postgresql://u:p@localhost:5432/db")
    assert u.startswith("postgresql+asyncpg://")
    u2 = _normalize_async_database_url("postgres://u:p@h/d")
    assert u2.startswith("postgresql+asyncpg://")
    u3 = _normalize_async_database_url("postgresql+asyncpg://u@h/d")
    assert u3 == "postgresql+asyncpg://u@h/d"


def test_init_git_workspace_initializes_repo(tmp_path):
    from mini_devin.api.app import _init_git_workspace

    root = tmp_path / "ws"
    root.mkdir()
    _init_git_workspace(str(root))
    assert (root / ".git").is_dir()
    _init_git_workspace(str(root))


def test_health_endpoints_respond():
    from mini_devin.api.app import app

    client = TestClient(app, raise_server_exceptions=False)
    r1 = client.get("/health")
    assert r1.status_code == 200
    r2 = client.get("/api/health")
    assert r2.status_code == 200


def test_websocket_message_type_has_core_kinds():
    from mini_devin.api.websocket import MessageType

    assert hasattr(MessageType, "TOKEN")
    assert hasattr(MessageType, "TASK_COMPLETED")
    assert hasattr(MessageType, "TASK_FAILED")


def test_rate_limit_allows_fresh_keys():
    from mini_devin.api.app import _check_rate_limit

    key = f"eval-rl-{__import__('uuid').uuid4().hex}"
    assert _check_rate_limit(key, max_calls=5, window_seconds=60) is True
    assert _check_rate_limit(key, max_calls=5, window_seconds=60) is True


def test_invalid_api_path_404():
    from mini_devin.api.app import app

    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/api/this-route-should-not-exist-eval-xyz")
    assert r.status_code == 404


def test_root_or_docs_accessible():
    from mini_devin.api.app import app

    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/")
    assert r.status_code == 200
    r2 = client.get("/docs")
    assert r2.status_code in (200, 404)


def test_cors_options_on_api_preflight():
    from mini_devin.api.app import app

    client = TestClient(app, raise_server_exceptions=False)
    r = client.options(
        "/api/models",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"},
    )
    assert r.status_code in (200, 204, 405)


def test_package_version_string():
    import mini_devin

    assert isinstance(mini_devin.__version__, str)
    assert len(mini_devin.__version__) >= 1


def test_agent_task_status_enum_importable():
    from mini_devin.schemas.state import TaskStatus as AgentTaskStatus

    assert AgentTaskStatus is not None


def test_safety_guards_module_loads():
    from mini_devin.safety import guards as g

    assert g is not None


def test_editor_tool_module_loads():
    from mini_devin.tools.editor import EditorTool

    assert EditorTool is not None


def test_terminal_tool_module_loads():
    from mini_devin.tools import terminal as t

    assert t is not None


def test_llm_provider_enum():
    from mini_devin.core.providers import Provider

    assert Provider.OPENAI.value == "openai"
    assert Provider.ANTHROPIC.value == "anthropic"
