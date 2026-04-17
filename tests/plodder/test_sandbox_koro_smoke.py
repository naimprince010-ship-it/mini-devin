"""
KORO sandbox smoke: .sql, pom.xml (Maven), composer.json (Composer).

Plan-level tests need no Docker. ``run_detected`` smokes use a mocked Docker client so
CI / laptops without Docker still exercise the full ``plan_container_run`` → ``_run_container`` path.

Run (from repo root):
  python -m pytest tests/plodder/test_sandbox_koro_smoke.py -v
"""

from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock

import pytest

from plodder.sandbox.container_manager import plan_container_run
from plodder.sandbox.toolchain_detect import build_toolchain_spec

_DEFAULT_IMAGES = dict(
    python_image="python:3.11-alpine",
    node_image="node:20-alpine",
    go_image="golang:1.22-alpine",
    rust_image="rust:alpine",
    alpine_image="alpine:3.19",
    cpp_image="gcc:12-bookworm",
    typescript_image="node:22-alpine",
    java_image="eclipse-temurin:21-jdk-alpine",
    php_image="php:8.3-cli-alpine",
    dotnet_image="mcr.microsoft.com/dotnet/sdk:8.0-alpine",
    maven_image="maven:3.9.9-eclipse-temurin-21-alpine",
    gradle_image="gradle:8.10.2-jdk21-alpine",
    composer_image="composer:2",
    postgres_client_image="postgres:16-alpine",
)


def _plan(**kwargs):
    return plan_container_run(
        language_hint=kwargs.pop("language_hint", None),
        language_key=kwargs.pop("language_key", None),
        docker_client=None,
        prefer_generic_if_image_missing=False,
        auto_pull_missing=False,
        **_DEFAULT_IMAGES,
        **kwargs,
    )


def _make_mocked_sandbox(monkeypatch: pytest.MonkeyPatch, *, exit_code: int, out: bytes, err: bytes):
    """``ExecutionSandbox`` with Docker API mocked (no ``docker`` PyPI pkg / daemon required)."""
    container = MagicMock()
    container.id = "mock-cid"
    container.wait.return_value = {"StatusCode": exit_code}
    container.logs.side_effect = [out, err]

    mock_client = MagicMock()
    mock_client.containers.create.return_value = container

    fake_docker = types.ModuleType("docker")
    fake_docker.from_env = lambda *a, **k: mock_client  # noqa: ARG005
    monkeypatch.setitem(sys.modules, "docker", fake_docker)

    from plodder.sandbox.execution_sandbox import ExecutionSandbox

    sb = ExecutionSandbox()
    return sb, mock_client


def test_build_toolchain_sql_no_url_uses_alpine_hint():
    spec = build_toolchain_spec(
        "migrations/001.sql",
        language_hint=None,
        sql_url=None,
        workspace_files={"migrations/001.sql": "SELECT 1;"},
        **_DEFAULT_IMAGES,
    )
    assert spec.language_id == "sql"
    assert spec.image == _DEFAULT_IMAGES["alpine_image"]
    joined = " ".join(spec.argv)
    assert "SANDBOX_SQL_URL" in joined or "DATABASE_URL" in joined


def test_build_toolchain_sql_with_url_uses_psql_env():
    spec = build_toolchain_spec(
        "migrations/001.sql",
        language_hint=None,
        sql_url="postgresql://u:p@db.example:5432/app",
        workspace_files={"migrations/001.sql": "SELECT 1;"},
        **_DEFAULT_IMAGES,
    )
    assert spec.language_id == "sql"
    assert spec.image == _DEFAULT_IMAGES["postgres_client_image"]
    assert dict(spec.container_env).get("PLODDER_SQL_URL", "").startswith("postgresql://")
    assert "psql" in " ".join(spec.argv)


def test_plan_maven_when_pom_in_ancestor():
    ws = {
        "pom.xml": """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>
  <groupId>local</groupId>
  <artifactId>koro-smoke</artifactId>
  <version>1.0-SNAPSHOT</version>
</project>""",
        "src/main/java/App.java": "class App { public static void main(String[] a) { } }",
    }
    planned = _plan(entry="src/main/java/App.java", workspace_files=ws)
    assert planned.image == _DEFAULT_IMAGES["maven_image"]
    cmd = " ".join(planned.argv)
    assert "mvn" in cmd and "package" in cmd


def test_plan_composer_when_composer_json_present():
    ws = {
        "composer.json": """{
  "name": "acme/koro-smoke",
  "description": "smoke",
  "require": {}
}""",
        "index.php": "<?php echo \"koro_php\\n\";",
    }
    planned = _plan(entry="index.php", workspace_files=ws)
    assert planned.image == _DEFAULT_IMAGES["composer_image"]
    cmd = " ".join(planned.argv)
    assert "composer install" in cmd and "php /workspace/index.php" in cmd


def test_run_detected_sql_without_db_url_mocked(monkeypatch):
    monkeypatch.delenv("SANDBOX_SQL_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    hint = b"Set SANDBOX_SQL_URL or DATABASE_URL (Postgres) for .sql runs.\n"
    sb, client = _make_mocked_sandbox(monkeypatch, exit_code=1, out=hint, err=b"")
    r = sb.run_detected({"init.sql": "SELECT 1;"}, entry="init.sql", network=False)
    assert r.exit_code == 1
    blob = (r.stderr or "") + (r.stdout or "")
    assert "SANDBOX_SQL_URL" in blob or "DATABASE_URL" in blob
    image = client.containers.create.call_args[0][0]
    assert image == sb.alpine_image


def test_run_detected_sql_with_url_passes_env_mocked(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db.internal:5432/app")
    sb, client = _make_mocked_sandbox(monkeypatch, exit_code=0, out=b"ok\n", err=b"")
    r = sb.run_detected({"init.sql": "SELECT 1;"}, entry="init.sql", network=False)
    assert r.exit_code == 0
    kwargs = client.containers.create.call_args[1]
    assert kwargs.get("environment") == {"PLODDER_SQL_URL": "postgresql://u:p@db.internal:5432/app"}
    assert client.containers.create.call_args[0][0] == sb.postgres_client_image


def test_run_detected_php_composer_mocked(monkeypatch):
    sb, client = _make_mocked_sandbox(monkeypatch, exit_code=0, out=b"koro_ok\n", err=b"")
    files = {
        "composer.json": """{
  "name": "acme/koro-smoke",
  "description": "smoke",
  "require": {}
}""",
        "index.php": "<?php echo \"koro_ok\\n\";",
    }
    r = sb.run_detected(files, entry="index.php", network=False, timeout_sec=120)
    assert r.exit_code == 0, r.stderr + r.stdout
    assert "koro_ok" in (r.stdout or "")
    assert client.containers.create.call_args[0][0] == sb.composer_image


def test_run_detected_maven_mocked(monkeypatch):
    sb, client = _make_mocked_sandbox(
        monkeypatch,
        exit_code=0,
        out=b"[INFO] BUILD SUCCESS\n",
        err=b"",
    )
    files = {
        "pom.xml": """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>
  <groupId>local</groupId>
  <artifactId>koro-smoke</artifactId>
  <version>1.0-SNAPSHOT</version>
  <properties>
    <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    <maven.compiler.release>11</maven.compiler.release>
  </properties>
</project>""",
        "src/main/java/App.java": """public class App {
  public static void main(String[] args) {
    System.out.println("koro_mvn");
  }
}""",
    }
    r = sb.run_detected(files, entry="src/main/java/App.java", network=True, timeout_sec=60)
    assert r.exit_code == 0, r.stderr + r.stdout
    combined = (r.stdout or "") + (r.stderr or "")
    assert "BUILD SUCCESS" in combined
    assert client.containers.create.call_args[0][0] == sb.maven_image


def _docker_client_ok() -> bool:
    try:
        import docker as docker_mod
    except ImportError:
        return False
    try:
        docker_mod.from_env().ping()
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _docker_client_ok(),
    reason="Optional: real Docker engine smoke (machine has no Docker).",
)
@pytest.mark.skipif(
    not os.environ.get("RUN_KORO_MAVEN_DOCKER"),
    reason="Optional: set RUN_KORO_MAVEN_DOCKER=1 for real Maven + network.",
)
def test_run_detected_maven_real_docker_optional(monkeypatch):
    """Manual / CI-with-Docker: hits Maven Central; off by default."""
    pytest.importorskip("docker")
    monkeypatch.delenv("SANDBOX_SQL_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from plodder.sandbox.execution_sandbox import ExecutionSandbox

    sb = ExecutionSandbox()
    files = {
        "pom.xml": """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>
  <groupId>local</groupId>
  <artifactId>koro-smoke</artifactId>
  <version>1.0-SNAPSHOT</version>
  <properties>
    <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    <maven.compiler.release>11</maven.compiler.release>
  </properties>
</project>""",
        "src/main/java/App.java": """public class App {
  public static void main(String[] args) {
    System.out.println("koro_mvn");
  }
}""",
    }
    r = sb.run_detected(files, entry="src/main/java/App.java", network=True, timeout_sec=300)
    assert r.exit_code == 0, r.stderr + r.stdout
    assert "BUILD SUCCESS" in ((r.stdout or "") + (r.stderr or ""))
