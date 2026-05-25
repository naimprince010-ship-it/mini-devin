from __future__ import annotations

import importlib

from mini_devin.reliability.startup_guard import run_startup_preflight


def test_config_corruption_is_reported(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_INIT_TIMEOUT", "not-a-number")

    report = run_startup_preflight()

    assert report.has_errors is True
    assert any(issue.code == "config.invalid.database_init_timeout" for issue in report.checks)


def test_partial_dependency_outage_marks_degraded(monkeypatch) -> None:
    monkeypatch.setenv("PLODDER_QUEUE_BACKEND", "redis_streams")

    report = run_startup_preflight()

    assert report.startup_mode == "degraded"
    assert any(issue.code.startswith("import.failed.redis") for issue in report.checks)


def test_strict_mode_requires_fail_fast_on_error(monkeypatch) -> None:
    monkeypatch.setenv("PLODDER_STRICT_STARTUP", "1")
    monkeypatch.setenv("CREATE_SESSION_DB_WAIT_SEC", "broken")

    report = run_startup_preflight()

    assert report.strict_mode is True
    assert report.should_fail_fast is True


def test_degraded_mode_startup_without_hard_failure(monkeypatch) -> None:
    monkeypatch.delenv("PLODDER_STRICT_STARTUP", raising=False)
    monkeypatch.setenv("PLODDER_QUEUE_BACKEND", "redis_streams")

    report = run_startup_preflight()

    assert report.startup_mode == "degraded"
    assert report.should_fail_fast is False


def test_required_sandbox_dependency_failure_is_fatal(monkeypatch) -> None:
    original_import = importlib.import_module

    def _patched_import(name: str, package=None):  # type: ignore[no-untyped-def]
        if name == "mini_devin.sandbox.factory":
            raise RuntimeError("sandbox bootstrap unavailable")
        return original_import(name, package)

    monkeypatch.setenv("PLODDER_REQUIRE_SANDBOX", "1")
    monkeypatch.setenv("PLODDER_STRICT_STARTUP", "1")
    monkeypatch.setattr(importlib, "import_module", _patched_import)

    report = run_startup_preflight()

    assert report.has_errors is True
    assert report.should_fail_fast is True
    assert any(issue.code == "import.failed.mini_devin.sandbox.factory" for issue in report.checks)
