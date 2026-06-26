from __future__ import annotations

from mini_devin.config.settings import GovernanceTelemetrySettings, Settings


def test_governance_settings_defaults(monkeypatch) -> None:
    monkeypatch.delenv("PLODDER_GOVERNANCE_TELEMETRY", raising=False)
    monkeypatch.delenv("PLODDER_GOVERNANCE_EMIT_BUDGET_SIGNALS", raising=False)
    monkeypatch.delenv("PLODDER_GOVERNANCE_EMIT_RETRY_SIGNALS", raising=False)
    monkeypatch.delenv("PLODDER_GOVERNANCE_EMIT_LOOP_SIGNALS", raising=False)
    monkeypatch.delenv("PLODDER_GOVERNANCE_SCHEMA_VERSION", raising=False)

    cfg = GovernanceTelemetrySettings.from_env()

    assert cfg.enabled is False
    assert cfg.emit_budget_signals is True
    assert cfg.emit_retry_signals is True
    assert cfg.emit_loop_signals is True
    assert cfg.schema_version == "governance.telemetry.v1"


def test_settings_from_env_includes_governance(monkeypatch) -> None:
    monkeypatch.setenv("PLODDER_GOVERNANCE_TELEMETRY", "true")
    monkeypatch.setenv("PLODDER_GOVERNANCE_EMIT_BUDGET_SIGNALS", "false")
    monkeypatch.setenv("PLODDER_GOVERNANCE_EMIT_RETRY_SIGNALS", "true")
    monkeypatch.setenv("PLODDER_GOVERNANCE_EMIT_LOOP_SIGNALS", "false")
    monkeypatch.setenv("PLODDER_GOVERNANCE_SCHEMA_VERSION", "governance.telemetry.v1")

    settings = Settings.from_env()
    as_dict = settings.to_dict()

    assert settings.governance.enabled is True
    assert settings.governance.emit_budget_signals is False
    assert settings.governance.emit_retry_signals is True
    assert settings.governance.emit_loop_signals is False
    assert as_dict["governance"]["enabled"] is True
    assert as_dict["governance"]["emit_budget_signals"] is False
