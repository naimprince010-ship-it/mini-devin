from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from mini_devin.orchestrator.agent import Agent
from mini_devin.orchestrator.session_events import load_session_events
from mini_devin.orchestrator.standard_events import AgentEventKind, AgentStreamEvent
from mini_devin.reliability.ops_telemetry import FileOpsTelemetryCollector, telemetry_config_from_env


def _build_status_event() -> AgentStreamEvent:
    return AgentStreamEvent(
        kind=AgentEventKind.STATUS,
        role="system",
        text="governance probe",
        legacy_type="test_status",
    )


def test_append_session_event_includes_governance_payload(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PLODDER_GOVERNANCE_TELEMETRY", "true")
    monkeypatch.setenv("PLODDER_GOVERNANCE_EMIT_BUDGET_SIGNALS", "true")
    monkeypatch.setenv("PLODDER_GOVERNANCE_EMIT_RETRY_SIGNALS", "true")
    monkeypatch.setenv("PLODDER_GOVERNANCE_EMIT_LOOP_SIGNALS", "true")
    monkeypatch.setenv("PLODDER_GOVERNANCE_TOKEN_BUDGET", "1000")
    monkeypatch.setenv("PLODDER_REPO_ROOT", str(tmp_path))

    agent = Agent(llm_client=MagicMock(), working_directory=str(tmp_path), auto_verify=False, verbose=False)
    agent._append_session_event(_build_status_event())

    rows = load_session_events(tmp_path, max_lines=50)
    assert rows
    row = rows[-1]
    assert row["governance_schema"] == "governance.telemetry.v1"
    assert row["governance_observe_only"] is True
    assert isinstance(row.get("governance_signals"), list)
    assert {signal["signal_type"] for signal in row["governance_signals"]} == {"budget", "retry", "loop"}
    for signal in row["governance_signals"]:
        assert signal["schema"] == "governance.telemetry.v1"
        assert signal["observe_only"] is True
        assert isinstance(signal.get("counters"), dict)


def test_governance_signals_bridge_into_ops_telemetry(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PLODDER_GOVERNANCE_TELEMETRY", "true")
    monkeypatch.setenv("PLODDER_GOVERNANCE_EMIT_BUDGET_SIGNALS", "true")
    monkeypatch.setenv("PLODDER_GOVERNANCE_EMIT_RETRY_SIGNALS", "true")
    monkeypatch.setenv("PLODDER_GOVERNANCE_EMIT_LOOP_SIGNALS", "true")
    monkeypatch.setenv("PLODDER_REPO_ROOT", str(tmp_path))

    agent = Agent(llm_client=MagicMock(), working_directory=str(tmp_path), auto_verify=False, verbose=False)
    agent._append_session_event(_build_status_event())

    telemetry_root = tmp_path / ".plodder" / "ops" / "telemetry"
    collector = FileOpsTelemetryCollector(
        events_file=telemetry_root / "events.jsonl",
        state_file=telemetry_root / "state.json",
        config=telemetry_config_from_env(),
    )
    export = collector.export(hours=1)
    kpis = export.get("kpis", {})
    assert kpis.get("governance_signals", 0) >= 1

    raw_lines = (telemetry_root / "events.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in raw_lines if line.strip()]
    governance_rows = [row for row in rows if row.get("event_type") == "governance.signal"]
    assert governance_rows
    assert all(row.get("metrics", {}).get("observe_only") is True for row in governance_rows)
    assert all(row.get("metrics", {}).get("schema") == "governance.telemetry.v1" for row in governance_rows)
