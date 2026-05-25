from __future__ import annotations

from mini_devin.reliability.incident_state import RuntimeIncidentTracker


def test_crash_loop_detection(tmp_path) -> None:
    tracker = RuntimeIncidentTracker(crash_loop_file=tmp_path / "crash_loop.json", threshold=2, window_seconds=600)

    tracker.record_failure("startup.db.exception", "db failed")
    tracker.record_failure("startup.db.exception", "db failed again")

    snapshot = tracker.crash_loop_snapshot()
    assert snapshot.active is True
    assert snapshot.failure_count >= 2


def test_runtime_incident_recovery(tmp_path) -> None:
    tracker = RuntimeIncidentTracker(crash_loop_file=tmp_path / "crash_loop.json", threshold=3, window_seconds=600)

    tracker.record_failure("startup.db.timeout", "timed out")
    recovered = tracker.record_recovery("startup.db.timeout", "recovered after retry")

    assert recovered is not None
    assert recovered.state == "resolved"
    diag = tracker.diagnostics()
    assert diag["incidents"][0]["state"] == "resolved"
