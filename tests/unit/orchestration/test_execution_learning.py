from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from mini_devin.orchestration.execution_learning import (
    ExecutionLearningMemory,
    LearningMemoryPolicy,
    LearningSignal,
    LocalJsonlLearningStore,
    build_failure_fingerprint,
)
from mini_devin.orchestration.observability import TimelineRecord, record_timeline_event


class _Clock:
    def __init__(self, start: datetime) -> None:
        self.current = start

    def now(self) -> datetime:
        return self.current


def _signal(
    *,
    strategy: str,
    fingerprint: str,
    success: bool,
    quality: float,
    verifier_passed: bool,
) -> LearningSignal:
    return LearningSignal(
        session_id="sess",
        unit_id="unit-1",
        strategy_key=strategy,
        fingerprint=fingerprint,
        success=success,
        quality=quality,
        verifier_passed=verifier_passed,
    )


def test_memory_scoring_prefers_verified_success(tmp_path: Path) -> None:
    memory = ExecutionLearningMemory(tmp_path)

    success = memory.remember(
        _signal(
            strategy="incremental patch",
            fingerprint="fp-1",
            success=True,
            quality=0.8,
            verifier_passed=True,
        )
    )
    failure = memory.remember(
        _signal(
            strategy="incremental patch",
            fingerprint="fp-1",
            success=False,
            quality=0.8,
            verifier_passed=False,
        )
    )

    assert success.score > failure.score


def test_decay_behavior_reduces_scores_over_time(tmp_path: Path) -> None:
    clock = _Clock(datetime(2026, 1, 1, tzinfo=timezone.utc))
    memory = ExecutionLearningMemory(
        tmp_path,
        policy=LearningMemoryPolicy(decay_per_day=1.0),
        now_fn=clock.now,
    )

    created = memory.remember(
        _signal(
            strategy="safe strategy",
            fingerprint="fp-2",
            success=True,
            quality=1.0,
            verifier_passed=True,
        )
    )
    baseline = created.score

    clock.current = clock.current + timedelta(days=2)
    refreshed = memory.retrieve_strategies(strategy_hint="safe strategy", limit=1)

    assert refreshed
    assert refreshed[0].score < baseline


def test_contradiction_detection(tmp_path: Path) -> None:
    memory = ExecutionLearningMemory(tmp_path)
    fingerprint = "fp-3"

    memory.remember(
        _signal(
            strategy="same strategy",
            fingerprint=fingerprint,
            success=False,
            quality=-0.4,
            verifier_passed=False,
        )
    )

    assert memory.detect_contradiction(strategy_key="same strategy", fingerprint=fingerprint, success=True)


def test_strategy_retrieval_returns_best_scored_match(tmp_path: Path) -> None:
    memory = ExecutionLearningMemory(tmp_path)

    memory.remember(
        _signal(
            strategy="pipeline optimize",
            fingerprint="fp-4",
            success=True,
            quality=0.3,
            verifier_passed=True,
        )
    )
    memory.remember(
        _signal(
            strategy="pipeline optimize",
            fingerprint="fp-4",
            success=True,
            quality=0.9,
            verifier_passed=True,
        )
    )

    picked = memory.retrieve_strategies(strategy_hint="pipeline optimize", limit=1)

    assert picked
    assert picked[0].strategy_key == "pipeline optimize"
    assert picked[0].outcome == "success"


def test_failure_fingerprint_reuse(tmp_path: Path) -> None:
    memory = ExecutionLearningMemory(tmp_path)
    fingerprint = build_failure_fingerprint(strategy_key="fix flaky tests", error="timeout", status="failed")

    memory.remember(
        _signal(
            strategy="fix flaky tests",
            fingerprint=fingerprint,
            success=False,
            quality=-0.9,
            verifier_passed=False,
        )
    )

    pattern = memory.retrieve_failure_pattern(fingerprint)

    assert pattern is not None
    assert pattern.fingerprint == fingerprint
    assert pattern.outcome == "failure"


def test_replay_learning_integration(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PLODDER_OBSERVABILITY", "1")
    monkeypatch.setenv("PLODDER_TIMELINE_RECORDING", "1")

    record_timeline_event(
        tmp_path,
        TimelineRecord(
            event_type="unit.promoted",
            source="coordination",
            session_id="sess-1",
            task_id="task-1",
            unit_id="unit-9",
            status="success",
            payload={"unit_id": "unit-9", "goal": "promote path"},
        ),
    )
    record_timeline_event(
        tmp_path,
        TimelineRecord(
            event_type="recovery.decision",
            source="coordination",
            session_id="sess-1",
            task_id="task-1",
            unit_id="unit-10",
            status="failed",
            payload={"unit_id": "unit-10", "goal": "recover path", "action": "replan"},
        ),
    )

    memory = ExecutionLearningMemory(tmp_path)
    learned = memory.ingest_replay_learning()

    entries = LocalJsonlLearningStore(tmp_path).load_all()

    assert learned >= 2
    assert any(entry.replay_driven for entry in entries)


def test_bounded_memory_growth(tmp_path: Path) -> None:
    memory = ExecutionLearningMemory(tmp_path, policy=LearningMemoryPolicy(max_entries=5))

    for index in range(25):
        memory.remember(
            LearningSignal(
                session_id="sess",
                unit_id=f"unit-{index}",
                strategy_key=f"strategy-{index}",
                fingerprint=f"fp-{index}",
                success=bool(index % 2),
                quality=0.2,
                verifier_passed=bool(index % 2),
            )
        )

    entries = LocalJsonlLearningStore(tmp_path).load_all()
    assert len(entries) <= 5
