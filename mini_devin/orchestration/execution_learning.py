"""Bounded execution learning memory for additive self-improvement.

This module is intentionally local-first and feature-flagged. It stores compact
execution outcomes, failure fingerprints, and strategy scores in a bounded file-
backed memory. The API is shaped so an external vector database backend can be
introduced later without changing runtime call sites.
"""

from __future__ import annotations

import json
import math
import os
import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from .observability import replay_timeline_records


def _flag_enabled(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def execution_learning_enabled() -> bool:
    return _flag_enabled("PLODDER_EXECUTION_LEARNING")


def execution_learning_replay_enabled() -> bool:
    return _flag_enabled("PLODDER_EXECUTION_LEARNING_REPLAY", True)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_strategy_key(value: str) -> str:
    return " ".join(value.strip().lower().split())[:400]


def _coerce_datetime(raw: Any) -> datetime:
    if isinstance(raw, datetime):
        if raw.tzinfo is None:
            return raw.replace(tzinfo=timezone.utc)
        return raw
    if isinstance(raw, str):
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        except ValueError:
            return _utcnow()
    return _utcnow()


def _bounded(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def build_failure_fingerprint(*, strategy_key: str, error: str | None, status: str | None) -> str:
    seed = "|".join(
        [
            _normalize_strategy_key(strategy_key),
            str(error or ""),
            str(status or ""),
        ]
    )
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:20]


@dataclass(frozen=True, slots=True)
class LearningMemoryPolicy:
    max_entries: int = 256
    decay_per_day: float = 0.04
    quality_weight: float = 0.6
    success_reward: float = 1.0
    failure_penalty: float = 0.8
    verifier_reward: float = 0.3
    verifier_penalty: float = 0.3
    replay_reward: float = 0.2
    contradiction_penalty: float = 0.7
    min_quality: float = -1.0
    max_quality: float = 1.0


@dataclass(frozen=True, slots=True)
class LearningSignal:
    session_id: str
    unit_id: str
    strategy_key: str
    fingerprint: str
    success: bool
    quality: float
    verifier_passed: bool
    replay_driven: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LearningEntry:
    entry_id: str
    ts: datetime
    session_id: str
    unit_id: str
    strategy_key: str
    fingerprint: str
    outcome: str
    quality: float
    verifier_passed: bool
    replay_driven: bool
    score: float
    contradictions: int = 0
    reuse_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def retention_score(self) -> float:
        # Prefer high-confidence successful patterns while retaining some failures for diagnostics.
        boost = 0.12 * float(self.reuse_count)
        contradiction_drag = 0.25 * float(self.contradictions)
        return float(self.score) + boost - contradiction_drag

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "ts": self.ts.isoformat(),
            "session_id": self.session_id,
            "unit_id": self.unit_id,
            "strategy_key": self.strategy_key,
            "fingerprint": self.fingerprint,
            "outcome": self.outcome,
            "quality": self.quality,
            "verifier_passed": self.verifier_passed,
            "replay_driven": self.replay_driven,
            "score": self.score,
            "contradictions": self.contradictions,
            "reuse_count": self.reuse_count,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> LearningEntry:
        return cls(
            entry_id=str(row.get("entry_id") or uuid.uuid4().hex),
            ts=_coerce_datetime(row.get("ts")),
            session_id=str(row.get("session_id") or ""),
            unit_id=str(row.get("unit_id") or ""),
            strategy_key=_normalize_strategy_key(str(row.get("strategy_key") or "")),
            fingerprint=str(row.get("fingerprint") or ""),
            outcome=str(row.get("outcome") or "failure"),
            quality=float(row.get("quality", 0.0)),
            verifier_passed=bool(row.get("verifier_passed")),
            replay_driven=bool(row.get("replay_driven")),
            score=float(row.get("score", 0.0)),
            contradictions=int(row.get("contradictions", 0)),
            reuse_count=int(row.get("reuse_count", 0)),
            metadata=dict(row.get("metadata") or {}),
        )


class LearningStoreBackend(Protocol):
    def load_all(self) -> list[LearningEntry]:
        ...

    def save_all(self, entries: list[LearningEntry]) -> None:
        ...


class LocalJsonlLearningStore:
    """Local bounded file store under ``.plodder/execution_learning.jsonl``."""

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace)

    def _root(self) -> Path:
        root = self.workspace.resolve()
        root.mkdir(parents=True, exist_ok=True)
        out = root / ".plodder"
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _path(self) -> Path:
        return self._root() / "execution_learning.jsonl"

    def _cursor_path(self) -> Path:
        return self._root() / "execution_learning_replay_cursor.txt"

    def load_all(self) -> list[LearningEntry]:
        path = self._path()
        if not path.is_file():
            return []
        rows: list[LearningEntry] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(LearningEntry.from_dict(row))
        except (OSError, json.JSONDecodeError):
            return []
        return rows

    def save_all(self, entries: list[LearningEntry]) -> None:
        path = self._path()
        with path.open("w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry.to_dict(), default=str, ensure_ascii=False) + "\n")

    def load_replay_cursor(self) -> int:
        path = self._cursor_path()
        if not path.is_file():
            return 0
        try:
            return max(0, int(path.read_text(encoding="utf-8").strip() or "0"))
        except (OSError, ValueError):
            return 0

    def save_replay_cursor(self, index: int) -> None:
        self._cursor_path().write_text(str(max(0, int(index))), encoding="utf-8")


def score_task_outcome_quality(
    *,
    success: bool,
    verifier_passed: bool,
    summary: str,
    error: str | None,
) -> float:
    quality = 0.65 if success else -0.65
    if verifier_passed:
        quality += 0.2
    else:
        quality -= 0.2
    if error:
        quality -= 0.2
    if summary.strip():
        quality += 0.05
    return _bounded(quality, -1.0, 1.0)


class ExecutionLearningMemory:
    """Bounded scored memory for execution outcomes and strategy adaptation."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        store: LearningStoreBackend | None = None,
        policy: LearningMemoryPolicy | None = None,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self.workspace = str(workspace)
        self.policy = policy or LearningMemoryPolicy()
        self.store = store or LocalJsonlLearningStore(workspace)
        self._now_fn = now_fn or _utcnow

    def _now(self) -> datetime:
        return self._now_fn()

    def _apply_decay(self, entries: list[LearningEntry]) -> list[LearningEntry]:
        now = self._now()
        out: list[LearningEntry] = []
        for entry in entries:
            age_days = max(0.0, (now - entry.ts).total_seconds() / 86400.0)
            decay = math.exp(-self.policy.decay_per_day * age_days)
            entry.score = float(entry.score) * decay
            out.append(entry)
        return out

    def _bounded_entries(self, entries: list[LearningEntry]) -> list[LearningEntry]:
        if len(entries) <= self.policy.max_entries:
            return entries
        ranked = sorted(entries, key=lambda item: (item.retention_score(), item.ts), reverse=True)
        kept = ranked[: self.policy.max_entries]
        return sorted(kept, key=lambda item: item.ts)

    def _score_signal(self, signal: LearningSignal, contradictions: int) -> float:
        quality = _bounded(signal.quality, self.policy.min_quality, self.policy.max_quality)
        score = quality * self.policy.quality_weight
        if signal.success:
            score += self.policy.success_reward
        else:
            score -= self.policy.failure_penalty
        if signal.verifier_passed:
            score += self.policy.verifier_reward
        else:
            score -= self.policy.verifier_penalty
        if signal.replay_driven:
            score += self.policy.replay_reward
        score -= float(contradictions) * self.policy.contradiction_penalty
        return float(score)

    def detect_contradiction(self, *, strategy_key: str, fingerprint: str, success: bool) -> bool:
        key = _normalize_strategy_key(strategy_key)
        for entry in self.store.load_all():
            if entry.strategy_key != key:
                continue
            if entry.fingerprint != fingerprint:
                continue
            if entry.outcome != ("success" if success else "failure"):
                return True
        return False

    def remember(self, signal: LearningSignal) -> LearningEntry:
        entries = self._apply_decay(self.store.load_all())
        strategy_key = _normalize_strategy_key(signal.strategy_key)
        contradiction_count = 0
        for entry in entries:
            if entry.strategy_key == strategy_key and entry.fingerprint == signal.fingerprint:
                if entry.outcome != ("success" if signal.success else "failure"):
                    contradiction_count += 1
                    entry.contradictions += 1

        scored = self._score_signal(signal, contradiction_count)
        created = LearningEntry(
            entry_id=uuid.uuid4().hex,
            ts=self._now(),
            session_id=signal.session_id,
            unit_id=signal.unit_id,
            strategy_key=strategy_key,
            fingerprint=signal.fingerprint,
            outcome="success" if signal.success else "failure",
            quality=float(_bounded(signal.quality, -1.0, 1.0)),
            verifier_passed=signal.verifier_passed,
            replay_driven=signal.replay_driven,
            score=scored,
            contradictions=contradiction_count,
            reuse_count=0,
            metadata=dict(signal.metadata),
        )
        entries.append(created)
        bounded = self._bounded_entries(entries)
        self.store.save_all(bounded)
        return created

    def retrieve_strategies(
        self,
        *,
        strategy_hint: str,
        fingerprint: str | None = None,
        success_only: bool = True,
        limit: int = 3,
    ) -> list[LearningEntry]:
        hint = _normalize_strategy_key(strategy_hint)
        entries = self._apply_decay(self.store.load_all())
        scored: list[tuple[float, LearningEntry]] = []
        for entry in entries:
            if success_only and entry.outcome != "success":
                continue
            if fingerprint and entry.fingerprint != fingerprint:
                continue
            affinity = 0.0
            if entry.strategy_key == hint:
                affinity = 1.0
            elif hint and hint in entry.strategy_key:
                affinity = 0.6
            elif entry.strategy_key and entry.strategy_key in hint:
                affinity = 0.5
            scored.append((entry.score + affinity, entry))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [entry for _, entry in scored[: max(1, int(limit))]]
        for entry in selected:
            entry.reuse_count += 1
        if selected:
            self.store.save_all(self._bounded_entries(entries))
        return selected

    def retrieve_failure_pattern(self, fingerprint: str) -> LearningEntry | None:
        entries = self._apply_decay(self.store.load_all())
        failures = [entry for entry in entries if entry.fingerprint == fingerprint and entry.outcome == "failure"]
        if not failures:
            return None
        failures.sort(key=lambda item: item.retention_score(), reverse=True)
        best = failures[0]
        best.reuse_count += 1
        self.store.save_all(self._bounded_entries(entries))
        return best

    def ingest_replay_learning(self, *, max_lines: int = 4000) -> int:
        if not execution_learning_replay_enabled():
            return 0

        entries = replay_timeline_records(self.workspace, max_lines=max_lines)
        store = self.store if isinstance(self.store, LocalJsonlLearningStore) else None
        cursor = store.load_replay_cursor() if store is not None else 0
        learned = 0
        newest_index = cursor

        for item in entries:
            if item.index < cursor:
                continue
            newest_index = max(newest_index, item.index + 1)
            record = item.record
            if record.event_type not in ("unit.promoted", "recovery.decision"):
                continue
            payload = dict(record.payload or {})
            unit_id = str(record.unit_id or payload.get("unit_id") or "unknown")
            strategy_key = _normalize_strategy_key(str(payload.get("goal") or unit_id))
            fingerprint = f"replay:{unit_id}:{record.event_type}"
            success = record.event_type == "unit.promoted"
            verifier_passed = success
            quality = 0.8 if success else -0.6
            self.remember(
                LearningSignal(
                    session_id=str(record.session_id or "replay"),
                    unit_id=unit_id,
                    strategy_key=strategy_key,
                    fingerprint=fingerprint,
                    success=success,
                    quality=quality,
                    verifier_passed=verifier_passed,
                    replay_driven=True,
                    metadata={
                        "event_type": record.event_type,
                        "status": record.status,
                    },
                )
            )
            learned += 1

        if store is not None:
            store.save_replay_cursor(newest_index)
        return learned
