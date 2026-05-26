"""Human-in-the-loop governance and approval boundaries.

This module is additive and local-first. It classifies execution actions, computes
risk, enforces bounded approval workflow, and stores tamper-evident audit records.
Interfaces are shaped for future RBAC/admin integrations without external auth
dependencies in the local scaffold.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol


def _flag_enabled(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def governance_enabled() -> bool:
    return _flag_enabled("PLODDER_HITL_GOVERNANCE")


def governance_safe_mode_enabled() -> bool:
    return _flag_enabled("PLODDER_SAFE_MODE_EXECUTION")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    OVERRIDDEN = "overridden"


@dataclass(frozen=True, slots=True)
class GovernancePolicy:
    approval_threshold: float = 0.75
    escalation_threshold: float = 0.9
    max_escalations: int = 2
    max_pending_approvals: int = 32
    safe_mode: bool = False
    allow_operator_override: bool = True
    reversible_keywords: tuple[str, ...] = ("dry-run", "read", "inspect", "list", "check")
    risky_keywords: tuple[str, ...] = (
        "delete",
        "drop",
        "truncate",
        "rm ",
        "destroy",
        "kill",
        "production",
        "migration",
        "deploy",
        "secret",
        "credential",
    )


@dataclass(frozen=True, slots=True)
class PolicyClassification:
    strategy_key: str
    risk_score: float
    risk_level: RiskLevel
    requires_approval: bool
    reversible: bool
    policy_tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    request_id: str
    session_id: str
    unit_id: str
    goal: str
    classification: PolicyClassification
    created_at: datetime = field(default_factory=_utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ApprovalResolution:
    status: ApprovalStatus
    actor_id: str
    reason: str
    override: bool = False


@dataclass(slots=True)
class ApprovalRecord:
    record_id: str
    request_id: str
    session_id: str
    unit_id: str
    status: ApprovalStatus
    actor_id: str
    reason: str
    risk_score: float
    risk_level: str
    reversible: bool
    escalation_level: int
    ts: datetime = field(default_factory=_utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    prev_hash: str = ""
    record_hash: str = ""

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "unit_id": self.unit_id,
            "status": self.status.value,
            "actor_id": self.actor_id,
            "reason": self.reason,
            "risk_score": float(self.risk_score),
            "risk_level": self.risk_level,
            "reversible": bool(self.reversible),
            "escalation_level": int(self.escalation_level),
            "ts": self.ts.isoformat(),
            "metadata": dict(self.metadata),
            "prev_hash": self.prev_hash,
        }

    def compute_hash(self) -> str:
        raw = json.dumps(self.canonical_payload(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def finalize(self) -> ApprovalRecord:
        self.record_hash = self.compute_hash()
        return self

    def to_dict(self) -> dict[str, Any]:
        return {**self.canonical_payload(), "record_hash": self.record_hash}

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> ApprovalRecord:
        ts_raw = row.get("ts")
        ts = _utcnow()
        if isinstance(ts_raw, str):
            try:
                ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except ValueError:
                ts = _utcnow()
        return cls(
            record_id=str(row.get("record_id") or uuid.uuid4().hex),
            request_id=str(row.get("request_id") or ""),
            session_id=str(row.get("session_id") or ""),
            unit_id=str(row.get("unit_id") or ""),
            status=ApprovalStatus(str(row.get("status") or ApprovalStatus.PENDING.value)),
            actor_id=str(row.get("actor_id") or ""),
            reason=str(row.get("reason") or ""),
            risk_score=float(row.get("risk_score", 0.0)),
            risk_level=str(row.get("risk_level") or RiskLevel.LOW.value),
            reversible=bool(row.get("reversible")),
            escalation_level=int(row.get("escalation_level", 0)),
            ts=ts,
            metadata=dict(row.get("metadata") or {}),
            prev_hash=str(row.get("prev_hash") or ""),
            record_hash=str(row.get("record_hash") or ""),
        )


class ApprovalAuthorizer(Protocol):
    def can_approve(self, *, actor_id: str, request: ApprovalRequest) -> bool:
        ...


class ApprovalStore:
    """Local append-only approval store with hash-chain integrity markers."""

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace)

    def _path(self) -> Path:
        root = self.workspace.resolve()
        root.mkdir(parents=True, exist_ok=True)
        out = root / ".plodder"
        out.mkdir(parents=True, exist_ok=True)
        return out / "approvals.jsonl"

    def list(self) -> list[ApprovalRecord]:
        path = self._path()
        if not path.is_file():
            return []
        out: list[ApprovalRecord] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    out.append(ApprovalRecord.from_dict(row))
        except (OSError, json.JSONDecodeError):
            return []
        return out

    def append(self, record: ApprovalRecord) -> ApprovalRecord:
        prior = self.list()
        record.prev_hash = prior[-1].record_hash if prior else ""
        record.finalize()
        path = self._path()
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), default=str, ensure_ascii=False) + "\n")
        return record

    def verify_integrity(self) -> bool:
        chain = self.list()
        prev = ""
        for record in chain:
            if record.prev_hash != prev:
                return False
            if record.compute_hash() != record.record_hash:
                return False
            prev = record.record_hash
        return True


@dataclass(frozen=True, slots=True)
class GovernanceDecision:
    allowed: bool
    classification: PolicyClassification
    approval_required: bool
    approval_status: ApprovalStatus | None
    escalation_level: int = 0
    reason: str = ""
    reversible_marker: bool = False
    request_id: str | None = None


OperatorReviewHook = Callable[[ApprovalRequest, int], ApprovalResolution | None]


class HITLGovernor:
    def __init__(
        self,
        workspace: str | Path,
        *,
        policy: GovernancePolicy | None = None,
        approval_store: ApprovalStore | None = None,
        operator_hook: OperatorReviewHook | None = None,
        authorizer: ApprovalAuthorizer | None = None,
    ) -> None:
        base_policy = policy or GovernancePolicy()
        if base_policy.safe_mode is False and governance_safe_mode_enabled():
            base_policy = GovernancePolicy(**{**base_policy.__dict__, "safe_mode": True})
        self.policy = base_policy
        self.store = approval_store or ApprovalStore(workspace)
        self.operator_hook = operator_hook
        self.authorizer = authorizer

    @staticmethod
    def classify_action(
        *,
        goal: str,
        acceptance: list[str] | tuple[str, ...],
        prior_failures: int,
        policy: GovernancePolicy,
    ) -> PolicyClassification:
        text = " ".join([goal, " ".join(acceptance)]).lower()
        keyword_hits = sum(1 for token in policy.risky_keywords if token in text)
        risk = 0.2 + min(0.65, 0.2 * keyword_hits) + min(0.25, 0.08 * max(prior_failures, 0))
        risk = max(0.0, min(1.0, risk))
        if risk >= 0.9:
            level = RiskLevel.CRITICAL
        elif risk >= 0.75:
            level = RiskLevel.HIGH
        elif risk >= 0.45:
            level = RiskLevel.MEDIUM
        else:
            level = RiskLevel.LOW
        reversible = any(token in text for token in policy.reversible_keywords)
        tags: list[str] = []
        if keyword_hits:
            tags.append("risky_keywords")
        if prior_failures > 0:
            tags.append("prior_failures")
        if policy.safe_mode:
            tags.append("safe_mode")
        requires = policy.safe_mode or risk >= policy.approval_threshold or keyword_hits >= 2
        return PolicyClassification(
            strategy_key=goal.strip().lower()[:256],
            risk_score=risk,
            risk_level=level,
            requires_approval=requires,
            reversible=reversible,
            policy_tags=tuple(tags),
        )

    def evaluate(
        self,
        *,
        session_id: str,
        unit_id: str,
        goal: str,
        acceptance: list[str] | tuple[str, ...],
        prior_failures: int,
    ) -> GovernanceDecision:
        classification = self.classify_action(
            goal=goal,
            acceptance=acceptance,
            prior_failures=prior_failures,
            policy=self.policy,
        )
        if not classification.requires_approval:
            return GovernanceDecision(
                allowed=True,
                classification=classification,
                approval_required=False,
                approval_status=None,
                reason="approval_not_required",
                reversible_marker=classification.reversible,
            )

        if len(self.store.list()) >= self.policy.max_pending_approvals:
            return GovernanceDecision(
                allowed=False,
                classification=classification,
                approval_required=True,
                approval_status=ApprovalStatus.ESCALATED,
                escalation_level=self.policy.max_escalations,
                reason="approval_queue_bounded",
                reversible_marker=classification.reversible,
            )

        request = ApprovalRequest(
            request_id=f"apr-{uuid.uuid4().hex[:10]}",
            session_id=session_id,
            unit_id=unit_id,
            goal=goal,
            classification=classification,
            metadata={"acceptance": list(acceptance), "prior_failures": int(prior_failures)},
        )

        level = 0
        resolution: ApprovalResolution | None = None
        while level <= self.policy.max_escalations:
            resolution = self.operator_hook(request, level) if self.operator_hook is not None else None
            if resolution is not None:
                if self.authorizer is not None and not self.authorizer.can_approve(actor_id=resolution.actor_id, request=request):
                    resolution = ApprovalResolution(status=ApprovalStatus.REJECTED, actor_id=resolution.actor_id, reason="unauthorized")
                break
            level += 1

        if resolution is None:
            self.store.append(
                ApprovalRecord(
                    record_id=uuid.uuid4().hex,
                    request_id=request.request_id,
                    session_id=session_id,
                    unit_id=unit_id,
                    status=ApprovalStatus.ESCALATED,
                    actor_id="system",
                    reason="no_operator_decision",
                    risk_score=classification.risk_score,
                    risk_level=classification.risk_level.value,
                    reversible=classification.reversible,
                    escalation_level=self.policy.max_escalations,
                    metadata={"goal": goal, "policy_tags": list(classification.policy_tags)},
                )
            )
            return GovernanceDecision(
                allowed=False,
                classification=classification,
                approval_required=True,
                approval_status=ApprovalStatus.ESCALATED,
                escalation_level=self.policy.max_escalations,
                reason="approval_unresolved",
                reversible_marker=classification.reversible,
                request_id=request.request_id,
            )

        status = resolution.status
        if resolution.override and self.policy.allow_operator_override and status == ApprovalStatus.APPROVED:
            status = ApprovalStatus.OVERRIDDEN

        self.store.append(
            ApprovalRecord(
                record_id=uuid.uuid4().hex,
                request_id=request.request_id,
                session_id=session_id,
                unit_id=unit_id,
                status=status,
                actor_id=resolution.actor_id,
                reason=resolution.reason,
                risk_score=classification.risk_score,
                risk_level=classification.risk_level.value,
                reversible=classification.reversible,
                escalation_level=level,
                metadata={"goal": goal, "policy_tags": list(classification.policy_tags)},
            )
        )
        allowed = status in (ApprovalStatus.APPROVED, ApprovalStatus.OVERRIDDEN)
        return GovernanceDecision(
            allowed=allowed,
            classification=classification,
            approval_required=True,
            approval_status=status,
            escalation_level=level,
            reason=resolution.reason,
            reversible_marker=classification.reversible,
            request_id=request.request_id,
        )
