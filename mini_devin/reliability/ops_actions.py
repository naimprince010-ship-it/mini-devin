from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field, field_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw))
    except Exception:
        return None


def _bounded_int(raw: str | None, default: int, *, low: int, high: int) -> int:
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(low, min(high, value))


def generate_action_id(now: datetime | None = None) -> str:
    current = now or _utcnow()
    return f"act-{current.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:10]}"


class OperatorActionType(str, Enum):
    ACKNOWLEDGE_INCIDENT = "acknowledge_incident"
    MARK_INVESTIGATION_STARTED = "mark_investigation_started"
    PAUSE_RUNTIME = "pause_runtime"
    RESUME_RUNTIME = "resume_runtime"
    RETRY_TASK = "retry_task"
    REPLAY_SESSION = "replay_session"
    QUARANTINE_RUNTIME = "quarantine_runtime"
    JUMP_TO_DIAGNOSTICS = "jump_to_diagnostics"


class OperatorActionDecision(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class OperatorActionStatus(str, Enum):
    RECEIVED = "received"
    POLICY_REJECTED = "policy_rejected"
    ACCEPTED = "accepted"
    DRY_RUN_COMPLETED = "dry_run_completed"


class OperatorActionRequest(BaseModel):
    action_type: OperatorActionType
    target: str = "runtime.main"
    operator_id: str = "operator.local"
    reason: str
    confirmation_token: str = Field(default="")
    dry_run: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("target", "operator_id", "reason", "confirmation_token")
    @classmethod
    def _strip(cls, value: str) -> str:
        return str(value).strip()


class OperatorActionResponse(BaseModel):
    schema_name: str = Field(alias="schema")
    action_id: str
    decision: OperatorActionDecision
    accepted: bool
    status: OperatorActionStatus
    dry_run: bool
    requested_at: str
    policy: dict[str, Any] = Field(default_factory=dict)
    lifecycle: list[dict[str, Any]] = Field(default_factory=list)
    audit: dict[str, Any] = Field(default_factory=dict)
    runtime_hook: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OperatorActionPolicyConfig:
    enabled: bool = True
    enforce_dry_run: bool = True
    min_reason_chars: int = 8
    required_confirmation_token: str = "APPROVE"
    allowed_actions: set[str] = field(default_factory=set)


def policy_config_from_env() -> OperatorActionPolicyConfig:
    enabled = (os.getenv("PLODDER_OPS_ACTIONS_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"})
    enforce_dry_run = (os.getenv("PLODDER_OPS_ACTIONS_ENFORCE_DRY_RUN", "true").strip().lower() in {"1", "true", "yes", "on"})
    min_reason_chars = _bounded_int(
        os.getenv("PLODDER_OPS_ACTIONS_MIN_REASON_CHARS"),
        default=8,
        low=1,
        high=500,
    )
    required_token = (os.getenv("PLODDER_OPS_ACTIONS_CONFIRM_TOKEN", "APPROVE") or "APPROVE").strip()
    allow_raw = (os.getenv("PLODDER_OPS_ACTIONS_ALLOWLIST") or "").strip()
    if allow_raw:
        allow = {item.strip().lower() for item in allow_raw.split(",") if item.strip()}
    else:
        allow = {item.value for item in OperatorActionType}
    return OperatorActionPolicyConfig(
        enabled=enabled,
        enforce_dry_run=enforce_dry_run,
        min_reason_chars=min_reason_chars,
        required_confirmation_token=required_token,
        allowed_actions=allow,
    )


@dataclass(frozen=True, slots=True)
class OperatorActionStoreConfig:
    retention_hours: int = 24 * 30
    max_events: int = 200_000


def store_config_from_env() -> OperatorActionStoreConfig:
    return OperatorActionStoreConfig(
        retention_hours=_bounded_int(
            os.getenv("PLODDER_OPS_ACTIONS_RETENTION_HOURS"),
            default=24 * 30,
            low=1,
            high=24 * 180,
        ),
        max_events=_bounded_int(
            os.getenv("PLODDER_OPS_ACTIONS_MAX_EVENTS"),
            default=200_000,
            low=1_000,
            high=2_000_000,
        ),
    )


class RuntimeControlHookPlanner(Protocol):
    def build_hook(self, request: OperatorActionRequest) -> dict[str, Any]:
        ...


class NoopRuntimeControlHookPlanner:
    def build_hook(self, request: OperatorActionRequest) -> dict[str, Any]:
        return {
            "ready": False,
            "mutating": False,
            "executor": "noop",
            "message": "Runtime control integration hook is scaffolded only.",
            "next_step": "Wire a policy-authorized runtime executor in a future slice.",
            "action_type": request.action_type.value,
        }


class FileOperatorActionIntake:
    """File-backed operator action intake with policy-gated dry-run-only lifecycle."""

    schema_version = "ops.action.v1"

    def __init__(
        self,
        *,
        events_file: Path,
        state_file: Path,
        policy: OperatorActionPolicyConfig,
        store: OperatorActionStoreConfig,
        runtime_hook_planner: RuntimeControlHookPlanner | None = None,
    ) -> None:
        self.events_file = events_file
        self.state_file = state_file
        self.policy = policy
        self.store = store
        self.runtime_hook_planner = runtime_hook_planner or NoopRuntimeControlHookPlanner()

        self.events_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        if not self.state_file.exists():
            return {"append_count": 0}
        try:
            payload = json.loads(self.state_file.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return {
                    "append_count": int(payload.get("append_count") or 0),
                }
        except Exception:
            pass
        return {"append_count": 0}

    def _save_state(self) -> None:
        self.state_file.write_text(json.dumps(self.state, ensure_ascii=True), encoding="utf-8")

    def _append_record(self, row: dict[str, Any], *, now: datetime) -> None:
        with self.events_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        self.state["append_count"] = int(self.state.get("append_count") or 0) + 1
        if int(self.state["append_count"]) % 25 == 0:
            self._enforce_retention(now=now)
        self._save_state()

    def _enforce_retention(self, *, now: datetime | None = None) -> None:
        if not self.events_file.exists():
            return
        current = now or _utcnow()
        cutoff = current - timedelta(hours=self.store.retention_hours)
        kept: list[str] = []
        with self.events_file.open("r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                ts = _parse_iso(str(row.get("requested_at") or ""))
                if ts is None or ts < cutoff:
                    continue
                kept.append(json.dumps(row, ensure_ascii=True))
        if len(kept) > self.store.max_events:
            kept = kept[-self.store.max_events :]
        self.events_file.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

    def _validate_policy(self, request: OperatorActionRequest) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        if not self.policy.enabled:
            issues.append(
                {
                    "code": "policy.disabled",
                    "message": "Operator action intake is disabled by policy.",
                    "component": "ops_action_policy",
                }
            )
        if request.action_type.value.lower() not in self.policy.allowed_actions:
            issues.append(
                {
                    "code": "policy.action_not_allowed",
                    "message": f"Action '{request.action_type.value}' is not allowlisted.",
                    "component": "ops_action_policy",
                }
            )
        if self.policy.enforce_dry_run and not request.dry_run:
            issues.append(
                {
                    "code": "policy.dry_run_required",
                    "message": "Dry-run is required; mutating actions are blocked in this slice.",
                    "component": "ops_action_policy",
                }
            )
        if len(request.reason.strip()) < self.policy.min_reason_chars:
            issues.append(
                {
                    "code": "policy.reason_too_short",
                    "message": f"Reason must be at least {self.policy.min_reason_chars} characters.",
                    "component": "ops_action_policy",
                }
            )
        expected = self.policy.required_confirmation_token.strip().upper()
        provided = request.confirmation_token.strip().upper()
        if expected and provided != expected:
            issues.append(
                {
                    "code": "policy.confirmation_token_mismatch",
                    "message": "Confirmation token mismatch.",
                    "component": "ops_action_policy",
                }
            )
        return issues

    def intake(self, request: OperatorActionRequest, *, now: datetime | None = None) -> OperatorActionResponse:
        current = now or _utcnow()
        action_id = generate_action_id(current)

        audit = {
            "action_id": action_id,
            "action_type": request.action_type.value,
            "target": request.target,
            "operator_id": request.operator_id,
            "reason": request.reason,
            "dry_run": bool(request.dry_run),
            "metadata": dict(request.metadata),
            "confirmation_gate": "explicit",
            "requested_at": current.isoformat(),
        }
        lifecycle: list[dict[str, Any]] = [
            {
                "status": OperatorActionStatus.RECEIVED.value,
                "time": current.isoformat(),
                "message": "Action request received.",
            }
        ]

        policy_issues = self._validate_policy(request)
        runtime_hook = self.runtime_hook_planner.build_hook(request)

        if policy_issues:
            lifecycle.append(
                {
                    "status": OperatorActionStatus.POLICY_REJECTED.value,
                    "time": current.isoformat(),
                    "message": "Action rejected by policy gates.",
                }
            )
            response = OperatorActionResponse(
                schema=self.schema_version,
                action_id=action_id,
                decision=OperatorActionDecision.REJECTED,
                accepted=False,
                status=OperatorActionStatus.POLICY_REJECTED,
                dry_run=bool(request.dry_run),
                requested_at=current.isoformat(),
                policy={"ok": False, "issues": policy_issues},
                lifecycle=lifecycle,
                audit=audit,
                runtime_hook=runtime_hook,
            )
        else:
            lifecycle.append(
                {
                    "status": OperatorActionStatus.ACCEPTED.value,
                    "time": current.isoformat(),
                    "message": "Action accepted for dry-run intake.",
                }
            )
            lifecycle.append(
                {
                    "status": OperatorActionStatus.DRY_RUN_COMPLETED.value,
                    "time": current.isoformat(),
                    "message": "Dry-run execution completed; no runtime mutation performed.",
                }
            )
            response = OperatorActionResponse(
                schema=self.schema_version,
                action_id=action_id,
                decision=OperatorActionDecision.ACCEPTED,
                accepted=True,
                status=OperatorActionStatus.DRY_RUN_COMPLETED,
                dry_run=bool(request.dry_run),
                requested_at=current.isoformat(),
                policy={"ok": True, "issues": []},
                lifecycle=lifecycle,
                audit=audit,
                runtime_hook=runtime_hook,
            )

        row = response.model_dump(mode="json", by_alias=True)
        self._append_record(row, now=current)
        return response

    def list_actions(
        self,
        *,
        hours: int = 24,
        limit: int = 100,
        decision: str | None = None,
        action_type: str | None = None,
        status: str | None = None,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        if not self.events_file.exists():
            return []
        current = now or _utcnow()
        cutoff = current - timedelta(hours=max(1, int(hours)))
        safe_limit = max(1, min(int(limit), 500))

        decision_filter = (decision or "").strip().lower() or None
        action_filter = (action_type or "").strip().lower() or None
        status_filter = (status or "").strip().lower() or None

        rows: list[dict[str, Any]] = []
        with self.events_file.open("r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                ts = _parse_iso(str(row.get("requested_at") or ""))
                if ts is None or ts < cutoff:
                    continue
                row_decision = str(row.get("decision") or "").lower()
                row_status = str(row.get("status") or "").lower()
                audit = row.get("audit") if isinstance(row.get("audit"), dict) else {}
                row_action = str(audit.get("action_type") or "").lower()
                if decision_filter and row_decision != decision_filter:
                    continue
                if action_filter and row_action != action_filter:
                    continue
                if status_filter and row_status != status_filter:
                    continue
                rows.append(row)
        rows.sort(key=lambda item: str(item.get("requested_at") or ""), reverse=True)
        return rows[:safe_limit]

    def get_action(self, action_id: str) -> dict[str, Any] | None:
        target = (action_id or "").strip()
        if not target:
            return None
        for row in self.list_actions(hours=self.store.retention_hours, limit=500):
            if str(row.get("action_id") or "") == target:
                return row
        return None
