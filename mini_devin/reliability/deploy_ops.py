from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .startup_guard import run_startup_preflight


@dataclass(frozen=True, slots=True)
class DeployValidationIssue:
    code: str
    level: str
    message: str
    component: str
    hint: str | None = None


@dataclass(slots=True)
class RollbackGuardReport:
    safe_to_deploy: bool
    checks: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "safe_to_deploy": self.safe_to_deploy,
            "checks": list(self.checks),
        }


@dataclass(slots=True)
class DeployPreflightReport:
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    mode: str = "local"
    issues: list[DeployValidationIssue] = field(default_factory=list)
    dependency_graph: dict[str, list[str]] = field(default_factory=dict)
    startup_sequence_valid: bool = True
    rollback: RollbackGuardReport = field(default_factory=lambda: RollbackGuardReport(safe_to_deploy=True))

    @property
    def has_errors(self) -> bool:
        return any(item.level == "error" for item in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(item.level == "warning" for item in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "mode": self.mode,
            "issues": [
                {
                    "code": item.code,
                    "level": item.level,
                    "message": item.message,
                    "component": item.component,
                    "hint": item.hint,
                }
                for item in self.issues
            ],
            "dependency_graph": dict(self.dependency_graph),
            "startup_sequence_valid": self.startup_sequence_valid,
            "rollback": self.rollback.to_dict(),
            "status": "failed" if self.has_errors else ("degraded" if self.has_warnings else "ok"),
        }


def _flag_enabled(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def startup_stage_order() -> list[str]:
    return [
        "boot.begin",
        "preflight.complete",
        "db.init.start",
        "db.init.complete",
        "session.cleanup.complete",
        "monitor.registration.complete",
        "ready",
    ]


def validate_startup_sequence(stage_history: list[str]) -> list[DeployValidationIssue]:
    if not stage_history:
        return []

    issues: list[DeployValidationIssue] = []
    order = startup_stage_order()
    index_map = {name: idx for idx, name in enumerate(order)}
    last = -1
    for name in stage_history:
        if name not in index_map:
            issues.append(
                DeployValidationIssue(
                    code="startup.sequence.unknown_stage",
                    level="warning",
                    message=f"Unknown startup stage: {name}",
                    component="startup_sequencing",
                    hint="Keep startup stage names aligned with startup_stage_order().",
                )
            )
            continue
        idx = index_map[name]
        if idx < last:
            issues.append(
                DeployValidationIssue(
                    code="startup.sequence.order_violation",
                    level="error",
                    message=f"Stage '{name}' appeared out of order in startup history.",
                    component="startup_sequencing",
                    hint="Verify startup hooks and background task lifecycle ordering.",
                )
            )
            break
        last = idx
    return issues


def build_runtime_dependency_graph(queue_backend: str | None = None) -> dict[str, list[str]]:
    backend = (queue_backend or os.environ.get("PLODDER_QUEUE_BACKEND") or "memory").strip().lower()
    graph: dict[str, list[str]] = {
        "api": ["startup_preflight", "database", "session_manager"],
        "session_manager": ["database"],
        "startup_preflight": [],
        "database": [],
        "monitor": ["session_manager"],
    }
    if backend == "redis_streams":
        graph["queue_transport"] = ["redis"]
        graph["api"].append("queue_transport")
        graph["redis"] = []
    else:
        graph["queue_transport"] = []
        graph["api"].append("queue_transport")
    return graph


def validate_dependency_graph(graph: dict[str, list[str]]) -> list[DeployValidationIssue]:
    issues: list[DeployValidationIssue] = []

    for node, deps in graph.items():
        for dep in deps:
            if dep not in graph:
                issues.append(
                    DeployValidationIssue(
                        code="dependency_graph.missing_node",
                        level="error",
                        message=f"Node '{node}' depends on missing node '{dep}'.",
                        component="dependency_graph",
                        hint="Ensure all runtime dependencies are explicitly declared.",
                    )
                )

    visiting: set[str] = set()
    visited: set[str] = set()

    def _visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for dep in graph.get(node, []):
            if dep in graph and _visit(dep):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    for node in graph:
        if _visit(node):
            issues.append(
                DeployValidationIssue(
                    code="dependency_graph.cycle_detected",
                    level="error",
                    message=f"Cycle detected around node '{node}'.",
                    component="dependency_graph",
                    hint="Break import/runtime cycles by lazy-loading or interface boundaries.",
                )
            )
            break

    return issues


def evaluate_rollback_guard(repo_root: str | Path) -> RollbackGuardReport:
    root = Path(repo_root)
    checks: list[dict[str, Any]] = []
    enabled = _flag_enabled("PLODDER_ROLLBACK_GUARDS", False)

    if not enabled:
        checks.append(
            {
                "name": "rollback_guards_enabled",
                "ok": True,
                "message": "Rollback guard checks disabled by feature flag.",
            }
        )
        return RollbackGuardReport(safe_to_deploy=True, checks=checks)

    git_ok = shutil.which("git") is not None and (root / ".git").exists()
    checks.append(
        {
            "name": "git_checkpoint_capable",
            "ok": git_ok,
            "message": "git available and repository initialized." if git_ok else "git or .git metadata missing.",
        }
    )

    require_marker = _flag_enabled("PLODDER_REQUIRE_ROLLBACK_MARKER", False)
    marker = root / ".plodder" / "ops" / "rollback_marker.json"
    marker_ok = marker.exists()
    if require_marker:
        checks.append(
            {
                "name": "rollback_marker_present",
                "ok": marker_ok,
                "message": "rollback marker exists" if marker_ok else "rollback marker missing",
            }
        )

    safe = all(bool(item.get("ok")) for item in checks)
    return RollbackGuardReport(safe_to_deploy=safe, checks=checks)


def build_operational_runbook_scaffold() -> dict[str, Any]:
    return {
        "title": "Plodder V2 Operational Runbook (Scaffold)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sections": [
            "Pre-deploy preflight checklist",
            "Startup sequencing and readiness checks",
            "Rollback decision matrix and guard conditions",
            "Degraded-mode operation policy",
            "Crash-loop containment workflow",
            "Incident recovery verification checklist",
            "Future container orchestrator integration points",
        ],
    }


def run_deploy_preflight(
    *,
    repo_root: str | Path,
    startup_stage_history: list[str] | None = None,
    mode: str | None = None,
) -> DeployPreflightReport:
    report = DeployPreflightReport(mode=mode or (os.environ.get("PLODDER_DEPLOY_MODE") or "local").strip() or "local")

    startup_report = run_startup_preflight()
    for item in startup_report.checks:
        report.issues.append(
            DeployValidationIssue(
                code=f"startup_preflight.{item.code}",
                level=item.level,
                message=item.message,
                component="startup_preflight",
                hint="Review startup preflight diagnostics before deploy.",
            )
        )

    history = list(startup_stage_history or [])
    seq_issues = validate_startup_sequence(history)
    report.issues.extend(seq_issues)
    report.startup_sequence_valid = not any(i.level == "error" for i in seq_issues)

    graph = build_runtime_dependency_graph()
    report.dependency_graph = graph
    report.issues.extend(validate_dependency_graph(graph))

    queue_policy = (os.environ.get("PLODDER_QUEUE_FAILOVER_POLICY") or "legacy").strip().lower()
    if queue_policy not in {"legacy", "safe_memory", "strict"}:
        report.issues.append(
            DeployValidationIssue(
                code="config.invalid.queue_failover_policy",
                level="error",
                message=f"Unsupported PLODDER_QUEUE_FAILOVER_POLICY value: {queue_policy}",
                component="deploy_config",
                hint="Use one of: legacy, safe_memory, strict.",
            )
        )

    report.rollback = evaluate_rollback_guard(repo_root)
    if not report.rollback.safe_to_deploy:
        report.issues.append(
            DeployValidationIssue(
                code="rollback.guard.blocked",
                level="error",
                message="Rollback guard checks failed.",
                component="rollback_safety",
                hint="Fix rollback guard checks before promotion.",
            )
        )

    return report
