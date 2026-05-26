from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True, slots=True)
class StartupIssue:
    code: str
    level: str
    message: str


@dataclass(slots=True)
class StartupPreflightReport:
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    checks: list[StartupIssue] = field(default_factory=list)
    strict_mode: bool = False

    @property
    def has_errors(self) -> bool:
        return any(item.level == "error" for item in self.checks)

    @property
    def has_warnings(self) -> bool:
        return any(item.level == "warning" for item in self.checks)

    @property
    def startup_mode(self) -> str:
        if self.has_errors or self.has_warnings:
            return "degraded"
        return "normal"

    @property
    def should_fail_fast(self) -> bool:
        return self.strict_mode and self.has_errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "strict_mode": self.strict_mode,
            "startup_mode": self.startup_mode,
            "checks": [
                {
                    "code": item.code,
                    "level": item.level,
                    "message": item.message,
                }
                for item in self.checks
            ],
        }


def _flag_enabled(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _check_numeric_env(
    report: StartupPreflightReport,
    *,
    name: str,
    minimum: int,
    maximum: int,
    default: int,
) -> None:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return
    try:
        value = int(raw)
    except ValueError:
        report.checks.append(
            StartupIssue(
                code=f"config.invalid.{name.lower()}",
                level="error",
                message=f"{name} must be an integer; got '{raw}'.",
            )
        )
        return

    if value < minimum or value > maximum:
        report.checks.append(
            StartupIssue(
                code=f"config.out_of_range.{name.lower()}",
                level="warning",
                message=(
                    f"{name}={value} is outside safe range [{minimum}, {maximum}]. "
                    f"Runtime will use bounded defaults near {default}."
                ),
            )
        )


def _check_import(report: StartupPreflightReport, module_name: str, *, required: bool) -> None:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        report.checks.append(
            StartupIssue(
                code=f"import.failed.{module_name}",
                level="error" if required else "warning",
                message=f"Module import failed for {module_name}: {exc}",
            )
        )


def run_startup_preflight() -> StartupPreflightReport:
    report = StartupPreflightReport(strict_mode=_flag_enabled("PLODDER_STRICT_STARTUP", False))

    _check_numeric_env(report, name="DATABASE_INIT_TIMEOUT", minimum=5, maximum=900, default=120)
    _check_numeric_env(report, name="CREATE_SESSION_DB_WAIT_SEC", minimum=5, maximum=900, default=120)
    _check_numeric_env(report, name="PLODDER_QUEUE_LEASE_SECONDS", minimum=1, maximum=600, default=45)
    _check_numeric_env(report, name="PLODDER_QUEUE_HEARTBEAT_SECONDS", minimum=1, maximum=300, default=10)
    _check_numeric_env(report, name="PLODDER_QUEUE_MAX_ATTEMPTS", minimum=1, maximum=20, default=3)

    _check_import(report, "mini_devin.orchestration.runtime_contracts", required=True)
    _check_import(report, "mini_devin.orchestrator.agent", required=True)
    _check_import(report, "mini_devin.orchestration.autonomous_coordination", required=True)

    backend = (os.environ.get("PLODDER_QUEUE_BACKEND") or "memory").strip().lower()
    if backend == "redis_streams":
        _check_import(report, "redis", required=False)

    if _flag_enabled("PLODDER_REQUIRE_SANDBOX", False):
        _check_import(report, "mini_devin.sandbox.factory", required=True)

    return report