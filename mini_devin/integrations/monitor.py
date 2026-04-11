"""
Self-Healing CI/CD Monitor

Polls deployed app health endpoints and cloud logs.
On crash detection it triggers the agent to fix the code and redeploys.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import aiohttp
import logging

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    url: str
    status: HealthStatus
    status_code: Optional[int]
    response_time_ms: float
    error: Optional[str]
    checked_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class MonitoredApp:
    name: str
    health_url: str
    logs_url: Optional[str] = None          # Cloud provider log endpoint
    platform: str = "generic"               # digitalocean | railway | docker | generic
    platform_config: Dict[str, Any] = field(default_factory=dict)
    check_interval_seconds: int = 60
    failure_threshold: int = 3              # Consecutive failures before auto-heal
    session_id: Optional[str] = None        # Agent session to use for healing
    enabled: bool = True
    # Internal state
    consecutive_failures: int = 0
    last_heal_at: Optional[str] = None
    last_status: HealthStatus = HealthStatus.UNKNOWN


# Singleton monitoring state
_apps: Dict[str, MonitoredApp] = {}
_monitor_task: Optional[asyncio.Task] = None
_heal_callbacks: List[Callable] = []    # registered by the agent/session manager
_history: List[Dict[str, Any]] = []
_MAX_HISTORY = 200


# ---------------------------------------------------------------------------
# Registration helpers
# ---------------------------------------------------------------------------

def register_app(app: MonitoredApp) -> None:
    _apps[app.name] = app
    logger.info(f"[monitor] Registered app: {app.name} → {app.health_url}")


def unregister_app(name: str) -> bool:
    return bool(_apps.pop(name, None))


def register_heal_callback(cb: Callable) -> None:
    """Called as cb(app_name, logs_excerpt) when a crash is detected."""
    _heal_callbacks.append(cb)


def get_status() -> Dict[str, Any]:
    return {
        "running": _monitor_task is not None and not _monitor_task.done(),
        "apps": {
            name: {
                "name": app.name,
                "health_url": app.health_url,
                "platform": app.platform,
                "status": app.last_status.value,
                "consecutive_failures": app.consecutive_failures,
                "last_heal_at": app.last_heal_at,
                "check_interval_seconds": app.check_interval_seconds,
                "enabled": app.enabled,
            }
            for name, app in _apps.items()
        },
        "history": _history[-50:],
    }


# ---------------------------------------------------------------------------
# Health checking
# ---------------------------------------------------------------------------

async def _check_health(app: MonitoredApp, session: aiohttp.ClientSession) -> HealthCheckResult:
    start = time.monotonic()
    try:
        async with session.get(app.health_url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            elapsed = (time.monotonic() - start) * 1000
            if resp.status < 400:
                return HealthCheckResult(
                    url=app.health_url,
                    status=HealthStatus.HEALTHY,
                    status_code=resp.status,
                    response_time_ms=elapsed,
                    error=None,
                )
            else:
                return HealthCheckResult(
                    url=app.health_url,
                    status=HealthStatus.DEGRADED,
                    status_code=resp.status,
                    response_time_ms=elapsed,
                    error=f"HTTP {resp.status}",
                )
    except Exception as exc:
        elapsed = (time.monotonic() - start) * 1000
        return HealthCheckResult(
            url=app.health_url,
            status=HealthStatus.DOWN,
            status_code=None,
            response_time_ms=elapsed,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Log readers
# ---------------------------------------------------------------------------

async def _fetch_logs(app: MonitoredApp, session: aiohttp.ClientSession, lines: int = 50) -> str:
    """Fetch recent logs from the platform. Returns plain text."""
    platform = app.platform.lower()

    try:
        if platform == "digitalocean":
            return await _fetch_do_logs(app, session, lines)
        elif platform == "docker":
            return await _fetch_docker_logs(app, lines)
        elif platform == "railway":
            return await _fetch_railway_logs(app, session, lines)
        elif app.logs_url:
            async with session.get(app.logs_url, timeout=aiohttp.ClientTimeout(total=20)) as r:
                return (await r.text())[-4000:]
    except Exception as exc:
        return f"[log fetch error: {exc}]"

    return ""


async def _fetch_do_logs(app: MonitoredApp, session: aiohttp.ClientSession, lines: int) -> str:
    """DigitalOcean App Platform log API."""
    token = app.platform_config.get("do_token") or os.getenv("DO_API_TOKEN", "")
    app_id = app.platform_config.get("app_id") or os.getenv("DO_APP_ID", "")
    if not token or not app_id:
        return "[DigitalOcean: DO_API_TOKEN or DO_APP_ID not configured]"

    url = f"https://api.digitalocean.com/v2/apps/{app_id}/logs"
    params = {"type": "RUN", "tail_lines": lines}
    headers = {"Authorization": f"Bearer {token}"}
    try:
        async with session.get(url, params=params, headers=headers,
                               timeout=aiohttp.ClientTimeout(total=20)) as r:
            data = await r.json()
            # DO returns a live_url for streaming; fall back to historic_urls
            historic = data.get("historic_urls", [])
            if historic:
                async with session.get(historic[0], timeout=aiohttp.ClientTimeout(total=15)) as lr:
                    return (await lr.text())[-4000:]
            return json.dumps(data)
    except Exception as exc:
        return f"[DO log error: {exc}]"


async def _fetch_docker_logs(app: MonitoredApp, lines: int) -> str:
    """Read docker logs via subprocess."""
    container = app.platform_config.get("container_name", app.name)
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "logs", "--tail", str(lines), container,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        return out.decode(errors="replace")
    except Exception as exc:
        return f"[docker logs error: {exc}]"


async def _fetch_railway_logs(app: MonitoredApp, session: aiohttp.ClientSession, lines: int) -> str:
    """Railway GraphQL API for deployment logs."""
    token = app.platform_config.get("railway_token") or os.getenv("RAILWAY_TOKEN", "")
    service_id = app.platform_config.get("service_id") or os.getenv("RAILWAY_SERVICE_ID", "")
    if not token or not service_id:
        return "[Railway: RAILWAY_TOKEN or RAILWAY_SERVICE_ID not configured]"

    query = """
    query Logs($serviceId: String!) {
      serviceDeploymentLogs(serviceId: $serviceId, limit: 50) { message timestamp }
    }"""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"query": query, "variables": {"serviceId": service_id}}
    try:
        async with session.post("https://backboard.railway.app/graphql/v2",
                                json=payload, headers=headers,
                                timeout=aiohttp.ClientTimeout(total=20)) as r:
            data = await r.json()
            logs = data.get("data", {}).get("serviceDeploymentLogs", [])
            return "\n".join(f"[{l['timestamp']}] {l['message']}" for l in logs[-lines:])
    except Exception as exc:
        return f"[Railway log error: {exc}]"


# ---------------------------------------------------------------------------
# Error pattern detection
# ---------------------------------------------------------------------------

CRASH_PATTERNS = [
    "Traceback (most recent call last)",
    "CRITICAL",
    "FATAL",
    "OOMKilled",
    "signal: killed",
    "exit code 1",
    "exit status 1",
    "Application crashed",
    "Error: Cannot find module",
    "SyntaxError:",
    "ImportError:",
    "ModuleNotFoundError:",
    "RuntimeError:",
    "Segmentation fault",
    "killed",
    "died",
    "unhealthy",
]


def _detect_crash_in_logs(logs: str) -> Optional[str]:
    """Returns a short excerpt around the first crash pattern found, or None."""
    lines = logs.splitlines()
    for i, line in enumerate(lines):
        for pattern in CRASH_PATTERNS:
            if pattern.lower() in line.lower():
                start = max(0, i - 5)
                end = min(len(lines), i + 15)
                excerpt = "\n".join(lines[start:end])
                return excerpt
    return None


# ---------------------------------------------------------------------------
# Auto-heal trigger
# ---------------------------------------------------------------------------

async def _trigger_heal(app: MonitoredApp, logs_excerpt: str) -> None:
    """Fire registered heal callbacks (agent auto-fix)."""
    app.last_heal_at = datetime.now(timezone.utc).isoformat()
    app.consecutive_failures = 0

    event = {
        "type": "heal_triggered",
        "app": app.name,
        "at": app.last_heal_at,
        "logs_excerpt": logs_excerpt[:800],
    }
    _history.append(event)
    if len(_history) > _MAX_HISTORY:
        _history.pop(0)

    logger.warning(f"[monitor] Auto-heal triggered for {app.name}")

    for cb in _heal_callbacks:
        try:
            if asyncio.iscoroutinefunction(cb):
                await cb(app.name, logs_excerpt, app.session_id)
            else:
                cb(app.name, logs_excerpt, app.session_id)
        except Exception as exc:
            logger.error(f"[monitor] heal callback error: {exc}")


# ---------------------------------------------------------------------------
# Main poll loop
# ---------------------------------------------------------------------------

async def _poll_loop() -> None:
    logger.info("[monitor] Poll loop started")
    connector = aiohttp.TCPConnector(ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        while True:
            for app in list(_apps.values()):
                if not app.enabled:
                    continue
                try:
                    result = await _check_health(app, session)
                    app.last_status = result.status

                    event: Dict[str, Any] = {
                        "type": "health_check",
                        "app": app.name,
                        "status": result.status.value,
                        "status_code": result.status_code,
                        "response_ms": round(result.response_time_ms),
                        "error": result.error,
                        "at": result.checked_at,
                    }
                    _history.append(event)
                    if len(_history) > _MAX_HISTORY:
                        _history.pop(0)

                    if result.status == HealthStatus.DOWN:
                        app.consecutive_failures += 1
                        logger.warning(
                            f"[monitor] {app.name} DOWN (failure #{app.consecutive_failures})"
                        )
                        if app.consecutive_failures >= app.failure_threshold:
                            logs = await _fetch_logs(app, session)
                            excerpt = _detect_crash_in_logs(logs) or logs[-600:]
                            await _trigger_heal(app, excerpt)
                    else:
                        app.consecutive_failures = 0

                except Exception as exc:
                    logger.error(f"[monitor] Error checking {app.name}: {exc}")

            await asyncio.sleep(min(app.check_interval_seconds for app in _apps.values()) if _apps else 60)


def start_monitor() -> None:
    global _monitor_task
    if _monitor_task and not _monitor_task.done():
        return
    loop = asyncio.get_event_loop()
    _monitor_task = loop.create_task(_poll_loop())
    logger.info("[monitor] Monitor started")


def stop_monitor() -> None:
    global _monitor_task
    if _monitor_task:
        _monitor_task.cancel()
        _monitor_task = None
    logger.info("[monitor] Monitor stopped")


# ---------------------------------------------------------------------------
# Quick one-shot helpers (for agent tool use)
# ---------------------------------------------------------------------------

async def check_app_health(url: str) -> Dict[str, Any]:
    """One-shot health check — returns dict."""
    connector = aiohttp.TCPConnector(ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        dummy = MonitoredApp(name="onetime", health_url=url)
        result = await _check_health(dummy, session)
        return {
            "status": result.status.value,
            "status_code": result.status_code,
            "response_time_ms": round(result.response_time_ms),
            "error": result.error,
        }


async def fetch_app_logs(platform: str, config: Dict[str, Any], lines: int = 50) -> str:
    """One-shot log fetch — returns log text."""
    dummy = MonitoredApp(
        name="onetime",
        health_url="",
        platform=platform,
        platform_config=config,
    )
    connector = aiohttp.TCPConnector(ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        return await _fetch_logs(dummy, session, lines)
