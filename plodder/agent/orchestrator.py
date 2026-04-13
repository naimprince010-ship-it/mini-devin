"""
Plodder session orchestration — failure streaks, diagnostic context, post-mortem,
and an OpenHands-style **worklog event stream** (action–observation pairs).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from mini_devin.reliability.post_mortem import (
    DiagnosticTriggerRecord,
    FailureStreakRecord,
    PostMortemPayload,
    RecoveryPathRecord,
    format_post_mortem,
    infer_recovery_path_summary,
)
from mini_devin.reliability.self_correction import (
    ErrorType,
    error_fingerprint,
    incremental_recovery_hint,
)


def _observation_text(result: dict[str, Any]) -> str:
    if not result.get("ok", True):
        return str(
            result.get("error")
            or result.get("stderr")
            or result.get("stdout")
            or result.get("message")
            or ""
        )
    return str(result.get("stderr") or result.get("stdout") or "")[:800]


def _exit_code(result: dict[str, Any]) -> int | None:
    raw = result.get("exit_code")
    return int(raw) if isinstance(raw, int) else None


def recovery_hint_for_failed_tool(tool_name: str, result: dict[str, Any]) -> str:
    """Incremental recovery hint for a single failed observation (no streak required)."""
    obs = _observation_text(result)
    ec = _exit_code(result)
    argv = result.get("argv")
    if isinstance(argv, list):
        cmd = " ".join(str(x) for x in argv)
    else:
        cmd = str(result.get("command") or "")
    args = {"command": cmd} if cmd else {}
    hint_tool = "terminal" if tool_name in ("sandbox_run", "sandbox_shell", "terminal") else tool_name
    et = ErrorType.FILE_NOT_FOUND if "not found" in obs.lower() else ErrorType.COMMAND_FAILED
    return incremental_recovery_hint(
        hint_tool,
        args,
        et,
        obs,
        last_failed_command=cmd,
    )


def _action_display_name(tool_name: str) -> str:
    if tool_name in ("sandbox_run", "sandbox_shell"):
        return "terminal_run"
    if tool_name in ("fs_write", "fs_delete"):
        return "editor_edit"
    if tool_name == "fs_read":
        return "file_read"
    if tool_name == "fs_list":
        return "workspace_list"
    if tool_name == "github":
        return "github_action"
    if tool_name == "gitleaks":
        return "security_scan"
    if tool_name == "atomic_edit":
        return "hands_atomic_edit"
    if tool_name == "lsp_check":
        return "eyes_lsp"
    if tool_name == "playwright_observe":
        return "eyes_playwright"
    return tool_name


def _observation_payload(result: dict[str, Any]) -> dict[str, Any]:
    """Structured observation for replay (stdout/stderr split when present)."""
    out: dict[str, Any] = {
        "ok": bool(result.get("ok", True)),
        "tool": result.get("tool"),
        "raw": result,
    }
    if "stdout" in result:
        out["stdout"] = result.get("stdout")
    if "stderr" in result:
        out["stderr"] = result.get("stderr")
    if "exit_code" in result:
        out["exit_code"] = result.get("exit_code")
    if "timed_out" in result:
        out["timed_out"] = result.get("timed_out")
    if "error" in result and result.get("error"):
        out["error"] = result.get("error")
    if "content" in result:
        out["content_preview"] = str(result.get("content", ""))[:2000]
    return out


WorklogState = Literal["ok", "failure", "diagnostic_trigger", "self_heal"]


@dataclass
class OrchestratorHintBundle:
    """Optional user-message fragments after a synthetic diagnostic (Plodder JSON loop)."""

    system_block: str = ""
    incremental_hint: str = ""
    error_fingerprint: str | None = None
    streak_length: int | None = None


@dataclass
class PlodderWorklog:
    """
    OpenHands-style **event stream**: action–observation pairs, diagnostics, running summary.

    Append-only; serialize with :meth:`to_dict` or :func:`export_worklog_json`.
    """

    goal: str
    workspace_root: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)
    summary_of_progress: str = ""
    _seq: int = field(default=0, repr=False)

    def _next_id(self) -> int:
        self._seq += 1
        return self._seq

    def _append_summary_line(self, line: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        if self.summary_of_progress:
            self.summary_of_progress += "\n"
        self.summary_of_progress += f"[{ts}] {line}"

    def record_action_observation(
        self,
        *,
        round_idx: int,
        thought: str,
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any],
        prior_round_had_diagnostic_injection: bool = False,
        diagnostic_bundle: OrchestratorHintBundle | None = None,
    ) -> int:
        """
        Record one tool call and its observation.

        Returns the new event **id** (for linking ``diagnostic_trigger`` rows).
        """
        ok = bool(result.get("ok", True))
        eid = self._next_id()
        state: WorklogState
        if diagnostic_bundle and diagnostic_bundle.error_fingerprint:
            state = "diagnostic_trigger"
        elif prior_round_had_diagnostic_injection and ok:
            state = "self_heal"
        elif ok:
            state = "ok"
        else:
            state = "failure"

        hint = ""
        if not ok:
            hint = recovery_hint_for_failed_tool(tool_name, result)

        entry: dict[str, Any] = {
            "id": eid,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "round": round_idx,
            "event_type": "action_observation",
            "action": {
                "display_name": _action_display_name(tool_name),
                "tool": tool_name,
                "arguments": args,
            },
            "thought": (thought or "").strip()[:8000],
            "observation": _observation_payload(result),
            "state": state,
            "incremental_recovery_hint": hint if hint else None,
            "diagnostic": None,
        }

        if diagnostic_bundle and diagnostic_bundle.error_fingerprint:
            entry["diagnostic"] = {
                "error_fingerprint": diagnostic_bundle.error_fingerprint,
                "streak_length": diagnostic_bundle.streak_length,
                "system_block": diagnostic_bundle.system_block,
                "incremental_recovery_hint": diagnostic_bundle.incremental_hint or hint,
            }

        self.events.append(entry)

        if ok:
            self._append_summary_line(
                f"R{round_idx} {tool_name} OK — {_action_display_name(tool_name)}"
                + (" (after diagnostic / self-heal context)" if state == "self_heal" else "")
            )
        elif state == "diagnostic_trigger" and diagnostic_bundle and diagnostic_bundle.error_fingerprint:
            fp = diagnostic_bundle.error_fingerprint
            short = fp[:80] + ("…" if len(fp) > 80 else "")
            self._append_summary_line(
                f"R{round_idx} {tool_name} FAILED → diagnostic_trigger fingerprint={short}"
            )
        else:
            self._append_summary_line(
                f"R{round_idx} {tool_name} FAILED — hint recorded for replay"
            )

        self._append_diagnostic_stream_event(diagnostic_bundle, ref_event_id=eid, round_idx=round_idx)

        return eid

    def _append_diagnostic_stream_event(
        self,
        bundle: OrchestratorHintBundle | None,
        *,
        ref_event_id: int,
        round_idx: int,
    ) -> None:
        """Separate high-signal row for OpenHands-style session replay."""
        if not bundle or not bundle.error_fingerprint:
            return
        self.events.append(
            {
                "id": self._next_id(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "round": round_idx,
                "event_type": "diagnostic_trigger",
                "ref_event_id": ref_event_id,
                "error_fingerprint": bundle.error_fingerprint,
                "streak_length": bundle.streak_length,
                "incremental_recovery_hint": bundle.incremental_hint,
                "system_block": bundle.system_block,
                "thought": "",
                "action": None,
                "observation": None,
                "state": "diagnostic_trigger",
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "plodder_worklog/1.0",
            "goal": self.goal,
            "workspace_root": self.workspace_root,
            "summary_of_progress": self.summary_of_progress,
            "events": list(self.events),
            "event_count": len(self.events),
        }


def export_worklog_json(
    worklog: PlodderWorklog | dict[str, Any],
    workspace_root: str | Path,
    *,
    filename: str = "worklog.json",
) -> Path:
    """
    Persist the full worklog stream to ``<workspace_root>/<filename>`` (OpenHands replay shape).

    Accepts a :class:`PlodderWorklog` or an already-serialized dict.
    """
    root = Path(workspace_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    payload = worklog.to_dict() if isinstance(worklog, PlodderWorklog) else dict(worklog)
    path = root / filename
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


class SessionRecoveryTracker:
    """
    Action-aware failure context for the unified driver tool loop.

    Fingerprints ``tool`` + ``exit_code`` + normalized observation text (same as mini_devin).
    """

    def __init__(self, *, streak_threshold: int = 3) -> None:
        self._threshold = streak_threshold
        self.failure_streaks: list[FailureStreakRecord] = []
        self.diagnostic_triggers: list[DiagnosticTriggerRecord] = []
        self.recovery_paths: list[RecoveryPathRecord] = []
        self._streak_tool: str | None = None
        self._streak_fp: str | None = None
        self._streak_n: int = 0
        self._pending_fp: str | None = None

    def record_tool_observation(self, tool_name: str, result: dict[str, Any]) -> OrchestratorHintBundle | None:
        """
        Update streaks from one tool result dict (``ok``, ``error``, ``stderr``, ``exit_code``, …).

        On reaching ``streak_threshold`` identical failures, records diagnostic data and returns
        hints you may inject into the next user message (driver does not auto-inject yet).
        """
        ok = bool(result.get("ok", True))
        if ok:
            if self._pending_fp is not None:
                self.recovery_paths.append(
                    infer_recovery_path_summary(
                        after_error_fingerprint=self._pending_fp,
                        next_tool_name=tool_name,
                        next_args={k: v for k, v in result.items() if k != "ok"},
                    )
                )
                self._pending_fp = None
            self._streak_tool = None
            self._streak_fp = None
            self._streak_n = 0
            return None

        obs = _observation_text(result)
        ec = _exit_code(result)
        fp = error_fingerprint(tool_name, obs, ec)
        if tool_name == self._streak_tool and fp == self._streak_fp:
            self._streak_n += 1
        else:
            self._streak_tool = tool_name
            self._streak_fp = fp
            self._streak_n = 1

        if self._streak_n < self._threshold:
            return None

        streak_len = self._streak_n
        self.failure_streaks.append(
            FailureStreakRecord(
                tool_name=tool_name,
                error_fingerprint=fp,
                streak_length=streak_len,
            )
        )
        self.diagnostic_triggers.append(
            DiagnosticTriggerRecord(
                error_fingerprint=fp,
                tool_name=tool_name,
                exit_code=ec,
                output_preview=obs[:400],
            )
        )
        self._pending_fp = fp
        self._streak_tool = None
        self._streak_fp = None
        self._streak_n = 0

        argv = result.get("argv")
        if isinstance(argv, list):
            cmd = " ".join(str(x) for x in argv)
        else:
            cmd = str(result.get("command") or "")
        args = {"command": cmd} if cmd else {}
        hint_tool = "terminal" if tool_name in ("sandbox_run", "sandbox_shell", "terminal") else tool_name
        et = ErrorType.FILE_NOT_FOUND if "not found" in obs.lower() else ErrorType.COMMAND_FAILED
        hint = incremental_recovery_hint(
            hint_tool,
            args,
            et,
            obs,
            last_failed_command=cmd,
        )
        return OrchestratorHintBundle(
            system_block=(
                f"## System observation (repeated failure)\n\n"
                f"Tool `{tool_name}` failed **{self._threshold}** times with the same fingerprint.\n"
                f"**error_fingerprint:** `{fp}`\n"
            ),
            incremental_hint=hint,
            error_fingerprint=fp,
            streak_length=streak_len,
        )

    def to_payload(self, *, goal: str) -> PostMortemPayload:
        return PostMortemPayload(
            goal_or_task_id=goal,
            diagnostic_triggers=list(self.diagnostic_triggers),
            recovery_paths=list(self.recovery_paths),
            failure_streaks=list(self.failure_streaks),
        )

    def format_report(self, *, goal: str) -> str:
        return format_post_mortem(self.to_payload(goal=goal))


__all__ = [
    "PlodderWorklog",
    "export_worklog_json",
    "recovery_hint_for_failed_tool",
    "SessionRecoveryTracker",
    "OrchestratorHintBundle",
    "format_post_mortem",
    "PostMortemPayload",
    "DiagnosticTriggerRecord",
    "FailureStreakRecord",
    "RecoveryPathRecord",
    "infer_recovery_path_summary",
    "incremental_recovery_hint",
]
