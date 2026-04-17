"""
Run ``mini_devin`` GitHub tool against a real workspace root (used by ``UnifiedSessionDriver``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mini_devin.core.tool_interface import ToolPolicy
from mini_devin.tools.github import GitHubAction, GitHubTool, GitHubToolInput


def _resolve_base_arg(raw: Any) -> str | None:
    if raw is None:
        return None
    t = str(raw).strip()
    if not t or t.lower() == "default":
        return None
    return t


def _coerce_pr_number(raw: Any) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _coerce_issue_list(raw: Any) -> list[int] | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        out: list[int] = []
        for x in raw:
            try:
                out.append(int(x))
            except (TypeError, ValueError):
                continue
        return out or None
    return None


async def run_github_tool_for_workspace(workspace_root: str, args: dict[str, Any]) -> dict[str, Any]:
    """Execute one ``github`` tool action; returns a dict suitable for Plodder tool JSON."""
    if not (os.getenv("GITHUB_TOKEN") or "").strip():
        return {
            "tool": "github",
            "ok": False,
            "error": "GITHUB_TOKEN (or GH_TOKEN) is not set — cannot call GitHub API.",
        }
    root = str(Path(workspace_root).resolve())
    action_s = str(args.get("action") or "").strip()
    if not action_s:
        return {"tool": "github", "ok": False, "error": "action is required"}

    try:
        action = GitHubAction(action_s)
    except ValueError:
        return {"tool": "github", "ok": False, "error": f"unknown action {action_s!r}"}

    tool = GitHubTool()
    tool.policy = ToolPolicy(default_timeout_seconds=180, max_timeout_seconds=900)

    assignees = args.get("assignees")
    if assignees is not None and not isinstance(assignees, list):
        assignees = None

    inp = GitHubToolInput(
        action=action,
        branch_name=args.get("branch_name"),
        base_branch=_resolve_base_arg(args.get("base_branch")),
        commit_message=args.get("commit_message"),
        files=args.get("files") if isinstance(args.get("files"), list) else None,
        pr_title=args.get("pr_title"),
        pr_description=args.get("pr_description"),
        task_description=args.get("task_description"),
        pr_number=_coerce_pr_number(args.get("pr_number")),
        merge_method=str(args.get("merge_method") or "squash"),
        draft=bool(args.get("draft", False)),
        assignees=[str(x) for x in assignees] if assignees else None,
        linked_issues=_coerce_issue_list(args.get("linked_issues")),
        repo_path=root,
    )

    out = await tool.execute(inp)
    return {
        "tool": "github",
        "ok": bool(out.success),
        "message": out.message,
        "pr_url": out.pr_url,
    }
