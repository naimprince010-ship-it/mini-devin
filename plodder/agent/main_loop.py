"""
Unified agent entry point — RAG, container planning, workspace tree, tool loop, sandbox.

``run_agent`` wires ``AgentSessionDriver`` (extends ``UnifiedSessionDriver``) with optional
``DocumentationStore`` and Docker-backed ``ExecutionSandbox`` for ``docs_retrieve`` and
``container_verify`` tools.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from plodder.core.universal_prompt_engine import PolyglotSystemPrompt, UniversalPromptEngine
from plodder.orchestration.session_driver import (
    ChatMessage,
    LLMFn,
    SupportsSessionSandbox,
    UnifiedSessionDriver,
    UnifiedSessionResult,
)
from plodder.sandbox.container_manager import docker_image_exists, plan_container_run, pull_suggestion
from plodder.sandbox.execution_sandbox import ExecutionSandbox
from plodder.workspace.session_workspace import SessionWorkspace

_TREE_IGNORE = frozenset(
    {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache", ".pytest_cache", "dist", "build"}
)

_AGENT_WORKFLOW_HINT = """## Suggested multi-step workflow (complex tasks)
1. **`docs_retrieve`** — pull indexed best practices (set `language_key` when aligned with the stack, e.g. `go`).
2. **`container_verify`** — confirm the resolved Docker **image** for your `entry` + optional `language_key` before heavy edits.
3. **`fs_list` / `fs_write`** — shape the project tree; keep paths consistent with the tree below.
4. **`sandbox_run`** — verify; on errors use **`fs_read`** then minimal **`fs_write`**, then re-run (self-heal)."""


_AGENT_EXTENSION_TOOLS_MD = """## Additional tools (autonomous loop)

| name | args | description |
|------|------|-------------|
| `docs_retrieve` | `query` (required), `language_key` (optional), `language_display` (optional), `n_results` (optional int, default 8) | Search the local documentation index (RAG) for snippets relevant to the query. |
| `container_verify` | `entry` (required), `language` (optional), `language_key` (optional) | Resolve the Docker image and argv **without** running code; reports whether the image exists locally and surfaces `docker pull` hints. |"""


class ToolCall(TypedDict):
    """Standard tool invocation shape (matches driver JSON)."""

    name: str
    args: dict[str, Any]


class ParsedTurn(TypedDict):
    """Normalized assistant turn after JSON parse."""

    status: str
    rationale: str
    tool_calls: list[ToolCall]


def normalize_tool_call(raw: dict[str, Any]) -> ToolCall:
    """Validate a single tool call dict from model output."""
    if not isinstance(raw, dict) or "name" not in raw:
        raise ValueError("tool_call must be a dict with name")
    args = raw.get("args")
    if args is not None and not isinstance(args, dict):
        raise ValueError("tool_call.args must be a dict")
    return ToolCall(name=str(raw["name"]), args=dict(args or {}))


def build_workspace_tree_block(
    workspace: SessionWorkspace,
    *,
    max_depth: int = 14,
    max_paths: int = 400,
    ignore_dir_names: frozenset[str] | None = None,
) -> str:
    """
    Build a sorted path listing under ``workspace.root`` (depth-limited) so the model
    retains project layout without listing every byte of every file.
    """
    ignore = ignore_dir_names or _TREE_IGNORE
    root = Path(workspace.root).resolve()
    paths: list[str] = []

    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        rel = Path(dirpath).relative_to(root)
        depth = len(rel.parts)
        if depth >= max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = sorted(n for n in dirnames if n not in ignore)
        for name in sorted(dirnames):
            if len(paths) >= max_paths:
                break
            p = rel / name
            paths.append(str(p).replace("\\", "/") + "/")
        if len(paths) >= max_paths:
            break
        for name in sorted(filenames):
            if len(paths) >= max_paths:
                break
            if name.startswith("."):
                continue
            p = rel / name
            paths.append(str(p).replace("\\", "/"))

    if len(paths) >= max_paths:
        paths.append("…(truncated)…")

    body = "\n".join(paths) if paths else "(empty workspace)"
    return f"## Workspace file tree (relative paths)\n```\n{body}\n```"


@dataclass
class AgentLoopConfig:
    """Dependencies for ``run_agent``."""

    llm: LLMFn
    workspace: SessionWorkspace
    sandbox: SupportsSessionSandbox | None = None
    engine: UniversalPromptEngine | None = None
    documentation_store: Any | None = None
    max_rounds: int = 24
    max_tool_calls_per_turn: int = 8
    sandbox_timeout_sec: int | None = 120
    inject_logic_plan: bool = True
    include_workspace_tree: bool = True


class AgentSessionDriver(UnifiedSessionDriver):
    """
    Extends the unified session driver with ``docs_retrieve``, ``container_verify``,
    and an initial workspace tree block.
    """

    def __init__(
        self,
        *,
        documentation_store: Any | None = None,
        include_workspace_tree: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._documentation_store = documentation_store
        self._include_workspace_tree = include_workspace_tree

    def allowed_tools(self) -> frozenset[str]:
        names = set(super().allowed_tools())
        if self._documentation_store is not None:
            names.add("docs_retrieve")
        if self._sandbox is not None:
            names.add("container_verify")
        return frozenset(names)

    def _system_content(self) -> str:
        base = super()._system_content()
        if self._documentation_store is not None or self._sandbox is not None:
            base += "\n\n" + _AGENT_EXTENSION_TOOLS_MD
        return base

    def _build_seed_user(self, goal: str, plan_md: str) -> str:
        blocks: list[str] = []
        if self._include_workspace_tree:
            blocks.append(build_workspace_tree_block(self._ws))
        if self._documentation_store is not None or self._sandbox is not None:
            blocks.append(_AGENT_WORKFLOW_HINT)
        blocks.append(super()._build_seed_user(goal, plan_md))
        return "\n\n".join(blocks)

    async def _exec_tool_async(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "docs_retrieve":
            return await self._tool_docs_retrieve_async(args)
        if name == "container_verify":
            return await self._tool_container_verify_async(args)
        return await super()._exec_tool_async(name, args)

    async def _tool_docs_retrieve_async(self, args: dict[str, Any]) -> dict[str, Any]:
        if self._documentation_store is None:
            return {"tool": "docs_retrieve", "ok": False, "error": "documentation store not configured"}
        q = str(args.get("query") or "").strip()
        if not q:
            return {"tool": "docs_retrieve", "ok": False, "error": "query is required"}
        lk_raw = args.get("language_key")
        language_key = str(lk_raw).strip() if lk_raw not in (None, "") else None
        ld_raw = args.get("language_display")
        language_display = str(ld_raw).strip() if ld_raw not in (None, "") else None
        n_raw = args.get("n_results", 8)
        try:
            n_results = int(n_raw)
        except (TypeError, ValueError):
            n_results = 8
        n_results = max(1, min(n_results, 24))

        block = self._engine.language_docs_retrieval_block(
            self._documentation_store,
            user_task=q,
            language_display=language_display,
            language_key=language_key,
            n_results=n_results,
            max_chars=9000,
        )
        if not block.strip():
            return {
                "tool": "docs_retrieve",
                "ok": True,
                "hits": 0,
                "context": "",
                "note": "No chunks returned (empty index or filter miss).",
            }
        hits = block.count("### Snippet")
        return {"tool": "docs_retrieve", "ok": True, "hits": hits, "context": block[:12000]}

    async def _tool_container_verify_async(self, args: dict[str, Any]) -> dict[str, Any]:
        if self._sandbox is None:
            return {"tool": "container_verify", "ok": False, "error": "sandbox not configured"}
        if not isinstance(self._sandbox, ExecutionSandbox):
            return {
                "tool": "container_verify",
                "ok": False,
                "error": "container_verify requires plodder.sandbox.ExecutionSandbox (Docker SDK)",
            }
        entry_raw = str(args.get("entry") or "").strip()
        if not entry_raw:
            return {"tool": "container_verify", "ok": False, "error": "entry is required"}
        lang = args.get("language")
        language = None if lang in (None, "", "auto") else str(lang)
        lk_raw = args.get("language_key")
        language_key = str(lk_raw).strip() if lk_raw not in (None, "") else None

        auto_pull = os.environ.get("PLODDER_DOCKER_AUTO_PULL", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        entry = entry_raw.replace("\\", "/").lstrip("/")
        planned = plan_container_run(
            entry=entry,
            language_hint=language,
            language_key=language_key,
            python_image=self._sandbox.python_image,
            node_image=self._sandbox.node_image,
            go_image=self._sandbox.go_image,
            rust_image=self._sandbox.rust_image,
            alpine_image=self._sandbox.alpine_image,
            cpp_image=self._sandbox.cpp_image,
            typescript_image=self._sandbox.typescript_image,
            docker_client=self._sandbox.docker_client,
            prefer_generic_if_image_missing=True,
            auto_pull_missing=auto_pull,
        )
        client = self._sandbox.docker_client
        present = docker_image_exists(client, planned.image) if client is not None else True
        return {
            "tool": "container_verify",
            "ok": True,
            "entry": entry,
            "image": planned.image,
            "argv": planned.argv,
            "language_hint_used": planned.language_hint_used,
            "image_present_local": present,
            "used_generic_fallback": planned.used_generic_fallback,
            "notes": planned.notes,
            "pull_hint": pull_suggestion(planned.image) if not present else "",
        }


async def run_agent(goal: str, config: AgentLoopConfig) -> UnifiedSessionResult:
    """
    Single entry point: planner pseudo-logic, workspace tree, multi-turn tools (including RAG + container verify).
    """
    driver = AgentSessionDriver(
        llm=config.llm,
        workspace=config.workspace,
        sandbox=config.sandbox,
        engine=config.engine or UniversalPromptEngine(PolyglotSystemPrompt()),
        max_rounds=config.max_rounds,
        max_tool_calls_per_turn=config.max_tool_calls_per_turn,
        sandbox_timeout_sec=config.sandbox_timeout_sec,
        inject_logic_plan=config.inject_logic_plan,
        documentation_store=config.documentation_store,
        include_workspace_tree=config.include_workspace_tree,
    )
    return await driver.run(goal)
