"""
Self-heal orchestration — pseudo-logic → code → sandbox → repair loop.

Also contains helpers for LSP-static repair bundles (see ``SelfHealBundle``).
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

from plodder.core.universal_prompt_engine import (
    AlgorithmSketch,
    DataStructureChoice,
    PolyglotSystemPrompt,
    PseudoLogicPlan,
    UniversalPromptEngine,
)
from plodder.lsp.bridge import DiagnosticsReport
from plodder.orchestration.session_driver import UnifiedSessionDriver, UnifiedSessionResult
from plodder.sandbox.execution_sandbox import SandboxResult


# ── LLM contract ────────────────────────────────────────────────────────────

ChatMessage = dict[str, Any]
LLMFn = Callable[[list[ChatMessage]], Awaitable[str]]


class SupportsRunPython(Protocol):
    def run_python(self, code: str, *, timeout_sec: int | None = None) -> SandboxResult: ...


# ── Plan / code parsing ─────────────────────────────────────────────────────

_PARADIGMS = frozenset(
    {"sequential", "divide_conquer", "greedy", "dp", "graph", "event_driven", "other"}
)


def _strip_json_fence(raw: str) -> str:
    t = raw.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def parse_pseudo_plan_json(raw: str, *, fallback_goal: str) -> PseudoLogicPlan:
    """Parse ``planner_json_user_prompt`` output into a ``PseudoLogicPlan``."""
    try:
        data = json.loads(_strip_json_fence(raw))
    except json.JSONDecodeError:
        return PseudoLogicPlan(
            goal=fallback_goal,
            constraints=["(LLM returned non-JSON; using minimal default plan)"],
            data_structures=[DataStructureChoice("TBD", "Refine after successful JSON plan", "")],
            algorithms=[
                AlgorithmSketch(
                    "Main",
                    "sequential",
                    ["Clarify requirements", "Implement", "Verify in sandbox"],
                    [],
                    [],
                )
            ],
        )

    def _ds(item: Any) -> DataStructureChoice:
        if not isinstance(item, dict):
            return DataStructureChoice("unknown", "", "")
        return DataStructureChoice(
            name=str(item.get("name", "unnamed")),
            role=str(item.get("role", "")),
            complexity_notes=str(item.get("complexity_notes", "")),
        )

    def _alg(item: Any) -> AlgorithmSketch:
        if not isinstance(item, dict):
            return AlgorithmSketch("step", "sequential", ["execute"], [], [])
        p = str(item.get("paradigm", "sequential"))
        if p not in _PARADIGMS:
            p = "sequential"
        return AlgorithmSketch(
            name=str(item.get("name", "algorithm")),
            paradigm=p,  # type: ignore[arg-type]
            steps=[str(s) for s in (item.get("steps") or []) if s is not None],
            invariants=[str(s) for s in (item.get("invariants") or []) if s is not None],
            edge_cases=[str(s) for s in (item.get("edge_cases") or []) if s is not None],
        )

    return PseudoLogicPlan(
        goal=str(data.get("goal", fallback_goal)),
        constraints=[str(c) for c in (data.get("constraints") or []) if c is not None],
        data_structures=[_ds(x) for x in (data.get("data_structures") or [])],
        algorithms=[_alg(x) for x in (data.get("algorithms") or [])],
        control_flow_mermaid=str(data.get("control_flow_mermaid", "")),
        modularity_boundaries=[str(m) for m in (data.get("modularity_boundaries") or []) if m is not None],
        memory_and_lifecycle_notes=str(data.get("memory_and_lifecycle_notes", "")),
        type_safety_strategy=str(data.get("type_safety_strategy", "")),
    )


def extract_code_fence(text: str, language: str = "python") -> str:
    """Pull the last fenced ``` block, else strip prose / repair bullets heuristically."""
    t = text.strip()
    lang = re.escape(language)
    blocks = re.findall(rf"```(?:{lang}|py|python)?\s*\n(.*?)```", t, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return blocks[-1].strip()
    blocks_any = re.findall(r"```(?:\w+)?\s*\n(.*?)```", t, flags=re.DOTALL | re.IGNORECASE)
    if blocks_any:
        return blocks_any[-1].strip()
    # repair prompt may end with raw program only
    lines = t.splitlines()
    while lines and lines[0].lstrip().startswith(("-", "*", "1.", "2.", "3.", "#")):
        lines.pop(0)
    return "\n".join(lines).strip()


# ── Master loop ─────────────────────────────────────────────────────────────


@dataclass
class SelfHealResult:
    success: bool
    goal: str
    plan: PseudoLogicPlan | None
    final_code: str | None
    sandbox_attempts: list[dict[str, Any]] = field(default_factory=list)
    plan_raw: str = ""
    last_repair_raw: str = ""


class SelfHealLoop:
    """
    Single-artifact loop: plan → one program string → ``run_python`` → stderr repair.

    For **multi-file projects** with filesystem + shell + auto-detected sandbox images,
    use ``UnifiedSessionDriver`` in ``plodder.orchestration.session_driver``.
    """

    def __init__(
        self,
        *,
        llm: LLMFn,
        sandbox: SupportsRunPython,
        engine: UniversalPromptEngine | None = None,
        target_language: str = "python",
        max_repair_attempts: int = 3,
        sandbox_timeout_sec: int | None = 30,
    ) -> None:
        self._llm = llm
        self._sandbox = sandbox
        self._engine = engine or UniversalPromptEngine(PolyglotSystemPrompt())
        self._target_language = target_language
        self._max_repair_attempts = max_repair_attempts
        self._sandbox_timeout_sec = sandbox_timeout_sec

    async def _complete(self, messages: list[ChatMessage]) -> str:
        return await self._llm(messages)

    async def run(self, goal: str) -> SelfHealResult:
        attempts: list[dict[str, Any]] = []

        plan_messages = [
            *self._engine.system_messages(),
            {"role": "user", "content": self._engine.planner_json_user_prompt(goal)},
        ]
        plan_raw = await self._complete(plan_messages)
        plan = parse_pseudo_plan_json(plan_raw, fallback_goal=goal)

        code_messages = [
            *self._engine.system_messages(),
            {"role": "user", "content": self._engine.coder_user_prompt(plan, self._target_language)},
        ]
        code_raw = await self._complete(code_messages)
        code = extract_code_fence(code_raw, self._target_language)

        last_repair_raw = ""
        max_runs = 1 + self._max_repair_attempts

        for run_idx in range(max_runs):
            if self._sandbox_timeout_sec is not None:
                result = await asyncio.to_thread(
                    self._sandbox.run_python,
                    code,
                    timeout_sec=self._sandbox_timeout_sec,
                )
            else:
                result = await asyncio.to_thread(self._sandbox.run_python, code)
            attempts.append(
                {
                    "run": run_idx,
                    "exit_code": result.exit_code,
                    "timed_out": result.timed_out,
                    "command": result.command,
                    "stderr": (result.stderr or "")[:8000],
                    "stdout": (result.stdout or "")[:4000],
                }
            )
            ok = result.exit_code == 0 and not result.timed_out
            if ok:
                return SelfHealResult(
                    success=True,
                    goal=goal,
                    plan=plan,
                    final_code=code,
                    sandbox_attempts=attempts,
                    plan_raw=plan_raw,
                    last_repair_raw=last_repair_raw,
                )
            if run_idx >= max_runs - 1:
                break

            fb = self._engine.execution_feedback_block(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
                command=result.command,
            )
            repair_messages = [
                *self._engine.system_messages(),
                {
                    "role": "user",
                    "content": self._engine.repair_user_prompt(
                        plan,
                        target_language=self._target_language,
                        prior_code=code,
                        execution_feedback_md=fb,
                        attempt=run_idx + 1,
                    ),
                },
            ]
            last_repair_raw = await self._complete(repair_messages)
            code = extract_code_fence(last_repair_raw, self._target_language)

        return SelfHealResult(
            success=False,
            goal=goal,
            plan=plan,
            final_code=code,
            sandbox_attempts=attempts,
            plan_raw=plan_raw,
            last_repair_raw=last_repair_raw,
        )


# ── LSP-static bundle (unchanged consumers) ─────────────────────────────────


@dataclass
class SelfHealBundle:
    """Everything the repair model needs for one iteration."""

    file_path: str
    source_text: str
    diagnostics: DiagnosticsReport
    extra_instruction: str = ""
    prior_attempts: list[str] = field(default_factory=list)

    def structured_issues(self) -> list[dict[str, Any]]:
        """JSON-serializable rows for tools / logging."""
        out: list[dict[str, Any]] = []
        for i in self.diagnostics.issues:
            out.append(
                {
                    "severity": i.severity,
                    "line": i.line,
                    "character": i.character,
                    "end_line": i.end_line,
                    "end_character": i.end_character,
                    "code": i.code,
                    "source": i.source,
                    "message": i.message,
                    "related_information": list(i.related_information),
                }
            )
        return out


def build_static_fix_prompt(bundle: SelfHealBundle) -> str:
    """
    Single user-style blob for an LLM repair pass (static errors only).

    Downstream: append sandbox feedback from ``UniversalPromptEngine.execution_feedback_block``
    if a run still fails after LSP is clean.
    """
    lsp_block = bundle.diagnostics.to_self_heal_prompt_block()
    prior = ""
    if bundle.prior_attempts:
        prior = "\n## Prior failed edits (avoid repeating)\n" + "\n---\n".join(bundle.prior_attempts[-3:])
    return (
        f"Fix the following file: `{bundle.file_path}`\n\n"
        f"{bundle.extra_instruction.strip()}\n\n"
        f"{lsp_block}\n\n"
        "## Current source\n```\n"
        + bundle.source_text
        + "\n```\n"
        "Return **only** the corrected full file content (no markdown fences around the whole file).\n"
        f"{prior}"
    ).strip()


def merge_static_and_runtime(*, static_lsp: str, sandbox_feedback: str) -> str:
    """Concatenate LSP block + sandbox block for a second repair hop."""
    parts = [p for p in (static_lsp.strip(), sandbox_feedback.strip()) if p]
    return "\n\n".join(parts)
