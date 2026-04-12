"""
Universal Prompt Engine — language-agnostic pseudo-logic before syntax.

Flow: intent → PseudoLogicPlan (structures, algorithms, flow) → then target syntax.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class DataStructureChoice:
    """Named structure with role in the solution (language-agnostic)."""

    name: str
    role: str
    complexity_notes: str = ""


@dataclass
class AlgorithmSketch:
    """High-level algorithmic intent without host-language syntax."""

    name: str
    paradigm: Literal["sequential", "divide_conquer", "greedy", "dp", "graph", "event_driven", "other"]
    steps: list[str]
    invariants: list[str] = field(default_factory=list)
    edge_cases: list[str] = field(default_factory=list)


@dataclass
class PseudoLogicPlan:
    """
    Universal logic artifact produced *before* any concrete language is chosen.

    Downstream agents compile this plan into a specific language's syntax,
    tests, and sandbox commands.
    """

    goal: str
    constraints: list[str]
    data_structures: list[DataStructureChoice]
    algorithms: list[AlgorithmSketch]
    control_flow_mermaid: str = ""
    modularity_boundaries: list[str] = field(default_factory=list)
    memory_and_lifecycle_notes: str = ""
    type_safety_strategy: str = ""

    def to_markdown_brief(self) -> str:
        """Dense representation for injection into LLM context."""
        lines: list[str] = [f"## Goal\n{self.goal}", "## Constraints"] + [f"- {c}" for c in self.constraints]
        lines.append("## Data structures")
        for ds in self.data_structures:
            lines.append(f"- **{ds.name}**: {ds.role}" + (f" ({ds.complexity_notes})" if ds.complexity_notes else ""))
        lines.append("## Algorithms")
        for alg in self.algorithms:
            lines.append(f"### {alg.name} ({alg.paradigm})")
            lines.extend(f"1. {s}" for s in alg.steps)
            if alg.invariants:
                lines.append("_Invariants:_ " + "; ".join(alg.invariants))
            if alg.edge_cases:
                lines.append("_Edge cases:_ " + "; ".join(alg.edge_cases))
        if self.control_flow_mermaid.strip():
            lines.append("## Control flow (Mermaid)\n```mermaid\n" + self.control_flow_mermaid.strip() + "\n```")
        if self.modularity_boundaries:
            lines.append("## Module boundaries")
            lines.extend(f"- {m}" for m in self.modularity_boundaries)
        if self.memory_and_lifecycle_notes:
            lines.append(f"## Memory / lifecycle\n{self.memory_and_lifecycle_notes}")
        if self.type_safety_strategy:
            lines.append(f"## Type-safety strategy\n{self.type_safety_strategy}")
        return "\n\n".join(lines)


@dataclass
class PolyglotSystemPrompt:
    """
    High-level compiler-architect persona: types, memory, modularity first.
    """

    product_name: str = "Plodder"

    def base_instruction(self) -> str:
        return f"""You are {self.product_name}, a senior **compiler-minded software architect** and polyglot engineer.

## Prime directives
1. **Universal logic first**: Before emitting code in *any* language, mentally produce a Pseudo-Logic Plan:
   explicit data structures, algorithm steps, invariants, edge cases, and control flow (Mermaid when helpful).
2. **Type safety**: Prefer explicit types, narrow contracts, and validation at boundaries. If the language lacks
   static types, emulate them with schemas, assertions, or tests.
3. **Memory & resources**: For managed runtimes, avoid leaks and unbounded retention. For native/unsafe tiers,
   call out ownership, lifetimes, RAII, or manual free discipline *before* coding.
4. **Modularity**: Small composable units, clear interfaces, dependency injection where idiomatic, minimal public API.
5. **Evidence over claims**: After code, specify how to verify (tests, property checks, sandbox runs).

## Polyglot behaviour
- You may be asked to work in hundreds of languages. If syntax is uncertain, **state assumptions** and prefer
  conservative, idiomatic patterns for that ecosystem.
- When a language is unfamiliar, request or use retrieved documentation snippets (RAG) and treat them as authoritative
  for syntax and style until contradicted by specs.

## Output shape (unless user overrides)
1. **Pseudo-Logic Plan** (bullets + optional Mermaid).
2. **Language choice** + rationale (if not fixed).
3. **Code** (minimal complete units).
4. **Verification** (commands or test outline).
"""


class UniversalPromptEngine:
    """
    Builds prompts and optional structured plans for the polyglot agent loop.

    Typical integration:
    - ``build_planner_messages(goal)`` → LLM returns JSON/text → parse into ``PseudoLogicPlan``.
    - ``build_coder_messages(plan, target_language)`` → LLM emits code + tests.
    """

    def __init__(self, persona: PolyglotSystemPrompt | None = None) -> None:
        self.persona = persona or PolyglotSystemPrompt()

    def system_messages(self) -> list[dict[str, Any]]:
        """Chat-style messages for models that support multi-system turns."""
        return [{"role": "system", "content": self.persona.base_instruction()}]

    def planner_user_prompt(self, goal: str, extra_context: str = "") -> str:
        return (
            f"Task:\n{goal.strip()}\n\n"
            f"{extra_context.strip()}\n\n"
            "Respond with a **Pseudo-Logic Plan** only (no code yet): "
            "data structures with roles, algorithm(s) with steps and invariants, edge cases, "
            "optional Mermaid for control flow, module boundaries, memory/lifecycle notes, type-safety strategy."
        )

    def planner_json_user_prompt(self, goal: str, extra_context: str = "") -> str:
        """Ask for machine-parseable JSON (used by the orchestration loop)."""
        schema = (
            '{"goal": str, "constraints": [str], '
            '"data_structures": [{"name": str, "role": str, "complexity_notes": str}], '
            '"algorithms": [{"name": str, "paradigm": "sequential|divide_conquer|greedy|dp|graph|event_driven|other", '
            '"steps": [str], "invariants": [str], "edge_cases": [str]}], '
            '"control_flow_mermaid": str, "modularity_boundaries": [str], '
            '"memory_and_lifecycle_notes": str, "type_safety_strategy": str}'
        )
        return (
            f"Task:\n{goal.strip()}\n\n{extra_context.strip()}\n\n"
            f"Return **only** a single JSON object (no markdown) matching this shape:\n{schema}\n"
            "Use empty strings or empty arrays where unknown."
        )

    def coder_user_prompt(
        self,
        plan: PseudoLogicPlan,
        target_language: str,
        task_hint: str = "",
        *,
        retrieved_context: str | None = None,
    ) -> str:
        body = plan.to_markdown_brief()
        hint = f"\n\nAdditional hint:\n{task_hint}" if task_hint.strip() else ""
        rag = ""
        if retrieved_context and retrieved_context.strip():
            rag = "\n\n## Retrieved documentation (trust for syntax & idioms)\n" + retrieved_context.strip()
        return (
            f"Target language: **{target_language}**\n\n"
            f"Compile the following universal plan into idiomatic, modular, well-typed code, "
            f"plus minimal tests or a verification command.{hint}{rag}\n\n{body}"
        )

    def language_docs_retrieval_block(
        self,
        store: Any,
        *,
        user_task: str,
        language_display: str | None = None,
        language_key: str | None = None,
        n_results: int = 8,
        max_chars: int = 8000,
    ) -> str:
        """
        Query Plodder's Lance ``DocumentationStore`` for cheat-sheet chunks.

        Typical wiring::

            store = DocumentationStore(docs_dir=\"docs/languages\")
            ctx = engine.language_docs_retrieval_block(
                store,
                user_task=\"Implement a small CLI parser\",
                language_display=\"Rust\",
                language_key=\"rust\",
            )
            prompt = engine.coder_user_prompt(plan, \"rust\", retrieved_context=ctx)

        ``language_key`` should match YAML ``language_key`` from ``scripts/prepare_docs.py``.
        """
        fmt = getattr(store, "format_retrieval_block_for_language", None)
        if not callable(fmt):
            return ""
        return fmt(
            user_task,
            language_display=language_display,
            language_key=language_key,
            n_results=n_results,
            max_chars=max_chars,
        )

    def execution_feedback_block(
        self,
        *,
        stdout: str,
        stderr: str,
        exit_code: int,
        command: str,
    ) -> str:
        """Format sandbox output for the self-healing loop."""
        return (
            "## Execution feedback (sandbox)\n"
            f"- **Command**: `{command}`\n"
            f"- **Exit code**: {exit_code}\n"
            "### STDOUT\n```\n" + (stdout or "(empty)") + "\n```\n"
            "### STDERR\n```\n" + (stderr or "(empty)") + "\n```\n"
            "Analyze failures, map stack traces to your modules, propose a minimal fix, then restate tests."
        )

    def repair_user_prompt(
        self,
        plan: PseudoLogicPlan,
        *,
        target_language: str,
        prior_code: str,
        execution_feedback_md: str,
        attempt: int,
    ) -> str:
        """Prompt for a sandbox repair pass (stderr-driven) while keeping the pseudo-logic contract."""
        return (
            f"## Repair attempt {attempt}\n"
            f"Target language: **{target_language}**\n\n"
            "The last run failed in the sandbox. Produce a **minimal Repair Plan** in 2–4 bullets, "
            "then output **only** the full corrected program as plain text (no markdown fences).\n\n"
            "### Original pseudo-logic (do not contradict)\n"
            + plan.to_markdown_brief()
            + "\n\n"
            + execution_feedback_md
            + "\n\n### Code to fix\n```text\n"
            + prior_code
            + "\n```\n"
        )

    def session_unified_driver_contract(self) -> str:
        """
        JSON-only tool loop contract for ``UnifiedSessionDriver``.

        The model must pick filesystem vs sandbox actions from execution feedback.
        """
        return (
            "## Response contract (strict JSON only)\n"
            "Every assistant message must be **one** JSON object, no markdown fences, no prose:\n"
            "```\n"
            "{\n"
            '  "status": "continue" | "done",\n'
            '  "rationale": "1-3 sentences: what you will do or why you stop",\n'
            '  "tool_calls": [\n'
            '    {"name": "<tool>", "args": { ... }}\n'
            "  ]\n"
            "}\n"
            "```\n"
            "- If ``status`` is ``done``, set ``tool_calls`` to ``[]`` and briefly state success/failure in ``rationale``.\n"
            "- If ``continue``, emit one or more tool calls; you will receive structured results next turn.\n"
            "- Use **relative paths** from the workspace root (POSIX ``/``).\n"
            "- After stderr names a file/line, prefer ``fs_read`` that path, then ``fs_write`` a minimal fix.\n"
        )

    def session_unified_tools_catalog(self) -> str:
        """Human-readable tool reference injected into the system prompt."""
        return """## Tools (names and args)

| name | args | description |
|------|------|-------------|
| `fs_list` | `path` (optional, default `.`) | List files and subdirs under `path`. |
| `fs_read` | `path` | Read a UTF-8 text file. |
| `fs_write` | `path`, `content` | Create/overwrite a file (creates parent dirs). |
| `fs_delete` | `path` | Delete a file or empty/non-empty directory tree. |
| `sandbox_run` | `entry` (required), `language` (optional: `auto`/`python`/`javascript`/`typescript`/…), `language_key` (optional: doc slug e.g. `rust`/`python` — selects Docker image + hints), `network` (bool, default false) | Snapshot workspace text files into the sandbox and run `entry` with an **auto-detected** Docker image; `language_key` aligns with indexed docs metadata. |
| `sandbox_shell` | `argv` (list of strings), `language_hint` (optional), `network` (bool) | Run a shell command in the sandbox with the workspace snapshot (e.g. `["sh","-c","cd /workspace && npm install"]`). Use `network: true` for installs. |

**Notes:** `sandbox_*` requires Docker. For `sandbox_shell`, `argv` runs as the container entrypoint; working directory is `/workspace`."""
