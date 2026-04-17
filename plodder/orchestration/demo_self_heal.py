"""
Demo: pseudo-logic → buggy Python → sandbox fail → LLM repair → success.

No Docker / no API keys required — uses ``DemoMockLLM`` + ``CodeSniffingSandbox``.

Run from repo root::

    python -m plodder.orchestration.demo_self_heal
"""

from __future__ import annotations

import asyncio
import json
import sys

from plodder.orchestration.self_heal import SelfHealLoop, SelfHealResult
from plodder.sandbox.execution_sandbox import SandboxResult


class DemoMockLLM:
    """Deterministic responses: JSON plan → buggy code → fixed code on repair."""

    async def __call__(self, messages: list[dict[str, object]]) -> str:
        user = str(messages[-1].get("content", ""))
        if "single JSON object" in user or "matching this shape" in user:
            plan = {
                "goal": "Print the result of 1+1",
                "constraints": ["Must run under CPython", "No external deps"],
                "data_structures": [{"name": "none", "role": "scalar arithmetic", "complexity_notes": ""}],
                "algorithms": [
                    {
                        "name": "add_and_print",
                        "paradigm": "sequential",
                        "steps": ["compute 1+1", "print result"],
                        "invariants": [],
                        "edge_cases": [],
                    }
                ],
                "control_flow_mermaid": "",
                "modularity_boundaries": ["single script"],
                "memory_and_lifecycle_notes": "",
                "type_safety_strategy": "none (dynamic Python)",
            }
            return json.dumps(plan)
        if "Repair attempt" in user:
            return (
                "- Replace division-by-zero with integer addition.\n"
                "- Keep output visible.\n"
                "```python\nprint('1+1 =', 1 + 1)\n```"
            )
        # initial codegen
        return "```python\n# intentionally broken for demo\nprint(1/0)\n```"


class CodeSniffingSandbox:
    """Simulates sandbox: fail while code contains ``1/0``, else success."""

    def run_python(self, code: str, *, timeout_sec: int | None = None) -> SandboxResult:
        cmd = "python /workspace/main.py"
        if "1/0" in code or "1 // 0" in code:
            return SandboxResult(
                stdout="",
                stderr="Traceback (most recent call last):\n  ZeroDivisionError: division by zero\n",
                exit_code=1,
                timed_out=False,
                container_id=None,
                command=cmd,
            )
        return SandboxResult(
            stdout="1+1 = 2\n",
            stderr="",
            exit_code=0,
            timed_out=False,
            container_id=None,
            command=cmd,
        )


async def _main() -> SelfHealResult:
    loop = SelfHealLoop(
        llm=DemoMockLLM(),
        sandbox=CodeSniffingSandbox(),
        max_repair_attempts=3,
        sandbox_timeout_sec=None,
    )
    return await loop.run("Print the integer result of 1 plus 1")


def main() -> None:
    r = asyncio.run(_main())
    print("success:", r.success)
    print("sandbox runs:", len(r.sandbox_attempts))
    for a in r.sandbox_attempts:
        print(f"  run {a['run']}: exit={a['exit_code']} stderr_head={a['stderr'][:60]!r}")
    print("final_code:\n", (r.final_code or "").strip())
    if not r.success:
        sys.exit(1)


if __name__ == "__main__":
    main()
