"""
Demo ``UnifiedSessionDriver`` with a scripted mock LLM (no Docker).

Run::

    python -m plodder.orchestration.demo_session_driver
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from plodder.orchestration.session_driver import UnifiedSessionDriver
from plodder.sandbox.execution_sandbox import SandboxResult
from plodder.workspace.session_workspace import SessionWorkspace


class StubSandbox:
    """Fake sandbox: succeed when ``main.py`` has no ``1/0``."""

    def run_detected(
        self,
        files: dict[str, str],
        *,
        entry: str,
        language: str | None = None,
        language_key: str | None = None,
        timeout_sec: int | None = None,
        network: bool = False,
    ) -> SandboxResult:
        text = files.get(entry, "")
        if "1/0" in text or "1//0" in text:
            return SandboxResult(
                "",
                "ZeroDivisionError: division by zero\n",
                1,
                False,
                None,
                f"python /workspace/{entry}",
            )
        return SandboxResult("ok\n", "", 0, False, None, f"python /workspace/{entry}")

    def run_shell_in_workspace(
        self,
        files: dict[str, str],
        argv: list[str],
        *,
        language_hint: str | None = None,
        timeout_sec: int | None = None,
        network: bool = False,
    ) -> SandboxResult:
        joined = " ".join(argv)
        return SandboxResult(f"(stub shell) {joined}\n", "", 0, False, None, joined)


class ScriptedLLM:
    """Emit JSON tool turns without calling a real API."""

    def __init__(self) -> None:
        self._step = 0

    async def __call__(self, messages: list[dict[str, object]]) -> str:
        last = str(messages[-1].get("content", ""))
        if "matching this shape" in last or "single JSON object" in last:
            return json.dumps(
                {
                    "goal": "demo",
                    "constraints": [],
                    "data_structures": [],
                    "algorithms": [],
                    "control_flow_mermaid": "",
                    "modularity_boundaries": [],
                    "memory_and_lifecycle_notes": "",
                    "type_safety_strategy": "",
                }
            )
        self._step += 1
        if self._step == 1:
            return json.dumps(
                {
                    "status": "continue",
                    "rationale": "Create a broken main.py then list workspace.",
                    "tool_calls": [
                        {
                            "name": "fs_write",
                            "args": {"path": "main.py", "content": "print('hello')\nprint(1/0)\n"},
                        },
                        {"name": "fs_list", "args": {"path": "."}},
                    ],
                }
            )
        if self._step == 2:
            return json.dumps(
                {
                    "status": "continue",
                    "rationale": "Run sandbox to capture stderr, then fix file.",
                    "tool_calls": [{"name": "sandbox_run", "args": {"entry": "main.py", "language": "python"}}],
                }
            )
        if self._step == 3:
            return json.dumps(
                {
                    "status": "continue",
                    "rationale": "Repair main.py after ZeroDivisionError.",
                    "tool_calls": [
                        {
                            "name": "fs_write",
                            "args": {"path": "main.py", "content": "print('hello')\nprint(1+1)\n"},
                        }
                    ],
                }
            )
        if self._step == 4:
            return json.dumps(
                {
                    "status": "continue",
                    "rationale": "Re-run and smoke shell.",
                    "tool_calls": [
                        {"name": "sandbox_run", "args": {"entry": "main.py"}},
                        {
                            "name": "sandbox_shell",
                            "args": {
                                "argv": ["sh", "-c", "echo ping"],
                                "language_hint": "python",
                                "network": False,
                            },
                        },
                    ],
                }
            )
        return json.dumps(
            {
                "status": "done",
                "rationale": "Sandbox verified successfully; task complete.",
                "tool_calls": [],
            }
        )


async def _main() -> None:
    root = Path(tempfile.mkdtemp(prefix="plodder_session_"))
    ws = SessionWorkspace(root)
    driver = UnifiedSessionDriver(
        llm=ScriptedLLM(),
        workspace=ws,
        sandbox=StubSandbox(),
        max_rounds=12,
        inject_logic_plan=True,
    )
    r = await driver.run("Build a tiny Python demo under main.py and verify it runs.")
    print("success:", r.success)
    print("rounds:", r.rounds)
    print("rationale:", r.rationale)
    print("main.py:", (root / "main.py").read_text(encoding="utf-8"))


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
