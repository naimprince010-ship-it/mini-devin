#!/usr/bin/env python3
"""
End-to-end: Plodder ``run_agent`` with ``PLODDER_FORCE_PROCESS_SANDBOX=1``.

Default: deterministic mock LLM (no API key) that runs ``fs_write`` + ``sandbox_shell``
``git init`` then ``done``. Use ``--real`` for a live model when ``OPENAI_API_KEY`` is set.

Example:
  set PLODDER_FORCE_PROCESS_SANDBOX=1
  python scripts/run_plodder_process_sandbox_e2e.py
  python scripts/run_plodder_process_sandbox_e2e.py --real
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Force process sandbox before importing plodder sandbox stack (per user request)
os.environ["PLODDER_FORCE_PROCESS_SANDBOX"] = "1"

from plodder.agent.main_loop import AgentLoopConfig, run_agent, _create_default_session_sandbox
from plodder.workspace.session_workspace import SessionWorkspace


GOAL = (
    "Initialize a new git repo in the current workspace and create a README.md file "
    "with a project description."
)

README_BODY = """# Plodder process-sandbox E2E

This repository was initialized by an automated Plodder ``run_agent`` smoke test using
``ProcessExecutionSandbox`` (host ``bash``, no Docker).

"""


def _mock_llm_factory() -> object:
    """Two-turn scripted agent: write README, ``git init``, then done."""
    step = {"n": 0}

    async def llm(messages: list[dict]) -> str:  # type: ignore[type-arg]
        del messages  # contract only
        i = step["n"]
        step["n"] += 1
        if i == 0:
            return json.dumps(
                {
                    "status": "continue",
                    "rationale": "Create README.md then initialize git in the workspace via sandbox_shell.",
                    "tool_calls": [
                        {
                            "name": "fs_write",
                            "args": {"path": "README.md", "content": README_BODY},
                        },
                        {
                            "name": "sandbox_shell",
                            "args": {
                                "argv": ["git", "init"],
                                "language_hint": "shell",
                            },
                        },
                    ],
                }
            )
        return json.dumps(
            {
                "status": "done",
                "rationale": "README.md written and git repository initialized successfully.",
                "tool_calls": [],
            }
        )

    return llm


def _real_llm_factory(model: str):
    async def llm(messages: list[dict]) -> str:  # type: ignore[type-arg]
        try:
            import litellm
        except ImportError as e:
            raise RuntimeError("Install litellm for --real mode: poetry add litellm") from e
        resp = await litellm.acompletion(model=model, messages=messages, temperature=0.2)
        choice = resp.choices[0]
        content = choice.message.content
        if not content:
            raise RuntimeError("Empty model response")
        return str(content)

    return llm


async def main_async(workspace: Path, *, use_real: bool, model: str) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
        force=True,
    )
    log = logging.getLogger("e2e_process_sandbox")

    ws = SessionWorkspace(workspace)
    sb = _create_default_session_sandbox(ws)
    log.info("Sandbox backend: %s", type(sb).__name__ if sb else "(none)")
    if sb is None:
        log.error("No sandbox (Docker and process sandbox both failed). Is bash installed?")
        return 2

    if use_real:
        if not (os.getenv("OPENAI_API_KEY") or "").strip():
            log.error("--real requires OPENAI_API_KEY in the environment.")
            return 2
        llm = _real_llm_factory(model)
        max_rounds = 16
        inject_plan = True
    else:
        llm = _mock_llm_factory()
        max_rounds = 4
        inject_plan = False

    cfg = AgentLoopConfig(
        llm=llm,
        workspace=ws,
        sandbox=None,
        max_rounds=max_rounds,
        inject_logic_plan=inject_plan,
    )
    log.info("PLODDER_FORCE_PROCESS_SANDBOX=%r", os.environ.get("PLODDER_FORCE_PROCESS_SANDBOX"))
    log.info("Goal: %s", GOAL)
    log.info("Workspace: %s", workspace.resolve())

    result = await run_agent(GOAL, cfg)
    log.info("run_agent success=%s rounds=%s rationale=%s", result.success, result.rounds, result.rationale[:200])

    readme = workspace / "README.md"
    gitdir = workspace / ".git"
    ok_readme = readme.is_file()
    ok_git = gitdir.is_dir()
    log.info("Host README.md exists=%s .git exists=%s", ok_readme, ok_git)
    if ok_readme:
        log.info("README head: %r", readme.read_text(encoding="utf-8")[:200])
    return 0 if (result.success and ok_readme and ok_git) else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Empty directory for the run (default: temp dir)",
    )
    p.add_argument("--real", action="store_true", help="Use litellm + OPENAI_API_KEY (non-deterministic).")
    p.add_argument("--model", default="gpt-4o-mini", help="Model for --real")
    args = p.parse_args()
    ws = args.workspace
    if ws is None:
        ws = Path(tempfile.mkdtemp(prefix="plodder_proc_e2e_"))
    else:
        ws = ws.resolve()
        ws.mkdir(parents=True, exist_ok=True)

    return asyncio.run(main_async(ws, use_real=args.real, model=args.model))


if __name__ == "__main__":
    raise SystemExit(main())
