"""
Vercel Auto-Deploy Tool for Plodder

Deploys a local project to Vercel using the Vercel CLI (`npx vercel`).
Requires VERCEL_TOKEN to be set in the environment (or passed at call time).
Returns the live production URL on success.
"""

import asyncio
import os
import re
import shlex
import time
from pathlib import Path
from typing import Optional

from pydantic import Field

from ..core.tool_interface import BaseTool
from ..schemas.tools import BaseToolInput, BaseToolOutput, ToolStatus

# ── Limits ────────────────────────────────────────────────────────────────────
_DEPLOY_TIMEOUT_SECONDS = int(os.environ.get("PLODDER_DEPLOY_TIMEOUT_SECONDS", "300"))
_OUTPUT_MAX_CHARS = 8_000

# ── URL extraction ─────────────────────────────────────────────────────────────
_URL_RE = re.compile(
    r"https://[a-zA-Z0-9][a-zA-Z0-9\-\.]*\.(vercel\.app|now\.sh)[^\s]*"
)


def _extract_url(output: str) -> Optional[str]:
    """Return the first Vercel deployment URL found in *output*, or None."""
    # Prefer lines that start with '✅' or contain 'Production:' / 'Inspect:'
    for line in output.splitlines():
        if "production:" in line.lower() or "✅" in line or "🔍" in line:
            m = _URL_RE.search(line)
            if m:
                return m.group(0)
    # Fall back to the last URL in any line
    urls = _URL_RE.findall(output)
    return urls[-1] if urls else None


def _truncate(text: str, max_chars: int = _OUTPUT_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + f"\n… [{len(text) - max_chars} chars omitted] …\n" + text[-half:]


# ── Schema ─────────────────────────────────────────────────────────────────────

class DeployInput(BaseToolInput):
    """Input for the Vercel deploy tool."""

    working_directory: str = Field(
        default=".",
        description="Path to the project directory to deploy. Defaults to current directory.",
    )
    project_name: Optional[str] = Field(
        default=None,
        description=(
            "Optional Vercel project name. If omitted, Vercel uses the directory name "
            "or whatever is in vercel.json."
        ),
    )
    vercel_token: Optional[str] = Field(
        default=None,
        description=(
            "Vercel token override. Normally the tool reads VERCEL_TOKEN from the "
            "environment; only pass this when the env var is not set."
        ),
    )
    production: bool = Field(
        default=True,
        description="Deploy to production (--prod). Set False to create a preview deployment.",
    )


class DeployOutput(BaseToolOutput):
    """Output from the Vercel deploy tool."""

    success: bool = Field(description="Whether the deployment succeeded")
    url: Optional[str] = Field(
        default=None, description="Live production URL returned by Vercel"
    )
    exit_code: int = Field(description="Exit code from the vercel CLI process")
    output: str = Field(description="Combined stdout + stderr from the deploy command")
    message: str = Field(description="Human-readable result summary")


# ── Tool implementation ────────────────────────────────────────────────────────

class DeployVercelTool(BaseTool[DeployInput, DeployOutput]):
    """Auto-deploy a project to Vercel and return the live URL."""

    name = "deploy_vercel"
    description = (
        "Deploy a project to Vercel using the Vercel CLI. "
        "Reads VERCEL_TOKEN from environment (required). "
        "Returns the live production URL on success.\n"
        "Usage: set working_directory to the project root, optionally set project_name."
    )
    input_schema = DeployInput
    output_schema = DeployOutput

    async def _execute(self, input_data: DeployInput) -> DeployOutput:
        start = time.monotonic()

        # ── Resolve token ──────────────────────────────────────────────────────
        token = (input_data.vercel_token or "").strip() or os.environ.get(
            "VERCEL_TOKEN", ""
        ).strip()
        if not token:
            return DeployOutput(
                status=ToolStatus.FAILURE,
                execution_time_ms=0,
                success=False,
                exit_code=-1,
                output="",
                message=(
                    "VERCEL_TOKEN is not set. "
                    "Add it to the environment (e.g. in .env or docker-compose.yml) "
                    "and restart the server."
                ),
            )

        # ── Resolve working directory ──────────────────────────────────────────
        cwd = Path(input_data.working_directory).resolve()
        if not cwd.is_dir():
            return DeployOutput(
                status=ToolStatus.FAILURE,
                execution_time_ms=0,
                success=False,
                exit_code=-1,
                output="",
                message=f"working_directory does not exist: {cwd}",
            )

        # ── Build command ──────────────────────────────────────────────────────
        cmd: list[str] = ["npx", "--yes", "vercel"]
        if input_data.production:
            cmd.append("--prod")
        cmd += ["--yes"]  # non-interactive
        if input_data.project_name:
            # Vercel project names must be lowercase alphanumeric + hyphens
            safe_name = re.sub(r"[^a-z0-9\-]", "-", input_data.project_name.lower()).strip("-")
            if safe_name:
                cmd += ["--name", safe_name]

        # Token is passed via environment, NOT as a CLI flag, so it stays out of
        # process listings and logs.
        env = {**os.environ, "VERCEL_TOKEN": token}

        # ── Run ────────────────────────────────────────────────────────────────
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(cwd),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,  # merge stderr → stdout
            )
            try:
                raw_out, _ = await asyncio.wait_for(
                    proc.communicate(), timeout=_DEPLOY_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                elapsed = int((time.monotonic() - start) * 1000)
                return DeployOutput(
                    status=ToolStatus.FAILURE,
                    execution_time_ms=elapsed,
                    success=False,
                    exit_code=-1,
                    output="[Deployment timed out]",
                    message=(
                        f"Vercel deployment timed out after {_DEPLOY_TIMEOUT_SECONDS}s."
                    ),
                )
        except FileNotFoundError:
            return DeployOutput(
                status=ToolStatus.FAILURE,
                execution_time_ms=0,
                success=False,
                exit_code=-1,
                output="",
                message=(
                    "npx is not available in this environment. "
                    "Node.js must be installed to use the Vercel deploy tool."
                ),
            )

        elapsed = int((time.monotonic() - start) * 1000)
        output_text = _truncate(raw_out.decode("utf-8", errors="replace"))
        exit_code: int = proc.returncode or 0
        succeeded = exit_code == 0
        url = _extract_url(output_text) if succeeded else None

        if succeeded and url:
            message = f"Deployed successfully. Live URL: {url}"
        elif succeeded:
            message = "Deployment command exited 0 but no URL was found in output."
        else:
            message = f"Deployment failed (exit code {exit_code}). See output for details."

        return DeployOutput(
            status=ToolStatus.SUCCESS if succeeded else ToolStatus.FAILURE,
            execution_time_ms=elapsed,
            success=succeeded,
            url=url,
            exit_code=exit_code,
            output=output_text,
            message=message,
        )


def create_deploy_tool() -> DeployVercelTool:
    """Factory — create a DeployVercelTool instance."""
    return DeployVercelTool()
