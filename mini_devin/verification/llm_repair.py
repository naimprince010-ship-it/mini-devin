"""
LLM-assisted verification repair: optional Ruff autofix and bounded search/replace edits.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from litellm import acompletion

    _LITELLM = True
except ImportError:
    _LITELLM = False


def llm_env_configured() -> bool:
    if (
        os.environ.get("GROQ_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    ):
        return True
    if os.environ.get("AZURE_API_KEY") and os.environ.get("AZURE_API_BASE"):
        return True
    if os.environ.get("OLLAMA_ENABLED", "true").lower() == "true":
        return True
    return False


def _repair_model_id() -> str:
    explicit = os.environ.get("LLM_MODEL") or os.environ.get("MINI_DEVIN_REPAIR_MODEL")
    if explicit:
        return explicit
    from mini_devin.core.providers import get_model_registry

    return get_model_registry().get_default_model()


def _litellm_name(model_id: str) -> str:
    from mini_devin.core.providers import get_litellm_model_name

    return get_litellm_model_name(model_id)


def _safe_repo_path(root: Path, rel: str) -> Path | None:
    rel = rel.replace("\\", "/").strip().lstrip("/")
    if not rel or ".." in Path(rel).parts:
        return None
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if fence:
        try:
            data = json.loads(fence.group(1))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass
    brace = re.search(r"\{[\s\S]*\}", text)
    if brace:
        try:
            data = json.loads(brace.group(0))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None
    return None


async def try_ruff_autofix(working_directory: str) -> tuple[bool, str]:
    """Run `ruff check --fix` if the Ruff CLI is available. Returns (ran_successfully, message)."""
    if not shutil.which("ruff"):
        return False, "ruff CLI not found on PATH"
    root = Path(working_directory)
    if not root.is_dir():
        return False, f"not a directory: {working_directory}"
    try:
        proc = await asyncio.create_subprocess_exec(
            "ruff",
            "check",
            "--fix",
            ".",
            cwd=str(root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        out = (stdout or b"").decode(errors="ignore") + (stderr or b"").decode(errors="ignore")
        if proc.returncode in (0, 1):
            return True, f"ruff check --fix completed (exit {proc.returncode}): {out[:500]}"
        return False, f"ruff failed (exit {proc.returncode}): {out[:500]}"
    except Exception as e:
        logger.warning("ruff autofix error: %s", e)
        return False, str(e)


async def run_llm_search_replace_repair(
    working_directory: str,
    instruction: str,
    failure_output: str,
    *,
    max_output_chars: int = 14000,
) -> tuple[bool, str, list[str]]:
    """
    Ask the LLM for JSON: {"edits": [{"path": "relative/path.py", "old_text": "...", "new_text": "..."}]}
    and apply edits when old_text is unique in the file. Paths must stay under working_directory.
    """
    if not _LITELLM:
        return False, "litellm is not installed", []
    if not llm_env_configured():
        return False, "no LLM API configured for repair", []

    root = Path(working_directory).resolve()
    if not root.is_dir():
        return False, f"not a directory: {working_directory}", []

    blob = failure_output[:max_output_chars] if failure_output else ""
    model_id = _repair_model_id()
    model = _litellm_name(model_id)

    system = (
        "You are a coding agent that outputs ONLY valid JSON (no markdown outside JSON). "
        'Schema: {"edits":[{"path":"relative/path/from/repo/root","old_text":"exact snippet",'
        '"new_text":"replacement"}]}. '
        "old_text must match the file exactly once. Use small, minimal snippets. "
        "If you cannot safely fix, return {\"edits\":[]}."
    )
    user = f"Task:\n{instruction}\n\nTool / test / lint output (truncated):\n{blob}"

    try:
        response = await acompletion(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=4096,
            timeout=120,
        )
    except Exception as e:
        logger.warning("LLM repair completion failed: %s", e)
        return False, f"LLM call failed: {e}", []

    content = ""
    try:
        choices = getattr(response, "choices", None) or []
        if choices:
            content = (choices[0].message.content or "") if choices[0].message else ""
    except Exception:
        content = str(response)

    parsed = _extract_json_object(content)
    if not parsed:
        return False, "LLM did not return parseable JSON edits", []

    edits = parsed.get("edits")
    if not isinstance(edits, list):
        return False, "LLM JSON missing edits array", []

    modified: list[str] = []
    for item in edits:
        if not isinstance(item, dict):
            continue
        rel = item.get("path") or item.get("file")
        old_text = item.get("old_text") or item.get("old")
        new_text = item.get("new_text") or item.get("new")
        if not isinstance(rel, str) or not isinstance(old_text, str) or not isinstance(new_text, str):
            continue
        path = _safe_repo_path(root, rel)
        if path is None or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        count = text.count(old_text)
        if count != 1:
            continue
        path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
        modified.append(str(path.relative_to(root)))

    if modified:
        return True, f"Applied {len(modified)} LLM edit(s)", modified
    return False, "LLM returned no applicable edits", []


async def run_verification_llm_repair(
    working_directory: str,
    repair_kind: str,
    error_blob: str,
) -> tuple[bool, str, list[str]]:
    """Convenience wrapper for lint vs test repair prompts."""
    if repair_kind == "lint":
        instruction = (
            "Fix the lint/style issues indicated by the output. Prefer minimal changes. "
            "Only modify source files under the repository root."
        )
    else:
        instruction = (
            "Fix the code so the failing tests or errors in the output would be resolved. "
            "Do not weaken tests unless the test is clearly wrong; prefer fixing implementation. "
            "Only modify files under the repository root."
        )
    return await run_llm_search_replace_repair(working_directory, instruction, error_blob)
