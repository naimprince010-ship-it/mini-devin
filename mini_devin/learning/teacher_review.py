"""
Post-task teacher review (OpenAI-first) and append-only JSONL for fine-tuning.

Later: add ``openrouter`` / ``moonshot`` backends by implementing the same
``_review_with_messages`` contract and switching on ``TEACHER_BACKEND``.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.llm_client import LLMClient, LLMConfig, LLMMessage
from ..schemas.state import TaskState


TEACHER_SYSTEM_PROMPT = """You are a strict senior engineer reviewing another agent's work.
You receive: the user task, a short agent summary, optional tool/command/file hints, and a truncated conversation excerpt.

Return ONLY valid JSON (no markdown fences) with this exact shape:
{
  "verdict": "pass" | "issues" | "fail",
  "issues": ["string", ...],
  "mistake_analysis": "string or empty",
  "correct_approach": "string or empty",
  "suggested_followups": ["string", ...],
  "confidence": 0.0
}
- Use "pass" only if the agent likely satisfied acceptance criteria and did not hallucinate success.
- Use "issues" for minor gaps; "fail" for wrong solution, skipped verification, or clear hallucination.
- confidence is your estimated probability 0..1 that your verdict is correct."""


@dataclass
class LearningSettings:
    """Learning / teacher pipeline (env-driven; OpenAI-only implementation for now)."""

    teacher_review_enabled: bool = False
    teacher_backend: str = "openai"
    teacher_model: str = "gpt-4o"
    teacher_api_key: str | None = None
    training_data_log_path: str = "data/training_logs/reviews.jsonl"
    teacher_max_excerpt_messages: int = 32
    teacher_max_chars_per_message: int = 6000

    @classmethod
    def from_env(cls) -> LearningSettings:
        base = Path(os.environ.get("MINI_DEVIN_DATA", "data"))
        default_log = str(base / "training_logs" / "reviews.jsonl")
        return cls(
            teacher_review_enabled=os.environ.get("TEACHER_REVIEW_ENABLED", "false").lower()
            == "true",
            teacher_backend=os.environ.get("TEACHER_BACKEND", "openai").lower().strip(),
            teacher_model=os.environ.get("TEACHER_MODEL", "gpt-4o").strip(),
            teacher_api_key=os.environ.get("TEACHER_API_KEY")
            or os.environ.get("OPENAI_API_KEY"),
            training_data_log_path=os.environ.get("TRAINING_DATA_LOG_PATH", default_log),
            teacher_max_excerpt_messages=int(os.environ.get("TEACHER_MAX_EXCERPT_MESSAGES", "32")),
            teacher_max_chars_per_message=int(
                os.environ.get("TEACHER_MAX_CHARS_PER_MESSAGE", "6000")
            ),
        )


def parse_teacher_json(raw: str | None) -> dict[str, Any]:
    """Extract JSON object from model output (handles optional ``` fences)."""
    if not raw or not raw.strip():
        return {
            "verdict": "issues",
            "issues": ["empty_teacher_response"],
            "mistake_analysis": "",
            "correct_approach": "",
            "suggested_followups": [],
            "confidence": 0.0,
        }
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}\s*$", text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return {
        "verdict": "issues",
        "issues": ["teacher_json_parse_failed"],
        "mistake_analysis": text[:4000],
        "correct_approach": "",
        "suggested_followups": [],
        "confidence": 0.0,
    }


def serialize_conversation_excerpt(
    messages: list[LLMMessage],
    *,
    max_messages: int,
    max_chars_per_message: int,
) -> list[dict[str, Any]]:
    """Last N messages, tool calls summarized, content truncated."""
    tail = messages[-max_messages:] if len(messages) > max_messages else messages
    out: list[dict[str, Any]] = []
    for m in tail:
        entry: dict[str, Any] = {"role": m.role}
        if m.content:
            c = m.content
            if len(c) > max_chars_per_message:
                c = c[: max_chars_per_message - 20] + "\n...[truncated]"
            entry["content"] = c
        if m.tool_calls:
            entry["tool_calls"] = [
                {"name": tc.name, "arguments": tc.arguments} for tc in m.tool_calls
            ]
        if m.tool_call_id:
            entry["tool_call_id"] = m.tool_call_id
        if m.name:
            entry["name"] = m.name
        out.append(entry)
    return out


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)


def _sft_messages_stub(
    task_description: str,
    agent_summary: str,
    teacher: dict[str, Any],
) -> list[dict[str, Any]]:
    """Shape compatible with later Unsloth / Llama-Factory chat JSONL exports."""
    user = (
        f"Task:\n{task_description}\n\n"
        f"Agent summary (to critique):\n{agent_summary[:8000]}"
    )
    assistant = json.dumps(teacher, ensure_ascii=False)
    return [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]


async def _review_openai(
    *,
    settings: LearningSettings,
    user_payload: str,
    verbose: bool,
) -> tuple[str, dict[str, Any]]:
    api_key = settings.teacher_api_key
    if not api_key:
        raise RuntimeError("TEACHER review needs TEACHER_API_KEY or OPENAI_API_KEY")

    # Fresh client (do not mutate agent LLM)
    cfg = LLMConfig(
        model=settings.teacher_model,
        temperature=0.0,
        max_tokens=2048,
        api_key=api_key,
    )
    review_client = LLMClient(cfg)
    review_client.set_system_prompt(TEACHER_SYSTEM_PROMPT)
    review_client.add_user_message(user_payload)
    resp = await review_client.complete(tools=None, stream=False)
    raw = (resp.content or "").strip()
    parsed = parse_teacher_json(raw)
    return raw, parsed


async def maybe_log_teacher_review(
    *,
    task: TaskState,
    agent_model: str,
    conversation: list[LLMMessage],
    summary: str,
    verbose: bool = True,
) -> None:
    """
    If TEACHER_REVIEW_ENABLED=true, call teacher LLM and append one JSONL record.

    Swallows errors so training never breaks the agent run.
    """
    settings = LearningSettings.from_env()
    if not settings.teacher_review_enabled:
        return

    if settings.teacher_backend != "openai":
        if verbose:
            print(
                f"[Teacher] Unsupported TEACHER_BACKEND={settings.teacher_backend!r} "
                "(only 'openai' implemented; skipping)."
            )
        return

    excerpt = serialize_conversation_excerpt(
        conversation,
        max_messages=settings.teacher_max_excerpt_messages,
        max_chars_per_message=settings.teacher_max_chars_per_message,
    )

    acceptance = "\n".join(f"- {c}" for c in (task.goal.acceptance_criteria or [])) or "(none)"
    user_payload = f"""## Task
{task.goal.description}

## Acceptance criteria
{acceptance}

## Final status
{task.status.value}
Last error (if any): {task.last_error or "(none)"}

## Commands executed (last 20)
{json.dumps(task.commands_executed[-20:], ensure_ascii=False)}

## Files touched
{json.dumps([fc.path for fc in (task.files_changed or [])][-30:], ensure_ascii=False)}

## Agent final summary (may be truncated by caller)
{summary[:12000]}

## Conversation excerpt (truncated)
{json.dumps(excerpt, ensure_ascii=False, default=str)}
"""

    record_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    try:
        raw_teacher, teacher_obj = await _review_openai(
            settings=settings,
            user_payload=user_payload,
            verbose=verbose,
        )
    except Exception as e:
        teacher_obj = {
            "verdict": "issues",
            "issues": ["teacher_call_failed"],
            "mistake_analysis": str(e)[:2000],
            "correct_approach": "",
            "suggested_followups": [],
            "confidence": 0.0,
        }
        raw_teacher = ""

    record: dict[str, Any] = {
        "schema_version": "1.0",
        "id": record_id,
        "timestamp": ts,
        "task_id": task.task_id,
        "agent_model": agent_model,
        "teacher_model": settings.teacher_model,
        "teacher_backend": settings.teacher_backend,
        "task_status": task.status.value,
        "teacher_review": teacher_obj,
        "raw_teacher_response": raw_teacher[:16000] if raw_teacher else "",
        "fine_tune_exports": {
            "sft_teacher_critique_messages": _sft_messages_stub(
                task.goal.description, summary, teacher_obj
            ),
        },
    }

    log_path = Path(settings.training_data_log_path)
    if not log_path.is_absolute():
        log_path = Path.cwd() / log_path

    try:
        _append_jsonl(log_path, record)
        if verbose:
            print(f"[Teacher] Logged review to {log_path}")
    except Exception as e:
        if verbose:
            print(f"[Teacher] Failed to write training log: {e}")
