"""
Context condenser — OpenHands-style long-conversation handling.

When history grows, replace the *middle* segment with a single summary while
keeping the initial goal (system + first user) and the last N tool
observations (with valid assistant/tool ordering) verbatim.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mini_devin.core.llm_client import LLMClient


def _tool_call_ids_for_assistant(msg: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for tc in msg.get("tool_calls") or []:
        if isinstance(tc, dict) and tc.get("id"):
            out.add(str(tc["id"]))
    return out


def find_suffix_start_for_last_n_tool_observations(
    messages: list[dict[str, Any]],
    *,
    last_n: int = 3,
) -> int | None:
    """
    Smallest index ``k`` such that ``messages[k:]`` is API-valid and includes
    the last ``last_n`` ``role=tool`` messages in full (each preceded by its
    assistant ``tool_calls`` block).
    """
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if not tool_indices:
        return None
    keep = tool_indices[-last_n:] if len(tool_indices) >= last_n else tool_indices
    first_keep = keep[0]
    tcid = messages[first_keep].get("tool_call_id")
    if not tcid:
        return first_keep
    for i in range(first_keep, -1, -1):
        m = messages[i]
        if m.get("role") != "assistant":
            continue
        if str(tcid) in _tool_call_ids_for_assistant(m):
            return i
    return first_keep


def split_goal_prefix(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Leading ``system`` messages plus the first ``user`` message (initial goal)."""
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(messages) and messages[i].get("role") == "system":
        out.append(messages[i])
        i += 1
    if i < len(messages) and messages[i].get("role") == "user":
        out.append(messages[i])
        i += 1
    return out, i


def _serialized_len(msgs: list[dict[str, Any]]) -> int:
    try:
        return len(json.dumps(msgs, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return sum(len(str(m)) for m in msgs)


async def condense_chat_messages(
    messages: list[dict[str, Any]],
    *,
    summarizer: LLMClient,
    last_n_observations: int | None = None,
    min_messages: int | None = None,
    min_chars: int | None = None,
) -> list[dict[str, Any]]:
    """
    If thresholds exceeded, replace ``middle`` with one ``user`` summary message.

    Prefix = system + first user. Suffix = from assistant issuing the Nth-last
    tool observation through end. Requires at least one ``tool`` message.
    """
    if not messages:
        return messages

    n_obs = last_n_observations
    if n_obs is None:
        raw = (os.environ.get("LLM_CONDENSE_LAST_OBSERVATIONS") or "").strip()
        n_obs = int(raw) if raw.isdigit() else 3

    min_m = min_messages
    if min_m is None:
        raw_m = (os.environ.get("LLM_CONDENSE_MIN_MESSAGES") or "").strip()
        min_m = int(raw_m) if raw_m.isdigit() else 40

    min_c = min_chars
    if min_c is None:
        raw_c = (os.environ.get("LLM_CONDENSE_MIN_CHARS") or "").strip()
        min_c = int(raw_c) if raw_c.isdigit() else 120_000

    suffix_start = find_suffix_start_for_last_n_tool_observations(messages, last_n=n_obs)
    if suffix_start is None:
        return messages

    prefix, prefix_end = split_goal_prefix(messages)
    if prefix_end >= suffix_start:
        return messages

    middle = messages[prefix_end:suffix_start]
    if len(middle) < 2:
        return messages

    if len(messages) < min_m and _serialized_len(messages) < min_c:
        return messages

    raw_middle = json.dumps(middle, ensure_ascii=False, default=str)
    cap = int((os.environ.get("LLM_CONDENSE_INPUT_MAX_CHARS") or "200000").strip() or "200000")
    if len(raw_middle) > cap:
        raw_middle = raw_middle[:cap] + "\n\n...(middle JSON truncated before summarization)"

    instruction = (
        "Summarize the following agent transcript segment for a coding agent that continues next. "
        "Preserve exact file paths, shell commands, exit codes, fingerprints, and error text where present. "
        "Use concise markdown sections (###). Omit pleasantries."
    )
    try:
        summary = await summarizer.completion_ephemeral(
            [
                {"role": "system", "content": instruction},
                {"role": "user", "content": raw_middle},
            ],
            temperature=0.15,
            max_tokens=min(8192, summarizer.config.max_tokens),
        )
    except Exception:
        summary = (
            "(Condenser summarization failed; middle truncated.)\n\n"
            + raw_middle[:8000]
            + ("\n…" if len(raw_middle) > 8000 else "")
        )

    condensed_user = {
        "role": "user",
        "content": "## Condensed prior context (middle segment)\n\n" + (summary.strip() or "(empty summary)"),
    }
    suffix = messages[suffix_start:]
    return [*prefix, condensed_user, *suffix]
