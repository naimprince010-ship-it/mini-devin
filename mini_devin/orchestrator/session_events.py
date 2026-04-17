"""
Append-only session event stream for resume / dashboards (OpenHands-style).

Events are written to ``<workspace>/.plodder/session_events.jsonl`` (one JSON object per line).

``think`` / ``observe`` rows may include LLM token totals and :func:`estimate_llm_cost_usd` metadata
(configurable via ``LLM_ESTIMATE_PRICE_*`` env vars — see that function's docstring).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def estimate_llm_cost_usd(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> tuple[float | None, str]:
    """
    Approximate USD spend from token counts (dashboard / JSONL only — not a bill).

    Override with env (USD per **million** tokens):

    - ``LLM_ESTIMATE_PRICE_INPUT_PER_MTOK_USD``
    - ``LLM_ESTIMATE_PRICE_OUTPUT_PER_MTOK_USD``

    Otherwise tries ``litellm.completion_cost`` when available, then a small
    built-in table for common API models.
    """
    if prompt_tokens < 0 or completion_tokens < 0:
        return None, "invalid_token_counts"

    inp = os.environ.get("LLM_ESTIMATE_PRICE_INPUT_PER_MTOK_USD", "").strip()
    outp = os.environ.get("LLM_ESTIMATE_PRICE_OUTPUT_PER_MTOK_USD", "").strip()
    if inp and outp:
        try:
            pi = float(inp)
            po = float(outp)
            cost = (prompt_tokens / 1_000_000.0) * pi + (completion_tokens / 1_000_000.0) * po
            return max(0.0, cost), "env_per_million_tokens"
        except ValueError:
            pass

    try:
        import litellm  # type: ignore

        fn = getattr(litellm, "completion_cost", None)
        if callable(fn):
            try:
                c = fn(model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
            except TypeError:
                c = None
            if c is not None and c >= 0:
                return float(c), "litellm.completion_cost"
    except Exception:
        pass

    m = (model or "").lower().replace(" ", "")
    if "ollama" in m or m.startswith("ollama/"):
        return 0.0, "local_model_assumed_zero"

    # USD per 1M input / output tokens — rough defaults when litellm/env unavailable
    table: list[tuple[str, float, float]] = [
        ("gpt-4o-mini", 0.15, 0.6),
        ("gpt-4o", 2.5, 10.0),
        ("gpt-4-turbo", 10.0, 30.0),
        ("gpt-3.5-turbo", 0.5, 1.5),
        ("claude-3-5-sonnet-20241022", 3.0, 15.0),
        ("claude-3-5-sonnet", 3.0, 15.0),
        ("claude-3-opus", 15.0, 75.0),
        ("claude-3-haiku", 0.25, 1.25),
        ("claude-3-sonnet", 3.0, 15.0),
    ]
    for key, pi, po in table:
        if key in m:
            cost = (prompt_tokens / 1_000_000.0) * pi + (completion_tokens / 1_000_000.0) * po
            return max(0.0, cost), f"approx_table:{key}"

    return None, "unknown_model"


def append_session_event(workspace: str | Path, event: dict[str, Any]) -> dict[str, Any] | None:
    """
    Append one JSON line.

    Returns the full row dict (including ``ts``) on success, or ``None`` on failure.
    """
    try:
        root = Path(workspace).resolve()
        root.mkdir(parents=True, exist_ok=True)
        log_dir = root / ".plodder"
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / "session_events.jsonl"
        row: dict[str, Any] = {"ts": datetime.now(timezone.utc).isoformat(), **event}
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
        return row
    except OSError:
        return None


def load_session_events(workspace: str | Path, *, max_lines: int = 800) -> list[dict[str, Any]]:
    """Read the tail of the JSONL event stream for resume / context injection."""
    path = Path(workspace).resolve() / ".plodder" / "session_events.jsonl"
    if not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    tail = lines[-max_lines:]
    out: list[dict[str, Any]] = []
    for line in tail:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
