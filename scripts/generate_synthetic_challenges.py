#!/usr/bin/env python3
"""
Generate synthetic coding challenges + pytest stubs for a repo (OpenAI-first).

Requires OPENAI_API_KEY. Optional: LLM_MODEL (default gpt-4o).

Later: point LLM_MODEL at OpenRouter/Moonshot via LiteLLM once those are wired
in the main app the same way.

Usage (from repo root ``mini-devin/``):
  python scripts/generate_synthetic_challenges.py --root . --count 3 --out ./data/synthetic
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path


def _load_json_object(text: str) -> dict:
    text = text.strip()
    try:
        out = json.loads(text)
        return out if isinstance(out, dict) else {}
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence:
        try:
            out = json.loads(fence.group(1).strip())
            return out if isinstance(out, dict) else {}
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{[\s\S]*\}\s*$", text)
    if m:
        try:
            out = json.loads(m.group(0))
            return out if isinstance(out, dict) else {}
        except json.JSONDecodeError:
            pass
    return {}


def _repo_digest(root: Path, max_files: int = 40) -> str:
    """Short text summary: top-level dirs + a few source files."""
    lines: list[str] = []
    for p in sorted(root.iterdir()):
        if p.name.startswith(".") or p.name in ("node_modules", "__pycache__", "dist", "build"):
            continue
        if p.is_dir():
            lines.append(f"dir: {p.name}/")
        elif p.suffix in {".py", ".ts", ".tsx", ".md"}:
            try:
                snippet = p.read_text(encoding="utf-8", errors="replace")[:1200]
                lines.append(f"file: {p.as_posix()}\n---\n{snippet}\n---")
            except OSError:
                continue
        if len(lines) >= max_files:
            break
    return "\n".join(lines)[:24000]


async def _generate_one(model: str, digest: str, index: int) -> dict:
    from mini_devin.core.llm_client import LLMClient, LLMConfig

    cfg = LLMConfig(model=model, temperature=0.3, max_tokens=4096)
    client = LLMClient(cfg)
    client.set_system_prompt(
        "You output ONLY valid JSON, no markdown. "
        "Each item must be realistic for the given codebase context."
    )
    client.add_user_message(
        f"""Project context (truncated):
{digest}

Produce ONE JSON object (not an array) with keys:
- "title": short string
- "description": coding task for a junior dev (2-6 sentences)
- "difficulty": "easy"|"medium"|"hard"
- "starter_notes": what files/functions to touch
- "test_file_content": full content of a single pytest file (test_challenge_{index}.py) with 2-5 tests that FAIL until the task is done. Use only stdlib + pytest. No network.
- "solution_sketch": brief bullet outline (not full code) of a correct approach

Challenge index: {index}
"""
    )
    resp = await client.complete(tools=None, stream=False)
    text = (resp.content or "").strip()
    if not text:
        return {}
    return _load_json_object(text)


async def main_async(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required.")

    model = os.environ.get("LLM_MODEL", "gpt-4o")
    digest = _repo_digest(root)
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.count):
        raw_path = out_dir / f"challenge_{i:03d}.raw.json"
        try:
            obj = await _generate_one(model, digest, i)
        except Exception as e:
            raw_path.write_text(json.dumps({"error": str(e)}), encoding="utf-8")
            print(f"[{i}] error -> {raw_path}", file=sys.stderr)
            continue
        raw_path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
        test_path = out_dir / f"challenge_{i:03d}_tests.py"
        tests = obj.get("test_file_content", "")
        if tests:
            test_path.write_text(tests, encoding="utf-8")
        print(f"[{i}] wrote {raw_path}" + (f" and {test_path}" if tests else ""))


def main() -> None:
    # Allow ``python scripts/foo.py`` from mini-devin root
    here = Path(__file__).resolve().parent.parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    p = argparse.ArgumentParser(description="Synthetic challenges via OpenAI")
    p.add_argument("--root", type=str, default=".", help="Repository root to summarize")
    p.add_argument("--count", type=int, default=3, help="Number of challenges")
    p.add_argument("--out", type=str, default="data/synthetic", help="Output directory")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
