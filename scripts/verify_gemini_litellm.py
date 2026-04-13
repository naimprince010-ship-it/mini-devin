"""
Smoke test: one non-streaming LiteLLM completion via Plodder's LLM client (Gemini).

Usage (from repo root):
  poetry run python scripts/verify_gemini_litellm.py

Requires GEMINI_API_KEY or GOOGLE_API_KEY in .env (replace YOUR_KEY_HERE).
Exit 0 on success, 1 on API/runtime failure, 2 if no usable API key.
"""

from __future__ import annotations

import asyncio
import os
import sys


async def _run() -> int:
    from dotenv import load_dotenv

    load_dotenv()

    key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    if not key or key == "YOUR_KEY_HERE":
        print(
            "No usable Gemini key: set GEMINI_API_KEY in .env (real key, not YOUR_KEY_HERE).",
            file=sys.stderr,
        )
        return 2

    import mini_devin.core.providers as prov

    prov._registry = None

    from mini_devin.core.llm_client import create_llm_client

    model = (os.environ.get("LLM_MODEL") or "gemini/gemini-1.5-flash").strip()
    client = create_llm_client(model=model)
    client.set_system_prompt("You are a test harness. Reply with exactly one word: OK")
    client.add_user_message("Ping.")
    response = await client.complete(tools=None, stream=False)
    text = (response.content or "").strip()
    print("model_returned:", response.model)
    print("content:", text[:500])
    if not text:
        print("Empty completion body.", file=sys.stderr)
        return 1
    return 0


def main() -> None:
    sys.exit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
