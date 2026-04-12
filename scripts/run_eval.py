#!/usr/bin/env python3
"""Run PR eval suite: fixed regression scenarios (no live LLM). Used by CI and locally."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "tests" / "eval" / "scenarios.json"


def main() -> int:
    p = argparse.ArgumentParser(description="Run Plodder eval pytest suite")
    p.add_argument(
        "--list",
        action="store_true",
        help="Print scenario ids from scenarios.json and exit",
    )
    args = p.parse_args()

    if args.list:
        data = json.loads(SCENARIOS.read_text(encoding="utf-8"))
        for s in data.get("scenarios", []):
            print(f"{s['id']}\t{s['pytest']}")
        return 0

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(ROOT / "tests" / "eval"),
        "-v",
        "--tb=short",
    ]
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
