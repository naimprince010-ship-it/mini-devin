#!/usr/bin/env python3
"""Run Plodder's HumanEval/MBPP code-generation benchmark."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mini_devin.integrations.code_bench import main


if __name__ == "__main__":
    raise SystemExit(main())
