#!/usr/bin/env python3
"""
Temporary Railway / local smoke test for ProcessSandbox (no Docker).

Run from repo root:
  python prod_test.py

Uses stdout logging so Railway ``railway run`` / release logs show each step.
"""

from __future__ import annotations

import json
import logging
import sys

from mini_devin.sandbox.railway_process_sandbox_test import run_railway_process_sandbox_check


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
        force=True,
    )
    result = run_railway_process_sandbox_check()
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
