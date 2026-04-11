#!/usr/bin/env python3
"""Wrapper: poetry run python scripts/local_bridge.py (same as -m mini_devin.bridge.local_runner)."""

from mini_devin.bridge.local_runner import main

if __name__ == "__main__":
    main()
