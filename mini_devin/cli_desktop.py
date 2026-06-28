"""Desktop helper entrypoints for local Electron app."""

from __future__ import annotations

import subprocess
from pathlib import Path


def _frontend_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "frontend"


def run_desktop_dev() -> None:
    """Run desktop app in development mode (Vite + Electron)."""
    subprocess.run(["npm", "run", "dev:desktop"], cwd=_frontend_dir(), check=True, shell=True)


def run_desktop_prod() -> None:
    """Build frontend and run desktop app in production mode."""
    subprocess.run(["npm", "run", "desktop"], cwd=_frontend_dir(), check=True, shell=True)
