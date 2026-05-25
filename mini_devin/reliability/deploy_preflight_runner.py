from __future__ import annotations

import json
from pathlib import Path

from .deploy_ops import run_deploy_preflight


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    report = run_deploy_preflight(repo_root=repo_root, startup_stage_history=[], mode="local")
    print(json.dumps(report.to_dict(), indent=2))
    return 1 if report.has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
