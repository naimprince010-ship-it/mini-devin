#!/usr/bin/env python3
"""Mandatory benchmark regression gate for CI/CD.

Runs deterministic canonical benchmark smoke tests and optionally a litellm smoke test
when credentials are available.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mini_devin.integrations.code_bench import run_code_benchmark


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


async def _run(args: argparse.Namespace) -> int:
    mbpp = await run_code_benchmark("mbpp", limit=args.mbpp_limit, mode="canonical", timeout=args.timeout)
    humaneval = await run_code_benchmark("humaneval", limit=args.humaneval_limit, mode="canonical", timeout=args.timeout)

    checks: list[tuple[str, float, float]] = [
        ("mbpp-canonical", mbpp.pass_rate, args.mbpp_min_pass_rate),
        ("humaneval-canonical", humaneval.pass_rate, args.humaneval_min_pass_rate),
    ]

    litellm_run = None
    if args.litellm_smoke and os.getenv("OPENAI_API_KEY"):
        model = args.litellm_model or os.getenv("BENCHMARK_DEFAULT_MODEL") or os.getenv("LLM_MODEL") or ""
        if model:
            litellm_run = await run_code_benchmark(
                "mbpp",
                limit=args.litellm_limit,
                mode="litellm",
                model=model,
                timeout=args.timeout,
            )
            checks.append(("mbpp-litellm-smoke", litellm_run.pass_rate, args.litellm_min_pass_rate))

    summary = {
        "mbpp": mbpp.to_dict(),
        "humaneval": humaneval.to_dict(),
        "litellm_smoke": litellm_run.to_dict() if litellm_run else None,
        "checks": [
            {"name": name, "actual": actual, "minimum": minimum, "ok": actual >= minimum}
            for name, actual, minimum in checks
        ],
    }

    out_dir = Path("runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "regression_gate.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for check in summary["checks"]:
        print(
            f"[gate] {check['name']}: actual={check['actual']} minimum={check['minimum']} ok={check['ok']}"
        )
    print(f"[gate] summary saved: {out_path}")

    return 0 if all(item["ok"] for item in summary["checks"]) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run benchmark regression gate checks")
    parser.add_argument("--mbpp-limit", type=int, default=5)
    parser.add_argument("--humaneval-limit", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--mbpp-min-pass-rate", type=float, default=_float_env("REGRESSION_MBPP_MIN_PASS_RATE", 100.0))
    parser.add_argument(
        "--humaneval-min-pass-rate",
        type=float,
        default=_float_env("REGRESSION_HUMANEVAL_MIN_PASS_RATE", 100.0),
    )
    parser.add_argument("--litellm-smoke", action="store_true", default=_bool_env("REGRESSION_LITELLM_SMOKE", False))
    parser.add_argument("--litellm-limit", type=int, default=5)
    parser.add_argument(
        "--litellm-min-pass-rate",
        type=float,
        default=_float_env("REGRESSION_LITELLM_MIN_PASS_RATE", 70.0),
    )
    parser.add_argument("--litellm-model", default=os.getenv("BENCHMARK_DEFAULT_MODEL", ""))
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
