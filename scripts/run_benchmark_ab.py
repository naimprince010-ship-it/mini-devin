#!/usr/bin/env python3
"""A/B + shadow benchmark rollout runner.

Runs the same benchmark against two models and writes a comparison artifact.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from mini_devin.integrations.code_bench import run_code_benchmark


async def _run(args: argparse.Namespace) -> int:
    baseline = await run_code_benchmark(
        args.benchmark,
        limit=args.limit,
        mode="litellm",
        model=args.model_a,
        timeout=args.timeout,
    )
    challenger = await run_code_benchmark(
        args.benchmark,
        limit=args.limit,
        mode="litellm",
        model=args.model_b,
        timeout=args.timeout,
    )

    winner = "model_a"
    if challenger.pass_rate > baseline.pass_rate:
        winner = "model_b"
    elif challenger.pass_rate == baseline.pass_rate:
        a_latency = sum(r.duration_s for r in baseline.results) / max(1, len(baseline.results))
        b_latency = sum(r.duration_s for r in challenger.results) / max(1, len(challenger.results))
        winner = "model_b" if b_latency < a_latency else "model_a"

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": args.benchmark,
        "limit": args.limit,
        "timeout": args.timeout,
        "shadow_mode": args.shadow,
        "model_a": args.model_a,
        "model_b": args.model_b,
        "baseline": baseline.to_dict(),
        "challenger": challenger.to_dict(),
        "winner": winner,
        "rollout_recommendation": (
            "keep-model-a" if args.shadow and winner != "model_b" else
            "promote-model-b" if winner == "model_b" else
            "keep-model-a"
        ),
    }

    out_dir = Path("runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"benchmark_ab_{args.benchmark}_{args.limit}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"winner={winner}")
    print(f"model_a={args.model_a}, pass_rate={baseline.pass_rate}")
    print(f"model_b={args.model_b}, pass_rate={challenger.pass_rate}")
    print(f"saved={out_path}")

    if args.shadow:
        return 0
    return 0 if winner == "model_b" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run benchmark A/B or shadow rollout evaluation")
    parser.add_argument("--benchmark", choices=["mbpp", "humaneval"], default="mbpp")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--model-a", required=True, help="Current production model")
    parser.add_argument("--model-b", required=True, help="Challenger model")
    parser.add_argument("--shadow", action="store_true", help="Do not fail process when challenger loses")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
