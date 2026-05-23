#!/usr/bin/env python3
"""Build unified benchmark observability dashboard (quality + latency + cost).

Reads data/code_bench/*.json run artifacts and writes a compact dashboard JSON.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


def _model_cost_per_1k_output_tokens() -> dict[str, float]:
    # Format: "model_a=0.01,model_b=0.003"
    raw = os.getenv("BENCHMARK_MODEL_PRICE_PER_1K_OUTPUT_TOKENS", "")
    table: dict[str, float] = {}
    for part in raw.split(","):
        item = part.strip()
        if not item or "=" not in item:
            continue
        model, price = item.split("=", 1)
        try:
            table[model.strip().lower()] = float(price.strip())
        except ValueError:
            continue
    return table


def _estimate_output_tokens(text: str) -> int:
    # Simple conservative approximation for dashboarding.
    return max(0, len(text or "") // 4)


def _load_runs(data_dir: Path) -> list[dict]:
    runs = []
    for path in sorted(data_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            runs.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return runs


def build_dashboard(data_dir: Path, max_runs: int) -> dict:
    runs = _load_runs(data_dir)[:max_runs]
    prices = _model_cost_per_1k_output_tokens()

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for run in runs:
        grouped[(run.get("benchmark", "unknown"), run.get("mode", "unknown"))].append(run)

    cards = []
    for (benchmark, mode), items in grouped.items():
        pass_rates = [float(r.get("pass_rate", 0.0) or 0.0) for r in items]
        durations = []
        est_costs = []
        model_usage: dict[str, int] = defaultdict(int)

        for run in items:
            for result in run.get("results", []):
                durations.append(float(result.get("duration_s", 0.0) or 0.0))
                code = result.get("generated_code", "") or ""
                tokens = _estimate_output_tokens(code)
                model_name = (run.get("model") or "").strip().lower()
                if model_name:
                    model_usage[model_name] += 1
                price = prices.get(model_name, 0.0)
                est_costs.append((tokens / 1000.0) * price if price > 0 else 0.0)

        cards.append(
            {
                "benchmark": benchmark,
                "mode": mode,
                "runs": len(items),
                "quality": {
                    "avg_pass_rate": round(mean(pass_rates), 2) if pass_rates else 0.0,
                    "max_pass_rate": round(max(pass_rates), 2) if pass_rates else 0.0,
                    "min_pass_rate": round(min(pass_rates), 2) if pass_rates else 0.0,
                },
                "latency": {
                    "avg_task_duration_s": round(mean(durations), 3) if durations else 0.0,
                    "p95_task_duration_s": round(sorted(durations)[int(0.95 * (len(durations) - 1))], 3)
                    if durations
                    else 0.0,
                },
                "cost": {
                    "estimated_total_usd": round(sum(est_costs), 6),
                    "estimated_avg_per_task_usd": round(mean(est_costs), 6) if est_costs else 0.0,
                    "pricing_source": "BENCHMARK_MODEL_PRICE_PER_1K_OUTPUT_TOKENS",
                },
                "model_usage": dict(model_usage),
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(data_dir),
        "cards": cards,
        "total_runs": len(runs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build benchmark observability dashboard")
    parser.add_argument("--data-dir", default="data/code_bench")
    parser.add_argument("--max-runs", type=int, default=200)
    parser.add_argument("--out", default="runs/benchmark_dashboard.json")
    args = parser.parse_args()

    dashboard = build_dashboard(Path(args.data_dir), args.max_runs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dashboard, indent=2), encoding="utf-8")
    print(f"saved={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
