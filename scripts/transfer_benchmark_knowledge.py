#!/usr/bin/env python3
"""Structured exam-to-chat knowledge transfer pipeline.

Converts benchmark run artifacts into compact lessons files that chat sessions can inject.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _classify(result: dict) -> str:
    blob = ((result.get("test_output") or "") + " " + (result.get("error") or "")).lower()
    if "syntaxerror" in blob:
        return "syntax"
    if "timed out" in blob or "timeoutexpired" in blob:
        return "timeout"
    if "missing required function" in blob or "nameerror" in blob:
        return "missing_function"
    if "assertionerror" in blob:
        return "assertion"
    if "traceback" in blob:
        return "runtime"
    return "unknown"


def _load_run(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_run(data_dir: Path) -> Path:
    paths = sorted(data_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not paths:
        raise FileNotFoundError(f"no benchmark runs found under {data_dir}")
    return paths[0]


def _build_lessons(run: dict) -> tuple[dict, str]:
    results = run.get("results", [])
    failed = [r for r in results if not r.get("passed")]
    by_type = Counter(_classify(r) for r in failed)

    rules = [
        "Return only valid Python code, never markdown fences.",
        "Define exact public function names expected by tests.",
        "For assertion-sensitive tasks, match exact output shape and ordering.",
        "Prefer deterministic tie-breakers and stable ordering when ranking items.",
    ]
    if by_type.get("timeout", 0) > 0:
        rules.append("Avoid expensive nested loops when linear alternatives exist.")
    if by_type.get("runtime", 0) > 0:
        rules.append("Harden edge-case checks to prevent runtime exceptions.")

    lesson_json = {
        "run_id": run.get("run_id"),
        "benchmark": run.get("benchmark"),
        "total": run.get("total"),
        "passed": run.get("passed"),
        "pass_rate": run.get("pass_rate"),
        "failed_by_type": dict(by_type),
        "rules": rules,
        "failed_tasks": [r.get("task_id") for r in failed],
    }

    md_lines = [
        "# Benchmark Lessons for Chat",
        "",
        f"- Source run: {run.get('run_id')}",
        f"- Benchmark: {run.get('benchmark')}",
        f"- Pass rate: {run.get('pass_rate')}%",
        "",
        "## Reliability Rules",
    ]
    md_lines.extend(f"- {rule}" for rule in rules)
    md_lines.append("")
    md_lines.append("## Failure Profile")
    for k, v in sorted(by_type.items()):
        md_lines.append(f"- {k}: {v}")

    return lesson_json, "\n".join(md_lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Transfer benchmark knowledge into chat lessons")
    parser.add_argument("--run", default="", help="Path to run JSON (default: latest in data/code_bench)")
    parser.add_argument("--data-dir", default="data/code_bench")
    parser.add_argument("--json-out", default="knowledge_base/benchmark_lessons.json")
    parser.add_argument("--md-out", default="knowledge_base/benchmark_lessons.md")
    args = parser.parse_args()

    run_path = Path(args.run) if args.run else _latest_run(Path(args.data_dir))
    run = _load_run(run_path)

    lesson_json, lesson_md = _build_lessons(run)

    json_out = Path(args.json_out)
    md_out = Path(args.md_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(lesson_json, indent=2), encoding="utf-8")
    md_out.write_text(lesson_md, encoding="utf-8")

    print(f"source={run_path}")
    print(f"json_out={json_out}")
    print(f"md_out={md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
