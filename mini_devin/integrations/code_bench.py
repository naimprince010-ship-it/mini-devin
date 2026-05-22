"""HumanEval/MBPP style code-generation benchmark harness for Plodder."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


_DATA_DIR = Path(os.environ.get("PLODDER_DATA") or os.environ.get("MINI_DEVIN_DATA", "data")) / "code_bench"
_DATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class CodeBenchTask:
    task_id: str
    benchmark: str
    prompt: str
    tests: str
    entry_point: str = ""
    canonical_solution: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CodeBenchResult:
    task_id: str
    benchmark: str
    passed: bool
    generated_code: str = ""
    test_output: str = ""
    error: str = ""
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CodeBenchRun:
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    benchmark: str = "humaneval"
    mode: str = "canonical"
    limit: int = 10
    status: str = "pending"
    started_at: str = ""
    finished_at: str = ""
    total: int = 0
    passed: int = 0
    results: list[CodeBenchResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return round((self.passed / self.total) * 100, 1) if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["pass_rate"] = self.pass_rate
        d["results"] = [r.to_dict() for r in self.results]
        return d


def _load_dataset_candidates(names: list[str], split: str):
    last_error: Exception | None = None
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        raise RuntimeError("Install the `datasets` package to load HumanEval/MBPP.") from exc

    for name in names:
        try:
            return load_dataset(name, split=split)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not load dataset candidates {names}: {last_error}")


def load_humaneval_tasks(limit: int = 10) -> list[CodeBenchTask]:
    ds = _load_dataset_candidates(["openai/openai_humaneval", "openai_humaneval"], "test")
    tasks: list[CodeBenchTask] = []
    for row in ds:
        task_id = str(row.get("task_id") or f"humaneval-{len(tasks)}")
        prompt = str(row.get("prompt") or "")
        test = str(row.get("test") or "")
        entry_point = str(row.get("entry_point") or "")
        tests = test + f"\n\ncheck({entry_point})\n"
        tasks.append(
            CodeBenchTask(
                task_id=task_id,
                benchmark="humaneval",
                prompt=prompt,
                tests=tests,
                entry_point=entry_point,
                canonical_solution=str(row.get("canonical_solution") or ""),
                metadata={"source": "openai/openai_humaneval"},
            )
        )
        if len(tasks) >= limit:
            break
    return tasks


def load_mbpp_tasks(limit: int = 10, split: str = "test") -> list[CodeBenchTask]:
    ds = _load_dataset_candidates(["google-research-datasets/mbpp", "mbpp"], split)
    tasks: list[CodeBenchTask] = []
    for row in ds:
        task_id = str(row.get("task_id") or row.get("id") or f"mbpp-{len(tasks)}")
        prompt = str(row.get("text") or row.get("prompt") or "")
        tests_raw = row.get("test_list") or row.get("test") or []
        tests = "\n".join(str(t) for t in tests_raw) if isinstance(tests_raw, list) else str(tests_raw)
        setup = str(row.get("test_setup_code") or "")
        if setup:
            tests = setup + "\n" + tests
        tasks.append(
            CodeBenchTask(
                task_id=f"mbpp-{task_id}",
                benchmark="mbpp",
                prompt=prompt,
                tests=tests,
                canonical_solution=str(row.get("code") or ""),
                metadata={"source": "google-research-datasets/mbpp"},
            )
        )
        if len(tasks) >= limit:
            break
    return tasks


def load_tasks(benchmark: str, limit: int = 10) -> list[CodeBenchTask]:
    key = benchmark.strip().lower()
    if key in {"humaneval", "human-eval"}:
        return load_humaneval_tasks(limit)
    if key == "mbpp":
        return load_mbpp_tasks(limit)
    raise ValueError("benchmark must be humaneval or mbpp")


def strip_markdown_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def canonical_generator(task: CodeBenchTask) -> str:
    return task.prompt + "\n" + task.canonical_solution if task.benchmark == "humaneval" else task.canonical_solution


async def litellm_generator(task: CodeBenchTask, model: str = "") -> str:
    import litellm  # type: ignore

    model_id = model or os.environ.get("LLM_MODEL") or "gpt-4o-mini"
    prompt = (
        "Solve this Python programming task. Return only valid Python code, no markdown.\n\n"
        f"Task:\n{task.prompt}\n\n"
        "The hidden tests will be run after your code. Include all required imports and helpers."
    )
    response = await litellm.acompletion(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=2048,
    )
    return strip_markdown_code(response.choices[0].message.content or "")


def run_python_tests(code: str, tests: str, *, timeout: int = 10) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="plodder_code_bench_") as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text(code.rstrip() + "\n\n" + tests.rstrip() + "\n", encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(path)],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode == 0, output[-4000:]


async def run_code_benchmark(
    benchmark: str,
    *,
    limit: int = 10,
    mode: str = "canonical",
    model: str = "",
    timeout: int = 10,
    run_id: str = "",
    generator: Callable[[CodeBenchTask], Any] | None = None,
) -> CodeBenchRun:
    tasks = load_tasks(benchmark, limit)
    run = CodeBenchRun(
        run_id=run_id or str(uuid.uuid4())[:12],
        benchmark=benchmark,
        mode=mode,
        limit=limit,
        status="running",
        started_at=datetime.now(timezone.utc).isoformat(),
        total=len(tasks),
    )
    save_run(run)

    for task in tasks:
        t0 = time.time()
        try:
            if generator:
                generated = await generator(task)
            elif mode == "canonical":
                generated = canonical_generator(task)
            elif mode == "litellm":
                generated = await litellm_generator(task, model=model)
            else:
                raise ValueError("mode must be canonical or litellm")
            passed, output = run_python_tests(generated, task.tests, timeout=timeout)
            result = CodeBenchResult(
                task_id=task.task_id,
                benchmark=task.benchmark,
                passed=passed,
                generated_code=generated,
                test_output=output,
                duration_s=round(time.time() - t0, 2),
            )
        except Exception as exc:
            result = CodeBenchResult(
                task_id=task.task_id,
                benchmark=task.benchmark,
                passed=False,
                error=str(exc),
                duration_s=round(time.time() - t0, 2),
            )
        run.results.append(result)
        if result.passed:
            run.passed += 1
        save_run(run)

    run.status = "completed"
    run.finished_at = datetime.now(timezone.utc).isoformat()
    save_run(run)
    return run


def save_run(run: CodeBenchRun) -> Path:
    path = _DATA_DIR / f"{run.run_id}.json"
    path.write_text(json.dumps(run.to_dict(), indent=2), encoding="utf-8")
    return path


def load_run(run_id: str) -> dict[str, Any] | None:
    path = _DATA_DIR / f"{run_id}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in sorted(_DATA_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            runs.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return runs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HumanEval/MBPP code benchmark")
    parser.add_argument("--benchmark", choices=["humaneval", "mbpp"], default="humaneval")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--mode", choices=["canonical", "litellm"], default="canonical")
    parser.add_argument("--model", default="")
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    return parser


async def async_main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run = await run_code_benchmark(
        args.benchmark,
        limit=args.limit,
        mode=args.mode,
        model=args.model,
        timeout=args.timeout,
    )
    payload = run.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Run: {run.run_id}")
        print(f"Benchmark: {run.benchmark} | mode: {run.mode}")
        print(f"Passed: {run.passed}/{run.total} ({run.pass_rate}%)")
        print(f"Saved: {_DATA_DIR / (run.run_id + '.json')}")
    return 0 if run.passed == run.total else 1


def main(argv: list[str] | None = None) -> int:
    import asyncio

    return asyncio.run(async_main(argv))
