from __future__ import annotations

import asyncio

from mini_devin.integrations import code_bench


def test_humaneval_canonical_solution_passes(monkeypatch) -> None:
    task = code_bench.CodeBenchTask(
        task_id="HumanEval/0",
        benchmark="humaneval",
        prompt="def add(a, b):\n    \"\"\"Return sum.\"\"\"\n",
        canonical_solution="    return a + b\n",
        tests="assert add(2, 3) == 5",
        entry_point="add",
    )
    monkeypatch.setattr(code_bench, "load_tasks", lambda benchmark, limit=10: [task])

    run = asyncio.run(code_bench.run_code_benchmark("humaneval", limit=1, mode="canonical"))

    assert run.passed == 1
    assert run.pass_rate == 100.0


def test_mbpp_canonical_solution_passes(monkeypatch) -> None:
    task = code_bench.CodeBenchTask(
        task_id="mbpp-1",
        benchmark="mbpp",
        prompt="Write a function to square a number.",
        canonical_solution="def square(x):\n    return x * x\n",
        tests="assert square(4) == 16",
    )
    monkeypatch.setattr(code_bench, "load_tasks", lambda benchmark, limit=10: [task])

    run = asyncio.run(code_bench.run_code_benchmark("mbpp", limit=1, mode="canonical"))

    assert run.passed == 1


def test_failing_solution_fails() -> None:
    passed, output = code_bench.run_python_tests("def add(a, b):\n    return a - b", "assert add(2, 3) == 5")

    assert passed is False
    assert "AssertionError" in output


def test_strip_markdown_code() -> None:
    raw = "Here:\n```python\ndef f():\n    return 1\n```"

    assert code_bench.strip_markdown_code(raw) == "def f():\n    return 1"
