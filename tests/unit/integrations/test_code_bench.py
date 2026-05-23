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


def test_strip_markdown_code_removes_stray_fence() -> None:
    raw = "def f():\n    return 1\n```"

    assert code_bench.strip_markdown_code(raw) == "def f():\n    return 1"


def test_infer_test_entry_points_from_asserts() -> None:
    tests = "\n".join(
        [
            "assert remove_first_last_occurrence([1, 2, 1], 1) == [2]",
            "assert helper(10) == 11",
            "assert remove_first_last_occurrence([7, 8, 7], 7) == [8]",
        ]
    )

    assert code_bench.infer_test_entry_points(tests) == ["remove_first_last_occurrence", "helper"]


def test_validate_candidate_code_reports_missing_required_function() -> None:
    task = code_bench.CodeBenchTask(
        task_id="mbpp-xyz",
        benchmark="mbpp",
        prompt="Write a function that removes first and last occurrence.",
        canonical_solution="",
        tests="assert remove_first_last_occurrence([1, 2, 1], 1) == [2]",
    )

    issue = code_bench.validate_candidate_code(task, "def other_name(x):\n    return x")

    assert issue is not None
    assert "Missing required function name(s)" in issue


def test_classify_failure_categories() -> None:
    assert code_bench.classify_failure("SyntaxError: invalid syntax") == "syntax"
    assert code_bench.classify_failure("Timed out after 20 seconds") == "timeout"
    assert code_bench.classify_failure("NameError: name 'f' is not defined") == "missing_function"
    assert code_bench.classify_failure("AssertionError") == "assertion"
    assert code_bench.classify_failure("Traceback (most recent call last): ...") == "runtime"


def test_build_litellm_repair_prompt_includes_failure_guidance() -> None:
    task = code_bench.CodeBenchTask(
        task_id="mbpp-guidance",
        benchmark="mbpp",
        prompt="Count common colors.",
        canonical_solution="",
        tests="assert count_common(['a', 'a', 'b']) == [('a', 2), ('b', 1)]",
    )

    prompt = code_bench.build_litellm_repair_prompt(
        task,
        "def count_common(words):\n    return []",
        "AssertionError",
    )

    assert "Failure type: assertion" in prompt
    assert "deterministic ordering" in prompt


def test_extract_assertion_expected_value() -> None:
    output = (
        "Traceback ...\n"
        "assert count_common(['a', 'a', 'b']) == [('a', 2), ('b', 1)]\n"
        "AssertionError\n"
    )

    expected = code_bench.extract_assertion_expected_value(output)

    assert expected == "[('a', 2), ('b', 1)]"


def test_failure_guidance_assertion_includes_expected_hint() -> None:
    output = "assert fn([1,2,2]) == [2,1]\nAssertionError"

    guidance = code_bench.failure_guidance("assertion", output)

    assert "One failing assertion expects" in guidance
    assert "[2,1]" in guidance


def test_failure_guidance_assertion_includes_stable_tie_break_hint() -> None:
    output = "assert count_common(['a','b','a','b']) == [('a', 2), ('b', 2)]\nAssertionError"

    guidance = code_bench.failure_guidance("assertion", output)

    assert "stable tie-break" in guidance
    assert "not alphabetical order" in guidance


def test_failure_guidance_assertion_includes_expected_length_hint() -> None:
    output = "assert fn([1,2,3]) == [('a', 2), ('b', 1)]\nAssertionError"

    guidance = code_bench.failure_guidance("assertion", output)

    assert "Return exactly 2 item(s)" in guidance


def test_build_litellm_prompt_includes_inferred_signature_hint() -> None:
    task = code_bench.CodeBenchTask(
        task_id="mbpp-xyz",
        benchmark="mbpp",
        prompt="Write a function that removes first and last occurrence.",
        canonical_solution="",
        tests="assert remove_first_last_occurrence([1, 2, 1], 1) == [2]",
    )

    prompt = code_bench.build_litellm_prompt(task)

    assert "Required public function name(s): `remove_first_last_occurrence`" in prompt


def test_build_litellm_repair_prompt_includes_failure_context() -> None:
    task = code_bench.CodeBenchTask(
        task_id="mbpp-xyz",
        benchmark="mbpp",
        prompt="Write a function that removes first and last occurrence.",
        canonical_solution="",
        tests="assert remove_first_last_occurrence([1, 2, 1], 1) == [2]",
    )

    prompt = code_bench.build_litellm_repair_prompt(
        task,
        "def remove_occ(xs, x):\n    return xs",
        "NameError: name 'remove_first_last_occurrence' is not defined",
    )

    assert "Required public function name(s): `remove_first_last_occurrence`" in prompt
    assert "Observed test failure output" in prompt
    assert "Your previous code" in prompt


def test_litellm_mode_retries_once_with_repair(monkeypatch) -> None:
    task = code_bench.CodeBenchTask(
        task_id="mbpp-1",
        benchmark="mbpp",
        prompt="Write a function to square a number.",
        canonical_solution="",
        tests="assert square(4) == 16",
    )
    monkeypatch.setattr(code_bench, "load_tasks", lambda benchmark, limit=10: [task])
    monkeypatch.setattr(code_bench, "code_bench_best_of_candidates", lambda: 1)

    calls: list[str] = []

    async def fake_litellm_generator(_task, model=""):
        calls.append("first")
        return "def sqr(x):\n    return x * x\n"

    async def fake_repair_generator(_task, previous_code, test_output, model=""):
        assert "def sqr" in previous_code
        assert "Missing required function name(s): square" in test_output
        calls.append("repair")
        return "def square(x):\n    return x * x\n"

    monkeypatch.setattr(code_bench, "litellm_generator", fake_litellm_generator)
    monkeypatch.setattr(code_bench, "litellm_repair_generator", fake_repair_generator)

    run = asyncio.run(code_bench.run_code_benchmark("mbpp", limit=1, mode="litellm", model="fake/model"))

    assert run.passed == 1
    assert calls == ["first", "repair"]


def test_run_python_tests_timeout_returns_failure() -> None:
    passed, output = code_bench.run_python_tests("while True:\n    pass", "assert True", timeout=1)

    assert passed is False
    assert "Timed out" in output


def test_litellm_mode_retries_twice_when_needed(monkeypatch) -> None:
    task = code_bench.CodeBenchTask(
        task_id="mbpp-2",
        benchmark="mbpp",
        prompt="Write a function to square a number.",
        canonical_solution="",
        tests="assert square(4) == 16",
    )
    monkeypatch.setattr(code_bench, "load_tasks", lambda benchmark, limit=10: [task])
    monkeypatch.setattr(code_bench, "code_bench_best_of_candidates", lambda: 1)

    calls: list[str] = []

    async def fake_litellm_generator(_task, model=""):
        calls.append("first")
        return "def sqr(x):\n    return x * x\n"

    async def fake_repair_generator(_task, previous_code, test_output, model=""):
        if "def sqr" in previous_code:
            calls.append("repair1")
            return "def square(x):\n    return x - x\n"
        calls.append("repair2")
        return "def square(x):\n    return x * x\n"

    monkeypatch.setattr(code_bench, "litellm_generator", fake_litellm_generator)
    monkeypatch.setattr(code_bench, "litellm_repair_generator", fake_repair_generator)

    run = asyncio.run(code_bench.run_code_benchmark("mbpp", limit=1, mode="litellm", model="fake/model"))

    assert run.passed == 1
    assert calls == ["first", "repair1", "repair2"]


def test_litellm_mode_uses_adaptive_timeout_after_timeout_output(monkeypatch) -> None:
    task = code_bench.CodeBenchTask(
        task_id="mbpp-3",
        benchmark="mbpp",
        prompt="Write a function to square a number.",
        canonical_solution="",
        tests="assert square(4) == 16",
    )
    monkeypatch.setattr(code_bench, "load_tasks", lambda benchmark, limit=10: [task])
    monkeypatch.setattr(code_bench, "code_bench_best_of_candidates", lambda: 1)

    async def fake_litellm_generator(_task, model=""):
        return "def square(x):\n    return x * x\n"

    async def fake_repair_generator(_task, previous_code, test_output, model=""):
        return previous_code

    timeouts: list[int] = []

    def fake_run_python_tests(_code, _tests, *, timeout=10):
        timeouts.append(timeout)
        if len(timeouts) == 1:
            return False, "Timed out after 20 seconds"
        return True, ""

    monkeypatch.setattr(code_bench, "litellm_generator", fake_litellm_generator)
    monkeypatch.setattr(code_bench, "litellm_repair_generator", fake_repair_generator)
    monkeypatch.setattr(code_bench, "run_python_tests", fake_run_python_tests)

    run = asyncio.run(code_bench.run_code_benchmark("mbpp", limit=1, mode="litellm", model="fake/model", timeout=20))

    assert run.passed == 1
    assert timeouts == [20, 40]


def test_litellm_mode_best_of_candidates_passes_without_repair(monkeypatch) -> None:
    task = code_bench.CodeBenchTask(
        task_id="mbpp-4",
        benchmark="mbpp",
        prompt="Write a function to square a number.",
        canonical_solution="",
        tests="assert square(4) == 16",
    )
    monkeypatch.setattr(code_bench, "load_tasks", lambda benchmark, limit=10: [task])
    monkeypatch.setattr(code_bench, "code_bench_best_of_candidates", lambda: 3)
    monkeypatch.setattr(code_bench, "code_bench_repair_attempt_limit", lambda: 1)

    attempts: list[int] = []
    repair_calls: list[int] = []

    async def fake_litellm_generator(_task, model="", attempt=0):
        attempts.append(attempt)
        if attempt < 2:
            return "def sqr(x):\n    return x * x\n"
        return "def square(x):\n    return x * x\n"

    async def fake_repair_generator(_task, previous_code, test_output, model=""):
        repair_calls.append(1)
        return previous_code

    monkeypatch.setattr(code_bench, "litellm_generator", fake_litellm_generator)
    monkeypatch.setattr(code_bench, "litellm_repair_generator", fake_repair_generator)

    run = asyncio.run(code_bench.run_code_benchmark("mbpp", limit=1, mode="litellm", model="fake/model"))

    assert run.passed == 1
    assert attempts == [0, 1, 2]
    assert repair_calls == []


def test_code_bench_best_of_candidates_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("CODE_BENCH_BEST_OF", "5")

    assert code_bench.code_bench_best_of_candidates() == 5
