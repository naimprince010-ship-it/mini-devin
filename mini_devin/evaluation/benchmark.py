"""
Benchmark Suite for Mini-Devin

This module provides benchmark management for:
- Predefined task collections
- Custom benchmark creation
- Result comparison across runs
- Regression detection
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field

from .harness import EvaluationTask, EvaluationResult, TaskDifficulty, TaskCategory


@dataclass
class BenchmarkRun:
    """A single benchmark run."""
    run_id: str
    benchmark_name: str
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    
    # Configuration
    model: str = "gpt-4o"
    
    # Results
    results: list[EvaluationResult] = field(default_factory=list)
    
    # Summary
    total_tasks: int = 0
    passed_tasks: int = 0
    failed_tasks: int = 0
    pass_rate: float = 0.0
    total_duration: float = 0.0
    total_tokens: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "benchmark_name": self.benchmark_name,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "model": self.model,
            "total_tasks": self.total_tasks,
            "passed_tasks": self.passed_tasks,
            "failed_tasks": self.failed_tasks,
            "pass_rate": self.pass_rate,
            "total_duration": self.total_duration,
            "total_tokens": self.total_tokens,
            "results": [r.to_dict() for r in self.results],
        }


@dataclass
class Benchmark:
    """A benchmark definition."""
    name: str
    description: str
    tasks: list[EvaluationTask] = field(default_factory=list)
    
    # Metadata
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: list[str] = field(default_factory=list)
    
    # Configuration
    default_timeout: int = 600
    default_max_iterations: int = 30
    
    def add_task(self, task: EvaluationTask) -> None:
        """Add a task to the benchmark."""
        self.tasks.append(task)
    
    def remove_task(self, task_id: str) -> bool:
        """Remove a task from the benchmark."""
        for i, task in enumerate(self.tasks):
            if task.task_id == task_id:
                self.tasks.pop(i)
                return True
        return False
    
    def get_task(self, task_id: str) -> EvaluationTask | None:
        """Get a task by ID."""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None
    
    def filter_by_difficulty(self, difficulty: TaskDifficulty) -> list[EvaluationTask]:
        """Filter tasks by difficulty."""
        return [t for t in self.tasks if t.difficulty == difficulty]
    
    def filter_by_category(self, category: TaskCategory) -> list[EvaluationTask]:
        """Filter tasks by category."""
        return [t for t in self.tasks if t.category == category]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "tags": self.tags,
            "default_timeout": self.default_timeout,
            "default_max_iterations": self.default_max_iterations,
            "tasks": [
                {
                    "task_id": t.task_id,
                    "name": t.name,
                    "description": t.description,
                    "repo_url": t.repo_url,
                    "repo_path": t.repo_path,
                    "difficulty": t.difficulty.value,
                    "category": t.category.value,
                    "setup_commands": t.setup_commands,
                    "test_command": t.test_command,
                    "expected_files_modified": t.expected_files_modified,
                    "verification_script": t.verification_script,
                    "tags": t.tags,
                    "timeout_seconds": t.timeout_seconds,
                    "max_iterations": t.max_iterations,
                }
                for t in self.tasks
            ],
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Benchmark":
        """Create from dictionary."""
        benchmark = cls(
            name=data["name"],
            description=data["description"],
            version=data.get("version", "1.0.0"),
            tags=data.get("tags", []),
            default_timeout=data.get("default_timeout", 600),
            default_max_iterations=data.get("default_max_iterations", 30),
        )
        
        if "created_at" in data:
            benchmark.created_at = datetime.fromisoformat(data["created_at"])
        
        for t_data in data.get("tasks", []):
            task = EvaluationTask(
                task_id=t_data["task_id"],
                name=t_data["name"],
                description=t_data["description"],
                repo_url=t_data.get("repo_url"),
                repo_path=t_data.get("repo_path"),
                difficulty=TaskDifficulty(t_data.get("difficulty", "medium")),
                category=TaskCategory(t_data.get("category", "bug_fix")),
                setup_commands=t_data.get("setup_commands", []),
                test_command=t_data.get("test_command", ""),
                expected_files_modified=t_data.get("expected_files_modified", []),
                verification_script=t_data.get("verification_script"),
                tags=t_data.get("tags", []),
                timeout_seconds=t_data.get("timeout_seconds", benchmark.default_timeout),
                max_iterations=t_data.get("max_iterations", benchmark.default_max_iterations),
            )
            benchmark.tasks.append(task)
        
        return benchmark
    
    def save(self, path: str) -> None:
        """Save benchmark to file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
    
    @classmethod
    def load(cls, path: str) -> "Benchmark":
        """Load benchmark from file."""
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)


class BenchmarkSuite:
    """
    Manages a collection of benchmarks.
    
    Features:
    - Multiple benchmark management
    - Run history tracking
    - Result comparison
    - Regression detection
    """
    
    def __init__(self, storage_dir: str = "./benchmarks"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Benchmark storage
        self.benchmarks: dict[str, Benchmark] = {}
        
        # Run history
        self.runs: list[BenchmarkRun] = []
        
        # Load existing benchmarks
        self._load_benchmarks()
    
    def _load_benchmarks(self) -> None:
        """Load benchmarks from storage directory."""
        for file in self.storage_dir.glob("*.json"):
            if file.name.startswith("run_"):
                continue  # Skip run files
            try:
                benchmark = Benchmark.load(str(file))
                self.benchmarks[benchmark.name] = benchmark
            except Exception:
                pass
    
    def add_benchmark(self, benchmark: Benchmark) -> None:
        """Add a benchmark to the suite."""
        self.benchmarks[benchmark.name] = benchmark
        benchmark.save(str(self.storage_dir / f"{benchmark.name}.json"))
    
    def get_benchmark(self, name: str) -> Benchmark | None:
        """Get a benchmark by name."""
        return self.benchmarks.get(name)
    
    def list_benchmarks(self) -> list[str]:
        """List all benchmark names."""
        return list(self.benchmarks.keys())
    
    def remove_benchmark(self, name: str) -> bool:
        """Remove a benchmark."""
        if name not in self.benchmarks:
            return False
        
        del self.benchmarks[name]
        
        # Remove file
        file_path = self.storage_dir / f"{name}.json"
        if file_path.exists():
            file_path.unlink()
        
        return True
    
    def record_run(self, run: BenchmarkRun) -> None:
        """Record a benchmark run."""
        self.runs.append(run)
        
        # Save run to file
        run_file = self.storage_dir / f"run_{run.run_id}.json"
        run_file.write_text(json.dumps(run.to_dict(), indent=2))
    
    def get_run_history(
        self,
        benchmark_name: str | None = None,
        limit: int = 10,
    ) -> list[BenchmarkRun]:
        """Get run history."""
        runs = self.runs
        
        if benchmark_name:
            runs = [r for r in runs if r.benchmark_name == benchmark_name]
        
        # Sort by start time descending
        runs = sorted(runs, key=lambda r: r.started_at, reverse=True)
        
        return runs[:limit]
    
    def compare_runs(
        self,
        run_id_1: str,
        run_id_2: str,
    ) -> dict[str, Any]:
        """Compare two benchmark runs."""
        run1 = next((r for r in self.runs if r.run_id == run_id_1), None)
        run2 = next((r for r in self.runs if r.run_id == run_id_2), None)
        
        if not run1 or not run2:
            return {"error": "Run not found"}
        
        # Build result maps
        results1 = {r.task_id: r for r in run1.results}
        results2 = {r.task_id: r for r in run2.results}
        
        # Compare
        comparison = {
            "run1": {
                "run_id": run1.run_id,
                "pass_rate": run1.pass_rate,
                "total_duration": run1.total_duration,
            },
            "run2": {
                "run_id": run2.run_id,
                "pass_rate": run2.pass_rate,
                "total_duration": run2.total_duration,
            },
            "pass_rate_diff": run2.pass_rate - run1.pass_rate,
            "duration_diff": run2.total_duration - run1.total_duration,
            "regressions": [],
            "improvements": [],
            "unchanged": [],
        }
        
        # Find regressions and improvements
        all_task_ids = set(results1.keys()) | set(results2.keys())
        
        for task_id in all_task_ids:
            r1 = results1.get(task_id)
            r2 = results2.get(task_id)
            
            if r1 and r2:
                if r1.passed and not r2.passed:
                    comparison["regressions"].append(task_id)
                elif not r1.passed and r2.passed:
                    comparison["improvements"].append(task_id)
                else:
                    comparison["unchanged"].append(task_id)
        
        return comparison
    
    def detect_regressions(
        self,
        benchmark_name: str,
        threshold: float = 0.05,
    ) -> dict[str, Any]:
        """
        Detect regressions compared to the best historical run.
        
        Args:
            benchmark_name: Name of the benchmark
            threshold: Pass rate drop threshold to flag as regression
            
        Returns:
            Regression analysis
        """
        runs = self.get_run_history(benchmark_name, limit=100)
        
        if len(runs) < 2:
            return {"error": "Not enough runs for comparison"}
        
        # Find best historical run
        best_run = max(runs[1:], key=lambda r: r.pass_rate)
        latest_run = runs[0]
        
        pass_rate_drop = best_run.pass_rate - latest_run.pass_rate
        
        return {
            "latest_run": latest_run.run_id,
            "best_run": best_run.run_id,
            "latest_pass_rate": latest_run.pass_rate,
            "best_pass_rate": best_run.pass_rate,
            "pass_rate_drop": pass_rate_drop,
            "is_regression": pass_rate_drop > threshold,
            "comparison": self.compare_runs(best_run.run_id, latest_run.run_id),
        }
    
    def get_statistics(self) -> dict[str, Any]:
        """Get overall statistics."""
        return {
            "total_benchmarks": len(self.benchmarks),
            "total_runs": len(self.runs),
            "total_tasks": sum(len(b.tasks) for b in self.benchmarks.values()),
            "benchmarks": {
                name: {
                    "tasks": len(b.tasks),
                    "runs": len([r for r in self.runs if r.benchmark_name == name]),
                }
                for name, b in self.benchmarks.items()
            },
        }


# Predefined benchmarks

def create_python_benchmark() -> Benchmark:
    """Create a Python-focused benchmark."""
    benchmark = Benchmark(
        name="python_basics",
        description="Basic Python coding tasks",
        tags=["python", "basics"],
    )
    
    # Add sample tasks
    benchmark.add_task(EvaluationTask(
        task_id="py_fix_syntax",
        name="Fix Syntax Error",
        description="Fix the syntax error in the Python file that prevents it from running.",
        difficulty=TaskDifficulty.TRIVIAL,
        category=TaskCategory.BUG_FIX,
        test_command="python -m py_compile main.py",
    ))
    
    benchmark.add_task(EvaluationTask(
        task_id="py_add_function",
        name="Add Function",
        description="Add a function called 'calculate_average' that takes a list of numbers and returns their average.",
        difficulty=TaskDifficulty.EASY,
        category=TaskCategory.FEATURE,
        test_command="pytest test_main.py -v",
    ))
    
    benchmark.add_task(EvaluationTask(
        task_id="py_fix_test",
        name="Fix Failing Test",
        description="The test_divide function is failing. Fix the divide function to handle division by zero correctly.",
        difficulty=TaskDifficulty.EASY,
        category=TaskCategory.BUG_FIX,
        test_command="pytest test_math.py -v",
    ))
    
    return benchmark


def create_javascript_benchmark() -> Benchmark:
    """Create a JavaScript-focused benchmark."""
    benchmark = Benchmark(
        name="javascript_basics",
        description="Basic JavaScript coding tasks",
        tags=["javascript", "basics"],
    )
    
    benchmark.add_task(EvaluationTask(
        task_id="js_fix_async",
        name="Fix Async Bug",
        description="Fix the async/await bug that causes the function to return undefined.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.BUG_FIX,
        test_command="npm test",
    ))
    
    benchmark.add_task(EvaluationTask(
        task_id="js_add_validation",
        name="Add Input Validation",
        description="Add input validation to the createUser function to ensure email is valid.",
        difficulty=TaskDifficulty.EASY,
        category=TaskCategory.FEATURE,
        test_command="npm test",
    ))
    
    return benchmark


def create_fullstack_benchmark() -> Benchmark:
    """Create a full-stack benchmark."""
    benchmark = Benchmark(
        name="fullstack_tasks",
        description="Full-stack development tasks",
        tags=["fullstack", "api", "frontend"],
    )
    
    benchmark.add_task(EvaluationTask(
        task_id="fs_fix_api",
        name="Fix API Endpoint",
        description="The /api/users endpoint returns 500. Fix the database query to return users correctly.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.BUG_FIX,
        test_command="pytest tests/test_api.py -v",
    ))
    
    benchmark.add_task(EvaluationTask(
        task_id="fs_add_endpoint",
        name="Add API Endpoint",
        description="Add a new POST /api/products endpoint that creates a product with name and price fields.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.FEATURE,
        test_command="pytest tests/test_api.py -v",
    ))
    
    return benchmark
