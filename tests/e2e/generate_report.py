"""
Test Report Generator for Mini-Devin E2E Tests (Phase 10).

This script runs all E2E tests and generates a comprehensive report.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_pytest_and_collect_results(test_dir: str) -> dict:
    """
    Run pytest on the test directory and collect results.
    
    Args:
        test_dir: Directory containing tests
        
    Returns:
        Dictionary with test results
    """
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            test_dir,
            "-v",
            "--tb=short",
            "-q",
            "--json-report",
            "--json-report-file=test_results.json",
        ],
        capture_output=True,
        text=True,
        cwd=Path(test_dir).parent,
    )
    
    json_report_path = Path(test_dir).parent / "test_results.json"
    if json_report_path.exists():
        with open(json_report_path) as f:
            return json.load(f)
    
    return {
        "summary": {
            "passed": 0,
            "failed": 0,
            "error": 0,
            "skipped": 0,
            "total": 0,
        },
        "tests": [],
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


def generate_markdown_report(results: dict, output_path: str) -> str:
    """
    Generate a markdown report from test results.
    
    Args:
        results: Test results dictionary
        output_path: Path to save the report
        
    Returns:
        Path to the generated report
    """
    summary = results.get("summary", {})
    tests = results.get("tests", [])
    
    lines = [
        "# Mini-Devin End-to-End Test Report",
        "",
        f"**Generated:** {datetime.utcnow().isoformat()}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Tests | {summary.get('total', 0)} |",
        f"| Passed | {summary.get('passed', 0)} |",
        f"| Failed | {summary.get('failed', 0)} |",
        f"| Errors | {summary.get('error', 0)} |",
        f"| Skipped | {summary.get('skipped', 0)} |",
        "",
    ]
    
    total = summary.get('total', 0)
    passed = summary.get('passed', 0)
    if total > 0:
        success_rate = (passed / total) * 100
        lines.append(f"**Success Rate:** {success_rate:.1f}%")
        lines.append("")
    
    lines.extend([
        "## Test Categories",
        "",
        "### Terminal & Editor Tools",
        "Tests for basic file operations and command execution.",
        "",
        "### Browser Tools",
        "Tests for web search, fetch, and interactive browser functionality.",
        "",
        "### Gates Integration",
        "Tests for planner and reviewer gates in the execution flow.",
        "",
        "## Detailed Results",
        "",
        "| Test | Status | Duration |",
        "|------|--------|----------|",
    ])
    
    for test in tests:
        name = test.get("nodeid", "Unknown")
        outcome = test.get("outcome", "unknown")
        duration = test.get("duration", 0)
        
        status_map = {
            "passed": "PASS",
            "failed": "FAIL",
            "error": "ERROR",
            "skipped": "SKIP",
        }
        status = status_map.get(outcome, outcome.upper())
        
        lines.append(f"| {name} | {status} | {duration:.3f}s |")
    
    if results.get("stdout"):
        lines.extend([
            "",
            "## Test Output",
            "",
            "```",
            results["stdout"][:5000],
            "```",
        ])
    
    report_content = "\n".join(lines)
    
    with open(output_path, "w") as f:
        f.write(report_content)
    
    return output_path


def main():
    """Run E2E tests and generate report."""
    print("Running Mini-Devin E2E Tests...")
    print("=" * 50)
    
    test_dir = Path(__file__).parent
    
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            str(test_dir),
            "-v",
            "--tb=short",
        ],
        capture_output=True,
        text=True,
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    lines = result.stdout.split("\n")
    
    results = {
        "summary": {
            "passed": 0,
            "failed": 0,
            "error": 0,
            "skipped": 0,
            "total": 0,
        },
        "tests": [],
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }
    
    for line in lines:
        if "PASSED" in line:
            results["summary"]["passed"] += 1
            results["summary"]["total"] += 1
            results["tests"].append({
                "nodeid": line.split("PASSED")[0].strip(),
                "outcome": "passed",
                "duration": 0,
            })
        elif "FAILED" in line:
            results["summary"]["failed"] += 1
            results["summary"]["total"] += 1
            results["tests"].append({
                "nodeid": line.split("FAILED")[0].strip(),
                "outcome": "failed",
                "duration": 0,
            })
        elif "ERROR" in line:
            results["summary"]["error"] += 1
            results["summary"]["total"] += 1
            results["tests"].append({
                "nodeid": line.split("ERROR")[0].strip(),
                "outcome": "error",
                "duration": 0,
            })
        elif "SKIPPED" in line:
            results["summary"]["skipped"] += 1
            results["summary"]["total"] += 1
            results["tests"].append({
                "nodeid": line.split("SKIPPED")[0].strip(),
                "outcome": "skipped",
                "duration": 0,
            })
    
    report_path = test_dir / "e2e_test_report.md"
    generate_markdown_report(results, str(report_path))
    
    print("=" * 50)
    print(f"Report generated: {report_path}")
    print(f"Total: {results['summary']['total']}")
    print(f"Passed: {results['summary']['passed']}")
    print(f"Failed: {results['summary']['failed']}")
    print(f"Errors: {results['summary']['error']}")
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
