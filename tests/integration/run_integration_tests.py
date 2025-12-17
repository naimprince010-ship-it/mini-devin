#!/usr/bin/env python3
"""
Run integration tests for Mini-Devin.

This script runs all integration tests and generates a report.
"""

import subprocess
import sys
import json
from datetime import datetime
from pathlib import Path


def run_tests():
    """Run integration tests and return results."""
    print("=" * 60)
    print("Mini-Devin Integration Test Suite")
    print(f"Started at: {datetime.now().isoformat()}")
    print("=" * 60)
    print()
    
    test_dir = Path(__file__).parent
    
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            str(test_dir / "test_agent_integration.py"),
            "-v",
            "--tb=short",
            "-x",
        ],
        capture_output=True,
        text=True,
        cwd=test_dir.parent.parent,
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    print()
    print("=" * 60)
    print(f"Finished at: {datetime.now().isoformat()}")
    print(f"Exit code: {result.returncode}")
    print("=" * 60)
    
    return result.returncode


def generate_report(exit_code: int) -> dict:
    """Generate a test report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "exit_code": exit_code,
        "status": "PASSED" if exit_code == 0 else "FAILED",
        "test_categories": [
            "Agent Initialization",
            "Tools Integration",
            "Memory Integration",
            "Parallel Execution",
            "Specialized Agents",
            "Skills Framework",
            "Safety Guards",
            "Verification",
            "API",
            "End-to-End Scenarios",
            "Configuration",
        ],
    }
    
    return report


if __name__ == "__main__":
    exit_code = run_tests()
    report = generate_report(exit_code)
    
    print()
    print("Test Report:")
    print(json.dumps(report, indent=2))
    
    sys.exit(exit_code)
