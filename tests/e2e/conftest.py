"""Shared fixtures for End-to-End tests."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from mini_devin.config.settings import AgentGatesSettings
from mini_devin.schemas.state import TaskState, TaskGoal


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory for tests."""
    workspace = tempfile.mkdtemp(prefix="mini_devin_e2e_")
    yield workspace
    shutil.rmtree(workspace, ignore_errors=True)


@pytest.fixture
def python_project(temp_workspace):
    """Create a minimal Python project in the workspace."""
    project_dir = Path(temp_workspace) / "python_project"
    project_dir.mkdir()
    
    (project_dir / "main.py").write_text('''
def greet(name: str) -> str:
    """Greet a person by name."""
    return f"Hello, {name}!"

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

if __name__ == "__main__":
    print(greet("World"))
''')
    
    (project_dir / "test_main.py").write_text('''
import pytest
from main import greet, add

def test_greet():
    assert greet("Alice") == "Hello, Alice!"

def test_add():
    assert add(2, 3) == 5
    
def test_add_negative():
    assert add(-1, 1) == 0
''')
    
    (project_dir / "requirements.txt").write_text("pytest>=7.0.0\n")
    
    return str(project_dir)


@pytest.fixture
def node_project(temp_workspace):
    """Create a minimal Node.js project in the workspace."""
    project_dir = Path(temp_workspace) / "node_project"
    project_dir.mkdir()
    
    (project_dir / "package.json").write_text('''{
  "name": "test-project",
  "version": "1.0.0",
  "scripts": {
    "test": "node test.js"
  }
}
''')
    
    (project_dir / "index.js").write_text('''
function greet(name) {
    return `Hello, ${name}!`;
}

function add(a, b) {
    return a + b;
}

module.exports = { greet, add };
''')
    
    (project_dir / "test.js").write_text('''
const { greet, add } = require('./index');

let passed = 0;
let failed = 0;

function test(name, fn) {
    try {
        fn();
        console.log(`PASS: ${name}`);
        passed++;
    } catch (e) {
        console.log(`FAIL: ${name} - ${e.message}`);
        failed++;
    }
}

function assertEqual(actual, expected) {
    if (actual !== expected) {
        throw new Error(`Expected ${expected}, got ${actual}`);
    }
}

test('greet returns correct message', () => {
    assertEqual(greet('Alice'), 'Hello, Alice!');
});

test('add returns correct sum', () => {
    assertEqual(add(2, 3), 5);
});

console.log(`\\nResults: ${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
''')
    
    return str(project_dir)


@pytest.fixture
def buggy_python_project(temp_workspace):
    """Create a Python project with intentional bugs for testing."""
    project_dir = Path(temp_workspace) / "buggy_project"
    project_dir.mkdir()
    
    (project_dir / "calculator.py").write_text('''
def divide(a: int, b: int) -> float:
    """Divide two numbers."""
    return a / b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a + b  # BUG: should be a * b
''')
    
    (project_dir / "test_calculator.py").write_text('''
import pytest
from calculator import divide, multiply

def test_divide():
    assert divide(10, 2) == 5.0

def test_divide_by_zero():
    with pytest.raises(ZeroDivisionError):
        divide(10, 0)

def test_multiply():
    assert multiply(3, 4) == 12  # This will fail due to bug
''')
    
    return str(project_dir)


@pytest.fixture
def gates_enabled():
    """Gates settings with all gates enabled."""
    return AgentGatesSettings(
        planning_required=True,
        max_plan_steps=5,
        review_required=True,
        block_on_high_severity=True,
        use_llm_planning=False,
    )


@pytest.fixture
def gates_disabled():
    """Gates settings with all gates disabled."""
    return AgentGatesSettings(
        planning_required=False,
        max_plan_steps=5,
        review_required=False,
        block_on_high_severity=False,
        use_llm_planning=False,
    )


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing without API calls."""
    client = MagicMock()
    client.complete = AsyncMock()
    client.add_user_message = MagicMock()
    client.add_assistant_message = MagicMock()
    client.add_tool_result = MagicMock()
    client.set_system_prompt = MagicMock()
    client.get_usage_stats = MagicMock(return_value={"total_tokens": 1000})
    client.conversation = []
    return client


def create_task(description: str, criteria: list[str] | None = None) -> TaskState:
    """Helper to create a task for testing."""
    import uuid
    return TaskState(
        task_id=str(uuid.uuid4()),
        goal=TaskGoal(
            description=description,
            acceptance_criteria=criteria or [],
        ),
    )
