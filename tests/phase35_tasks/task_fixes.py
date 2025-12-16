"""
Predefined fixes for Phase 3.5/3.6 tasks.

These fixes simulate what the agent would apply to fix the bugs in the test fixtures.
Each fix is a minimal, focused patch following diff discipline principles.
"""

from dataclasses import dataclass


@dataclass
class TaskFix:
    """A fix to apply to a task."""
    task_id: str
    file_path: str
    original: str
    replacement: str
    description: str


# Task 1: Python - Fix divide by zero
TASK_01_FIX = TaskFix(
    task_id="task_01_python_fix_test",
    file_path="src/calculator.py",
    original='''def divide(a: float, b: float) -> float:
    """Divide a by b.
    
    BUG: This function doesn't handle division by zero properly.
    It should raise a ValueError with a descriptive message.
    """
    # BUG: No check for division by zero
    return a / b''',
    replacement='''def divide(a: float, b: float) -> float:
    """Divide a by b.
    
    Args:
        a: The dividend.
        b: The divisor.
    
    Returns:
        The result of a / b.
    
    Raises:
        ValueError: If b is zero.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b''',
    description="Fix divide function to handle division by zero",
)

# Task 2: Python - Add power function
TASK_02_FIX_CALC = TaskFix(
    task_id="task_02_python_add_feature",
    file_path="src/calculator.py",
    original='''def divide(a: float, b: float) -> float:
    """Divide a by b.
    
    BUG: This function doesn't handle division by zero properly.
    It should raise a ValueError with a descriptive message.
    """
    # BUG: No check for division by zero
    return a / b''',
    replacement='''def divide(a: float, b: float) -> float:
    """Divide a by b.
    
    Args:
        a: The dividend.
        b: The divisor.
    
    Returns:
        The result of a / b.
    
    Raises:
        ValueError: If b is zero.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def power(base: float, exponent: float) -> float:
    """Calculate base raised to the power of exponent.
    
    Args:
        base: The base number.
        exponent: The exponent.
    
    Returns:
        base ** exponent.
    """
    return base ** exponent''',
    description="Add power function to calculator",
)

TASK_02_FIX_TEST = TaskFix(
    task_id="task_02_python_add_feature",
    file_path="tests/test_calculator.py",
    original='''from src.calculator import add, subtract, multiply, divide''',
    replacement='''from src.calculator import add, subtract, multiply, divide, power''',
    description="Import power function in tests",
)

TASK_02_FIX_TEST2 = TaskFix(
    task_id="task_02_python_add_feature",
    file_path="tests/test_calculator.py",
    original='''class TestDivide:
    def test_divide_positive_numbers(self):
        assert divide(6, 3) == 2
    
    def test_divide_by_zero(self):
        """This test will fail because divide doesn't handle zero properly."""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            divide(5, 0)''',
    replacement='''class TestDivide:
    def test_divide_positive_numbers(self):
        assert divide(6, 3) == 2
    
    def test_divide_by_zero(self):
        """This test will fail because divide doesn't handle zero properly."""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            divide(5, 0)


class TestPower:
    def test_power_positive(self):
        assert power(2, 3) == 8
    
    def test_power_zero_exponent(self):
        assert power(5, 0) == 1
    
    def test_power_negative_exponent(self):
        assert power(2, -1) == 0.5''',
    description="Add tests for power function",
)

# Task 3: Python - Refactor to class (complex, skip for now)
# This task is intentionally complex and may fail

# Task 4: Node - Fix capitalize empty string
TASK_04_FIX = TaskFix(
    task_id="task_04_node_fix_test",
    file_path="src/stringUtils.js",
    original='''function capitalize(str) {
  // BUG: This will throw an error for empty strings
  // because str[0] is undefined and toUpperCase() fails
  return str[0].toUpperCase() + str.slice(1);
}''',
    replacement='''function capitalize(str) {
  // Handle empty strings
  if (!str || str.length === 0) {
    return '';
  }
  return str[0].toUpperCase() + str.slice(1);
}''',
    description="Fix capitalize to handle empty strings",
)

# Task 5: Node - Add truncate function
TASK_05_FIX_UTILS = TaskFix(
    task_id="task_05_node_add_feature",
    file_path="src/stringUtils.js",
    original='''function capitalize(str) {
  // BUG: This will throw an error for empty strings
  // because str[0] is undefined and toUpperCase() fails
  return str[0].toUpperCase() + str.slice(1);
}''',
    replacement='''function capitalize(str) {
  // Handle empty strings
  if (!str || str.length === 0) {
    return '';
  }
  return str[0].toUpperCase() + str.slice(1);
}

/**
 * Truncate a string to a maximum length.
 * @param {string} str - The string to truncate.
 * @param {number} maxLength - The maximum length.
 * @returns {string} The truncated string with '...' if truncated.
 */
function truncate(str, maxLength) {
  if (!str || str.length <= maxLength) {
    return str || '';
  }
  return str.slice(0, maxLength) + '...';
}''',
    description="Add truncate function",
)

TASK_05_FIX_EXPORTS = TaskFix(
    task_id="task_05_node_add_feature",
    file_path="src/stringUtils.js",
    original='''module.exports = {
  capitalize,
  toLowerCase,
  toUpperCase,
  reverse,
};''',
    replacement='''module.exports = {
  capitalize,
  toLowerCase,
  toUpperCase,
  reverse,
  truncate,
};''',
    description="Export truncate function",
)

TASK_05_FIX_TEST = TaskFix(
    task_id="task_05_node_add_feature",
    file_path="tests/stringUtils.test.js",
    original='''const { capitalize, toLowerCase, toUpperCase, reverse } = require('../src/stringUtils');''',
    replacement='''const { capitalize, toLowerCase, toUpperCase, reverse, truncate } = require('../src/stringUtils');''',
    description="Import truncate in tests",
)

TASK_05_FIX_TEST2 = TaskFix(
    task_id="task_05_node_add_feature",
    file_path="tests/stringUtils.test.js",
    original='''describe('reverse', () => {
  test('should reverse a string', () => {
    expect(reverse('hello')).toBe('olleh');
  });

  test('should handle empty string', () => {
    expect(reverse('')).toBe('');
  });
});''',
    replacement='''describe('reverse', () => {
  test('should reverse a string', () => {
    expect(reverse('hello')).toBe('olleh');
  });

  test('should handle empty string', () => {
    expect(reverse('')).toBe('');
  });
});

describe('truncate', () => {
  test('should truncate long strings', () => {
    expect(truncate('hello world', 5)).toBe('hello...');
  });

  test('should not truncate short strings', () => {
    expect(truncate('hi', 5)).toBe('hi');
  });

  test('should handle empty string', () => {
    expect(truncate('', 5)).toBe('');
  });
});''',
    description="Add tests for truncate function",
)

# Task 6: Node - Refactor to class (complex, skip for now)
# This task is intentionally complex and may fail

# Task 7: Mixed - Backend validation (already passing)
# No fix needed

# Task 8: Mixed - Frontend validation (already passing)
# No fix needed

# Task 9: Python - Add documentation (already passing with existing docstrings)
# No fix needed

# Task 10: Node - Add error handling
TASK_10_FIX_UTILS = TaskFix(
    task_id="task_10_node_error_handling",
    file_path="src/stringUtils.js",
    original='''function capitalize(str) {
  // BUG: This will throw an error for empty strings
  // because str[0] is undefined and toUpperCase() fails
  return str[0].toUpperCase() + str.slice(1);
}

/**
 * Convert a string to lowercase.
 * @param {string} str - The string to convert.
 * @returns {string} The lowercase string.
 */
function toLowerCase(str) {
  return str.toLowerCase();
}

/**
 * Convert a string to uppercase.
 * @param {string} str - The string to convert.
 * @returns {string} The uppercase string.
 */
function toUpperCase(str) {
  return str.toUpperCase();
}

/**
 * Reverse a string.
 * @param {string} str - The string to reverse.
 * @returns {string} The reversed string.
 */
function reverse(str) {
  return str.split('').reverse().join('');
}''',
    replacement='''function capitalize(str) {
  if (typeof str !== 'string') {
    throw new TypeError('Input must be a string');
  }
  if (!str || str.length === 0) {
    return '';
  }
  return str[0].toUpperCase() + str.slice(1);
}

/**
 * Convert a string to lowercase.
 * @param {string} str - The string to convert.
 * @returns {string} The lowercase string.
 */
function toLowerCase(str) {
  if (typeof str !== 'string') {
    throw new TypeError('Input must be a string');
  }
  return str.toLowerCase();
}

/**
 * Convert a string to uppercase.
 * @param {string} str - The string to convert.
 * @returns {string} The uppercase string.
 */
function toUpperCase(str) {
  if (typeof str !== 'string') {
    throw new TypeError('Input must be a string');
  }
  return str.toUpperCase();
}

/**
 * Reverse a string.
 * @param {string} str - The string to reverse.
 * @returns {string} The reversed string.
 */
function reverse(str) {
  if (typeof str !== 'string') {
    throw new TypeError('Input must be a string');
  }
  return str.split('').reverse().join('');
}''',
    description="Add type checking to all functions",
)

TASK_10_FIX_TEST = TaskFix(
    task_id="task_10_node_error_handling",
    file_path="tests/stringUtils.test.js",
    original='''describe('reverse', () => {
  test('should reverse a string', () => {
    expect(reverse('hello')).toBe('olleh');
  });

  test('should handle empty string', () => {
    expect(reverse('')).toBe('');
  });
});''',
    replacement='''describe('reverse', () => {
  test('should reverse a string', () => {
    expect(reverse('hello')).toBe('olleh');
  });

  test('should handle empty string', () => {
    expect(reverse('')).toBe('');
  });
});

describe('error handling', () => {
  test('capitalize should throw TypeError for non-string', () => {
    expect(() => capitalize(123)).toThrow(TypeError);
  });

  test('toLowerCase should throw TypeError for non-string', () => {
    expect(() => toLowerCase(123)).toThrow(TypeError);
  });

  test('toUpperCase should throw TypeError for non-string', () => {
    expect(() => toUpperCase(123)).toThrow(TypeError);
  });

  test('reverse should throw TypeError for non-string', () => {
    expect(() => reverse(123)).toThrow(TypeError);
  });
});''',
    description="Add error handling tests",
)


# Map task IDs to their fixes
TASK_FIXES: dict[str, list[TaskFix]] = {
    "task_01_python_fix_test": [TASK_01_FIX],
    "task_02_python_add_feature": [TASK_02_FIX_CALC, TASK_02_FIX_TEST, TASK_02_FIX_TEST2],
    # task_03 is complex refactoring - skip
    "task_04_node_fix_test": [TASK_04_FIX],
    "task_05_node_add_feature": [TASK_05_FIX_UTILS, TASK_05_FIX_EXPORTS, TASK_05_FIX_TEST, TASK_05_FIX_TEST2],
    # task_06 is complex refactoring - skip
    # task_07, task_08, task_09 already pass
    "task_10_node_error_handling": [TASK_10_FIX_UTILS, TASK_10_FIX_TEST],
}


def apply_fixes(workspace_path: str, task_id: str) -> tuple[bool, list[str]]:
    """
    Apply fixes for a task to the workspace.
    
    Args:
        workspace_path: Path to the workspace directory
        task_id: The task ID to apply fixes for
        
    Returns:
        Tuple of (success, list of applied fix descriptions)
    """
    from pathlib import Path
    
    fixes = TASK_FIXES.get(task_id, [])
    if not fixes:
        return True, []  # No fixes needed
    
    applied = []
    workspace = Path(workspace_path)
    
    for fix in fixes:
        file_path = workspace / fix.file_path
        
        if not file_path.exists():
            continue
        
        try:
            content = file_path.read_text()
            
            if fix.original in content:
                new_content = content.replace(fix.original, fix.replacement)
                file_path.write_text(new_content)
                applied.append(fix.description)
        except Exception as e:
            return False, [f"Error applying fix: {e}"]
    
    return True, applied
