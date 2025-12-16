"""Simple calculator module with a bug in divide function."""


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float) -> float:
    """Divide a by b.
    
    BUG: This function doesn't handle division by zero properly.
    It should raise a ValueError with a descriptive message.
    """
    # BUG: No check for division by zero
    return a / b
