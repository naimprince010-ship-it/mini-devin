"""Tests for the calculator module."""

import pytest
from src.calculator import add, subtract, multiply, divide


class TestAdd:
    def test_add_positive_numbers(self):
        assert add(2, 3) == 5
    
    def test_add_negative_numbers(self):
        assert add(-2, -3) == -5
    
    def test_add_mixed_numbers(self):
        assert add(-2, 3) == 1


class TestSubtract:
    def test_subtract_positive_numbers(self):
        assert subtract(5, 3) == 2
    
    def test_subtract_negative_numbers(self):
        assert subtract(-5, -3) == -2


class TestMultiply:
    def test_multiply_positive_numbers(self):
        assert multiply(2, 3) == 6
    
    def test_multiply_by_zero(self):
        assert multiply(5, 0) == 0


class TestDivide:
    def test_divide_positive_numbers(self):
        assert divide(6, 3) == 2
    
    def test_divide_by_zero(self):
        """This test will fail because divide doesn't handle zero properly."""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            divide(5, 0)
