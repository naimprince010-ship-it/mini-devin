"""Tests for validation utilities."""

from utils.validation import validate_email, validate_password, validate_username


class TestValidateEmail:
    def test_valid_email(self):
        assert validate_email("test@example.com") is True
    
    def test_invalid_email_no_at(self):
        assert validate_email("testexample.com") is False
    
    def test_invalid_email_no_domain(self):
        assert validate_email("test@") is False
    
    def test_empty_email(self):
        assert validate_email("") is False


class TestValidatePassword:
    def test_valid_password(self):
        is_valid, msg = validate_password("Password123")
        assert is_valid is True
        assert msg == ""
    
    def test_password_too_short(self):
        is_valid, msg = validate_password("Pass1")
        assert is_valid is False
        assert "8 characters" in msg
    
    def test_password_no_uppercase(self):
        is_valid, msg = validate_password("password123")
        assert is_valid is False
        assert "uppercase" in msg
    
    def test_password_no_digit(self):
        is_valid, msg = validate_password("Password")
        assert is_valid is False
        assert "digit" in msg


class TestValidateUsername:
    def test_valid_username(self):
        is_valid, msg = validate_username("john_doe")
        assert is_valid is True
    
    def test_username_too_short(self):
        is_valid, msg = validate_username("ab")
        assert is_valid is False
        assert "3 characters" in msg
    
    def test_username_invalid_chars(self):
        is_valid, msg = validate_username("john@doe")
        assert is_valid is False
        assert "letters, numbers, and underscores" in msg
