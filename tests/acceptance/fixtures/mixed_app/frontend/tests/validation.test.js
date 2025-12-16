const { validateEmail, validatePassword, validateUsername } = require('../src/utils/validation');

describe('validateEmail', () => {
  test('should accept valid email', () => {
    expect(validateEmail('test@example.com')).toBe(true);
  });

  test('should reject email without @', () => {
    expect(validateEmail('testexample.com')).toBe(false);
  });

  test('should reject empty email', () => {
    expect(validateEmail('')).toBe(false);
  });
});

describe('validatePassword', () => {
  test('should accept valid password', () => {
    const result = validatePassword('Password123');
    expect(result.isValid).toBe(true);
  });

  test('should reject short password', () => {
    const result = validatePassword('Pass1');
    expect(result.isValid).toBe(false);
    expect(result.message).toContain('8 characters');
  });

  test('should reject password without uppercase', () => {
    const result = validatePassword('password123');
    expect(result.isValid).toBe(false);
    expect(result.message).toContain('uppercase');
  });
});

describe('validateUsername', () => {
  test('should accept valid username', () => {
    const result = validateUsername('john_doe');
    expect(result.isValid).toBe(true);
  });

  test('should reject short username', () => {
    const result = validateUsername('ab');
    expect(result.isValid).toBe(false);
    expect(result.message).toContain('3 characters');
  });

  test('should reject username with invalid chars', () => {
    const result = validateUsername('john@doe');
    expect(result.isValid).toBe(false);
  });
});
