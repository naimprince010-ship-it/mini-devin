const { capitalize, toLowerCase, toUpperCase, reverse } = require('../src/stringUtils');

describe('capitalize', () => {
  test('should capitalize first letter', () => {
    expect(capitalize('hello')).toBe('Hello');
  });

  test('should handle already capitalized string', () => {
    expect(capitalize('Hello')).toBe('Hello');
  });

  test('should handle empty string in capitalize', () => {
    // This test will fail because capitalize doesn't handle empty strings
    expect(capitalize('')).toBe('');
  });
});

describe('toLowerCase', () => {
  test('should convert to lowercase', () => {
    expect(toLowerCase('HELLO')).toBe('hello');
  });

  test('should handle mixed case', () => {
    expect(toLowerCase('HeLLo')).toBe('hello');
  });
});

describe('toUpperCase', () => {
  test('should convert to uppercase', () => {
    expect(toUpperCase('hello')).toBe('HELLO');
  });
});

describe('reverse', () => {
  test('should reverse a string', () => {
    expect(reverse('hello')).toBe('olleh');
  });

  test('should handle empty string', () => {
    expect(reverse('')).toBe('');
  });
});
