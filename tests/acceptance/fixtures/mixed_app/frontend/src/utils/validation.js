/**
 * Validation utilities for the frontend.
 * These should match the backend validation rules.
 */

/**
 * Validate an email address.
 * @param {string} email - The email to validate.
 * @returns {boolean} True if valid.
 */
function validateEmail(email) {
  if (!email) return false;
  const pattern = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
  return pattern.test(email);
}

/**
 * Validate a password.
 * @param {string} password - The password to validate.
 * @returns {{isValid: boolean, message: string}}
 */
function validatePassword(password) {
  if (password.length < 8) {
    return { isValid: false, message: 'Password must be at least 8 characters' };
  }
  if (!/[A-Z]/.test(password)) {
    return { isValid: false, message: 'Password must contain at least one uppercase letter' };
  }
  if (!/[a-z]/.test(password)) {
    return { isValid: false, message: 'Password must contain at least one lowercase letter' };
  }
  if (!/[0-9]/.test(password)) {
    return { isValid: false, message: 'Password must contain at least one digit' };
  }
  return { isValid: true, message: '' };
}

/**
 * Validate a username.
 * @param {string} username - The username to validate.
 * @returns {{isValid: boolean, message: string}}
 */
function validateUsername(username) {
  if (username.length < 3) {
    return { isValid: false, message: 'Username must be at least 3 characters' };
  }
  if (username.length > 20) {
    return { isValid: false, message: 'Username must be at most 20 characters' };
  }
  if (!/^[a-zA-Z0-9_]+$/.test(username)) {
    return { isValid: false, message: 'Username can only contain letters, numbers, and underscores' };
  }
  return { isValid: true, message: '' };
}

module.exports = {
  validateEmail,
  validatePassword,
  validateUsername,
};
