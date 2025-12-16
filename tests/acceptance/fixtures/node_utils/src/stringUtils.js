/**
 * String utility functions.
 * 
 * BUG: capitalize() doesn't handle empty strings properly.
 */

/**
 * Capitalize the first letter of a string.
 * @param {string} str - The string to capitalize.
 * @returns {string} The capitalized string.
 */
function capitalize(str) {
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
}

module.exports = {
  capitalize,
  toLowerCase,
  toUpperCase,
  reverse,
};
