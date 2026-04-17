/**
 * Approximate 1-based line span where two texts first/last differ (for Monaco decorations).
 */
export function computeChangedLineSpan(
  before: string | undefined,
  after: string,
): { startLine: number; endLine: number } | null {
  if (before === undefined || before === after) return null;
  const a = before.split('\n');
  const b = after.split('\n');
  let i = 0;
  const maxI = Math.min(a.length, b.length);
  while (i < maxI && a[i] === b[i]) i += 1;
  let ja = a.length - 1;
  let jb = b.length - 1;
  while (ja >= i && jb >= i && a[ja] === b[jb]) {
    ja -= 1;
    jb -= 1;
  }
  const startLine = i + 1;
  const endLine = Math.max(ja + 1, jb + 1, startLine);
  return { startLine, endLine };
}
