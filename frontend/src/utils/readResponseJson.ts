/** FastAPI-style `detail` from parsed JSON body. */
export function detailFromJsonBody(body: unknown): string | undefined {
  if (!body || typeof body !== 'object') return undefined;
  const d = (body as { detail?: unknown }).detail;
  if (typeof d === 'string') return d;
  if (Array.isArray(d)) {
    return d
      .map((x) =>
        typeof x === 'object' && x !== null && 'msg' in x
          ? String((x as { msg: unknown }).msg)
          : JSON.stringify(x)
      )
      .join('; ');
  }
  return undefined;
}

/**
 * Read body once (avoids empty-body `.json()` exceptions and double-read bugs).
 */
export async function readJsonResponse<T = unknown>(
  response: Response
): Promise<{ json: T | null; text: string }> {
  const text = await response.text();
  const trimmed = text.trim();
  if (!trimmed) return { json: null, text: '' };
  try {
    return { json: JSON.parse(trimmed) as T, text };
  } catch {
    return { json: null, text: trimmed.slice(0, 400) };
  }
}
