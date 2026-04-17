/**
 * Normalize FastAPI / Starlette error bodies (`detail` string, list, or object).
 */
export function formatApiDetail(detail: unknown): string {
  if (detail == null) {
    return 'Request failed';
  }
  if (typeof detail === 'string') {
    return detail;
  }
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (item && typeof item === 'object' && 'msg' in item) {
          const loc = Array.isArray((item as { loc?: unknown }).loc)
            ? (item as { loc: string[] }).loc.join('.')
            : '';
          const msg = String((item as { msg?: string }).msg || item);
          return loc ? `${loc}: ${msg}` : msg;
        }
        return typeof item === 'string' ? item : JSON.stringify(item);
      })
      .join('; ');
  }
  if (typeof detail === 'object' && detail !== null) {
    return JSON.stringify(detail);
  }
  return String(detail);
}

export async function readApiErrorMessage(response: Response, fallback: string): Promise<string> {
  try {
    const data = await response.json();
    if (data && typeof data === 'object' && 'detail' in data) {
      return formatApiDetail((data as { detail: unknown }).detail);
    }
  } catch {
    /* ignore */
  }
  return response.statusText || fallback;
}
