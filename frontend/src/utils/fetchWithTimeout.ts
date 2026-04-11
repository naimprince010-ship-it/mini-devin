/** Fetch that always settles: avoids infinite spinners when the API is down or the proxy hangs. */

const DEFAULT_MS = 15_000;

export async function fetchWithTimeout(
  input: RequestInfo | URL,
  init: RequestInit & { timeoutMs?: number } = {}
): Promise<Response> {
  const { timeoutMs = DEFAULT_MS, ...rest } = init;
  const controller = new AbortController();
  const tid = setTimeout(() => controller.abort(), timeoutMs);
  const signal = rest.signal ?? controller.signal;
  try {
    return await fetch(input, { ...rest, signal });
  } finally {
    clearTimeout(tid);
  }
}

export async function fetchJsonWithTimeout<T>(
  url: string,
  init: RequestInit & { timeoutMs?: number } = {}
): Promise<T> {
  const res = await fetchWithTimeout(url, {
    ...init,
    headers: { Accept: 'application/json', ...init.headers },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new Error((err as { detail?: string }).detail || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export function isAbortError(e: unknown): boolean {
  return e instanceof DOMException && e.name === 'AbortError';
}
