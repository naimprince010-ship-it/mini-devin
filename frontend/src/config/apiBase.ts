/**
 * Resolve the JSON API base path (no trailing slash).
 * - VITE_API_URL: absolute API origin + /api (split deploy).
 * - window.__DEVIN_API_BASE__: manual override (e.g. injected in index.html).
 * - Dashboard under /app: use /app/api; backend rewrites to /api (avoids POST 405 on some edges).
 * - Otherwise: /api (same-origin monolith).
 */
export function getApiBase(): string {
  const env = import.meta.env.VITE_API_URL;
  if (env !== undefined && env !== null && String(env).trim() !== '') {
    return `${String(env).replace(/\/$/, '')}/api`;
  }
  if (typeof window === 'undefined') {
    return '/api';
  }
  const custom = (window as unknown as { __DEVIN_API_BASE__?: string }).__DEVIN_API_BASE__;
  if (custom !== undefined && custom !== null && String(custom).trim() !== '') {
    return String(custom).replace(/\/$/, '');
  }
  const path = window.location.pathname || '';
  if (path === '/app' || path.startsWith('/app/')) {
    // Vite only proxies `/api` in dev; production uses `/app/api` + server rewrite for strict CDNs.
    if (import.meta.env.DEV) {
      return '/api';
    }
    return '/app/api';
  }
  return '/api';
}

/** SSE URL for session event stream (same JSON as WebSocket). */
export function getSessionSseUrl(sessionId: string): string {
  const base = getApiBase().replace(/\/$/, '');
  const suffix = `/sessions/${sessionId}/stream`;
  if (base.startsWith('http://') || base.startsWith('https://')) {
    return `${base}${suffix}`;
  }
  if (typeof window !== 'undefined') {
    return `${window.location.origin}${base.startsWith('/') ? base : `/${base}`}${suffix}`;
  }
  return `${base}${suffix}`;
}

/** WebSocket URL for /api/ws[/sessionId]. */
export function getApiWsUrl(sessionId?: string): string {
  const base = getApiBase().replace(/\/$/, '');
  const pathSuffix = sessionId ? `/ws/${sessionId}` : '/ws';

  if (base.startsWith('http://') || base.startsWith('https://')) {
    const u = new URL(base);
    const wsScheme = u.protocol === 'https:' ? 'wss:' : 'ws:';
    const originPath = (u.pathname || '').replace(/\/$/, '') || '';
    return `${wsScheme}//${u.host}${originPath}${pathSuffix}`;
  }

  const wsScheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${wsScheme}//${window.location.host}${base}${pathSuffix}`;
}
