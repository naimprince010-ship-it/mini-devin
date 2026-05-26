import { useCallback, useState } from 'react';
import { getApiBase } from '../config/apiBase';
import { fetchWithTimeout, isAbortError } from '../utils/fetchWithTimeout';

const DEFAULT_TIMEOUT_MS = 15_000;

export interface OpsDashboardWindow {
  hours: number;
  start_time: string;
  end_time: string;
}

export interface OpsDashboardPagination {
  page: number;
  page_size: number;
  total_items: number;
  total_pages: number;
}

export interface OpsDashboardSummary {
  schema: string;
  generated_at: string;
  window: OpsDashboardWindow;
  kpis: Record<string, unknown>;
  score: {
    value?: number;
    band?: string;
    components?: Record<string, number>;
  };
  timeline_counts: Record<string, number>;
}

export interface RuntimeHealthItem {
  time: string;
  status: string;
  readiness: boolean;
  degraded: boolean;
  queue_requested?: string | null;
  queue_active?: string | null;
}

export interface DeploymentEventItem {
  time: string;
  phase: string;
  tags: Record<string, unknown>;
}

export interface QueueDegradationItem {
  start_time: string;
  end_time?: string | null;
  duration_seconds: number;
  queue_requested?: string | null;
  queue_active?: string | null;
}

export interface IncidentLifecycleItem {
  time: string;
  source: string;
  event: string;
  incident_open_count?: number | null;
  crash_loop_failures?: number | null;
}

export interface ScoreHistoryItem {
  time: string;
  value: number;
  band: string;
  components: Record<string, number>;
}

export interface WarningFrequencyItem {
  bucket_start: string;
  warning_count: number;
  warning_frequency_per_min: number;
}

export interface RestartLoopItem {
  bucket_start: string;
  crash_loop_active: boolean;
  crash_loop_failures: number;
}

export interface OpsDashboardListResponse<T> {
  schema: string;
  generated_at: string;
  window: OpsDashboardWindow;
  pagination: OpsDashboardPagination;
  items: T[];
}

export interface OpsDashboardFilters {
  hours: number;
  start_time?: string;
  end_time?: string;
  page: number;
  page_size: number;
}

export interface OpsDashboardData {
  summary: OpsDashboardSummary;
  runtimeTimeline: OpsDashboardListResponse<RuntimeHealthItem>;
  deploymentTimeline: OpsDashboardListResponse<DeploymentEventItem>;
  queueTimeline: OpsDashboardListResponse<QueueDegradationItem>;
  incidentTimeline: OpsDashboardListResponse<IncidentLifecycleItem>;
  scoreHistory: OpsDashboardListResponse<ScoreHistoryItem>;
  warningTrend: OpsDashboardListResponse<WarningFrequencyItem>;
  restartTrend: OpsDashboardListResponse<RestartLoopItem>;
}

function toQueryString(filters: OpsDashboardFilters): string {
  const params = new URLSearchParams();
  params.set('hours', String(Math.max(1, filters.hours)));
  params.set('page', String(Math.max(1, filters.page)));
  params.set('page_size', String(Math.max(1, filters.page_size)));
  if (filters.start_time && filters.start_time.trim()) {
    params.set('start_time', filters.start_time.trim());
  }
  if (filters.end_time && filters.end_time.trim()) {
    params.set('end_time', filters.end_time.trim());
  }
  return params.toString();
}

async function fetchJson<T>(endpoint: string): Promise<T> {
  const apiBase = getApiBase();
  let response: Response;
  try {
    response = await fetchWithTimeout(`${apiBase}${endpoint}`, {
      method: 'GET',
      timeoutMs: DEFAULT_TIMEOUT_MS,
      headers: {
        Accept: 'application/json',
      },
    });
  } catch (e) {
    if (isAbortError(e)) {
      throw new Error('Dashboard request timed out.');
    }
    throw e;
  }

  if (!response.ok) {
    const text = await response.text().catch(() => `HTTP ${response.status}`);
    throw new Error(text || `HTTP ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export function useOpsDashboard() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<OpsDashboardData | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);

  const clearError = useCallback(() => setError(null), []);

  const load = useCallback(async (filters: OpsDashboardFilters): Promise<OpsDashboardData> => {
    setLoading(true);
    setError(null);
    const query = toQueryString(filters);
    try {
      const [
        summary,
        runtimeTimeline,
        deploymentTimeline,
        queueTimeline,
        incidentTimeline,
        scoreHistory,
        warningTrend,
        restartTrend,
      ] = await Promise.all([
        fetchJson<OpsDashboardSummary>(`/ops/dashboard/summary?${query}`),
        fetchJson<OpsDashboardListResponse<RuntimeHealthItem>>(`/ops/dashboard/timeline/runtime-health?${query}`),
        fetchJson<OpsDashboardListResponse<DeploymentEventItem>>(`/ops/dashboard/timeline/deployments?${query}`),
        fetchJson<OpsDashboardListResponse<QueueDegradationItem>>(`/ops/dashboard/timeline/queue-degradation?${query}`),
        fetchJson<OpsDashboardListResponse<IncidentLifecycleItem>>(`/ops/dashboard/timeline/incidents?${query}`),
        fetchJson<OpsDashboardListResponse<ScoreHistoryItem>>(`/ops/dashboard/trends/score-history?${query}`),
        fetchJson<OpsDashboardListResponse<WarningFrequencyItem>>(`/ops/dashboard/trends/warning-frequency?${query}`),
        fetchJson<OpsDashboardListResponse<RestartLoopItem>>(`/ops/dashboard/trends/restart-loops?${query}`),
      ]);

      const nextData: OpsDashboardData = {
        summary,
        runtimeTimeline,
        deploymentTimeline,
        queueTimeline,
        incidentTimeline,
        scoreHistory,
        warningTrend,
        restartTrend,
      };
      setData(nextData);
      setLastUpdated(new Date().toISOString());
      return nextData;
    } catch (e) {
      const message = e instanceof Error ? e.message : 'Failed to load dashboard data';
      setError(message);
      throw e;
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    data,
    loading,
    error,
    lastUpdated,
    clearError,
    load,
  };
}
