import { useCallback, useEffect, useMemo, useState } from 'react';
import { Activity, AlertTriangle, Gauge, RefreshCw } from 'lucide-react';
import { useApi } from '../hooks/useApi';
import type { WorkerPoolMetricsResponse } from '../types';

type WorkerPoolStatusCardProps = {
  sessionId: string;
  isDark: boolean;
};

const POLL_MS = 5000;

function relativeUpdated(ts: string): string {
  const ms = Date.parse(ts);
  if (!Number.isFinite(ms)) return 'just now';
  const diffSec = Math.max(0, Math.floor((Date.now() - ms) / 1000));
  if (diffSec < 5) return 'just now';
  if (diffSec < 60) return `${diffSec}s ago`;
  const mins = Math.floor(diffSec / 60);
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  return `${hours}h ago`;
}

export function WorkerPoolStatusCard({ sessionId, isDark }: WorkerPoolStatusCardProps) {
  const api = useApi();
  const [data, setData] = useState<WorkerPoolMetricsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(async (silent = false) => {
    if (!sessionId) return;
    if (!silent) setRefreshing(true);
    try {
      const res = await api.getWorkerPoolMetrics(sessionId);
      setData(res);
      setError(null);
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Unable to load worker pool metrics';
      if (msg.toLowerCase().includes('run orchestration first')) {
        setData(null);
        setError('No orchestration run yet');
      } else {
        setError(msg);
      }
    } finally {
      if (!silent) setRefreshing(false);
    }
  }, [api, sessionId]);

  useEffect(() => {
    setData(null);
    setError(null);
    void load(false);
    const timer = window.setInterval(() => {
      void load(true);
    }, POLL_MS);
    return () => window.clearInterval(timer);
  }, [load]);

  const roleRows = useMemo(() => {
    if (!data) return [];
    return Object.entries(data.pools)
      .map(([role, stats]) => ({ role, ...stats }))
      .sort((a, b) => {
        if (a.saturated !== b.saturated) return a.saturated ? -1 : 1;
        if (a.waiting !== b.waiting) return b.waiting - a.waiting;
        return b.utilization - a.utilization;
      })
      .slice(0, 4);
  }, [data]);

  const saturatedCount = data?.saturated_roles.length ?? 0;

  return (
    <div
      className={`min-w-[220px] rounded-xl border px-3 py-2 ${
        isDark
          ? 'border-[#2a2d34] bg-[#111318] text-[#d8dbe2]'
          : 'border-[#d9e0ea] bg-[#f8fafc] text-[#1f2937]'
      }`}
    >
      <div className="mb-1.5 flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <Gauge className={`h-3.5 w-3.5 ${isDark ? 'text-[#7dd3fc]' : 'text-[#0ea5e9]'}`} />
          <span className="text-[11px] font-semibold tracking-wide">Worker Pools</span>
        </div>
        <button
          type="button"
          onClick={() => void load(false)}
          className={`inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[10px] ${
            isDark ? 'bg-[#1b1f27] text-[#9aa4b2] hover:text-white' : 'bg-[#e5edf7] text-[#4b5563] hover:text-[#111827]'
          }`}
          title="Refresh metrics"
        >
          <RefreshCw className={`h-3 w-3 ${refreshing ? 'animate-spin' : ''}`} />
          {refreshing ? 'Polling' : 'Refresh'}
        </button>
      </div>

      {!data ? (
        <div className={`text-[11px] ${isDark ? 'text-[#9aa4b2]' : 'text-[#64748b]'}`}>
          {error || 'Loading metrics...'}
        </div>
      ) : (
        <>
          <div className="mb-2 flex items-center gap-2 text-[10px]">
            <span
              className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 ${
                saturatedCount > 0
                  ? isDark
                    ? 'bg-[#3b1f1f] text-[#fca5a5]'
                    : 'bg-[#fee2e2] text-[#b91c1c]'
                  : isDark
                    ? 'bg-[#183224] text-[#86efac]'
                    : 'bg-[#dcfce7] text-[#166534]'
              }`}
            >
              {saturatedCount > 0 ? <AlertTriangle className="h-3 w-3" /> : <Activity className="h-3 w-3" />}
              {saturatedCount > 0 ? `${saturatedCount} saturated` : 'healthy'}
            </span>
            <span className={isDark ? 'text-[#7c8696]' : 'text-[#64748b]'}>
              {data.source === 'active' ? 'live' : 'last run'} • {relativeUpdated(data.updated_at)}
            </span>
          </div>

          <div className="space-y-1">
            {roleRows.map((row) => (
              <div key={row.role} className="grid grid-cols-[72px_1fr_auto] items-center gap-2 text-[10px]">
                <span className={`truncate font-medium uppercase tracking-wide ${isDark ? 'text-[#c4c9d4]' : 'text-[#334155]'}`}>
                  {row.role}
                </span>
                <div className={`h-1.5 overflow-hidden rounded-full ${isDark ? 'bg-[#222733]' : 'bg-[#dbe7f5]'}`}>
                  <div
                    className={`h-full ${row.saturated ? 'bg-[#ef4444]' : 'bg-[#22c55e]'}`}
                    style={{ width: `${Math.max(6, Math.min(100, Math.round(row.utilization * 100)))}%` }}
                  />
                </div>
                <span className={isDark ? 'text-[#9aa4b2]' : 'text-[#475569]'}>
                  {row.active}/{row.limit}
                  {row.waiting > 0 ? ` (+${row.waiting})` : ''}
                </span>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
