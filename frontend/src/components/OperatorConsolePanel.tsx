import { type ReactNode, useCallback, useEffect, useMemo, useState } from 'react';
import {
  Activity,
  BadgeCheck,
  AlertTriangle,
  Clock3,
  PauseCircle,
  RefreshCw,
  RotateCcw,
  SearchCheck,
  ShieldAlert,
  ShieldX,
  SkipForward,
  SquareArrowOutUpRight,
  Timer,
  TrendingUp,
} from 'lucide-react';
import { getApiBase } from '../config/apiBase';
import OperatorActionConfirmDialog from './OperatorActionConfirmDialog';
import {
  type DeploymentEventItem,
  type IncidentLifecycleItem,
  type OpsDashboardFilters,
  type QueueDegradationItem,
  type RestartLoopItem,
  type RuntimeHealthItem,
  type ScoreHistoryItem,
  type WarningFrequencyItem,
  useOpsDashboard,
} from '../hooks/useOpsDashboard';
import {
  type OperatorActionType,
  useOperatorActions,
} from '../hooks/useOperatorActions';

interface ActionDialogState {
  actionType: OperatorActionType;
  title: string;
  description: string;
  target: string;
}

function formatTs(ts: string | undefined): string {
  if (!ts) {
    return 'n/a';
  }
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) {
    return ts;
  }
  return d.toLocaleString();
}

function fmtNumber(value: unknown, precision = 2): string {
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n)) {
    return '0';
  }
  return n.toFixed(precision);
}

function TrendBars({
  items,
  valueSelector,
  labelSelector,
  valueLabel,
}: {
  items: Array<WarningFrequencyItem | RestartLoopItem | ScoreHistoryItem>;
  valueSelector: (item: WarningFrequencyItem | RestartLoopItem | ScoreHistoryItem) => number;
  labelSelector: (item: WarningFrequencyItem | RestartLoopItem | ScoreHistoryItem) => string;
  valueLabel: (value: number) => string;
}) {
  if (items.length === 0) {
    return <p className="text-xs text-[#737373]">No data points in this window.</p>;
  }

  const values = items.map(valueSelector);
  const maxValue = Math.max(1, ...values);

  return (
    <div className="space-y-2">
      {items.slice(0, 8).map((item, idx) => {
        const value = valueSelector(item);
        const widthPct = Math.max(4, (value / maxValue) * 100);
        return (
          <div key={`${labelSelector(item)}-${idx}`} className="space-y-1">
            <div className="flex items-center justify-between text-[11px] text-[#a3a3a3]">
              <span>{labelSelector(item)}</span>
              <span>{valueLabel(value)}</span>
            </div>
            <div className="h-2 rounded bg-[#1a1a1a] overflow-hidden border border-[#2a2a2a]">
              <div className="h-full bg-[#00ff99]/70" style={{ width: `${widthPct}%` }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function TimelineList<T>({
  title,
  items,
  renderItem,
}: {
  title: string;
  items: T[];
  renderItem: (item: T, index: number) => ReactNode;
}) {
  return (
    <section className="rounded-xl border border-[#262626] bg-[#111111] p-3">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-[#a3a3a3] mb-3">{title}</h3>
      {items.length === 0 ? (
        <p className="text-xs text-[#737373]">No events in selected time window.</p>
      ) : (
        <div className="space-y-2">{items.slice(0, 8).map(renderItem)}</div>
      )}
    </section>
  );
}

export default function OperatorConsolePanel() {
  const dashboard = useOpsDashboard();
  const operatorActions = useOperatorActions();

  const [hoursInput, setHoursInput] = useState('24');
  const [startTimeInput, setStartTimeInput] = useState('');
  const [endTimeInput, setEndTimeInput] = useState('');
  const [pollSeconds, setPollSeconds] = useState(15);

  const [filters, setFilters] = useState<OpsDashboardFilters>({
    hours: 24,
    page: 1,
    page_size: 25,
  });
  const [actionDialog, setActionDialog] = useState<ActionDialogState | null>(null);

  const load = useCallback(
    async (next: OpsDashboardFilters) => {
      try {
        await dashboard.load(next);
      } catch {
        // surface via hook error state only
      }
    },
    [dashboard],
  );

  useEffect(() => {
    void load(filters);
  }, [filters, load]);

  useEffect(() => {
    if (pollSeconds <= 0) {
      return;
    }
    const id = setInterval(() => {
      void load(filters);
    }, Math.max(3, pollSeconds) * 1000);
    return () => clearInterval(id);
  }, [filters, load, pollSeconds]);

  const applyFilters = () => {
    const parsedHours = Number.parseInt(hoursInput, 10);
    setFilters((prev) => ({
      ...prev,
      hours: Number.isFinite(parsedHours) ? Math.max(1, Math.min(parsedHours, 24 * 30)) : prev.hours,
      start_time: startTimeInput ? new Date(startTimeInput).toISOString() : undefined,
      end_time: endTimeInput ? new Date(endTimeInput).toISOString() : undefined,
      page: 1,
    }));
  };

  const resetFilters = () => {
    setHoursInput('24');
    setStartTimeInput('');
    setEndTimeInput('');
    setFilters({ hours: 24, page: 1, page_size: 25 });
  };

  const summary = dashboard.data?.summary;
  const kpis = (summary?.kpis || {}) as Record<string, unknown>;

  const scoreValue = summary?.score?.value ?? 0;
  const scoreBand = summary?.score?.band ?? 'risk';
  const scoreBandColor = scoreBand === 'high' ? 'text-[#00ff99]' : scoreBand === 'conditional' ? 'text-[#fbbf24]' : 'text-[#f87171]';

  const pageInfo = useMemo(() => {
    const p = dashboard.data?.runtimeTimeline.pagination;
    if (!p) {
      return { page: filters.page, totalPages: 1 };
    }
    return { page: p.page, totalPages: p.total_pages };
  }, [dashboard.data?.runtimeTimeline.pagination, filters.page]);

  const movePage = (delta: number) => {
    setFilters((prev) => ({
      ...prev,
      page: Math.max(1, prev.page + delta),
    }));
  };

  const openActionDialog = (
    actionType: OperatorActionType,
    title: string,
    description: string,
    target: string,
  ) => {
    setActionDialog({ actionType, title, description, target });
  };

  const executeFromDialog = async (reason: string) => {
    if (!actionDialog) {
      return;
    }
    await operatorActions.executeAction({
      action_type: actionDialog.actionType,
      requested_by: 'operator.local',
      target: actionDialog.target,
      reason,
      metadata: {
        dashboard_window_hours: filters.hours,
        dashboard_page: filters.page,
      },
    });
    setActionDialog(null);
  };

  const openDiagnostics = (path: string) => {
    const apiBase = getApiBase().replace(/\/$/, '');
    let href = `${apiBase}${path}`;
    if (!href.startsWith('http://') && !href.startsWith('https://')) {
      const normalized = href.startsWith('/') ? href : `/${href}`;
      href = `${window.location.origin}${normalized}`;
    }
    window.open(href, '_blank', 'noopener,noreferrer');
    void operatorActions.executeAction({
      action_type: 'jump_to_diagnostics',
      requested_by: 'operator.local',
      target: path,
      reason: 'Quick diagnostics jump for human-in-the-loop inspection.',
      metadata: { source: 'operator_action_rail' },
    });
  };

  const runtimeBadgeColor =
    operatorActions.runtimeState === 'running'
      ? 'text-[#00ff99] border-[#00ff99]/30 bg-[#00ff99]/10'
      : operatorActions.runtimeState === 'paused'
        ? 'text-[#fbbf24] border-[#fbbf24]/30 bg-[#fbbf24]/10'
        : 'text-[#f87171] border-[#f87171]/30 bg-[#f87171]/10';

  const isBusy = (actionType: OperatorActionType): boolean =>
    Boolean(operatorActions.inFlightByAction[actionType]);

  return (
    <div className="h-full flex flex-col bg-[#0a0a0a] text-white overflow-y-auto p-4 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold flex items-center gap-2">
            <Activity size={15} className="text-[#00ff99]" />
            Operator Console
          </h2>
          <p className="text-[11px] text-[#737373] mt-1">
            Last update: {dashboard.lastUpdated ? formatTs(dashboard.lastUpdated) : 'not loaded yet'}
          </p>
        </div>
        <button
          type="button"
          onClick={() => void load(filters)}
          disabled={dashboard.loading}
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border border-[#2a2a2a] bg-[#151515] text-xs text-[#d4d4d4] hover:bg-[#1c1c1c] disabled:opacity-60"
        >
          <RefreshCw size={12} className={dashboard.loading ? 'animate-spin' : ''} />
          Refresh
        </button>
      </div>

      <section className="rounded-xl border border-[#262626] bg-[#111111] p-3">
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-3">
          <label className="text-xs text-[#a3a3a3]">
            Window (hours)
            <input
              type="number"
              min={1}
              max={24 * 30}
              value={hoursInput}
              onChange={(e) => setHoursInput(e.target.value)}
              className="mt-1 w-full rounded border border-[#2a2a2a] bg-[#0d0d0d] px-2 py-1.5 text-sm text-white"
            />
          </label>
          <label className="text-xs text-[#a3a3a3]">
            Start time
            <input
              type="datetime-local"
              value={startTimeInput}
              onChange={(e) => setStartTimeInput(e.target.value)}
              className="mt-1 w-full rounded border border-[#2a2a2a] bg-[#0d0d0d] px-2 py-1.5 text-sm text-white"
            />
          </label>
          <label className="text-xs text-[#a3a3a3]">
            End time
            <input
              type="datetime-local"
              value={endTimeInput}
              onChange={(e) => setEndTimeInput(e.target.value)}
              className="mt-1 w-full rounded border border-[#2a2a2a] bg-[#0d0d0d] px-2 py-1.5 text-sm text-white"
            />
          </label>
          <label className="text-xs text-[#a3a3a3]">
            Poll interval (seconds)
            <input
              type="number"
              min={0}
              max={300}
              value={pollSeconds}
              onChange={(e) => setPollSeconds(Math.max(0, Number.parseInt(e.target.value || '0', 10)))}
              className="mt-1 w-full rounded border border-[#2a2a2a] bg-[#0d0d0d] px-2 py-1.5 text-sm text-white"
            />
          </label>
          <div className="flex items-end gap-2">
            <button
              type="button"
              onClick={applyFilters}
              className="flex-1 rounded border border-[#00ff99]/30 bg-[#00ff99]/10 px-3 py-2 text-xs font-medium text-[#00ff99] hover:bg-[#00ff99]/20"
            >
              Apply
            </button>
            <button
              type="button"
              onClick={resetFilters}
              className="flex-1 rounded border border-[#2a2a2a] bg-[#151515] px-3 py-2 text-xs text-[#d4d4d4] hover:bg-[#1c1c1c]"
            >
              Reset
            </button>
          </div>
        </div>
      </section>

      {dashboard.error && (
        <div className="rounded-lg border border-[#7f1d1d] bg-[#220f0f] px-3 py-2 text-xs text-[#fca5a5] flex items-center justify-between gap-2">
          <span>{dashboard.error}</span>
          <button type="button" onClick={dashboard.clearError} className="text-[#fca5a5] underline">
            Dismiss
          </button>
        </div>
      )}

      <section className="rounded-xl border border-[#262626] bg-[#111111] p-3 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-[#a3a3a3]">Operator Action Rail</h3>
          <div className="flex items-center gap-2">
            <span className={`text-[10px] px-2 py-1 rounded border ${runtimeBadgeColor}`}>
              runtime: {operatorActions.runtimeState}
            </span>
            <span className="text-[10px] px-2 py-1 rounded border border-[#2a2a2a] bg-[#161616] text-[#d4d4d4]">
              pending: {operatorActions.pendingCount}
            </span>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-2">
          <button
            type="button"
            disabled={isBusy('acknowledge_incident')}
            onClick={() =>
              openActionDialog(
                'acknowledge_incident',
                'Acknowledge Incident',
                'This records operator acknowledgment only. No automation is triggered.',
                'incident.current',
              )
            }
            className="rounded border border-[#2a2a2a] bg-[#151515] px-3 py-2 text-xs text-[#d4d4d4] inline-flex items-center gap-2 disabled:opacity-50"
          >
            <BadgeCheck size={13} /> Acknowledge Incident
          </button>

          <button
            type="button"
            disabled={isBusy('mark_investigation_started')}
            onClick={() =>
              openActionDialog(
                'mark_investigation_started',
                'Mark Investigation Started',
                'This marks HITL investigation start for audit and handoff tracking.',
                'incident.current',
              )
            }
            className="rounded border border-[#2a2a2a] bg-[#151515] px-3 py-2 text-xs text-[#d4d4d4] inline-flex items-center gap-2 disabled:opacity-50"
          >
            <SearchCheck size={13} /> Start Investigation
          </button>

          <button
            type="button"
            disabled={isBusy('pause_runtime') || operatorActions.runtimeState !== 'running'}
            onClick={() =>
              openActionDialog(
                'pause_runtime',
                'Pause Runtime (Scaffold)',
                'Safe scaffold only: records a pause intent and updates operator state badge.',
                'runtime.main',
              )
            }
            className="rounded border border-[#2a2a2a] bg-[#151515] px-3 py-2 text-xs text-[#d4d4d4] inline-flex items-center gap-2 disabled:opacity-50"
          >
            <PauseCircle size={13} /> Pause Runtime
          </button>

          <button
            type="button"
            disabled={isBusy('resume_runtime') || operatorActions.runtimeState === 'running'}
            onClick={() =>
              openActionDialog(
                'resume_runtime',
                'Resume Runtime (Scaffold)',
                'Safe scaffold only: records resume intent without backend mutation.',
                'runtime.main',
              )
            }
            className="rounded border border-[#2a2a2a] bg-[#151515] px-3 py-2 text-xs text-[#d4d4d4] inline-flex items-center gap-2 disabled:opacity-50"
          >
            <SkipForward size={13} /> Resume Runtime
          </button>

          <button
            type="button"
            disabled={isBusy('retry_task')}
            onClick={() =>
              openActionDialog(
                'retry_task',
                'Retry Task (Scaffold)',
                'Creates a retry-task intent only. Backend execution is intentionally stubbed.',
                'task.pending',
              )
            }
            className="rounded border border-[#2a2a2a] bg-[#151515] px-3 py-2 text-xs text-[#d4d4d4] inline-flex items-center gap-2 disabled:opacity-50"
          >
            <RotateCcw size={13} /> Retry Task
          </button>

          <button
            type="button"
            disabled={isBusy('replay_session')}
            onClick={() =>
              openActionDialog(
                'replay_session',
                'Replay Session (Scaffold)',
                'Creates a replay-session intent only for controlled future backend integration.',
                'session.active',
              )
            }
            className="rounded border border-[#2a2a2a] bg-[#151515] px-3 py-2 text-xs text-[#d4d4d4] inline-flex items-center gap-2 disabled:opacity-50"
          >
            <RotateCcw size={13} /> Replay Session
          </button>

          <button
            type="button"
            disabled={isBusy('quarantine_runtime') || operatorActions.runtimeState === 'quarantined'}
            onClick={() =>
              openActionDialog(
                'quarantine_runtime',
                'Quarantine Runtime (Scaffold)',
                'Safe scaffold only: records quarantine intent with explicit confirmation gate.',
                'runtime.main',
              )
            }
            className="rounded border border-[#7f1d1d] bg-[#2a1212] px-3 py-2 text-xs text-[#fca5a5] inline-flex items-center gap-2 disabled:opacity-50"
          >
            <ShieldX size={13} /> Quarantine Runtime
          </button>
        </div>

        <div className="border-t border-[#222] pt-3">
          <p className="text-[11px] text-[#737373] mb-2">Jump to diagnostics quick actions</p>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => openDiagnostics('/ops/diagnostics')}
              className="rounded border border-[#2a2a2a] bg-[#151515] px-3 py-1.5 text-xs text-[#d4d4d4] inline-flex items-center gap-1"
            >
              <SquareArrowOutUpRight size={12} /> Runtime Diagnostics
            </button>
            <button
              type="button"
              onClick={() => openDiagnostics('/ops/preflight')}
              className="rounded border border-[#2a2a2a] bg-[#151515] px-3 py-1.5 text-xs text-[#d4d4d4] inline-flex items-center gap-1"
            >
              <SquareArrowOutUpRight size={12} /> Preflight Report
            </button>
            <button
              type="button"
              onClick={() => openDiagnostics('/ops/telemetry/export?hours=1')}
              className="rounded border border-[#2a2a2a] bg-[#151515] px-3 py-1.5 text-xs text-[#d4d4d4] inline-flex items-center gap-1"
            >
              <SquareArrowOutUpRight size={12} /> Telemetry Export
            </button>
          </div>
        </div>
      </section>

      <section className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-3">
        <div className="rounded-xl border border-[#262626] bg-[#111111] p-3">
          <p className="text-[11px] text-[#737373]">Operational Score</p>
          <p className={`text-2xl font-semibold mt-1 ${scoreBandColor}`}>{fmtNumber(scoreValue, 1)}</p>
          <p className="text-xs text-[#a3a3a3] mt-1 uppercase tracking-wide">{scoreBand}</p>
        </div>
        <div className="rounded-xl border border-[#262626] bg-[#111111] p-3">
          <p className="text-[11px] text-[#737373]">Readiness Ratio</p>
          <p className="text-2xl font-semibold mt-1">{fmtNumber(kpis.readiness_success_ratio, 2)}</p>
          <p className="text-xs text-[#a3a3a3] mt-1">Window success fraction</p>
        </div>
        <div className="rounded-xl border border-[#262626] bg-[#111111] p-3">
          <p className="text-[11px] text-[#737373]">Warnings / Min</p>
          <p className="text-2xl font-semibold mt-1">{fmtNumber(kpis.warning_frequency_per_min, 3)}</p>
          <p className="text-xs text-[#a3a3a3] mt-1">Noise trend</p>
        </div>
        <div className="rounded-xl border border-[#262626] bg-[#111111] p-3">
          <p className="text-[11px] text-[#737373]">Queue Degraded Dwell</p>
          <p className="text-2xl font-semibold mt-1">{fmtNumber(kpis.queue_degraded_dwell_seconds, 0)}s</p>
          <p className="text-xs text-[#a3a3a3] mt-1">Cumulative window dwell</p>
        </div>
        <div className="rounded-xl border border-[#262626] bg-[#111111] p-3">
          <p className="text-[11px] text-[#737373]">Incidents MTTR</p>
          <p className="text-2xl font-semibold mt-1">{fmtNumber(kpis.incident_mttr_seconds, 0)}s</p>
          <p className="text-xs text-[#a3a3a3] mt-1">Mean time to resolve</p>
        </div>
      </section>

      <section className="grid grid-cols-1 xl:grid-cols-2 gap-3">
        <TimelineList<RuntimeHealthItem>
          title="Runtime Health Timeline"
          items={dashboard.data?.runtimeTimeline.items ?? []}
          renderItem={(item, i) => (
            <div key={`${item.time}-${i}`} className="rounded-lg border border-[#222] bg-[#0d0d0d] px-3 py-2 text-xs">
              <div className="flex items-center justify-between">
                <span className={`font-medium ${item.degraded ? 'text-[#fbbf24]' : 'text-[#00ff99]'}`}>{item.status}</span>
                <span className="text-[#737373]">{formatTs(item.time)}</span>
              </div>
              <div className="mt-1 text-[#a3a3a3]">
                readiness: {String(item.readiness)} · queue: {item.queue_active || 'n/a'}
              </div>
            </div>
          )}
        />

        <TimelineList<DeploymentEventItem>
          title="Deployment Event Timeline"
          items={dashboard.data?.deploymentTimeline.items ?? []}
          renderItem={(item, i) => (
            <div key={`${item.time}-${i}`} className="rounded-lg border border-[#222] bg-[#0d0d0d] px-3 py-2 text-xs">
              <div className="flex items-center justify-between">
                <span className="font-medium text-[#60a5fa]">{item.phase}</span>
                <span className="text-[#737373]">{formatTs(item.time)}</span>
              </div>
            </div>
          )}
        />

        <TimelineList<QueueDegradationItem>
          title="Queue Degradation Timeline"
          items={dashboard.data?.queueTimeline.items ?? []}
          renderItem={(item, i) => (
            <div key={`${item.start_time}-${i}`} className="rounded-lg border border-[#222] bg-[#0d0d0d] px-3 py-2 text-xs">
              <div className="flex items-center justify-between">
                <span className="font-medium text-[#fbbf24] inline-flex items-center gap-1">
                  <ShieldAlert size={12} /> degraded
                </span>
                <span className="text-[#737373]">{fmtNumber(item.duration_seconds, 0)}s</span>
              </div>
              <div className="mt-1 text-[#a3a3a3]">
                {formatTs(item.start_time)}{item.end_time ? ` -> ${formatTs(item.end_time)}` : ' -> active'}
              </div>
            </div>
          )}
        />

        <TimelineList<IncidentLifecycleItem>
          title="Incident Lifecycle Timeline"
          items={dashboard.data?.incidentTimeline.items ?? []}
          renderItem={(item, i) => (
            <div key={`${item.time}-${item.source}-${i}`} className="rounded-lg border border-[#222] bg-[#0d0d0d] px-3 py-2 text-xs">
              <div className="flex items-center justify-between">
                <span className={`font-medium inline-flex items-center gap-1 ${item.event === 'opened' ? 'text-[#f87171]' : 'text-[#34d399]'}`}>
                  <AlertTriangle size={12} /> {item.source}.{item.event}
                </span>
                <span className="text-[#737373]">{formatTs(item.time)}</span>
              </div>
              <div className="mt-1 text-[#a3a3a3]">
                open={item.incident_open_count ?? '-'} crash_failures={item.crash_loop_failures ?? '-'}
              </div>
            </div>
          )}
        />
      </section>

      <section className="grid grid-cols-1 xl:grid-cols-3 gap-3">
        <section className="rounded-xl border border-[#262626] bg-[#111111] p-3">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-[#a3a3a3] mb-3 inline-flex items-center gap-1">
            <TrendingUp size={12} /> Score History
          </h3>
          <TrendBars
            items={dashboard.data?.scoreHistory.items ?? []}
            valueSelector={(item) => (item as ScoreHistoryItem).value}
            labelSelector={(item) => new Date((item as ScoreHistoryItem).time).toLocaleTimeString()}
            valueLabel={(v) => fmtNumber(v, 1)}
          />
        </section>

        <section className="rounded-xl border border-[#262626] bg-[#111111] p-3">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-[#a3a3a3] mb-3 inline-flex items-center gap-1">
            <Clock3 size={12} /> Warning Frequency
          </h3>
          <TrendBars
            items={dashboard.data?.warningTrend.items ?? []}
            valueSelector={(item) => (item as WarningFrequencyItem).warning_frequency_per_min}
            labelSelector={(item) => new Date((item as WarningFrequencyItem).bucket_start).toLocaleTimeString()}
            valueLabel={(v) => `${fmtNumber(v, 2)}/min`}
          />
        </section>

        <section className="rounded-xl border border-[#262626] bg-[#111111] p-3">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-[#a3a3a3] mb-3 inline-flex items-center gap-1">
            <Timer size={12} /> Restart Loops
          </h3>
          <TrendBars
            items={dashboard.data?.restartTrend.items ?? []}
            valueSelector={(item) => (item as RestartLoopItem).crash_loop_failures}
            labelSelector={(item) => new Date((item as RestartLoopItem).bucket_start).toLocaleTimeString()}
            valueLabel={(v) => `${fmtNumber(v, 0)} failures`}
          />
        </section>
      </section>

      <section className="rounded-xl border border-[#262626] bg-[#111111] p-3">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-[#a3a3a3] mb-3">Operator Activity Timeline</h3>
        {operatorActions.records.length === 0 ? (
          <p className="text-xs text-[#737373]">No operator actions yet.</p>
        ) : (
          <div className="space-y-2">
            {operatorActions.records.slice(0, 12).map((entry) => {
              const statusColor =
                entry.status === 'succeeded'
                  ? 'text-[#34d399] border-[#34d399]/30 bg-[#34d399]/10'
                  : entry.status === 'failed'
                    ? 'text-[#f87171] border-[#f87171]/30 bg-[#f87171]/10'
                    : 'text-[#fbbf24] border-[#fbbf24]/30 bg-[#fbbf24]/10';
              return (
                <div
                  key={entry.audit.action_id}
                  className="rounded-lg border border-[#222] bg-[#0d0d0d] px-3 py-2 text-xs space-y-1"
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-medium text-[#d4d4d4]">{entry.audit.action_type}</span>
                    <span className={`text-[10px] px-2 py-0.5 rounded border ${statusColor}`}>{entry.status}</span>
                  </div>
                  <p className="text-[#a3a3a3]">{entry.message}</p>
                  <p className="text-[10px] text-[#737373]">
                    requested={formatTs(entry.audit.requested_at)} · target={entry.audit.target} · dry_run=
                    {String(entry.audit.dry_run)}
                  </p>
                  <details>
                    <summary className="text-[10px] text-[#60a5fa] cursor-pointer">audit payload</summary>
                    <pre className="mt-1 p-2 text-[10px] text-[#9ca3af] border border-[#1f2937] rounded bg-[#0a0a0a] overflow-x-auto">
                      {JSON.stringify(entry.audit, null, 2)}
                    </pre>
                  </details>
                </div>
              );
            })}
          </div>
        )}
      </section>

      <section className="rounded-xl border border-[#262626] bg-[#111111] p-3 flex items-center justify-between">
        <p className="text-xs text-[#a3a3a3]">
          Page {pageInfo.page} / {pageInfo.totalPages}
        </p>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => movePage(-1)}
            disabled={filters.page <= 1 || dashboard.loading}
            className="rounded border border-[#2a2a2a] bg-[#151515] px-3 py-1.5 text-xs text-[#d4d4d4] disabled:opacity-50"
          >
            Prev
          </button>
          <button
            type="button"
            onClick={() => movePage(1)}
            disabled={filters.page >= pageInfo.totalPages || dashboard.loading}
            className="rounded border border-[#2a2a2a] bg-[#151515] px-3 py-1.5 text-xs text-[#d4d4d4] disabled:opacity-50"
          >
            Next
          </button>
        </div>
      </section>

      <OperatorActionConfirmDialog
        open={Boolean(actionDialog)}
        title={actionDialog?.title || 'Confirm action'}
        description={actionDialog?.description || ''}
        confirmLabel={actionDialog ? `Confirm ${actionDialog.actionType}` : 'Confirm'}
        onCancel={() => setActionDialog(null)}
        onConfirm={({ reason }) => {
          void executeFromDialog(reason);
        }}
      />
    </div>
  );
}
