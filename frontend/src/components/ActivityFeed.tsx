import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Activity,
  Brain,
  Terminal,
  FileCode,
  FolderGit2,
  Sparkles,
  AlertCircle,
  RefreshCw,
  ListOrdered,
} from 'lucide-react';
import { fetchWithTimeout, isAbortError } from '../utils/fetchWithTimeout';
import { getApiBase } from '../config/apiBase';
import type { ActivityFeedEvent } from '../types';

function truncate(s: string, max: number): string {
  const t = s.trim();
  if (t.length <= max) return t;
  return `${t.slice(0, max - 1)}…`;
}

/** Turn one JSONL row into a short title + optional detail for the UI. */
export function formatActivityEvent(ev: ActivityFeedEvent): {
  kind: string;
  title: string;
  detail?: string;
} {
  const t = String(ev.type ?? 'event');
  const planStep = ev.plan_step != null ? String(ev.plan_step) : '';

  if (t === 'think') {
    const raw = String(ev.text ?? '');
    return {
      kind: 'think',
      title: 'Agent thought',
      detail: truncate(raw.replace(/\*\*Think:\*\*/gi, '').trim(), 220) || '(no summary)',
    };
  }

  if (t === 'observe') {
    const delta = ev.filesystem_delta as { added_sorted?: string[]; removed_sorted?: string[] } | undefined;
    const added = delta?.added_sorted ?? [];
    const removed = delta?.removed_sorted ?? [];
    const tool = String(ev.tool ?? 'tool');
    const parts: string[] = [];
    if (added.length) parts.push(`added: ${added.slice(0, 12).join(', ')}${added.length > 12 ? '…' : ''}`);
    if (removed.length) parts.push(`removed: ${removed.slice(0, 8).join(', ')}${removed.length > 8 ? '…' : ''}`);
    const exit = ev.exit_code != null ? `exit ${ev.exit_code}` : '';
    return {
      kind: 'observe',
      title: `After ${tool}${exit ? ` (${exit})` : ''}`,
      detail: parts.length ? `Files: ${parts.join(' · ')}` : 'No file tree changes detected',
    };
  }

  if (t === 'act') {
    const tool = String(ev.tool ?? '');
    if (tool === 'terminal') {
      const cmd = String(ev.command ?? '');
      return {
        kind: 'terminal',
        title: 'Ran command',
        detail: truncate(cmd, 180),
      };
    }
    if (tool === 'editor') {
      const action = String(ev.action ?? 'edit');
      const path = String(ev.path ?? '');
      return {
        kind: 'editor',
        title: `Editor: ${action}`,
        detail: path || undefined,
      };
    }
    return { kind: 'act', title: `Action: ${tool}`, detail: planStep ? `Step ${planStep}` : undefined };
  }

  if (t === 'task_start') {
    return {
      kind: 'task',
      title: 'Task started',
      detail: truncate(String(ev.goal ?? ''), 200) || undefined,
    };
  }

  if (t === 'auto_verify') {
    const tool = String(ev.tool ?? 'verify');
    const code = ev.exit_code != null ? String(ev.exit_code) : '?';
    const path = ev.path != null ? String(ev.path) : '';
    return {
      kind: 'verify',
      title: `Auto-verify (${tool})`,
      detail: path ? `${path} · exit ${code}` : `exit ${code}`,
    };
  }

  return {
    kind: t,
    title: t.replace(/_/g, ' '),
    detail: planStep ? `Step ${planStep}` : undefined,
  };
}

function rowIcon(kind: string) {
  if (kind === 'think') return <Brain size={12} className="text-violet-400 flex-shrink-0 mt-0.5" />;
  if (kind === 'observe') return <FolderGit2 size={12} className="text-sky-400 flex-shrink-0 mt-0.5" />;
  if (kind === 'terminal') return <Terminal size={12} className="text-[#00ff99] flex-shrink-0 mt-0.5" />;
  if (kind === 'editor') return <FileCode size={12} className="text-amber-400 flex-shrink-0 mt-0.5" />;
  if (kind === 'task') return <ListOrdered size={12} className="text-[#3399ff] flex-shrink-0 mt-0.5" />;
  if (kind === 'verify') return <Sparkles size={12} className="text-emerald-400 flex-shrink-0 mt-0.5" />;
  return <Activity size={12} className="text-[#525252] flex-shrink-0 mt-0.5" />;
}

interface ActivityFeedProps {
  sessionId?: string;
  /** When the agent is running, poll a bit faster. */
  isRunning?: boolean;
}

export const ActivityFeed: React.FC<ActivityFeedProps> = ({ sessionId, isRunning }) => {
  const [events, setEvents] = useState<ActivityFeedEvent[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const load = useCallback(async () => {
    if (!sessionId) {
      setEvents([]);
      return;
    }
    setLoading(true);
    setErr(null);
    const apiBase = getApiBase();
    try {
      const res = await fetchWithTimeout(`${apiBase}/sessions/${sessionId}/activity-feed?limit=800`, {
        timeoutMs: 12_000,
        headers: { 'Content-Type': 'application/json' },
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error((body as { detail?: string }).detail || `HTTP ${res.status}`);
      }
      const data = (await res.json()) as { events?: ActivityFeedEvent[] };
      setEvents(Array.isArray(data.events) ? data.events : []);
    } catch (e) {
      if (isAbortError(e)) {
        setErr('Request timed out');
      } else {
        setErr(e instanceof Error ? e.message : 'Failed to load activity');
      }
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    if (!sessionId) return undefined;
    const ms = isRunning ? 2200 : 6000;
    const id = window.setInterval(() => void load(), ms);
    return () => window.clearInterval(id);
  }, [sessionId, isRunning, load]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [events.length]);

  if (!sessionId) {
    return (
      <div className="p-4 text-[#525252] text-xs italic">
        Select a session to see the activity timeline from <code className="text-[#737373]">session_events.jsonl</code>.
      </div>
    );
  }

  return (
    <div className="absolute inset-0 flex flex-col bg-[#050505]">
      <div className="flex items-center justify-between px-3 py-2 border-b border-[#1a1a1a] bg-[#080808] flex-shrink-0">
        <div className="flex items-center gap-2 min-w-0">
          <Activity size={14} className="text-[#00ff99] flex-shrink-0" />
          <span className="text-[10px] font-semibold text-[#737373] uppercase tracking-wider truncate">
            Activity feed
          </span>
          <span className="text-[10px] text-[#3a3a3a]">·</span>
          <span className="text-[10px] text-[#525252] truncate">.plodder/session_events.jsonl</span>
        </div>
        <button
          type="button"
          onClick={() => void load()}
          disabled={loading}
          className="p-1.5 rounded-md text-[#525252] hover:text-white hover:bg-[#1a1a1a] disabled:opacity-40 transition-colors"
          title="Refresh"
        >
          <RefreshCw size={12} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto custom-scrollbar p-3 space-y-2">
        {err && (
          <div className="flex items-start gap-2 rounded-lg border border-red-500/25 bg-red-500/5 px-3 py-2 text-[11px] text-red-300">
            <AlertCircle size={14} className="flex-shrink-0 mt-0.5" />
            <span>{err}</span>
          </div>
        )}

        {events.length === 0 && !loading && !err && (
          <p className="text-[#3a3a3a] text-xs italic px-1">
            No events yet. Run a task — thoughts, commands, and file changes will appear here.
          </p>
        )}

        {events.map((raw, i) => {
          const { kind, title, detail } = formatActivityEvent(raw);
          const ts = raw.ts ? String(raw.ts) : '';
          const timeShort = ts ? ts.replace('T', ' ').slice(11, 19) : '';
          const tokens =
            raw.llm_total_tokens != null ? (
              <span className="text-[9px] text-[#404040] ml-1">
                · {String(raw.llm_total_tokens)} tok
                {raw.llm_estimated_cost_usd_cumulative != null
                  ? ` · ~$${Number(raw.llm_estimated_cost_usd_cumulative).toFixed(4)}`
                  : ''}
              </span>
            ) : null;

          return (
            <div
              key={`${ts}-${i}`}
              className="rounded-lg border border-[#1f1f1f] bg-[#0c0c0c] px-2.5 py-2 hover:border-[#2a2a2a] transition-colors"
            >
              <div className="flex items-start gap-2">
                {rowIcon(kind)}
                <div className="flex-1 min-w-0">
                  <div className="flex items-baseline flex-wrap gap-x-2 gap-y-0">
                    <span className="text-[11px] font-medium text-[#e5e5e5]">{title}</span>
                    {timeShort ? (
                      <span className="text-[9px] text-[#404040] font-mono">{timeShort}</span>
                    ) : null}
                    {tokens}
                  </div>
                  {detail ? (
                    <p className="text-[10px] text-[#737373] mt-1 leading-relaxed break-words">{detail}</p>
                  ) : null}
                </div>
              </div>
            </div>
          );
        })}
        <div ref={bottomRef} />
      </div>
    </div>
  );
};
