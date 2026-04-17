import React, { useState, useEffect, useCallback } from 'react';
import {
  FlaskConical,
  Play,
  Trash2,
  RefreshCw,
  ChevronDown,
  ChevronRight,
  CheckCircle2,
  XCircle,
  Clock,
  AlertTriangle,
  BarChart3,
  Filter,
  StopCircle,
  ExternalLink,
  Code2,
} from 'lucide-react';
import { getApiBase } from '../config/apiBase';

// ── Types ─────────────────────────────────────────────────────────────────────

interface SWETask {
  task_id: string;
  repo: string;
  instance_id: string;
  problem_statement: string;
  hints_text: string;
  fail_to_pass: string[];
  pass_to_pass: string[];
  version: string;
}

interface TaskResult {
  result_id: string;
  run_id: string;
  task_id: string;
  instance_id: string;
  repo: string;
  status: 'pending' | 'running' | 'resolved' | 'unresolved' | 'error' | 'skipped';
  patch: string;
  agent_log: string;
  test_output: string;
  fail_to_pass_results: Record<string, boolean>;
  pass_to_pass_results: Record<string, boolean>;
  error: string;
  duration_s: number;
  started_at: string;
  finished_at: string;
}

interface BenchmarkRun {
  run_id: string;
  name: string;
  split: string;
  limit: number;
  repo_filter: string;
  status: 'pending' | 'running' | 'completed' | 'cancelled';
  task_ids: string[];
  result_ids: string[];
  resolved: number;
  total: number;
  resolve_rate: number;
  started_at: string;
  finished_at: string;
}

interface BenchmarkStats {
  total_runs: number;
  completed_runs: number;
  total_tasks_evaluated: number;
  total_resolved: number;
  overall_resolve_rate: number;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const STATUS_CONFIG = {
  resolved:   { color: 'text-green-400',  bg: 'bg-green-500/10 border-green-500/20',  icon: CheckCircle2, label: 'Resolved' },
  unresolved: { color: 'text-red-400',    bg: 'bg-red-500/10 border-red-500/20',      icon: XCircle,      label: 'Unresolved' },
  running:    { color: 'text-yellow-400', bg: 'bg-yellow-500/10 border-yellow-500/20',icon: RefreshCw,    label: 'Running' },
  pending:    { color: 'text-[#525252]',  bg: 'bg-[#1a1a1a] border-[#2a2a2a]',        icon: Clock,        label: 'Pending' },
  error:      { color: 'text-orange-400', bg: 'bg-orange-500/10 border-orange-500/20',icon: AlertTriangle,label: 'Error' },
  skipped:    { color: 'text-[#525252]',  bg: 'bg-[#1a1a1a] border-[#2a2a2a]',        icon: Clock,        label: 'Skipped' },
};

function StatusBadge({ status }: { status: string }) {
  const cfg = STATUS_CONFIG[status as keyof typeof STATUS_CONFIG] || STATUS_CONFIG.pending;
  const Icon = cfg.icon;
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold border ${cfg.bg} ${cfg.color}`}>
      <Icon size={9} className={status === 'running' ? 'animate-spin' : ''} />
      {cfg.label}
    </span>
  );
}

function ResolveBar({ resolved, total }: { resolved: number; total: number }) {
  const pct = total > 0 ? (resolved / total) * 100 : 0;
  const color = pct >= 50 ? 'bg-green-500' : pct >= 25 ? 'bg-yellow-500' : 'bg-red-500';
  return (
    <div className="w-full">
      <div className="flex justify-between text-[10px] mb-1">
        <span className="text-[#737373]">Resolved</span>
        <span className="font-bold text-white">{resolved}/{total} ({pct.toFixed(1)}%)</span>
      </div>
      <div className="h-1.5 rounded-full bg-[#2a2a2a] overflow-hidden">
        <div className={`h-full rounded-full transition-all duration-500 ${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────────

export const BenchmarkPanel: React.FC = () => {
  const [tab, setTab] = useState<'runs' | 'tasks' | 'stats'>('runs');

  // Runs state
  const [runs, setRuns] = useState<BenchmarkRun[]>([]);
  const [expandedRun, setExpandedRun] = useState<string | null>(null);
  const [runResults, setRunResults] = useState<Record<string, TaskResult[]>>({});
  const [expandedResult, setExpandedResult] = useState<string | null>(null);

  // Task browser state
  const [tasks, setTasks] = useState<SWETask[]>([]);
  const [taskFilter, setTaskFilter] = useState('');
  const [loadingTasks, setLoadingTasks] = useState(false);

  // Stats
  const [stats, setStats] = useState<BenchmarkStats | null>(null);

  // New run form
  const [formOpen, setFormOpen] = useState(false);
  const [formSplit, setFormSplit] = useState('lite');
  const [formLimit, setFormLimit] = useState(5);
  const [formRepo, setFormRepo] = useState('');
  const [formName, setFormName] = useState('');
  const [formUseAgent, setFormUseAgent] = useState(true);
  const [isStarting, setIsStarting] = useState(false);

  const [isRefreshing, setIsRefreshing] = useState(false);

  // ── Data fetching ────────────────────────────────────────────────────────

  const fetchRuns = useCallback(async () => {
    try {
      const res = await fetch(`${getApiBase()}/benchmark/runs`);
      const data = await res.json();
      setRuns(data.runs || []);
    } catch { /* ignore */ }
  }, []);

  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch(`${getApiBase()}/benchmark/stats`);
      const data = await res.json();
      setStats(data);
    } catch { /* ignore */ }
  }, []);

  const fetchTasks = useCallback(async () => {
    setLoadingTasks(true);
    try {
      const params = new URLSearchParams({ limit: '20' });
      if (taskFilter) params.set('repo_filter', taskFilter);
      const res = await fetch(`${getApiBase()}/benchmark/tasks?${params}`);
      const data = await res.json();
      setTasks(data.tasks || []);
    } catch { /* ignore */ } finally {
      setLoadingTasks(false);
    }
  }, [taskFilter]);

  const fetchRunResults = useCallback(async (runId: string) => {
    if (runResults[runId]) return;
    try {
      const res = await fetch(`${getApiBase()}/benchmark/runs/${runId}/results`);
      const data = await res.json();
      setRunResults(prev => ({ ...prev, [runId]: data.results || [] }));
    } catch { /* ignore */ }
  }, [runResults]);

  useEffect(() => { fetchRuns(); fetchStats(); }, []);
  useEffect(() => { if (tab === 'tasks') fetchTasks(); }, [tab, taskFilter]);

  // Auto-refresh running runs
  useEffect(() => {
    const hasRunning = runs.some(r => r.status === 'running');
    if (!hasRunning) return;
    const t = setInterval(() => {
      fetchRuns();
      fetchStats();
      // Refresh results for running runs
      runs.filter(r => r.status === 'running').forEach(r => {
        setRunResults(prev => { const copy = { ...prev }; delete copy[r.run_id]; return copy; });
      });
    }, 3000);
    return () => clearInterval(t);
  }, [runs]);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await Promise.all([fetchRuns(), fetchStats()]);
    setIsRefreshing(false);
  };

  // ── Actions ──────────────────────────────────────────────────────────────

  const handleStartRun = async () => {
    setIsStarting(true);
    try {
      await fetch(`${getApiBase()}/benchmark/runs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          split: formSplit,
          limit: formLimit,
          repo_filter: formRepo,
          name: formName,
          use_agent: formUseAgent,
        }),
      });
      setFormOpen(false);
      setFormName('');
      setFormRepo('');
      await fetchRuns();
    } finally {
      setIsStarting(false);
    }
  };

  const handleDeleteRun = async (runId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm('Delete this benchmark run?')) return;
    await fetch(`${getApiBase()}/benchmark/runs/${runId}`, { method: 'DELETE' });
    setRuns(prev => prev.filter(r => r.run_id !== runId));
    setRunResults(prev => { const copy = { ...prev }; delete copy[runId]; return copy; });
    fetchStats();
  };

  const handleCancelRun = async () => {
    await fetch(`${getApiBase()}/benchmark/runs/cancel`, { method: 'POST' });
    fetchRuns();
  };

  const toggleRun = async (runId: string) => {
    if (expandedRun === runId) {
      setExpandedRun(null);
    } else {
      setExpandedRun(runId);
      await fetchRunResults(runId);
    }
  };

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="h-full flex flex-col bg-[#0f0f0f] overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-3 border-b border-[#1a1a1a] flex-shrink-0">
        <div className="flex items-center gap-2.5">
          <FlaskConical size={16} className="text-[#00ff99]" />
          <div>
            <h2 className="text-sm font-bold text-white">SWE-bench</h2>
            <p className="text-[10px] text-[#525252]">Standardized Benchmarking — Real GitHub Issues</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleRefresh}
            className="p-1.5 text-[#525252] hover:text-[#a3a3a3] hover:bg-[#1a1a1a] rounded-md transition-colors"
          >
            <RefreshCw size={13} className={isRefreshing ? 'animate-spin' : ''} />
          </button>
          <button
            onClick={() => setFormOpen(v => !v)}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-[#00ff99] text-black text-xs font-bold rounded-lg hover:bg-[#00e588] transition-colors"
          >
            <Play size={11} />
            New Run
          </button>
        </div>
      </div>

      {/* Stats bar */}
      {stats && (
        <div className="flex items-center gap-4 px-5 py-2 border-b border-[#1a1a1a] bg-[#0a0a0a] flex-shrink-0">
          <div className="text-center">
            <p className="text-lg font-bold text-white">{stats.overall_resolve_rate}%</p>
            <p className="text-[9px] text-[#525252] uppercase tracking-wider">Overall</p>
          </div>
          <div className="w-px h-8 bg-[#1a1a1a]" />
          <div className="text-center">
            <p className="text-sm font-bold text-white">{stats.total_resolved}</p>
            <p className="text-[9px] text-[#525252] uppercase tracking-wider">Resolved</p>
          </div>
          <div className="text-center">
            <p className="text-sm font-bold text-white">{stats.total_tasks_evaluated}</p>
            <p className="text-[9px] text-[#525252] uppercase tracking-wider">Evaluated</p>
          </div>
          <div className="text-center">
            <p className="text-sm font-bold text-white">{stats.completed_runs}</p>
            <p className="text-[9px] text-[#525252] uppercase tracking-wider">Runs</p>
          </div>
          {/* Leaderboard comparison */}
          <div className="ml-auto flex items-center gap-4 text-[10px]">
            <span className="text-[#525252]">vs others:</span>
            {[
              { name: 'Devin 2', pct: 55 },
              { name: 'Claude', pct: 49 },
              { name: 'OpenHands', pct: 38 },
            ].map(cmp => (
              <span key={cmp.name} className="text-[#737373]">
                <span className="text-[#a3a3a3] font-medium">{cmp.name}</span> {cmp.pct}%
              </span>
            ))}
            {stats.overall_resolve_rate > 0 && (
              <span className={`font-bold ${
                stats.overall_resolve_rate >= 50 ? 'text-green-400' :
                stats.overall_resolve_rate >= 30 ? 'text-yellow-400' : 'text-red-400'
              }`}>
                Plodder: {stats.overall_resolve_rate}%
              </span>
            )}
          </div>
        </div>
      )}

      {/* New Run form */}
      {formOpen && (
        <div className="px-5 py-4 border-b border-[#1a1a1a] bg-[#111] flex-shrink-0 space-y-3">
          <p className="text-xs font-bold text-white">Configure Benchmark Run</p>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-[10px] uppercase tracking-wider text-[#525252] font-bold block mb-1">Split</label>
              <select
                value={formSplit}
                onChange={e => setFormSplit(e.target.value)}
                className="w-full bg-[#0f0f0f] border border-[#2a2a2a] rounded-lg px-3 py-2 text-xs text-white outline-none"
              >
                <option value="lite">SWE-bench Lite (300 tasks)</option>
                <option value="full">SWE-bench Full (2294 tasks)</option>
              </select>
            </div>
            <div>
              <label className="text-[10px] uppercase tracking-wider text-[#525252] font-bold block mb-1">Task Limit</label>
              <input
                type="number"
                min={1}
                max={50}
                value={formLimit}
                onChange={e => setFormLimit(Number(e.target.value))}
                className="w-full bg-[#0f0f0f] border border-[#2a2a2a] rounded-lg px-3 py-2 text-xs text-white outline-none"
              />
            </div>
            <div>
              <label className="text-[10px] uppercase tracking-wider text-[#525252] font-bold block mb-1">Repo Filter (optional)</label>
              <input
                type="text"
                placeholder="e.g. django"
                value={formRepo}
                onChange={e => setFormRepo(e.target.value)}
                className="w-full bg-[#0f0f0f] border border-[#2a2a2a] rounded-lg px-3 py-2 text-xs text-white outline-none placeholder:text-[#3a3a3a]"
              />
            </div>
            <div>
              <label className="text-[10px] uppercase tracking-wider text-[#525252] font-bold block mb-1">Run Name</label>
              <input
                type="text"
                placeholder="Optional label"
                value={formName}
                onChange={e => setFormName(e.target.value)}
                className="w-full bg-[#0f0f0f] border border-[#2a2a2a] rounded-lg px-3 py-2 text-xs text-white outline-none placeholder:text-[#3a3a3a]"
              />
            </div>
          </div>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={formUseAgent}
              onChange={e => setFormUseAgent(e.target.checked)}
              className="accent-[#00ff99]"
            />
            <span className="text-xs text-[#a3a3a3]">Run Plodder agent on each task</span>
          </label>
          <div className="flex gap-2">
            <button
              onClick={handleStartRun}
              disabled={isStarting}
              className="flex items-center gap-1.5 px-4 py-2 bg-[#00ff99] text-black text-xs font-bold rounded-lg hover:bg-[#00e588] disabled:opacity-50 transition-colors"
            >
              {isStarting ? <RefreshCw size={11} className="animate-spin" /> : <Play size={11} />}
              {isStarting ? 'Starting…' : 'Start Benchmark'}
            </button>
            <button
              onClick={() => setFormOpen(false)}
              className="px-4 py-2 text-xs text-[#737373] hover:text-white transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Tab bar */}
      <div className="flex gap-1 px-4 pt-3 border-b border-[#1a1a1a] flex-shrink-0">
        {(['runs', 'tasks', 'stats'] as const).map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-3 py-1.5 text-xs font-medium rounded-t-md border-b-2 transition-all ${
              tab === t
                ? 'text-white border-[#00ff99]'
                : 'text-[#525252] border-transparent hover:text-[#a3a3a3]'
            }`}
          >
            {t === 'runs' ? 'Runs' : t === 'tasks' ? 'Task Browser' : 'Leaderboard'}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto custom-scrollbar">
        {/* ── RUNS TAB ── */}
        {tab === 'runs' && (
          <div className="p-4 space-y-2">
            {runs.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-16 gap-3 text-center">
                <FlaskConical size={40} className="text-[#2a2a2a]" />
                <p className="text-[#525252] text-sm font-medium">No benchmark runs yet</p>
                <p className="text-[#3a3a3a] text-xs max-w-xs">
                  Click <span className="text-white font-semibold">New Run</span> to evaluate Plodder on
                  real GitHub issues from the SWE-bench dataset.
                </p>
                <button
                  onClick={() => setFormOpen(true)}
                  className="mt-2 flex items-center gap-1.5 px-4 py-2 bg-[#00ff99] text-black text-xs font-bold rounded-lg hover:bg-[#00e588] transition-colors"
                >
                  <Play size={11} />
                  Start First Run
                </button>
              </div>
            ) : (
              runs.map(run => (
                <div key={run.run_id} className="border border-[#1e1e1e] rounded-xl overflow-hidden">
                  {/* Run header */}
                  <div
                    className="flex items-center gap-3 px-4 py-3 bg-[#111] cursor-pointer hover:bg-[#141414] transition-colors"
                    onClick={() => toggleRun(run.run_id)}
                  >
                    <button className="text-[#525252]">
                      {expandedRun === run.run_id ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                    </button>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-xs font-semibold text-white truncate">{run.name}</span>
                        <StatusBadge status={run.status} />
                      </div>
                      <div className="text-[10px] text-[#525252] font-mono">
                        {run.run_id} · {run.split} · {run.total} tasks
                        {run.repo_filter && ` · ${run.repo_filter}`}
                      </div>
                    </div>
                    <div className="flex-shrink-0 w-36">
                      <ResolveBar resolved={run.resolved} total={run.total} />
                    </div>
                    <div className="flex items-center gap-1 ml-2">
                      {run.status === 'running' && (
                        <button
                          onClick={e => { e.stopPropagation(); handleCancelRun(); }}
                          className="p-1 text-red-400 hover:bg-red-500/10 rounded transition-colors"
                          title="Cancel run"
                        >
                          <StopCircle size={13} />
                        </button>
                      )}
                      <button
                        onClick={e => handleDeleteRun(run.run_id, e)}
                        className="p-1 text-[#3a3a3a] hover:text-red-400 hover:bg-red-500/10 rounded transition-colors"
                      >
                        <Trash2 size={13} />
                      </button>
                    </div>
                  </div>

                  {/* Results */}
                  {expandedRun === run.run_id && (
                    <div className="border-t border-[#1a1a1a] divide-y divide-[#1a1a1a]">
                      {!runResults[run.run_id] ? (
                        <div className="px-4 py-3 text-xs text-[#525252] italic">Loading results…</div>
                      ) : runResults[run.run_id].length === 0 ? (
                        <div className="px-4 py-3 text-xs text-[#525252] italic">No results yet — run is in progress.</div>
                      ) : (
                        runResults[run.run_id].map(result => (
                          <div key={result.result_id}>
                            <div
                              className="flex items-center gap-3 px-4 py-2.5 cursor-pointer hover:bg-[#0d0d0d] transition-colors"
                              onClick={() => setExpandedResult(
                                expandedResult === result.result_id ? null : result.result_id
                              )}
                            >
                              <span className="text-[#525252]">
                                {expandedResult === result.result_id ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                              </span>
                              <StatusBadge status={result.status} />
                              <span className="text-xs font-mono text-[#a3a3a3] flex-1 truncate">{result.instance_id}</span>
                              <span className="text-[10px] text-[#525252]">{result.repo}</span>
                              <span className="text-[10px] text-[#3a3a3a]">{result.duration_s}s</span>
                            </div>

                            {expandedResult === result.result_id && (
                              <div className="px-4 pb-4 pt-2 space-y-3 bg-[#080808]">
                                {/* Fail-to-pass tests */}
                                {Object.keys(result.fail_to_pass_results).length > 0 && (
                                  <div>
                                    <p className="text-[10px] uppercase tracking-wider text-[#525252] font-bold mb-2">
                                      Fail → Pass Tests
                                    </p>
                                    {Object.entries(result.fail_to_pass_results).map(([tid, passed]) => (
                                      <div key={tid} className="flex items-center gap-2 text-[11px] font-mono mb-1">
                                        {passed
                                          ? <CheckCircle2 size={11} className="text-green-400 flex-shrink-0" />
                                          : <XCircle size={11} className="text-red-400 flex-shrink-0" />
                                        }
                                        <span className={passed ? 'text-green-400' : 'text-red-400'}>
                                          {tid.split('::').pop()}
                                        </span>
                                      </div>
                                    ))}
                                  </div>
                                )}

                                {/* Agent log */}
                                {result.agent_log && (
                                  <div>
                                    <p className="text-[10px] uppercase tracking-wider text-[#525252] font-bold mb-1">Agent Summary</p>
                                    <p className="text-[11px] text-[#a3a3a3] leading-relaxed whitespace-pre-wrap bg-[#0f0f0f] rounded-lg p-3 max-h-24 overflow-y-auto custom-scrollbar">
                                      {result.agent_log}
                                    </p>
                                  </div>
                                )}

                                {/* Patch */}
                                {result.patch && (
                                  <div>
                                    <p className="text-[10px] uppercase tracking-wider text-[#525252] font-bold mb-1 flex items-center gap-1">
                                      <Code2 size={10} /> Generated Patch
                                    </p>
                                    <pre className="text-[10px] font-mono bg-[#050505] rounded-lg p-3 overflow-x-auto max-h-40 custom-scrollbar leading-5">
                                      {result.patch.split('\n').map((line, i) => (
                                        <span key={i} className={
                                          line.startsWith('+') && !line.startsWith('+++') ? 'text-green-400 block' :
                                          line.startsWith('-') && !line.startsWith('---') ? 'text-red-400 block' :
                                          line.startsWith('@@') ? 'text-blue-400 block' :
                                          'text-[#737373] block'
                                        }>{line}</span>
                                      ))}
                                    </pre>
                                  </div>
                                )}

                                {/* Error */}
                                {result.error && (
                                  <div className="bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
                                    <p className="text-[10px] text-red-400 font-mono">{result.error}</p>
                                  </div>
                                )}
                              </div>
                            )}
                          </div>
                        ))
                      )}
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        )}

        {/* ── TASK BROWSER TAB ── */}
        {tab === 'tasks' && (
          <div className="p-4">
            <div className="flex items-center gap-2 mb-4">
              <Filter size={12} className="text-[#525252]" />
              <input
                type="text"
                placeholder="Filter by repo (e.g. django, requests)…"
                value={taskFilter}
                onChange={e => setTaskFilter(e.target.value)}
                className="flex-1 bg-[#111] border border-[#1e1e1e] rounded-lg px-3 py-2 text-xs text-white outline-none placeholder:text-[#3a3a3a] focus:border-[#00ff99]/30"
              />
              <button
                onClick={fetchTasks}
                className="p-2 text-[#525252] hover:text-[#a3a3a3] hover:bg-[#1a1a1a] rounded-lg transition-colors"
              >
                <RefreshCw size={12} className={loadingTasks ? 'animate-spin' : ''} />
              </button>
            </div>

            {tasks.length === 0 && !loadingTasks ? (
              <p className="text-[#3a3a3a] text-xs italic text-center py-8">No tasks found.</p>
            ) : (
              <div className="space-y-2">
                {tasks.map(task => (
                  <div key={task.task_id} className="border border-[#1e1e1e] rounded-xl p-4 bg-[#111] space-y-2">
                    <div className="flex items-start justify-between gap-2">
                      <div>
                        <p className="text-xs font-bold text-white font-mono">{task.instance_id}</p>
                        <p className="text-[10px] text-[#525252]">{task.repo} · v{task.version}</p>
                      </div>
                      <a
                        href={`https://github.com/${task.repo}/issues`}
                        target="_blank"
                        rel="noreferrer"
                        className="text-[#525252] hover:text-[#a3a3a3] transition-colors"
                      >
                        <ExternalLink size={12} />
                      </a>
                    </div>
                    <p className="text-[11px] text-[#a3a3a3] leading-relaxed line-clamp-3">
                      {task.problem_statement}
                    </p>
                    {task.fail_to_pass.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {task.fail_to_pass.map(t => (
                          <span key={t} className="px-2 py-0.5 bg-red-500/10 border border-red-500/20 rounded text-[9px] font-mono text-red-400">
                            {t.split('::').pop()}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ── STATS / LEADERBOARD TAB ── */}
        {tab === 'stats' && (
          <div className="p-6 space-y-6">
            {/* Overall gauge */}
            <div className="text-center space-y-1">
              <p className="text-[10px] uppercase tracking-widest text-[#525252] font-bold">Plodder Score</p>
              <p className="text-5xl font-black text-white">
                {stats?.overall_resolve_rate ?? 0}
                <span className="text-2xl text-[#525252]">%</span>
              </p>
              <p className="text-xs text-[#525252]">
                {stats?.total_resolved ?? 0} of {stats?.total_tasks_evaluated ?? 0} tasks resolved
              </p>
            </div>

            {/* Leaderboard comparison */}
            <div className="border border-[#1e1e1e] rounded-xl overflow-hidden">
              <div className="px-4 py-2 bg-[#111] border-b border-[#1e1e1e]">
                <p className="text-[10px] uppercase tracking-wider text-[#525252] font-bold flex items-center gap-1">
                  <BarChart3 size={10} /> SWE-bench Lite Leaderboard (2025)
                </p>
              </div>
              <div className="divide-y divide-[#1a1a1a]">
                {[
                  { rank: 1, name: 'Devin 2.0',         org: 'Cognition',      pct: 55.0, isUs: false },
                  { rank: 2, name: 'Claude Sonnet',      org: 'Anthropic',      pct: 49.0, isUs: false },
                  { rank: 3, name: 'OpenHands',          org: 'AllHands AI',    pct: 38.0, isUs: false },
                  { rank: 4, name: 'GPT-4o (SWE-agent)', org: 'OpenAI',         pct: 23.7, isUs: false },
                  { rank: 5, name: 'Plodder',         org: 'You',            pct: stats?.overall_resolve_rate ?? 0, isUs: true },
                ]
                  .sort((a, b) => b.pct - a.pct)
                  .map((entry, i) => (
                    <div key={entry.name}
                      className={`flex items-center gap-3 px-4 py-2.5 ${entry.isUs ? 'bg-[#00ff99]/5' : ''}`}
                    >
                      <span className="text-[10px] text-[#3a3a3a] w-4 text-right">{i + 1}</span>
                      <div className="flex-1 min-w-0">
                        <p className={`text-xs font-semibold ${entry.isUs ? 'text-[#00ff99]' : 'text-white'}`}>
                          {entry.name} {entry.isUs && '⭐'}
                        </p>
                        <p className="text-[10px] text-[#525252]">{entry.org}</p>
                      </div>
                      <div className="flex items-center gap-3 flex-shrink-0">
                        <div className="w-28 h-1.5 bg-[#1a1a1a] rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full ${entry.isUs ? 'bg-[#00ff99]' : 'bg-[#525252]'}`}
                            style={{ width: `${(entry.pct / 60) * 100}%` }}
                          />
                        </div>
                        <span className={`text-xs font-bold w-10 text-right ${entry.isUs ? 'text-[#00ff99]' : 'text-white'}`}>
                          {entry.pct}%
                        </span>
                      </div>
                    </div>
                  ))}
              </div>
            </div>

            {/* Run history */}
            {runs.filter(r => r.status === 'completed').length > 0 && (
              <div>
                <p className="text-[10px] uppercase tracking-wider text-[#525252] font-bold mb-3">Run History</p>
                <div className="space-y-2">
                  {runs.filter(r => r.status === 'completed').map(run => (
                    <div key={run.run_id} className="flex items-center gap-3 px-4 py-2.5 bg-[#111] rounded-xl border border-[#1e1e1e]">
                      <span className="text-[10px] text-[#525252] font-mono">{run.run_id}</span>
                      <span className="flex-1 text-xs text-[#a3a3a3] truncate">{run.name}</span>
                      <ResolveBar resolved={run.resolved} total={run.total} />
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
