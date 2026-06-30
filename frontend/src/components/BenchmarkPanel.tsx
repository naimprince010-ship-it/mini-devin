import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Clock,
  Code2,
  ExternalLink,
  Filter,
  FlaskConical,
  Play,
  RefreshCw,
  StopCircle,
  Trash2,
  XCircle,
} from 'lucide-react';
import { getApiBase } from '../config/apiBase';

type CodeBenchmarkName = 'humaneval' | 'mbpp';
type CodeBenchmarkMode = 'canonical' | 'litellm';

interface CodeBenchmarkResult {
  task_id: string;
  benchmark: string;
  passed: boolean;
  generated_code: string;
  test_output: string;
  error: string;
  duration_s: number;
}

interface CodeBenchmarkRun {
  run_id: string;
  benchmark: string;
  mode: string;
  limit: number;
  status: 'starting' | 'running' | 'completed' | 'error' | string;
  started_at?: string;
  finished_at?: string;
  total?: number;
  passed?: number;
  pass_rate?: number;
  results?: CodeBenchmarkResult[];
}

interface SWETask {
  task_id: string;
  repo: string;
  instance_id: string;
  problem_statement: string;
  fail_to_pass: string[];
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
  error: string;
  duration_s: number;
}

interface BenchmarkRun {
  run_id: string;
  name: string;
  split: string;
  limit: number;
  repo_filter: string;
  status: 'pending' | 'running' | 'completed' | 'cancelled';
  resolved: number;
  total: number;
  resolve_rate: number;
}

interface BenchmarkStats {
  total_runs: number;
  completed_runs: number;
  total_tasks_evaluated: number;
  total_resolved: number;
  overall_resolve_rate: number;
}

const STATUS_CONFIG = {
  resolved: { color: 'text-green-400', bg: 'bg-green-500/10 border-green-500/20', icon: CheckCircle2, label: 'Resolved' },
  completed: { color: 'text-green-400', bg: 'bg-green-500/10 border-green-500/20', icon: CheckCircle2, label: 'Completed' },
  passed: { color: 'text-green-400', bg: 'bg-green-500/10 border-green-500/20', icon: CheckCircle2, label: 'Passed' },
  unresolved: { color: 'text-red-400', bg: 'bg-red-500/10 border-red-500/20', icon: XCircle, label: 'Unresolved' },
  failed: { color: 'text-red-400', bg: 'bg-red-500/10 border-red-500/20', icon: XCircle, label: 'Failed' },
  running: { color: 'text-yellow-400', bg: 'bg-yellow-500/10 border-yellow-500/20', icon: RefreshCw, label: 'Running' },
  starting: { color: 'text-yellow-400', bg: 'bg-yellow-500/10 border-yellow-500/20', icon: RefreshCw, label: 'Starting' },
  pending: { color: 'text-[#737373]', bg: 'bg-[#1a1a1a] border-[#2a2a2a]', icon: Clock, label: 'Pending' },
  error: { color: 'text-orange-400', bg: 'bg-orange-500/10 border-orange-500/20', icon: AlertTriangle, label: 'Error' },
  skipped: { color: 'text-[#737373]', bg: 'bg-[#1a1a1a] border-[#2a2a2a]', icon: Clock, label: 'Skipped' },
  cancelled: { color: 'text-[#737373]', bg: 'bg-[#1a1a1a] border-[#2a2a2a]', icon: StopCircle, label: 'Cancelled' },
};

function StatusBadge({ status }: { status: string }) {
  const cfg = STATUS_CONFIG[status as keyof typeof STATUS_CONFIG] || STATUS_CONFIG.pending;
  const Icon = cfg.icon;
  return (
    <span className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-bold ${cfg.bg} ${cfg.color}`}>
      <Icon size={9} className={status === 'running' || status === 'starting' ? 'animate-spin' : ''} />
      {cfg.label}
    </span>
  );
}

function ScoreBar({ value, label }: { value: number; label: string }) {
  const pct = Math.max(0, Math.min(100, value || 0));
  const color = pct >= 70 ? 'bg-green-500' : pct >= 35 ? 'bg-yellow-500' : 'bg-red-500';
  return (
    <div className="w-full min-w-[120px]">
      <div className="mb-1 flex justify-between text-[10px]">
        <span className="text-[#737373]">{label}</span>
        <span className="font-bold text-white">{pct.toFixed(1)}%</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-[#2a2a2a]">
        <div className={`h-full rounded-full transition-all duration-500 ${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

async function readJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${getApiBase()}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...init?.headers,
    },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
    throw new Error(body.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

function CodeBenchmarkView() {
  const [runs, setRuns] = useState<CodeBenchmarkRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [expandedResult, setExpandedResult] = useState<string | null>(null);
  const [benchmark, setBenchmark] = useState<CodeBenchmarkName>('humaneval');
  const [mode, setMode] = useState<CodeBenchmarkMode>('canonical');
  const [limit, setLimit] = useState(5);
  const [timeout, setTimeoutValue] = useState(10);
  const [model, setModel] = useState('openai/llama3.3-70b-instruct');
  const [loading, setLoading] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState('');

  const selectedRun = useMemo(
    () => runs.find((run) => run.run_id === selectedRunId) || runs[0] || null,
    [runs, selectedRunId],
  );

  const refreshRuns = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const data = await readJson<{ runs: CodeBenchmarkRun[] }>('/code-benchmark/runs');
      setRuns(data.runs || []);
      if (!selectedRunId && data.runs?.length) {
        setSelectedRunId(data.runs[0].run_id);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load code benchmark runs');
    } finally {
      setLoading(false);
    }
  }, [selectedRunId]);

  const refreshRun = useCallback(async (runId: string) => {
    try {
      const run = await readJson<CodeBenchmarkRun>(`/code-benchmark/runs/${runId}`);
      setRuns((prev) => {
        const exists = prev.some((item) => item.run_id === runId);
        if (!exists) return [run, ...prev];
        return prev.map((item) => (item.run_id === runId ? run : item));
      });
    } catch {
      await refreshRuns();
    }
  }, [refreshRuns]);

  useEffect(() => {
    void refreshRuns();
  }, []);

  useEffect(() => {
    const active = runs.some((run) => run.status === 'starting' || run.status === 'running');
    if (!active) return;
    const timer = window.setInterval(() => {
      runs
        .filter((run) => run.status === 'starting' || run.status === 'running')
        .forEach((run) => void refreshRun(run.run_id));
    }, 3000);
    return () => window.clearInterval(timer);
  }, [runs, refreshRun]);

  const startRun = async () => {
    if (mode === 'litellm' && limit > 20) {
      const ok = window.confirm(`This will ask the LLM to solve ${limit} tasks and may use many tokens. Continue?`);
      if (!ok) return;
    }
    setStarting(true);
    setError('');
    try {
      const run = await readJson<CodeBenchmarkRun>('/code-benchmark/runs', {
        method: 'POST',
        body: JSON.stringify({ benchmark, limit, mode, model: mode === 'litellm' ? model : '', timeout }),
      });
      setSelectedRunId(run.run_id);
      setRuns((prev) => [run, ...prev]);
      window.setTimeout(() => void refreshRun(run.run_id), 2500);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to start benchmark');
    } finally {
      setStarting(false);
    }
  };

  const passRate = selectedRun?.pass_rate ?? (
    selectedRun?.total ? ((selectedRun.passed || 0) / selectedRun.total) * 100 : 0
  );
  const resultCount = selectedRun?.results?.length || 0;

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="grid gap-3 border-b border-[#1a1a1a] bg-[#101010] p-4 lg:grid-cols-[1.2fr_1fr]">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <label className="space-y-1">
            <span className="block text-[10px] font-bold uppercase tracking-wider text-[#525252]">Exam</span>
            <select
              value={benchmark}
              onChange={(e) => setBenchmark(e.target.value as CodeBenchmarkName)}
              className="w-full rounded-lg border border-[#2a2a2a] bg-[#0a0a0a] px-3 py-2 text-xs text-white outline-none focus:border-[#00ff99]/40"
            >
              <option value="humaneval">HumanEval</option>
              <option value="mbpp">MBPP</option>
            </select>
          </label>
          <label className="space-y-1">
            <span className="block text-[10px] font-bold uppercase tracking-wider text-[#525252]">Mode</span>
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value as CodeBenchmarkMode)}
              className="w-full rounded-lg border border-[#2a2a2a] bg-[#0a0a0a] px-3 py-2 text-xs text-white outline-none focus:border-[#00ff99]/40"
            >
              <option value="canonical">Harness check</option>
              <option value="litellm">Plodder model</option>
            </select>
          </label>
          <label className="space-y-1">
            <span className="block text-[10px] font-bold uppercase tracking-wider text-[#525252]">Limit</span>
            <input
              type="number"
              min={1}
              max={benchmark === 'humaneval' ? 164 : 1000}
              value={limit}
              onChange={(e) => setLimit(Math.max(1, Math.min(1000, Number(e.target.value) || 1)))}
              className="w-full rounded-lg border border-[#2a2a2a] bg-[#0a0a0a] px-3 py-2 text-xs text-white outline-none focus:border-[#00ff99]/40"
            />
          </label>
          <label className="space-y-1">
            <span className="block text-[10px] font-bold uppercase tracking-wider text-[#525252]">Timeout</span>
            <input
              type="number"
              min={1}
              max={120}
              value={timeout}
              onChange={(e) => setTimeoutValue(Math.max(1, Math.min(120, Number(e.target.value) || 10)))}
              className="w-full rounded-lg border border-[#2a2a2a] bg-[#0a0a0a] px-3 py-2 text-xs text-white outline-none focus:border-[#00ff99]/40"
            />
          </label>
        </div>

        <div className="flex flex-col gap-2">
          {mode === 'litellm' && (
            <input
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="w-full rounded-lg border border-[#2a2a2a] bg-[#0a0a0a] px-3 py-2 text-xs text-white outline-none placeholder:text-[#3a3a3a] focus:border-[#00ff99]/40"
              placeholder="Model name"
            />
          )}
          <div className="flex items-center gap-2">
            <button
              onClick={startRun}
              disabled={starting}
              className="inline-flex items-center gap-1.5 rounded-lg bg-[#00ff99] px-4 py-2 text-xs font-bold text-black transition-colors hover:bg-[#00e588] disabled:opacity-50"
            >
              {starting ? <RefreshCw size={12} className="animate-spin" /> : <Play size={12} />}
              {starting ? 'Starting' : 'Run Exam'}
            </button>
            <button
              onClick={() => void refreshRuns()}
              className="inline-flex items-center gap-1.5 rounded-lg border border-[#2a2a2a] px-3 py-2 text-xs font-semibold text-[#a3a3a3] transition-colors hover:border-[#3a3a3a] hover:text-white"
            >
              <RefreshCw size={12} className={loading ? 'animate-spin' : ''} />
              Refresh
            </button>
          </div>
          {error && (
            <div className="rounded-lg border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs text-red-300">
              {error}
            </div>
          )}
        </div>
      </div>

      <div className="grid min-h-0 flex-1 lg:grid-cols-[320px_1fr]">
        <div className="min-h-0 overflow-y-auto border-r border-[#1a1a1a] p-3 custom-scrollbar">
          {runs.length === 0 ? (
            <div className="flex flex-col items-center justify-center gap-3 py-16 text-center">
              <FlaskConical size={36} className="text-[#2a2a2a]" />
              <p className="text-sm font-medium text-[#737373]">No code benchmark runs yet</p>
              <button
                onClick={startRun}
                className="inline-flex items-center gap-1.5 rounded-lg bg-[#00ff99] px-4 py-2 text-xs font-bold text-black hover:bg-[#00e588]"
              >
                <Play size={12} />
                Start First Run
              </button>
            </div>
          ) : (
            <div className="space-y-2">
              {runs.map((run) => {
                const rate = run.pass_rate ?? (run.total ? ((run.passed || 0) / run.total) * 100 : 0);
                return (
                  <button
                    key={run.run_id}
                    onClick={() => setSelectedRunId(run.run_id)}
                    className={`w-full rounded-xl border p-3 text-left transition-colors ${
                      selectedRun?.run_id === run.run_id
                        ? 'border-[#00ff99]/30 bg-[#00ff99]/5'
                        : 'border-[#1e1e1e] bg-[#111] hover:bg-[#141414]'
                    }`}
                  >
                    <div className="mb-2 flex items-center justify-between gap-2">
                      <span className="truncate text-xs font-bold text-white">{run.benchmark}</span>
                      <StatusBadge status={run.status} />
                    </div>
                    <div className="mb-2 text-[10px] font-mono text-[#525252]">
                      {run.run_id} · {run.mode} · limit {run.limit}
                    </div>
                    <ScoreBar value={rate} label={`${run.passed || 0}/${run.total || run.limit}`} />
                  </button>
                );
              })}
            </div>
          )}
        </div>

        <div className="min-h-0 overflow-y-auto p-4 custom-scrollbar">
          {!selectedRun ? (
            <div className="flex h-full items-center justify-center text-sm text-[#525252]">Select a run to inspect results.</div>
          ) : (
            <div className="space-y-4">
              <div className="rounded-xl border border-[#1e1e1e] bg-[#111] p-4">
                <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <div className="mb-1 flex items-center gap-2">
                      <h3 className="text-sm font-bold capitalize text-white">{selectedRun.benchmark}</h3>
                      <StatusBadge status={selectedRun.status} />
                    </div>
                    <p className="text-[10px] font-mono text-[#525252]">
                      {selectedRun.run_id} · {selectedRun.mode} · {resultCount} visible results
                    </p>
                  </div>
                  <button
                    onClick={() => void refreshRun(selectedRun.run_id)}
                    className="rounded-lg border border-[#2a2a2a] p-2 text-[#737373] transition-colors hover:text-white"
                    title="Refresh run"
                  >
                    <RefreshCw size={13} />
                  </button>
                </div>
                <ScoreBar value={passRate} label={`Passed ${selectedRun.passed || 0}/${selectedRun.total || selectedRun.limit}`} />
              </div>

              {!selectedRun.results?.length ? (
                <div className="rounded-xl border border-[#1e1e1e] bg-[#0a0a0a] p-8 text-center text-sm text-[#525252]">
                  {selectedRun.status === 'starting' || selectedRun.status === 'running'
                    ? 'Run is still working. Results will appear here.'
                    : 'No task results saved for this run.'}
                </div>
              ) : (
                selectedRun.results.map((result) => {
                  const status = result.passed ? 'passed' : 'failed';
                  const isOpen = expandedResult === result.task_id;
                  return (
                    <div key={result.task_id} className="overflow-hidden rounded-xl border border-[#1e1e1e] bg-[#111]">
                      <button
                        onClick={() => setExpandedResult(isOpen ? null : result.task_id)}
                        className="flex w-full items-center gap-3 px-4 py-3 text-left transition-colors hover:bg-[#141414]"
                      >
                        {isOpen ? <ChevronDown size={14} className="text-[#525252]" /> : <ChevronRight size={14} className="text-[#525252]" />}
                        <StatusBadge status={status} />
                        <span className="min-w-0 flex-1 truncate text-xs font-mono text-[#a3a3a3]">{result.task_id}</span>
                        <span className="text-[10px] text-[#525252]">{result.duration_s.toFixed(2)}s</span>
                      </button>
                      {isOpen && (
                        <div className="space-y-3 border-t border-[#1a1a1a] bg-[#080808] p-4">
                          {result.error && (
                            <div className="rounded-lg border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs text-red-300">
                              {result.error}
                            </div>
                          )}
                          {result.test_output && (
                            <pre className="max-h-40 overflow-auto rounded-lg bg-[#050505] p-3 text-[10px] leading-5 text-[#a3a3a3] custom-scrollbar">
                              {result.test_output}
                            </pre>
                          )}
                          <div>
                            <p className="mb-1 flex items-center gap-1 text-[10px] font-bold uppercase tracking-wider text-[#525252]">
                              <Code2 size={10} /> Generated Code
                            </p>
                            <pre className="max-h-80 overflow-auto rounded-lg bg-[#050505] p-3 text-[10px] leading-5 text-[#a3a3a3] custom-scrollbar">
                              {result.generated_code || 'No code captured.'}
                            </pre>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function SweBenchmarkView() {
  const [tab, setTab] = useState<'runs' | 'tasks' | 'stats'>('runs');
  const [runs, setRuns] = useState<BenchmarkRun[]>([]);
  const [expandedRun, setExpandedRun] = useState<string | null>(null);
  const [runResults, setRunResults] = useState<Record<string, TaskResult[]>>({});
  const [expandedResult, setExpandedResult] = useState<string | null>(null);
  const [tasks, setTasks] = useState<SWETask[]>([]);
  const [taskFilter, setTaskFilter] = useState('');
  const [loadingTasks, setLoadingTasks] = useState(false);
  const [stats, setStats] = useState<BenchmarkStats | null>(null);
  const [formOpen, setFormOpen] = useState(false);
  const [formSplit, setFormSplit] = useState('lite');
  const [formLimit, setFormLimit] = useState(3);
  const [formRepo, setFormRepo] = useState('');
  const [formName, setFormName] = useState('');
  const [formUseAgent, setFormUseAgent] = useState(true);
  const [benchmarkConfirmed, setBenchmarkConfirmed] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [startError, setStartError] = useState('');

  const fetchRuns = useCallback(async () => {
    const data = await readJson<{ runs: BenchmarkRun[] }>('/benchmark/runs');
    setRuns(data.runs || []);
  }, []);

  const fetchStats = useCallback(async () => {
    const data = await readJson<BenchmarkStats>('/benchmark/stats');
    setStats(data);
  }, []);

  const fetchTasks = useCallback(async () => {
    setLoadingTasks(true);
    try {
      const params = new URLSearchParams({ limit: '20' });
      if (taskFilter) params.set('repo_filter', taskFilter);
      const data = await readJson<{ tasks: SWETask[] }>(`/benchmark/tasks?${params}`);
      setTasks(data.tasks || []);
    } finally {
      setLoadingTasks(false);
    }
  }, [taskFilter]);

  const fetchRunResults = useCallback(async (runId: string) => {
    if (runResults[runId]) return;
    const data = await readJson<{ results: TaskResult[] }>(`/benchmark/runs/${runId}/results`);
    setRunResults((prev) => ({ ...prev, [runId]: data.results || [] }));
  }, [runResults]);

  useEffect(() => {
    void Promise.all([fetchRuns(), fetchStats()]);
  }, []);

  useEffect(() => {
    if (tab === 'tasks') void fetchTasks();
  }, [tab, taskFilter]);

  useEffect(() => {
    const hasRunning = runs.some((run) => run.status === 'running');
    if (!hasRunning) return;
    const timer = window.setInterval(() => {
      void Promise.all([fetchRuns(), fetchStats()]);
      runs
        .filter((run) => run.status === 'running')
        .forEach((run) => {
          setRunResults((prev) => {
            const copy = { ...prev };
            delete copy[run.run_id];
            return copy;
          });
        });
    }, 3000);
    return () => window.clearInterval(timer);
  }, [runs, fetchRuns, fetchStats]);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await Promise.all([fetchRuns(), fetchStats()]);
    setIsRefreshing(false);
  };

  const handleStartRun = async () => {
    if (formUseAgent && !benchmarkConfirmed) return;
    setIsStarting(true);
    setStartError('');
    try {
      await readJson('/benchmark/runs', {
        method: 'POST',
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
      setBenchmarkConfirmed(false);
      await Promise.all([fetchRuns(), fetchStats()]);
    } catch (e) {
      setStartError(e instanceof Error ? e.message : 'Failed to start benchmark run');
    } finally {
      setIsStarting(false);
    }
  };

  const handleDeleteRun = async (runId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!window.confirm('Delete this benchmark run?')) return;
    await readJson(`/benchmark/runs/${runId}`, { method: 'DELETE' });
    setRuns((prev) => prev.filter((run) => run.run_id !== runId));
    setRunResults((prev) => {
      const copy = { ...prev };
      delete copy[runId];
      return copy;
    });
    void fetchStats();
  };

  const handleCancelRun = async (e: React.MouseEvent) => {
    e.stopPropagation();
    await readJson('/benchmark/runs/cancel', { method: 'POST' });
    void fetchRuns();
  };

  const toggleRun = async (runId: string) => {
    if (expandedRun === runId) {
      setExpandedRun(null);
    } else {
      setExpandedRun(runId);
      await fetchRunResults(runId);
    }
  };

  const hasRunning = runs.some((r) => r.status === 'running' || r.status === 'pending');

  return (
    <div className="flex min-h-0 flex-1 flex-col">

      {/* ── Top bar ─────────────────────────────────────────────────── */}
      <div className="flex flex-shrink-0 items-center justify-between gap-3 border-b border-[#1a1a1a] px-5 py-3">
        <div>
          <h3 className="text-sm font-bold text-white">SWE-bench Dashboard</h3>
          <p className="text-[10px] text-[#525252]">Real GitHub issue repair benchmark · Plodder agent</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleRefresh}
            title="Refresh"
            className="rounded-lg border border-[#1e1e1e] p-2 text-[#525252] transition-colors hover:border-[#2a2a2a] hover:text-[#a3a3a3]"
          >
            <RefreshCw size={13} className={isRefreshing ? 'animate-spin' : ''} />
          </button>
          <button
            onClick={() => { setFormOpen((v) => !v); setStartError(''); }}
            className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-bold transition-colors ${
              formOpen
                ? 'border border-[#2a2a2a] bg-[#1a1a1a] text-[#a3a3a3] hover:text-white'
                : 'bg-[#00ff99] text-black hover:bg-[#00e588]'
            }`}
          >
            <Play size={11} />
            {formOpen ? 'Cancel' : 'New Run'}
          </button>
        </div>
      </div>

      {/* ── Stats overview bar ──────────────────────────────────────── */}
      {stats && (
        <div className="flex flex-shrink-0 flex-wrap items-center gap-x-6 gap-y-2 border-b border-[#1a1a1a] bg-[#080808] px-5 py-2.5">
          <div className="flex items-baseline gap-1.5">
            <span className="text-2xl font-black text-[#00ff99]">
              {stats.overall_resolve_rate.toFixed(1)}
              <span className="text-sm text-[#00ff99]/60">%</span>
            </span>
            <span className="text-[9px] font-bold uppercase tracking-wider text-[#525252]">Overall Score</span>
          </div>
          <div className="h-6 w-px bg-[#1e1e1e]" />
          {[
            { label: 'Resolved', value: stats.total_resolved, accent: true },
            { label: 'Evaluated', value: stats.total_tasks_evaluated, accent: false },
            { label: 'Completed Runs', value: stats.completed_runs, accent: false },
            { label: 'Total Runs', value: stats.total_runs, accent: false },
          ].map(({ label, value, accent }) => (
            <div key={label} className="flex flex-col">
              <span className={`text-base font-bold ${accent ? 'text-[#00ff99]' : 'text-white'}`}>{value}</span>
              <span className="text-[9px] uppercase tracking-wider text-[#525252]">{label}</span>
            </div>
          ))}
          {hasRunning && (
            <div className="ml-auto flex items-center gap-1.5">
              <span className="h-2 w-2 animate-ping rounded-full bg-yellow-400" />
              <span className="text-[10px] font-bold uppercase tracking-wide text-yellow-400">Run in progress</span>
            </div>
          )}
        </div>
      )}

      {/* ── Start Benchmark Form ────────────────────────────────────── */}
      {formOpen && (
        <div className="flex-shrink-0 border-b border-[#1a1a1a] bg-gradient-to-b from-[#0d1117] to-[#0a0a0a] px-5 py-5">
          <div className="mx-auto max-w-2xl space-y-4">
            {/* Warning */}
            <div className="flex items-start gap-2.5 rounded-xl border border-yellow-500/20 bg-yellow-500/5 px-4 py-3">
              <AlertTriangle size={14} className="mt-0.5 flex-shrink-0 text-yellow-400" />
              <div className="text-xs text-yellow-200/80 leading-relaxed">
                <span className="font-bold text-yellow-300">Token usage warning: </span>
                Each SWE-bench task runs a full Plodder agent loop. Start with 1–3 tasks to estimate cost.
              </div>
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              {/* Split */}
              <div className="space-y-1.5">
                <label className="block text-[10px] font-bold uppercase tracking-wider text-[#525252]">Split</label>
                <select
                  value={formSplit}
                  onChange={(e) => setFormSplit(e.target.value)}
                  className="w-full rounded-lg border border-[#2a2a2a] bg-[#111] px-3 py-2.5 text-sm text-white outline-none focus:border-[#00ff99]/40 transition-colors"
                >
                  <option value="lite">SWE-bench Lite (300 tasks)</option>
                  <option value="full">SWE-bench Full (2294 tasks)</option>
                </select>
              </div>

              {/* Limit */}
              <div className="space-y-1.5">
                <label className="block text-[10px] font-bold uppercase tracking-wider text-[#525252]">
                  Limit
                  <span className="ml-1 font-normal normal-case text-[#3a3a3a]">(max {formUseAgent ? 10 : 50})</span>
                </label>
                <input
                  type="number"
                  min={1}
                  max={formUseAgent ? 10 : 50}
                  value={formLimit}
                  onChange={(e) => setFormLimit(Math.max(1, Math.min(formUseAgent ? 10 : 50, Number(e.target.value) || 1)))}
                  className="w-full rounded-lg border border-[#2a2a2a] bg-[#111] px-3 py-2.5 text-sm text-white outline-none focus:border-[#00ff99]/40 transition-colors"
                />
              </div>

              {/* Run name */}
              <div className="space-y-1.5">
                <label className="block text-[10px] font-bold uppercase tracking-wider text-[#525252]">Run Name <span className="font-normal normal-case text-[#3a3a3a]">(optional)</span></label>
                <input
                  placeholder="e.g. gpt-4o-mini baseline"
                  value={formName}
                  onChange={(e) => setFormName(e.target.value)}
                  className="w-full rounded-lg border border-[#2a2a2a] bg-[#111] px-3 py-2.5 text-sm text-white outline-none placeholder:text-[#3a3a3a] focus:border-[#00ff99]/40 transition-colors"
                />
              </div>

              {/* Repo filter */}
              <div className="space-y-1.5">
                <label className="block text-[10px] font-bold uppercase tracking-wider text-[#525252]">Repo Filter <span className="font-normal normal-case text-[#3a3a3a]">(optional)</span></label>
                <input
                  placeholder="e.g. django/django"
                  value={formRepo}
                  onChange={(e) => setFormRepo(e.target.value)}
                  className="w-full rounded-lg border border-[#2a2a2a] bg-[#111] px-3 py-2.5 text-sm text-white outline-none placeholder:text-[#3a3a3a] focus:border-[#00ff99]/40 transition-colors"
                />
              </div>
            </div>

            {/* Checkboxes + submit */}
            <div className="flex flex-wrap items-center gap-4 rounded-xl border border-[#1e1e1e] bg-[#0d0d0d] px-4 py-3">
              <label className="flex cursor-pointer items-center gap-2 text-xs text-[#a3a3a3] hover:text-white transition-colors">
                <input type="checkbox" checked={formUseAgent} onChange={(e) => setFormUseAgent(e.target.checked)} className="accent-[#00ff99]" />
                Run with Plodder agent
              </label>
              {formUseAgent && (
                <label className="flex cursor-pointer items-center gap-2 text-xs text-yellow-300 hover:text-yellow-200 transition-colors">
                  <input type="checkbox" checked={benchmarkConfirmed} onChange={(e) => setBenchmarkConfirmed(e.target.checked)} className="accent-yellow-400" />
                  I understand token usage
                </label>
              )}
              <button
                onClick={handleStartRun}
                disabled={isStarting || (formUseAgent && !benchmarkConfirmed)}
                className="ml-auto inline-flex items-center gap-2 rounded-lg bg-[#00ff99] px-5 py-2 text-xs font-bold text-black transition-all hover:bg-[#00e588] disabled:opacity-40 disabled:cursor-not-allowed"
              >
                {isStarting
                  ? <><RefreshCw size={12} className="animate-spin" />Starting…</>
                  : <><Play size={12} />Start Benchmark</>
                }
              </button>
            </div>

            {startError && (
              <div className="rounded-xl border border-red-500/20 bg-red-500/10 px-4 py-3 text-xs text-red-300">
                {startError}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Tab bar ─────────────────────────────────────────────────── */}
      <div className="flex flex-shrink-0 gap-1 border-b border-[#1a1a1a] px-4 pt-3">
        {(['runs', 'tasks', 'stats'] as const).map((item) => (
          <button
            key={item}
            onClick={() => setTab(item)}
            className={`rounded-t-md border-b-2 px-3 py-1.5 text-xs font-semibold transition-all ${
              tab === item ? 'border-[#00ff99] text-white' : 'border-transparent text-[#525252] hover:text-[#a3a3a3]'
            }`}
          >
            {item === 'runs' ? `Runs${runs.length ? ` (${runs.length})` : ''}` : item === 'tasks' ? 'Task Browser' : 'Stats'}
          </button>
        ))}
      </div>

      {/* ── Tab content ─────────────────────────────────────────────── */}
      <div className="min-h-0 flex-1 overflow-y-auto custom-scrollbar">

        {/* RUNS TAB */}
        {tab === 'runs' && (
          <div className="p-4">
            {runs.length === 0 ? (
              <div className="flex flex-col items-center justify-center gap-4 py-20 text-center">
                <div className="flex h-16 w-16 items-center justify-center rounded-2xl border border-[#1e1e1e] bg-[#0d0d0d]">
                  <FlaskConical size={32} className="text-[#2a2a2a]" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-[#737373]">No SWE-bench runs yet</p>
                  <p className="mt-1 text-xs text-[#3a3a3a]">Start a run to evaluate Plodder on real GitHub issues.</p>
                </div>
                <button
                  onClick={() => setFormOpen(true)}
                  className="inline-flex items-center gap-1.5 rounded-xl bg-[#00ff99] px-5 py-2.5 text-sm font-bold text-black hover:bg-[#00e588] transition-colors"
                >
                  <Play size={14} />
                  Start First Run
                </button>
              </div>
            ) : (
              <div className="space-y-3">
                {/* Table header */}
                <div className="hidden grid-cols-[1fr_80px_100px_110px_90px_60px] items-center gap-3 rounded-lg border border-[#1a1a1a] bg-[#090909] px-4 py-2 text-[9px] font-bold uppercase tracking-wider text-[#404040] sm:grid">
                  <span>Run</span>
                  <span>Split</span>
                  <span>Status</span>
                  <span>Tasks</span>
                  <span className="text-center">Score</span>
                  <span />
                </div>

                {runs.map((run) => {
                  const rate = Number(run.resolve_rate) || 0;
                  const rateColor = rate >= 30 ? 'text-[#00ff99]' : rate >= 15 ? 'text-yellow-400' : 'text-[#737373]';
                  const isExpanded = expandedRun === run.run_id;
                  return (
                    <div key={run.run_id} className={`overflow-hidden rounded-xl border transition-colors ${
                      isExpanded ? 'border-[#00ff99]/20 bg-[#050f05]' : 'border-[#1e1e1e] bg-[#0d0d0d] hover:border-[#2a2a2a]'
                    }`}>
                      {/* Run row */}
                      <div
                        className="grid cursor-pointer grid-cols-[1fr_auto] items-center gap-3 px-4 py-3.5 sm:grid-cols-[1fr_80px_100px_110px_90px_60px]"
                        onClick={() => void toggleRun(run.run_id)}
                      >
                        {/* Name + ID */}
                        <div className="min-w-0">
                          <div className="flex items-center gap-2">
                            {isExpanded ? <ChevronDown size={13} className="flex-shrink-0 text-[#525252]" /> : <ChevronRight size={13} className="flex-shrink-0 text-[#525252]" />}
                            <span className="truncate text-sm font-semibold text-white">
                              {run.name || `Run ${run.run_id.slice(0, 8)}`}
                            </span>
                          </div>
                          <p className="ml-5 mt-0.5 font-mono text-[10px] text-[#404040]">{run.run_id.slice(0, 12)}…</p>
                        </div>

                        {/* Split badge */}
                        <span className={`hidden rounded-full border px-2 py-0.5 text-center text-[10px] font-bold sm:inline-block ${
                          run.split === 'full' ? 'border-purple-500/30 text-purple-400' : 'border-sky-500/30 text-sky-400'
                        }`}>
                          {run.split === 'full' ? 'Full' : 'Lite'}
                        </span>

                        {/* Status */}
                        <div className="hidden sm:block">
                          <StatusBadge status={run.status} />
                        </div>

                        {/* Tasks */}
                        <div className="hidden items-baseline gap-1 sm:flex">
                          <span className="text-sm font-bold text-white">{run.resolved}</span>
                          <span className="text-xs text-[#404040]">/</span>
                          <span className="text-sm text-[#737373]">{run.total}</span>
                          <span className="text-[9px] text-[#3a3a3a]">tasks</span>
                        </div>

                        {/* Score % */}
                        <div className="flex flex-col items-center justify-center">
                          <span className={`text-lg font-black leading-none ${rateColor}`}>
                            {rate.toFixed(1)}<span className="text-xs font-bold opacity-60">%</span>
                          </span>
                          <span className="mt-0.5 text-[8px] uppercase tracking-wider text-[#3a3a3a]">score</span>
                        </div>

                        {/* Actions */}
                        <div className="flex items-center justify-end gap-1" onClick={(e) => e.stopPropagation()}>
                          {run.status === 'running' && (
                            <button onClick={handleCancelRun} className="rounded-md p-1.5 text-yellow-400/60 hover:bg-yellow-500/10 hover:text-yellow-300 transition-colors" title="Cancel run">
                              <StopCircle size={13} />
                            </button>
                          )}
                          <button onClick={(e) => void handleDeleteRun(run.run_id, e)} className="rounded-md p-1.5 text-[#3a3a3a] hover:bg-red-500/10 hover:text-red-400 transition-colors" title="Delete run">
                            <Trash2 size={13} />
                          </button>
                        </div>
                      </div>

                      {/* Resolve rate mini-bar */}
                      <div className="mx-4 mb-3 -mt-1 hidden h-0.5 overflow-hidden rounded-full bg-[#1a1a1a] sm:block">
                        <div
                          className={`h-full rounded-full transition-all duration-700 ${
                            rate >= 30 ? 'bg-[#00ff99]/60' : rate >= 15 ? 'bg-yellow-400/60' : 'bg-[#3a3a3a]'
                          }`}
                          style={{ width: `${Math.min(100, rate)}%` }}
                        />
                      </div>

                      {/* Expanded: task results */}
                      {isExpanded && (
                        <div className="border-t border-[#1a1a1a]">
                          {!runResults[run.run_id] ? (
                            <div className="flex items-center gap-2 px-5 py-4 text-xs italic text-[#525252]">
                              <RefreshCw size={12} className="animate-spin" />
                              Loading task results…
                            </div>
                          ) : runResults[run.run_id].length === 0 ? (
                            <p className="px-5 py-4 text-xs italic text-[#3a3a3a]">No results recorded yet.</p>
                          ) : (
                            <div className="divide-y divide-[#111]">
                              {/* Results sub-header */}
                              <div className="grid grid-cols-[1fr_90px_80px_50px] gap-3 px-5 py-2 text-[9px] font-bold uppercase tracking-wider text-[#3a3a3a]">
                                <span>Instance</span>
                                <span>Status</span>
                                <span>Repo</span>
                                <span className="text-right">Duration</span>
                              </div>
                              {runResults[run.run_id].map((result) => {
                                const isOpen = expandedResult === result.result_id;
                                return (
                                  <div key={result.result_id}>
                                    <div
                                      className="grid cursor-pointer grid-cols-[1fr_90px_80px_50px] items-center gap-3 px-5 py-2.5 transition-colors hover:bg-[#0a0f0a]"
                                      onClick={() => setExpandedResult(isOpen ? null : result.result_id)}
                                    >
                                      <div className="flex items-center gap-2 min-w-0">
                                        {isOpen ? <ChevronDown size={11} className="flex-shrink-0 text-[#3a3a3a]" /> : <ChevronRight size={11} className="flex-shrink-0 text-[#3a3a3a]" />}
                                        <span className="truncate font-mono text-[11px] text-[#a3a3a3]">{result.instance_id}</span>
                                      </div>
                                      <div><StatusBadge status={result.status} /></div>
                                      <span className="truncate text-[10px] text-[#525252]">{result.repo?.split('/')[1] || result.repo}</span>
                                      <span className="text-right font-mono text-[10px] text-[#3a3a3a]">{Number(result.duration_s).toFixed(1)}s</span>
                                    </div>
                                    {isOpen && (
                                      <div className="space-y-3 bg-[#060c06] px-5 pb-4 pt-2">
                                        {result.error && (
                                          <div className="rounded-lg border border-red-500/20 bg-red-500/5 px-3 py-2 text-[11px] text-red-300">{result.error}</div>
                                        )}
                                        {result.agent_log && (
                                          <div>
                                            <p className="mb-1 text-[9px] font-bold uppercase tracking-wider text-[#3a3a3a]">Agent log</p>
                                            <pre className="max-h-28 overflow-auto rounded-lg bg-[#050505] p-3 text-[10px] leading-relaxed text-[#737373] custom-scrollbar">{result.agent_log}</pre>
                                          </div>
                                        )}
                                        {result.patch && (
                                          <div>
                                            <p className="mb-1 text-[9px] font-bold uppercase tracking-wider text-[#3a3a3a]">Patch</p>
                                            <pre className="max-h-48 overflow-auto rounded-lg bg-[#050505] p-3 text-[10px] leading-relaxed text-[#a3a3a3] custom-scrollbar">{result.patch}</pre>
                                          </div>
                                        )}
                                        {result.test_output && (
                                          <div>
                                            <p className="mb-1 text-[9px] font-bold uppercase tracking-wider text-[#3a3a3a]">Test output</p>
                                            <pre className="max-h-28 overflow-auto rounded-lg bg-[#050505] p-3 text-[10px] leading-relaxed text-[#737373] custom-scrollbar">{result.test_output}</pre>
                                          </div>
                                        )}
                                      </div>
                                    )}
                                  </div>
                                );
                              })}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* TASKS BROWSER TAB */}
        {tab === 'tasks' && (
          <div className="p-4">
            <div className="mb-4 flex items-center gap-2">
              <div className="relative flex-1">
                <Filter size={11} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#3a3a3a]" />
                <input
                  value={taskFilter}
                  onChange={(e) => setTaskFilter(e.target.value)}
                  placeholder="Filter by repo (e.g. django/django)…"
                  className="w-full rounded-lg border border-[#1e1e1e] bg-[#0d0d0d] py-2 pl-8 pr-3 text-xs text-white outline-none placeholder:text-[#3a3a3a] focus:border-[#00ff99]/30 transition-colors"
                />
              </div>
              <button onClick={() => void fetchTasks()} className="rounded-lg border border-[#1e1e1e] p-2 text-[#525252] transition-colors hover:border-[#2a2a2a] hover:text-[#a3a3a3]">
                <RefreshCw size={12} className={loadingTasks ? 'animate-spin' : ''} />
              </button>
            </div>
            {loadingTasks ? (
              <div className="flex items-center justify-center gap-2 py-16 text-xs text-[#525252]">
                <RefreshCw size={14} className="animate-spin" /> Loading tasks…
              </div>
            ) : tasks.length === 0 ? (
              <p className="py-16 text-center text-xs italic text-[#3a3a3a]">No tasks found.</p>
            ) : (
              <div className="space-y-2">
                {tasks.map((task) => (
                  <div key={task.task_id} className="rounded-xl border border-[#1e1e1e] bg-[#0d0d0d] p-4 transition-colors hover:border-[#2a2a2a]">
                    <div className="mb-2 flex items-start justify-between gap-2">
                      <div>
                        <p className="font-mono text-xs font-bold text-white">{task.instance_id}</p>
                        <p className="mt-0.5 text-[10px] text-[#525252]">{task.repo} · v{task.version}</p>
                      </div>
                      <a href={`https://github.com/${task.repo}/issues`} target="_blank" rel="noreferrer" className="text-[#3a3a3a] transition-colors hover:text-[#737373]">
                        <ExternalLink size={12} />
                      </a>
                    </div>
                    <p className="line-clamp-3 text-[11px] leading-relaxed text-[#737373]">{task.problem_statement}</p>
                    {task.fail_to_pass.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1">
                        {task.fail_to_pass.slice(0, 4).map((t) => (
                          <span key={t} className="rounded bg-[#1a1a1a] px-1.5 py-0.5 font-mono text-[9px] text-[#525252]">{t.split('::').pop()}</span>
                        ))}
                        {task.fail_to_pass.length > 4 && (
                          <span className="rounded bg-[#1a1a1a] px-1.5 py-0.5 text-[9px] text-[#3a3a3a]">+{task.fail_to_pass.length - 4} more</span>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* STATS TAB */}
        {tab === 'stats' && (
          <div className="space-y-6 p-5">
            {/* Big score */}
            <div className="rounded-2xl border border-[#1e1e1e] bg-gradient-to-b from-[#050f05] to-[#0a0a0a] p-8 text-center">
              <p className="mb-1 text-[10px] font-bold uppercase tracking-[0.2em] text-[#525252]">Plodder · SWE-bench Resolve Rate</p>
              <p className="text-6xl font-black text-[#00ff99]">
                {stats?.overall_resolve_rate.toFixed(1) ?? '0.0'}
                <span className="text-2xl text-[#00ff99]/50">%</span>
              </p>
              <p className="mt-2 text-xs text-[#525252]">
                {stats?.total_resolved ?? 0} of {stats?.total_tasks_evaluated ?? 0} tasks resolved
              </p>
              <div className="mx-auto mt-4 h-1.5 max-w-xs overflow-hidden rounded-full bg-[#1a1a1a]">
                <div
                  className="h-full rounded-full bg-[#00ff99] transition-all duration-700"
                  style={{ width: `${stats?.overall_resolve_rate ?? 0}%` }}
                />
              </div>
            </div>

            {/* Leaderboard comparison */}
            <div className="overflow-hidden rounded-2xl border border-[#1e1e1e]">
              <div className="border-b border-[#1e1e1e] bg-[#0d0d0d] px-5 py-3">
                <p className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-wider text-[#525252]">
                  <BarChart3 size={11} /> Industry comparison (SWE-bench Lite)
                </p>
              </div>
              <div className="divide-y divide-[#111]">
                {[
                  { name: 'Devin 2.0', pct: 55.0, org: 'Cognition' },
                  { name: 'Claude Sonnet 3.7', pct: 49.0, org: 'Anthropic' },
                  { name: 'OpenHands', pct: 38.0, org: 'All-Hands' },
                  { name: 'SWE-agent', pct: 23.7, org: 'Princeton' },
                  { name: 'Plodder', pct: stats?.overall_resolve_rate ?? 0, org: 'You', isUs: true },
                ].map((entry) => {
                  const w = (entry.pct / 60) * 100;
                  return (
                    <div
                      key={entry.name}
                      className={`flex items-center gap-4 px-5 py-3 ${(entry as { isUs?: boolean }).isUs ? 'bg-[#00ff99]/5' : ''}`}
                    >
                      <div className="w-32 flex-shrink-0">
                        <p className={`text-xs font-semibold ${(entry as { isUs?: boolean }).isUs ? 'text-[#00ff99]' : 'text-white'}`}>{entry.name}</p>
                        <p className="text-[9px] text-[#3a3a3a]">{entry.org}</p>
                      </div>
                      <div className="flex-1">
                        <div className="h-2 overflow-hidden rounded-full bg-[#1a1a1a]">
                          <div
                            className={`h-full rounded-full transition-all duration-700 ${(entry as { isUs?: boolean }).isUs ? 'bg-[#00ff99]' : 'bg-[#404040]'}`}
                            style={{ width: `${w}%` }}
                          />
                        </div>
                      </div>
                      <span className={`w-12 text-right text-sm font-bold ${(entry as { isUs?: boolean }).isUs ? 'text-[#00ff99]' : 'text-white'}`}>
                        {entry.pct.toFixed(1)}%
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export const BenchmarkPanel: React.FC = () => {
  const [suite, setSuite] = useState<'code' | 'swe'>('code');

  return (
    <div className="flex h-full flex-col overflow-hidden bg-[#0f0f0f]">
      <div className="flex flex-shrink-0 flex-wrap items-center justify-between gap-3 border-b border-[#1a1a1a] px-5 py-3">
        <div className="flex items-center gap-2.5">
          <FlaskConical size={16} className="text-[#00ff99]" />
          <div>
            <h2 className="text-sm font-bold text-white">Benchmarks</h2>
            <p className="text-[10px] text-[#525252]">HumanEval, MBPP, and SWE-bench exams</p>
          </div>
        </div>
        <div className="flex rounded-lg border border-[#262626] bg-[#0a0a0a] p-0.5">
          <button
            onClick={() => setSuite('code')}
            className={`rounded-md px-3 py-1.5 text-xs font-bold transition-colors ${suite === 'code' ? 'bg-[#1a1a1a] text-[#00ff99]' : 'text-[#737373] hover:text-white'}`}
          >
            Code Bench
          </button>
          <button
            onClick={() => setSuite('swe')}
            className={`rounded-md px-3 py-1.5 text-xs font-bold transition-colors ${suite === 'swe' ? 'bg-[#1a1a1a] text-[#00ff99]' : 'text-[#737373] hover:text-white'}`}
          >
            SWE-bench
          </button>
        </div>
      </div>
      {suite === 'code' ? <CodeBenchmarkView /> : <SweBenchmarkView />}
    </div>
  );
};
