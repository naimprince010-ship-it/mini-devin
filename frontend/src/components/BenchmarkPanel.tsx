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
    if (formUseAgent && !benchmarkConfirmed) {
      window.alert('SWE-bench can spend a lot of LLM tokens. Confirm the warning before starting.');
      return;
    }
    if (formUseAgent && formLimit > 3) {
      const ok = window.confirm(`This will run ${formLimit} agent tasks and may spend quota quickly. Continue?`);
      if (!ok) return;
    }
    setIsStarting(true);
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
      await fetchRuns();
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

  const handleCancelRun = async () => {
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

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex items-center justify-between border-b border-[#1a1a1a] px-5 py-3">
        <div>
          <h3 className="text-sm font-bold text-white">SWE-bench</h3>
          <p className="text-[10px] text-[#525252]">Real GitHub issue repair benchmark</p>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={handleRefresh} className="rounded-md p-1.5 text-[#525252] transition-colors hover:bg-[#1a1a1a] hover:text-[#a3a3a3]">
            <RefreshCw size={13} className={isRefreshing ? 'animate-spin' : ''} />
          </button>
          <button onClick={() => setFormOpen((value) => !value)} className="inline-flex items-center gap-1.5 rounded-lg bg-[#00ff99] px-3 py-1.5 text-xs font-bold text-black hover:bg-[#00e588]">
            <Play size={11} />
            New Run
          </button>
        </div>
      </div>

      {stats && (
        <div className="flex items-center gap-4 border-b border-[#1a1a1a] bg-[#0a0a0a] px-5 py-2">
          <div className="text-center">
            <p className="text-lg font-bold text-white">{stats.overall_resolve_rate}%</p>
            <p className="text-[9px] uppercase tracking-wider text-[#525252]">Overall</p>
          </div>
          <div className="h-8 w-px bg-[#1a1a1a]" />
          <div className="text-center">
            <p className="text-sm font-bold text-white">{stats.total_resolved}</p>
            <p className="text-[9px] uppercase tracking-wider text-[#525252]">Resolved</p>
          </div>
          <div className="text-center">
            <p className="text-sm font-bold text-white">{stats.total_tasks_evaluated}</p>
            <p className="text-[9px] uppercase tracking-wider text-[#525252]">Evaluated</p>
          </div>
          <div className="text-center">
            <p className="text-sm font-bold text-white">{stats.completed_runs}</p>
            <p className="text-[9px] uppercase tracking-wider text-[#525252]">Runs</p>
          </div>
        </div>
      )}

      {formOpen && (
        <div className="space-y-3 border-b border-[#1a1a1a] bg-[#111] px-5 py-4">
          <div className="flex items-start gap-2 rounded-lg border border-yellow-500/25 bg-yellow-500/10 px-3 py-2 text-xs text-yellow-200">
            <AlertTriangle size={14} className="mt-0.5 flex-shrink-0" />
            <span>SWE-bench starts full agent loops. Start with 1-3 tasks.</span>
          </div>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <select value={formSplit} onChange={(e) => setFormSplit(e.target.value)} className="rounded-lg border border-[#2a2a2a] bg-[#0f0f0f] px-3 py-2 text-xs text-white outline-none">
              <option value="lite">SWE-bench Lite</option>
              <option value="full">SWE-bench Full</option>
            </select>
            <input type="number" min={1} max={formUseAgent ? 10 : 50} value={formLimit} onChange={(e) => setFormLimit(Math.max(1, Math.min(formUseAgent ? 10 : 50, Number(e.target.value) || 1)))} className="rounded-lg border border-[#2a2a2a] bg-[#0f0f0f] px-3 py-2 text-xs text-white outline-none" />
            <input placeholder="Repo filter" value={formRepo} onChange={(e) => setFormRepo(e.target.value)} className="rounded-lg border border-[#2a2a2a] bg-[#0f0f0f] px-3 py-2 text-xs text-white outline-none placeholder:text-[#3a3a3a]" />
            <input placeholder="Run name" value={formName} onChange={(e) => setFormName(e.target.value)} className="rounded-lg border border-[#2a2a2a] bg-[#0f0f0f] px-3 py-2 text-xs text-white outline-none placeholder:text-[#3a3a3a]" />
          </div>
          <div className="flex flex-wrap items-center gap-4">
            <label className="flex cursor-pointer items-center gap-2 text-xs text-[#a3a3a3]">
              <input type="checkbox" checked={formUseAgent} onChange={(e) => setFormUseAgent(e.target.checked)} className="accent-[#00ff99]" />
              Run Plodder agent
            </label>
            {formUseAgent && (
              <label className="flex cursor-pointer items-center gap-2 text-xs text-[#a3a3a3]">
                <input type="checkbox" checked={benchmarkConfirmed} onChange={(e) => setBenchmarkConfirmed(e.target.checked)} className="accent-yellow-400" />
                I understand token usage
              </label>
            )}
            <button onClick={handleStartRun} disabled={isStarting || (formUseAgent && !benchmarkConfirmed)} className="inline-flex items-center gap-1.5 rounded-lg bg-[#00ff99] px-4 py-2 text-xs font-bold text-black hover:bg-[#00e588] disabled:opacity-50">
              {isStarting ? <RefreshCw size={11} className="animate-spin" /> : <Play size={11} />}
              Start Benchmark
            </button>
          </div>
        </div>
      )}

      <div className="flex gap-1 border-b border-[#1a1a1a] px-4 pt-3">
        {(['runs', 'tasks', 'stats'] as const).map((item) => (
          <button key={item} onClick={() => setTab(item)} className={`rounded-t-md border-b-2 px-3 py-1.5 text-xs font-medium transition-all ${tab === item ? 'border-[#00ff99] text-white' : 'border-transparent text-[#525252] hover:text-[#a3a3a3]'}`}>
            {item === 'runs' ? 'Runs' : item === 'tasks' ? 'Task Browser' : 'Stats'}
          </button>
        ))}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-4 custom-scrollbar">
        {tab === 'runs' && (
          <div className="space-y-2">
            {runs.length === 0 ? (
              <div className="py-16 text-center text-sm text-[#525252]">No SWE-bench runs yet.</div>
            ) : runs.map((run) => (
              <div key={run.run_id} className="overflow-hidden rounded-xl border border-[#1e1e1e]">
                <div className="flex cursor-pointer items-center gap-3 bg-[#111] px-4 py-3 transition-colors hover:bg-[#141414]" onClick={() => void toggleRun(run.run_id)}>
                  {expandedRun === run.run_id ? <ChevronDown size={14} className="text-[#525252]" /> : <ChevronRight size={14} className="text-[#525252]" />}
                  <div className="min-w-0 flex-1">
                    <div className="mb-1 flex items-center gap-2">
                      <span className="truncate text-xs font-semibold text-white">{run.name}</span>
                      <StatusBadge status={run.status} />
                    </div>
                    <div className="text-[10px] font-mono text-[#525252]">{run.run_id} · {run.split} · {run.total} tasks</div>
                  </div>
                  <div className="w-36 flex-shrink-0">
                    <ScoreBar value={run.resolve_rate} label={`${run.resolved}/${run.total}`} />
                  </div>
                  {run.status === 'running' && (
                    <button onClick={(e) => { e.stopPropagation(); void handleCancelRun(); }} className="rounded p-1 text-red-400 hover:bg-red-500/10" title="Cancel run">
                      <StopCircle size={13} />
                    </button>
                  )}
                  <button onClick={(e) => void handleDeleteRun(run.run_id, e)} className="rounded p-1 text-[#3a3a3a] hover:bg-red-500/10 hover:text-red-400">
                    <Trash2 size={13} />
                  </button>
                </div>
                {expandedRun === run.run_id && (
                  <div className="divide-y divide-[#1a1a1a] border-t border-[#1a1a1a]">
                    {!runResults[run.run_id] ? (
                      <div className="px-4 py-3 text-xs italic text-[#525252]">Loading results...</div>
                    ) : runResults[run.run_id].length === 0 ? (
                      <div className="px-4 py-3 text-xs italic text-[#525252]">No results yet.</div>
                    ) : runResults[run.run_id].map((result) => (
                      <div key={result.result_id}>
                        <div className="flex cursor-pointer items-center gap-3 px-4 py-2.5 transition-colors hover:bg-[#0d0d0d]" onClick={() => setExpandedResult(expandedResult === result.result_id ? null : result.result_id)}>
                          {expandedResult === result.result_id ? <ChevronDown size={12} className="text-[#525252]" /> : <ChevronRight size={12} className="text-[#525252]" />}
                          <StatusBadge status={result.status} />
                          <span className="min-w-0 flex-1 truncate text-xs font-mono text-[#a3a3a3]">{result.instance_id}</span>
                          <span className="text-[10px] text-[#525252]">{result.duration_s}s</span>
                        </div>
                        {expandedResult === result.result_id && (
                          <div className="space-y-3 bg-[#080808] px-4 pb-4 pt-2">
                            {result.error && <div className="rounded-lg border border-red-500/20 bg-red-500/10 px-3 py-2 text-[10px] text-red-300">{result.error}</div>}
                            {result.agent_log && <pre className="max-h-24 overflow-auto rounded-lg bg-[#0f0f0f] p-3 text-[11px] text-[#a3a3a3] custom-scrollbar">{result.agent_log}</pre>}
                            {result.patch && <pre className="max-h-40 overflow-auto rounded-lg bg-[#050505] p-3 text-[10px] leading-5 text-[#a3a3a3] custom-scrollbar">{result.patch}</pre>}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {tab === 'tasks' && (
          <div>
            <div className="mb-4 flex items-center gap-2">
              <Filter size={12} className="text-[#525252]" />
              <input value={taskFilter} onChange={(e) => setTaskFilter(e.target.value)} placeholder="Filter by repo..." className="flex-1 rounded-lg border border-[#1e1e1e] bg-[#111] px-3 py-2 text-xs text-white outline-none placeholder:text-[#3a3a3a] focus:border-[#00ff99]/30" />
              <button onClick={() => void fetchTasks()} className="rounded-lg p-2 text-[#525252] transition-colors hover:bg-[#1a1a1a] hover:text-[#a3a3a3]">
                <RefreshCw size={12} className={loadingTasks ? 'animate-spin' : ''} />
              </button>
            </div>
            <div className="space-y-2">
              {tasks.length === 0 && !loadingTasks ? (
                <p className="py-8 text-center text-xs italic text-[#3a3a3a]">No tasks found.</p>
              ) : tasks.map((task) => (
                <div key={task.task_id} className="space-y-2 rounded-xl border border-[#1e1e1e] bg-[#111] p-4">
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <p className="font-mono text-xs font-bold text-white">{task.instance_id}</p>
                      <p className="text-[10px] text-[#525252]">{task.repo} · v{task.version}</p>
                    </div>
                    <a href={`https://github.com/${task.repo}/issues`} target="_blank" rel="noreferrer" className="text-[#525252] transition-colors hover:text-[#a3a3a3]">
                      <ExternalLink size={12} />
                    </a>
                  </div>
                  <p className="max-h-16 overflow-hidden text-[11px] leading-relaxed text-[#a3a3a3]">{task.problem_statement}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {tab === 'stats' && (
          <div className="space-y-6 p-2">
            <div className="space-y-1 text-center">
              <p className="text-[10px] font-bold uppercase tracking-widest text-[#525252]">Plodder SWE Score</p>
              <p className="text-5xl font-black text-white">{stats?.overall_resolve_rate ?? 0}<span className="text-2xl text-[#525252]">%</span></p>
              <p className="text-xs text-[#525252]">{stats?.total_resolved ?? 0} of {stats?.total_tasks_evaluated ?? 0} tasks resolved</p>
            </div>
            <div className="overflow-hidden rounded-xl border border-[#1e1e1e]">
              <div className="border-b border-[#1e1e1e] bg-[#111] px-4 py-2">
                <p className="flex items-center gap-1 text-[10px] font-bold uppercase tracking-wider text-[#525252]"><BarChart3 size={10} /> Reference comparison</p>
              </div>
              {[
                { name: 'Devin 2.0', pct: 55 },
                { name: 'Claude Sonnet', pct: 49 },
                { name: 'OpenHands', pct: 38 },
                { name: 'Plodder', pct: stats?.overall_resolve_rate ?? 0 },
              ].map((entry) => (
                <div key={entry.name} className={`flex items-center gap-3 border-b border-[#1a1a1a] px-4 py-2.5 last:border-b-0 ${entry.name === 'Plodder' ? 'bg-[#00ff99]/5' : ''}`}>
                  <span className={`min-w-0 flex-1 truncate text-xs font-semibold ${entry.name === 'Plodder' ? 'text-[#00ff99]' : 'text-white'}`}>{entry.name}</span>
                  <div className="h-1.5 w-28 overflow-hidden rounded-full bg-[#1a1a1a]">
                    <div className={`h-full rounded-full ${entry.name === 'Plodder' ? 'bg-[#00ff99]' : 'bg-[#525252]'}`} style={{ width: `${(entry.pct / 60) * 100}%` }} />
                  </div>
                  <span className={`w-10 text-right text-xs font-bold ${entry.name === 'Plodder' ? 'text-[#00ff99]' : 'text-white'}`}>{entry.pct}%</span>
                </div>
              ))}
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
