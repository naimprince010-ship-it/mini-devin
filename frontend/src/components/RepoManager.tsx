import { useState, useCallback, useEffect, useRef } from 'react';
import {
  GitBranch, GitFork, Trash2, RefreshCw, Plus, ExternalLink,
  Loader2, Check, AlertCircle, Link2, Github, LogOut, User,
  Play, GitPullRequest, Bug, GitCommit, Key, ChevronDown, ChevronRight, X
} from 'lucide-react';
import { getApiBase } from '../config/apiBase';
import { detailFromJsonBody, readJsonResponse } from '../utils/readResponseJson';

interface GitHubOAuthStatus {
  connected: boolean;
  github_login?: string;
  github_email?: string;
  github_avatar_url?: string;
  connected_at?: string;
  scopes?: string[];
  github_configured?: boolean;
}

interface Repo {
  repo_id: string;
  repo_url: string;
  repo_name: string;
  owner: string;
  default_branch: string;
  local_path: string | null;
  created_at: string;
  last_synced: string | null;
  status: string;
  has_token: boolean;
  note?: string;
}

interface PR { number: number; title: string; state: string; url: string; author: string; head: string; base: string; created_at: string; }
interface Issue { number: number; title: string; state: string; url: string; author: string; created_at: string; labels: string[]; }
interface Commit { sha: string; message: string; author: string; date: string; }

interface RepoManagerProps {
  apiBaseUrl?: string;
  sessionId?: string;
  onRepoLinked?: (repoId: string) => void;
  onOpenInSession?: (localPath: string, repoName: string) => void;
}

export function RepoManager({ apiBaseUrl = getApiBase(), sessionId, onRepoLinked, onOpenInSession }: RepoManagerProps) {
  const [repos, setRepos] = useState<Repo[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showAddForm, setShowAddForm] = useState(false);
  const [newRepoUrl, setNewRepoUrl] = useState('');
  const [newRepoToken, setNewRepoToken] = useState('');
  const [newRepoBranch, setNewRepoBranch] = useState('main');
  const [adding, setAdding] = useState(false);
  const [cloning, setCloning] = useState<string | null>(null);
  const [expandedRepo, setExpandedRepo] = useState<string | null>(null);
  const [githubStatus, setGithubStatus] = useState<GitHubOAuthStatus | null>(null);
  const [activeTab, setActiveTab] = useState<Record<string, 'pulls' | 'issues' | 'commits' | 'branches'>>({});
  const [pulls, setPulls] = useState<Record<string, PR[]>>({});
  const [issues, setIssues] = useState<Record<string, Issue[]>>({});
  const [commits, setCommits] = useState<Record<string, Commit[]>>({});
  const [branches, setBranches] = useState<Record<string, string[]>>({});
  const [tabLoading, setTabLoading] = useState<string | null>(null);
  const [showCreatePR, setShowCreatePR] = useState<string | null>(null);
  const [showCreateIssue, setShowCreateIssue] = useState<string | null>(null);
  const [showCreateBranch, setShowCreateBranch] = useState<string | null>(null);
  const [showCreateRepo, setShowCreateRepo] = useState(false);
  const [showTokenForm, setShowTokenForm] = useState<string | null>(null);
  const [tokenInput, setTokenInput] = useState('');
  const [prForm, setPrForm] = useState({ title: '', body: '', head: '', base: '' });
  const [issueForm, setIssueForm] = useState({ title: '', body: '' });
  const [branchForm, setBranchForm] = useState({ name: '', from: '' });
  const [newRepoForm, setNewRepoForm] = useState({ name: '', description: '', private: false, token: '' });
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [oauthBusy, setOauthBusy] = useState(false);

  const loadGitHubStatus = useCallback(async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/github/oauth/status`);
      const { json } = await readJsonResponse<GitHubOAuthStatus>(response);
      if (response.ok && json) setGithubStatus(json);
      else setGithubStatus({ connected: false, github_configured: false });
    } catch { setGithubStatus({ connected: false, github_configured: false }); }
  }, [apiBaseUrl]);

  useEffect(() => { loadGitHubStatus(); }, [loadGitHubStatus]);

  const loadRepos = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${apiBaseUrl}/repos`);
      if (response.status === 404 || response.status === 501) { setRepos([]); return; }
      const { json, text } = await readJsonResponse<{ repos?: Repo[] }>(response);
      if (!response.ok) throw new Error(detailFromJsonBody(json) || text || 'Failed to load repositories');
      setRepos(json?.repos || []);
    } catch (e) {
      console.warn('Repos API unavailable:', e);
      setRepos([]);
    } finally { setLoading(false); }
  }, [apiBaseUrl]);

  const applyOAuthTokenToRepos = useCallback(
    async (accessToken: string) => {
      const targets = repos.filter(r => !r.has_token);
      for (const r of targets) {
        try {
          const res = await fetch(`${apiBaseUrl}/repos/${r.repo_id}/token`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ token: accessToken }),
          });
          if (!res.ok) continue;
        } catch {
          continue;
        }
      }
      await loadRepos();
    },
    [apiBaseUrl, repos, loadRepos],
  );

  const handleGitHubOAuthConnect = useCallback(async () => {
    setOauthBusy(true);
    setError(null);
    try {
      const start = await fetch(`${apiBaseUrl}/github/oauth/start`);
      const { json: startJson, text } = await readJsonResponse<{ authorize_url?: string; state?: string }>(start);
      if (!start.ok) {
        throw new Error(detailFromJsonBody(startJson) || text || `OAuth start failed (${start.status})`);
      }
      const authorizeUrl = startJson?.authorize_url;
      const state = startJson?.state;
      if (!authorizeUrl || !state) throw new Error('Invalid OAuth start response');

      // Do not use noopener: the callback page postsMessage to window.opener.
      const popup = window.open(authorizeUrl, 'github_oauth', 'width=560,height=720');
      if (!popup) throw new Error('Popup blocked — allow popups for this site');

      const finalize = async (st: string) => {
        const res = await fetch(`${apiBaseUrl}/github/oauth/result?state=${encodeURIComponent(st)}`);
        const { json, text: rt } = await readJsonResponse<{ access_token?: string }>(res);
        if (!res.ok || !json?.access_token) {
          throw new Error(detailFromJsonBody(json) || rt || 'Could not read OAuth token');
        }
        setNewRepoToken(json.access_token);
        await applyOAuthTokenToRepos(json.access_token);
        await loadGitHubStatus();
        try { popup.close(); } catch { /* ignore */ }
      };

      await new Promise<void>((resolve, reject) => {
        const timer = window.setTimeout(() => {
          window.removeEventListener('message', onMsg);
          reject(new Error('GitHub authorization timed out — try again'));
        }, 120_000);

        const onMsg = (ev: MessageEvent) => {
          const d = ev.data;
          if (!d || typeof d !== 'object') return;
          if (d.type !== 'github_oauth_done' || d.state !== state) return;
          window.removeEventListener('message', onMsg);
          window.clearTimeout(timer);
          void finalize(String(d.state))
            .then(() => resolve())
            .catch(reject);
        };
        window.addEventListener('message', onMsg);
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : 'GitHub OAuth failed');
    } finally {
      setOauthBusy(false);
    }
  }, [apiBaseUrl, applyOAuthTokenToRepos, loadGitHubStatus]);

  useEffect(() => { loadRepos(); }, [loadRepos]);

  useEffect(() => {
    const hasPending = repos.some(r => r.status === 'pending');
    if (hasPending && !pollRef.current) {
      pollRef.current = setInterval(() => loadRepos(), 3000);
    } else if (!hasPending && pollRef.current) {
      clearInterval(pollRef.current); pollRef.current = null;
    }
    return () => { if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; } };
  }, [repos, loadRepos]);

  const handleAddRepo = async () => {
    if (!newRepoUrl.trim()) return;
    setAdding(true); setError(null);
    try {
      const response = await fetch(`${apiBaseUrl}/repos`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ repo_url: newRepoUrl, github_token: newRepoToken || null, branch: newRepoBranch })
      });
      const { json, text } = await readJsonResponse<Repo>(response);
      if (!response.ok) {
        throw new Error(detailFromJsonBody(json) || text || `HTTP ${response.status}: Failed to add repository`);
      }
      if (!json || typeof json !== 'object' || !('repo_id' in json)) {
        throw new Error(text ? 'Server returned invalid data' : 'Server returned empty response (is the API running on port 8000?)');
      }
      setRepos(prev => [json as Repo, ...prev]);
      setShowAddForm(false); setNewRepoUrl(''); setNewRepoToken(''); setNewRepoBranch('main');
    } catch (e) { setError(e instanceof Error ? e.message : 'Failed to add repository'); }
    finally { setAdding(false); }
  };

  const handleClone = async (repoId: string) => {
    setCloning(repoId);
    try {
      await fetch(`${apiBaseUrl}/repos/${repoId}/clone`, { method: 'POST' });
      await loadRepos();
    } catch (e) { setError(e instanceof Error ? e.message : 'Failed to clone'); }
    finally { setCloning(null); }
  };

  const handlePull = async (repoId: string) => {
    setCloning(repoId);
    try {
      await fetch(`${apiBaseUrl}/repos/${repoId}/pull`, { method: 'POST' });
      await loadRepos();
    } catch (e) { setError(e instanceof Error ? e.message : 'Failed to pull'); }
    finally { setCloning(null); }
  };

  const handleDelete = async (repoId: string) => {
    if (!confirm('Delete this repository?')) return;
    try {
      await fetch(`${apiBaseUrl}/repos/${repoId}`, { method: 'DELETE' });
      setRepos(prev => prev.filter(r => r.repo_id !== repoId));
    } catch (e) { setError(e instanceof Error ? e.message : 'Failed to delete'); }
  };

  const handleSetToken = async (repoId: string) => {
    if (!tokenInput.trim()) return;
    try {
      await fetch(`${apiBaseUrl}/repos/${repoId}/token`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token: tokenInput })
      });
      setRepos(prev => prev.map(r => r.repo_id === repoId ? { ...r, has_token: true } : r));
      setShowTokenForm(null); setTokenInput('');
    } catch (e) { setError(e instanceof Error ? e.message : 'Failed to set token'); }
  };

  const loadTab = async (repoId: string, tab: 'pulls' | 'issues' | 'commits' | 'branches') => {
    setActiveTab(prev => ({ ...prev, [repoId]: tab }));
    setTabLoading(`${repoId}-${tab}`);
    try {
      if (tab === 'pulls') {
        const r = await fetch(`${apiBaseUrl}/repos/${repoId}/pulls`);
        const { json, text } = await readJsonResponse<{ pulls?: PR[] }>(r);
        if (r.ok && json?.pulls) setPulls(prev => ({ ...prev, [repoId]: json.pulls! }));
        else setError(detailFromJsonBody(json) || text || 'Failed to load pulls');
      } else if (tab === 'issues') {
        const r = await fetch(`${apiBaseUrl}/repos/${repoId}/issues`);
        const { json, text } = await readJsonResponse<{ issues?: Issue[] }>(r);
        if (r.ok && json?.issues) setIssues(prev => ({ ...prev, [repoId]: json.issues! }));
        else setError(detailFromJsonBody(json) || text || 'Failed to load issues');
      } else if (tab === 'commits') {
        const r = await fetch(`${apiBaseUrl}/repos/${repoId}/commits`);
        const { json, text } = await readJsonResponse<{ commits?: Commit[] }>(r);
        if (r.ok && json?.commits) setCommits(prev => ({ ...prev, [repoId]: json.commits! }));
        else setError(detailFromJsonBody(json) || text || 'Failed to load commits');
      } else if (tab === 'branches') {
        const r = await fetch(`${apiBaseUrl}/repos/${repoId}/branches`);
        const { json, text } = await readJsonResponse<{ branches?: string[] }>(r);
        if (r.ok && json?.branches) setBranches(prev => ({ ...prev, [repoId]: json.branches! }));
        else setError(detailFromJsonBody(json) || text || 'Failed to load branches');
      }
    } catch { setError('GitHub API failed. Make sure you have added a token.'); }
    finally { setTabLoading(null); }
  };

  const handleCreatePR = async (repoId: string) => {
    try {
      const r = await fetch(`${apiBaseUrl}/repos/${repoId}/pulls`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(prForm)
      });
      const { json, text } = await readJsonResponse<{ detail?: string; url?: string }>(r);
      if (!r.ok) throw new Error(detailFromJsonBody(json) || text || 'Failed to create PR');
      if (json?.url) window.open(json.url, '_blank');
      setShowCreatePR(null); setPrForm({ title: '', body: '', head: '', base: '' });
      loadTab(repoId, 'pulls');
    } catch (e) { setError(e instanceof Error ? e.message : 'Failed to create PR'); }
  };

  const handleCreateIssue = async (repoId: string) => {
    try {
      const r = await fetch(`${apiBaseUrl}/repos/${repoId}/issues`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(issueForm)
      });
      const { json, text } = await readJsonResponse<{ detail?: string; url?: string }>(r);
      if (!r.ok) throw new Error(detailFromJsonBody(json) || text || 'Failed to create issue');
      if (json?.url) window.open(json.url, '_blank');
      setShowCreateIssue(null); setIssueForm({ title: '', body: '' });
      loadTab(repoId, 'issues');
    } catch (e) { setError(e instanceof Error ? e.message : 'Failed to create issue'); }
  };

  const handleCreateBranch = async (repoId: string) => {
    const repo = repos.find(r => r.repo_id === repoId);
    try {
      const r = await fetch(`${apiBaseUrl}/repos/${repoId}/branches`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ branch_name: branchForm.name, from_branch: branchForm.from || repo?.default_branch })
      });
      const { json, text } = await readJsonResponse<{ detail?: string }>(r);
      if (!r.ok) throw new Error(detailFromJsonBody(json) || text || 'Failed to create branch');
      setShowCreateBranch(null); setBranchForm({ name: '', from: '' });
      loadTab(repoId, 'branches');
    } catch (e) { setError(e instanceof Error ? e.message : 'Failed to create branch'); }
  };

  const handleCreateRepo = async () => {
    try {
      const r = await fetch(`${apiBaseUrl}/github/create-repo`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newRepoForm)
      });
      const { json, text } = await readJsonResponse<{ detail?: string; repo_url?: string }>(r);
      if (!r.ok) throw new Error(detailFromJsonBody(json) || text || 'Failed to create repo');
      if (!json?.repo_url) throw new Error('Server did not return repo_url');
      setShowCreateRepo(false);
      setNewRepoForm({ name: '', description: '', private: false, token: '' });
      // Auto-add the new repo
      setNewRepoUrl(json.repo_url);
      setShowAddForm(true);
    } catch (e) { setError(e instanceof Error ? e.message : 'Failed to create repo'); }
  };

  const getStatusBadge = (status: string) => {
    if (status === 'cloned') return <span className="flex items-center gap-1 px-2 py-0.5 bg-green-900/40 text-green-400 text-xs rounded-full"><Check size={10} /> Ready</span>;
    if (status === 'pending') return <span className="flex items-center gap-1 px-2 py-0.5 bg-yellow-900/40 text-yellow-400 text-xs rounded-full"><Loader2 size={10} className="animate-spin" /> Cloning...</span>;
    return <span className="flex items-center gap-1 px-2 py-0.5 bg-red-900/40 text-red-400 text-xs rounded-full"><AlertCircle size={10} /> Failed</span>;
  };

  return (
    <div className="p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-white flex items-center gap-2"><GitFork size={20} /> Repositories</h2>
        <div className="flex items-center gap-2">
          <button onClick={() => setShowCreateRepo(true)} className="flex items-center gap-1 px-3 py-1.5 text-xs bg-purple-600/20 hover:bg-purple-600/30 text-purple-400 rounded-lg" title="Create new GitHub repo"><Plus size={12} /> New Repo</button>
          <button onClick={loadRepos} disabled={loading} className="p-2 text-gray-400 hover:text-white rounded-lg hover:bg-gray-700" title="Refresh"><RefreshCw size={16} className={loading ? 'animate-spin' : ''} /></button>
          <button onClick={() => setShowAddForm(!showAddForm)} className="p-2 text-gray-400 hover:text-white rounded-lg hover:bg-gray-700" title="Add Repository"><Plus size={16} /></button>
        </div>
      </div>

      {error && (
        <div className="p-3 bg-red-900/50 border border-red-700 rounded-lg text-red-200 text-sm flex justify-between">
          <span>{error}</span>
          <button onClick={() => setError(null)}><X size={14} /></button>
        </div>
      )}

      {githubStatus?.github_configured && !githubStatus.connected && (
        <div className="p-3 bg-gray-800/80 border border-gray-600 rounded-lg flex flex-wrap items-center justify-between gap-2">
          <p className="text-xs text-gray-300">
            Sign in with GitHub to obtain a token. It will be saved on each repo that does not have a token yet (same as manual key entry).
          </p>
          <button
            type="button"
            onClick={() => void handleGitHubOAuthConnect()}
            disabled={oauthBusy}
            className="shrink-0 flex items-center gap-1.5 px-3 py-1.5 text-xs bg-gray-700 hover:bg-gray-600 text-white rounded-lg disabled:opacity-50"
          >
            {oauthBusy ? <Loader2 size={14} className="animate-spin" /> : <Github size={14} />}
            Connect GitHub
          </button>
        </div>
      )}

      {/* Create New Repo Modal */}
      {showCreateRepo && (
        <div className="p-4 bg-gray-700 rounded-lg space-y-3 border border-purple-500/30">
          <h3 className="text-sm font-medium text-white flex items-center gap-2"><Github size={14} /> Create New GitHub Repository</h3>
          <input className="w-full px-3 py-2 bg-gray-600 text-white rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-purple-500" placeholder="Repository name" value={newRepoForm.name} onChange={e => setNewRepoForm(p => ({ ...p, name: e.target.value }))} />
          <input className="w-full px-3 py-2 bg-gray-600 text-white rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-purple-500" placeholder="Description (optional)" value={newRepoForm.description} onChange={e => setNewRepoForm(p => ({ ...p, description: e.target.value }))} />
          <input type="password" className="w-full px-3 py-2 bg-gray-600 text-white rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-purple-500" placeholder="GitHub Token (ghp_...)" value={newRepoForm.token} onChange={e => setNewRepoForm(p => ({ ...p, token: e.target.value }))} />
          <label className="flex items-center gap-2 text-sm text-gray-300 cursor-pointer"><input type="checkbox" checked={newRepoForm.private} onChange={e => setNewRepoForm(p => ({ ...p, private: e.target.checked }))} /> Private repository</label>
          <div className="flex gap-2">
            <button onClick={handleCreateRepo} disabled={!newRepoForm.name || !newRepoForm.token} className="flex-1 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 text-white rounded-lg text-sm">Create Repository</button>
            <button onClick={() => setShowCreateRepo(false)} className="px-4 py-2 bg-gray-600 hover:bg-gray-500 text-white rounded-lg text-sm">Cancel</button>
          </div>
        </div>
      )}

      {/* Add Existing Repo Form */}
      {showAddForm && (
        <div className="p-4 bg-gray-700 rounded-lg space-y-3">
          <h3 className="text-sm font-medium text-white">Add GitHub Repository</h3>
          <p className="text-xs text-gray-400">Paste any GitHub repo URL. Public repos clone instantly — no login needed!</p>
          <input autoFocus type="text" value={newRepoUrl} onChange={e => setNewRepoUrl(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleAddRepo()}
            placeholder="https://github.com/your-username/your-repo"
            className="w-full px-3 py-2 bg-gray-600 text-white rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500" />
          <input type="password" value={newRepoToken} onChange={e => setNewRepoToken(e.target.value)}
            placeholder="GitHub Token (optional — needed for PRs/Issues/Branches)"
            className="w-full px-3 py-2 bg-gray-600 text-white rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500" />
          <input type="text" value={newRepoBranch} onChange={e => setNewRepoBranch(e.target.value)} placeholder="main"
            className="w-full px-3 py-2 bg-gray-600 text-white rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500" />
          <div className="flex gap-2">
            <button onClick={handleAddRepo} disabled={adding || !newRepoUrl.trim()} className="flex-1 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded-lg text-sm flex items-center justify-center gap-2">
              {adding ? <><Loader2 size={14} className="animate-spin" /> Adding...</> : <><Plus size={14} /> Add Repository</>}
            </button>
            <button onClick={() => setShowAddForm(false)} className="px-4 py-2 bg-gray-600 hover:bg-gray-500 text-white rounded-lg text-sm">Cancel</button>
          </div>
        </div>
      )}

      {/* Repo List */}
      <div className="space-y-2">
        {loading && repos.length === 0 ? (
          <p className="text-gray-400 text-sm text-center py-4">Loading repositories...</p>
        ) : repos.length === 0 ? (
          <div className="text-center py-8 space-y-2">
            <GitFork size={32} className="mx-auto text-gray-600" />
            <p className="text-gray-400 text-sm">No repositories connected. Click + to add one.</p>
            <p className="text-gray-500 text-xs">You can clone any GitHub repo and Plodder will work inside it.</p>
          </div>
        ) : (
          repos.map(repo => (
            <div key={repo.repo_id} className="bg-gray-800 rounded-xl overflow-hidden border border-gray-700">
              {/* Repo Header */}
              <div className="p-3 flex items-center justify-between hover:bg-gray-750 cursor-pointer" onClick={() => setExpandedRepo(expandedRepo === repo.repo_id ? null : repo.repo_id)}>
                <div className="flex items-center gap-2 min-w-0">
                  <button onClick={e => { e.stopPropagation(); setExpandedRepo(expandedRepo === repo.repo_id ? null : repo.repo_id); }} className="text-gray-500 hover:text-gray-300">
                    {expandedRepo === repo.repo_id ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                  </button>
                  <GitFork size={14} className="text-gray-400 shrink-0" />
                  <span className="text-white font-medium text-sm truncate">{repo.owner}/{repo.repo_name}</span>
                  {getStatusBadge(repo.status)}
                  {repo.note && <span className="text-xs text-blue-400 truncate">(empty repo)</span>}
                </div>
                <div className="flex items-center gap-1 shrink-0" onClick={e => e.stopPropagation()}>
                  {/* Token */}
                  <button onClick={() => { setShowTokenForm(showTokenForm === repo.repo_id ? null : repo.repo_id); setTokenInput(''); }}
                    className={`p-1.5 rounded ${repo.has_token ? 'text-green-400' : 'text-gray-500 hover:text-yellow-400'}`} title={repo.has_token ? 'Token set — click to update' : 'Add GitHub token for PR/Issue/Branch ops'}>
                    <Key size={13} />
                  </button>
                  {/* Open in session */}
                  {repo.status === 'cloned' && onOpenInSession && repo.local_path && (
                    <button onClick={() => onOpenInSession(repo.local_path!, repo.repo_name)} className="p-1.5 text-gray-400 hover:text-green-400 rounded" title="Open in new session"><Play size={13} /></button>
                  )}
                  {/* Pull */}
                  {repo.status === 'cloned' && (
                    <button onClick={() => handlePull(repo.repo_id)} disabled={cloning === repo.repo_id} className="p-1.5 text-gray-400 hover:text-blue-400 rounded" title="Pull latest">
                      {cloning === repo.repo_id ? <Loader2 size={13} className="animate-spin" /> : <RefreshCw size={13} />}
                    </button>
                  )}
                  {/* GitHub link */}
                  <a href={repo.repo_url} target="_blank" rel="noopener noreferrer" onClick={e => e.stopPropagation()} className="p-1.5 text-gray-400 hover:text-blue-400 rounded"><ExternalLink size={13} /></a>
                  {/* Delete */}
                  <button onClick={() => handleDelete(repo.repo_id)} className="p-1.5 text-gray-400 hover:text-red-400 rounded"><Trash2 size={13} /></button>
                </div>
              </div>

              {/* Token Form */}
              {showTokenForm === repo.repo_id && (
                <div className="px-4 pb-3 flex gap-2 bg-gray-750 border-t border-gray-700">
                  <input type="password" value={tokenInput} onChange={e => setTokenInput(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleSetToken(repo.repo_id)}
                    placeholder="ghp_xxxxxxxxxxxx" className="flex-1 mt-3 px-3 py-1.5 bg-gray-600 text-white rounded-lg text-sm focus:outline-none focus:ring-1 focus:ring-yellow-500" autoFocus />
                  <button onClick={() => handleSetToken(repo.repo_id)} className="mt-3 px-3 py-1.5 bg-yellow-600 hover:bg-yellow-700 text-white rounded-lg text-sm">Save</button>
                </div>
              )}

              {/* Expanded: GitHub Operations */}
              {expandedRepo === repo.repo_id && (
                <div className="border-t border-gray-700">
                  {/* Tabs */}
                  <div className="flex gap-0 border-b border-gray-700">
                    {(['branches', 'pulls', 'issues', 'commits'] as const).map(tab => (
                      <button key={tab} onClick={() => loadTab(repo.repo_id, tab)}
                        className={`flex-1 flex items-center justify-center gap-1 px-2 py-2 text-xs transition-colors ${activeTab[repo.repo_id] === tab ? 'text-white bg-gray-700 border-b-2 border-blue-500' : 'text-gray-400 hover:text-white hover:bg-gray-750'}`}>
                        {tab === 'branches' && <GitBranch size={11} />}
                        {tab === 'pulls' && <GitPullRequest size={11} />}
                        {tab === 'issues' && <Bug size={11} />}
                        {tab === 'commits' && <GitCommit size={11} />}
                        {tab.charAt(0).toUpperCase() + tab.slice(1)}
                      </button>
                    ))}
                  </div>

                  <div className="p-3 max-h-72 overflow-y-auto">
                    {tabLoading?.startsWith(repo.repo_id) ? (
                      <div className="flex justify-center py-4"><Loader2 size={20} className="animate-spin text-gray-400" /></div>
                    ) : (
                      <>
                        {/* Branches Tab */}
                        {activeTab[repo.repo_id] === 'branches' && (
                          <div className="space-y-2">
                            <div className="flex justify-between items-center mb-2">
                              <span className="text-xs text-gray-400">{branches[repo.repo_id]?.length || 0} branches</span>
                              <button onClick={() => setShowCreateBranch(showCreateBranch === repo.repo_id ? null : repo.repo_id)}
                                className="flex items-center gap-1 text-xs px-2 py-1 bg-blue-600/20 hover:bg-blue-600/30 text-blue-400 rounded">
                                <Plus size={10} /> New Branch
                              </button>
                            </div>
                            {showCreateBranch === repo.repo_id && (
                              <div className="space-y-2 p-2 bg-gray-700 rounded-lg mb-2">
                                <input className="w-full px-2 py-1.5 bg-gray-600 text-white rounded text-xs focus:outline-none" placeholder="branch-name" value={branchForm.name} onChange={e => setBranchForm(p => ({ ...p, name: e.target.value }))} />
                                <input className="w-full px-2 py-1.5 bg-gray-600 text-white rounded text-xs focus:outline-none" placeholder={`from: ${repo.default_branch}`} value={branchForm.from} onChange={e => setBranchForm(p => ({ ...p, from: e.target.value }))} />
                                <div className="flex gap-2">
                                  <button onClick={() => handleCreateBranch(repo.repo_id)} disabled={!branchForm.name} className="flex-1 py-1.5 bg-blue-600 text-white rounded text-xs disabled:opacity-50">Create</button>
                                  <button onClick={() => setShowCreateBranch(null)} className="px-3 py-1.5 bg-gray-600 text-white rounded text-xs">Cancel</button>
                                </div>
                              </div>
                            )}
                            {(branches[repo.repo_id] || []).map(b => (
                              <div key={b} className="flex items-center gap-2 py-1.5 px-2 bg-gray-700 rounded-lg">
                                <GitBranch size={12} className="text-gray-400" />
                                <span className="text-sm text-white">{b}</span>
                                {b === repo.default_branch && <span className="ml-auto text-xs text-green-400 bg-green-900/30 px-1.5 py-0.5 rounded">default</span>}
                              </div>
                            ))}
                            {branches[repo.repo_id]?.length === 0 && <p className="text-gray-500 text-xs text-center py-2">No branches found</p>}
                          </div>
                        )}

                        {/* Pull Requests Tab */}
                        {activeTab[repo.repo_id] === 'pulls' && (
                          <div className="space-y-2">
                            <div className="flex justify-between items-center mb-2">
                              <span className="text-xs text-gray-400">{pulls[repo.repo_id]?.length || 0} open PRs</span>
                              <button onClick={() => setShowCreatePR(showCreatePR === repo.repo_id ? null : repo.repo_id)}
                                className="flex items-center gap-1 text-xs px-2 py-1 bg-green-600/20 hover:bg-green-600/30 text-green-400 rounded">
                                <Plus size={10} /> New PR
                              </button>
                            </div>
                            {showCreatePR === repo.repo_id && (
                              <div className="space-y-2 p-2 bg-gray-700 rounded-lg mb-2">
                                <input className="w-full px-2 py-1.5 bg-gray-600 text-white rounded text-xs focus:outline-none" placeholder="PR title" value={prForm.title} onChange={e => setPrForm(p => ({ ...p, title: e.target.value }))} />
                                <div className="flex gap-2">
                                  <input className="flex-1 px-2 py-1.5 bg-gray-600 text-white rounded text-xs focus:outline-none" placeholder="head branch (feature)" value={prForm.head} onChange={e => setPrForm(p => ({ ...p, head: e.target.value }))} />
                                  <input className="flex-1 px-2 py-1.5 bg-gray-600 text-white rounded text-xs focus:outline-none" placeholder={`base (${repo.default_branch})`} value={prForm.base} onChange={e => setPrForm(p => ({ ...p, base: e.target.value }))} />
                                </div>
                                <textarea className="w-full px-2 py-1.5 bg-gray-600 text-white rounded text-xs focus:outline-none" rows={2} placeholder="Description..." value={prForm.body} onChange={e => setPrForm(p => ({ ...p, body: e.target.value }))} />
                                <div className="flex gap-2">
                                  <button onClick={() => handleCreatePR(repo.repo_id)} disabled={!prForm.title || !prForm.head} className="flex-1 py-1.5 bg-green-600 text-white rounded text-xs disabled:opacity-50">Create PR</button>
                                  <button onClick={() => setShowCreatePR(null)} className="px-3 py-1.5 bg-gray-600 text-white rounded text-xs">Cancel</button>
                                </div>
                              </div>
                            )}
                            {(pulls[repo.repo_id] || []).map(pr => (
                              <a key={pr.number} href={pr.url} target="_blank" rel="noopener noreferrer"
                                className="flex items-start gap-2 py-2 px-2 bg-gray-700 hover:bg-gray-650 rounded-lg group">
                                <GitPullRequest size={14} className="text-green-400 mt-0.5 shrink-0" />
                                <div className="flex-1 min-w-0">
                                  <p className="text-sm text-white truncate group-hover:text-blue-300">#{pr.number} {pr.title}</p>
                                  <p className="text-xs text-gray-400">{pr.head} → {pr.base} · by {pr.author}</p>
                                </div>
                                <ExternalLink size={11} className="text-gray-500 group-hover:text-blue-400 shrink-0 mt-1" />
                              </a>
                            ))}
                            {(pulls[repo.repo_id] === undefined || pulls[repo.repo_id]?.length === 0) && (
                              <p className="text-gray-500 text-xs text-center py-2">
                                {pulls[repo.repo_id] === undefined ? 'Could not load PRs — GitHub token required.' : 'No open pull requests'}
                              </p>
                            )}
                          </div>
                        )}

                        {/* Issues Tab */}
                        {activeTab[repo.repo_id] === 'issues' && (
                          <div className="space-y-2">
                            <div className="flex justify-between items-center mb-2">
                              <span className="text-xs text-gray-400">{issues[repo.repo_id]?.length || 0} open issues</span>
                              <button onClick={() => setShowCreateIssue(showCreateIssue === repo.repo_id ? null : repo.repo_id)}
                                className="flex items-center gap-1 text-xs px-2 py-1 bg-red-600/20 hover:bg-red-600/30 text-red-400 rounded">
                                <Plus size={10} /> New Issue
                              </button>
                            </div>
                            {showCreateIssue === repo.repo_id && (
                              <div className="space-y-2 p-2 bg-gray-700 rounded-lg mb-2">
                                <input className="w-full px-2 py-1.5 bg-gray-600 text-white rounded text-xs focus:outline-none" placeholder="Issue title" value={issueForm.title} onChange={e => setIssueForm(p => ({ ...p, title: e.target.value }))} />
                                <textarea className="w-full px-2 py-1.5 bg-gray-600 text-white rounded text-xs focus:outline-none" rows={2} placeholder="Description..." value={issueForm.body} onChange={e => setIssueForm(p => ({ ...p, body: e.target.value }))} />
                                <div className="flex gap-2">
                                  <button onClick={() => handleCreateIssue(repo.repo_id)} disabled={!issueForm.title} className="flex-1 py-1.5 bg-red-600 text-white rounded text-xs disabled:opacity-50">Create Issue</button>
                                  <button onClick={() => setShowCreateIssue(null)} className="px-3 py-1.5 bg-gray-600 text-white rounded text-xs">Cancel</button>
                                </div>
                              </div>
                            )}
                            {(issues[repo.repo_id] || []).map(issue => (
                              <a key={issue.number} href={issue.url} target="_blank" rel="noopener noreferrer"
                                className="flex items-start gap-2 py-2 px-2 bg-gray-700 hover:bg-gray-650 rounded-lg group">
                                <Bug size={14} className="text-red-400 mt-0.5 shrink-0" />
                                <div className="flex-1 min-w-0">
                                  <p className="text-sm text-white truncate group-hover:text-blue-300">#{issue.number} {issue.title}</p>
                                  <p className="text-xs text-gray-400">by {issue.author} {issue.labels.length > 0 && `· ${issue.labels.join(', ')}`}</p>
                                </div>
                                <ExternalLink size={11} className="text-gray-500 group-hover:text-blue-400 shrink-0 mt-1" />
                              </a>
                            ))}
                            {(issues[repo.repo_id] === undefined || issues[repo.repo_id]?.length === 0) && (
                              <p className="text-gray-500 text-xs text-center py-2">
                                {issues[repo.repo_id] === undefined ? 'Could not load issues — GitHub token required.' : 'No open issues'}
                              </p>
                            )}
                          </div>
                        )}

                        {/* Commits Tab */}
                        {activeTab[repo.repo_id] === 'commits' && (
                          <div className="space-y-1">
                            {(commits[repo.repo_id] || []).map(c => (
                              <div key={c.sha} className="flex items-start gap-2 py-1.5 px-2 bg-gray-700 rounded-lg">
                                <GitCommit size={12} className="text-gray-400 mt-1 shrink-0" />
                                <div className="flex-1 min-w-0">
                                  <p className="text-sm text-white truncate">{c.message}</p>
                                  <p className="text-xs text-gray-400">{c.sha} · {c.author} · {new Date(c.date).toLocaleDateString()}</p>
                                </div>
                              </div>
                            ))}
                            {(commits[repo.repo_id] === undefined || commits[repo.repo_id]?.length === 0) && (
                              <p className="text-gray-500 text-xs text-center py-2">
                                {commits[repo.repo_id] === undefined ? 'Could not load commits — GitHub token required.' : 'No commits found'}
                              </p>
                            )}
                          </div>
                        )}

                        {!activeTab[repo.repo_id] && (
                          <p className="text-gray-500 text-xs text-center py-4">Select a tab above to view branches, PRs, issues, or commits.<br />
                            <span className="text-yellow-400">⚠ GitHub token required</span> for private repos and write operations.</p>
                        )}
                      </>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
