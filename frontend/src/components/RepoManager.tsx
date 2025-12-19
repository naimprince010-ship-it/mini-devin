import { useState, useCallback, useEffect } from 'react';
import { GitBranch, GitFork, Trash2, RefreshCw, Plus, ExternalLink, Loader2, Check, AlertCircle, Link2, Github, LogOut, User } from 'lucide-react';

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
  local_status?: string;
  current_branch?: string;
}

interface RepoManagerProps {
  apiBaseUrl?: string;
  sessionId?: string;
  onRepoLinked?: (repoId: string) => void;
}

export function RepoManager({ apiBaseUrl = 'http://localhost:8000/api', sessionId, onRepoLinked }: RepoManagerProps) {
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
  const [connectingGithub, setConnectingGithub] = useState(false);
  const [disconnectingGithub, setDisconnectingGithub] = useState(false);

  const loadGitHubStatus = useCallback(async () => {
    try {
      const token = localStorage.getItem('auth_token');
      const headers: Record<string, string> = {};
      if (token) {
        headers['Authorization'] = `Bearer ${token}`;
      }
      const response = await fetch(`${apiBaseUrl}/github/oauth/status`, { headers });
      if (response.ok) {
        const data = await response.json();
        setGithubStatus(data);
      }
    } catch (e) {
      console.error('Failed to load GitHub status:', e);
    }
  }, [apiBaseUrl]);

  const handleConnectGitHub = async () => {
    setConnectingGithub(true);
    try {
      const token = localStorage.getItem('auth_token');
      const headers: Record<string, string> = {};
      if (token) {
        headers['Authorization'] = `Bearer ${token}`;
      }
      const response = await fetch(`${apiBaseUrl}/github/oauth/start`, { headers });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to start GitHub OAuth');
      }
      const data = await response.json();
      window.location.href = data.auth_url;
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to connect GitHub');
      setConnectingGithub(false);
    }
  };

  const handleDisconnectGitHub = async () => {
    if (!confirm('Are you sure you want to disconnect your GitHub account?')) return;
    
    setDisconnectingGithub(true);
    try {
      const token = localStorage.getItem('auth_token');
      const headers: Record<string, string> = { 'Content-Type': 'application/json' };
      if (token) {
        headers['Authorization'] = `Bearer ${token}`;
      }
      const response = await fetch(`${apiBaseUrl}/github/oauth/disconnect`, {
        method: 'DELETE',
        headers
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to disconnect GitHub');
      }
      setGithubStatus({ connected: false, github_configured: true });
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to disconnect GitHub');
    } finally {
      setDisconnectingGithub(false);
    }
  };

  useEffect(() => {
    loadGitHubStatus();
    
    const urlParams = new URLSearchParams(window.location.search);
    const oauthStatus = urlParams.get('github_oauth');
    if (oauthStatus === 'success') {
      loadGitHubStatus();
      window.history.replaceState({}, '', window.location.pathname);
    } else if (oauthStatus === 'error') {
      const message = urlParams.get('message') || 'GitHub OAuth failed';
      setError(message.replace(/\+/g, ' '));
      window.history.replaceState({}, '', window.location.pathname);
    }
  }, [loadGitHubStatus]);

  const loadRepos = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${apiBaseUrl}/repos`);
      if (!response.ok) throw new Error('Failed to load repositories');
      const data = await response.json();
      setRepos(data.repos || []);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load repositories');
    } finally {
      setLoading(false);
    }
  }, [apiBaseUrl]);

  useEffect(() => {
    loadRepos();
  }, [loadRepos]);

  const handleAddRepo = async () => {
    if (!newRepoUrl.trim()) return;
    
    setAdding(true);
    setError(null);
    try {
      const response = await fetch(`${apiBaseUrl}/repos`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          repo_url: newRepoUrl,
          github_token: newRepoToken || null,
          branch: newRepoBranch
        })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to add repository');
      }
      
      const data = await response.json();
      setRepos(prev => [data, ...prev]);
      setShowAddForm(false);
      setNewRepoUrl('');
      setNewRepoToken('');
      setNewRepoBranch('main');
      
      // Auto-clone after adding
      handleClone(data.repo_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to add repository');
    } finally {
      setAdding(false);
    }
  };

  const handleClone = async (repoId: string) => {
    setCloning(repoId);
    setError(null);
    try {
      const response = await fetch(`${apiBaseUrl}/repos/${repoId}/clone`, {
        method: 'POST'
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to clone repository');
      }
      
      await loadRepos();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to clone repository');
    } finally {
      setCloning(null);
    }
  };

  const handlePull = async (repoId: string) => {
    setCloning(repoId);
    setError(null);
    try {
      const response = await fetch(`${apiBaseUrl}/repos/${repoId}/pull`, {
        method: 'POST'
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to pull repository');
      }
      
      await loadRepos();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to pull repository');
    } finally {
      setCloning(null);
    }
  };

  const handleDelete = async (repoId: string) => {
    if (!confirm('Are you sure you want to delete this repository?')) return;
    
    try {
      const response = await fetch(`${apiBaseUrl}/repos/${repoId}`, {
        method: 'DELETE'
      });
      
      if (!response.ok) throw new Error('Failed to delete repository');
      
      setRepos(prev => prev.filter(r => r.repo_id !== repoId));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to delete repository');
    }
  };

  const handleLinkToSession = async (repoId: string) => {
    if (!sessionId) {
      setError('No session selected. Create or select a session first.');
      return;
    }
    
    try {
      const response = await fetch(`${apiBaseUrl}/sessions/${sessionId}/repos`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ repo_id: repoId, branch: 'main' })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to link repository');
      }
      
      onRepoLinked?.(repoId);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to link repository');
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'cloned':
        return <Check size={14} className="text-green-400" />;
      case 'pending':
        return <AlertCircle size={14} className="text-yellow-400" />;
      case 'clone_failed':
        return <AlertCircle size={14} className="text-red-400" />;
      default:
        return <GitBranch size={14} className="text-gray-400" />;
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'cloned':
        return 'Ready';
      case 'pending':
        return 'Not cloned';
      case 'clone_failed':
        return 'Clone failed';
      default:
        return status;
    }
  };

  return (
    <div className="p-4">
      {/* GitHub OAuth Connection Section */}
      <div className="mb-6 p-4 bg-gray-800 rounded-lg border border-gray-700">
        <div className="flex items-center gap-2 mb-3">
          <Github size={20} className="text-white" />
          <h3 className="text-sm font-semibold text-white">GitHub Connection</h3>
        </div>
        
        {githubStatus?.connected ? (
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {githubStatus.github_avatar_url ? (
                <img 
                  src={githubStatus.github_avatar_url} 
                  alt={githubStatus.github_login} 
                  className="w-10 h-10 rounded-full"
                />
              ) : (
                <div className="w-10 h-10 rounded-full bg-gray-600 flex items-center justify-center">
                  <User size={20} className="text-gray-400" />
                </div>
              )}
              <div>
                <p className="text-white font-medium">{githubStatus.github_login}</p>
                <p className="text-gray-400 text-xs">{githubStatus.github_email || 'Connected via OAuth'}</p>
              </div>
            </div>
            <button
              onClick={handleDisconnectGitHub}
              disabled={disconnectingGithub}
              className="flex items-center gap-2 px-3 py-1.5 bg-red-600/20 hover:bg-red-600/30 text-red-400 rounded-lg text-sm transition-colors"
            >
              {disconnectingGithub ? (
                <Loader2 size={14} className="animate-spin" />
              ) : (
                <LogOut size={14} />
              )}
              Disconnect
            </button>
          </div>
        ) : (
          <div>
            <p className="text-gray-400 text-sm mb-3">
              Connect your GitHub account to access private repositories and enable automatic authentication.
            </p>
            <button
              onClick={handleConnectGitHub}
              disabled={connectingGithub || githubStatus?.github_configured === false}
              className="flex items-center gap-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-500 text-white rounded-lg text-sm transition-colors"
            >
              {connectingGithub ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  Connecting...
                </>
              ) : (
                <>
                  <Github size={16} />
                  Connect GitHub Account
                </>
              )}
            </button>
            {githubStatus?.github_configured === false && (
              <p className="text-yellow-400 text-xs mt-2">
                GitHub OAuth is not configured on the server. Contact the administrator.
              </p>
            )}
          </div>
        )}
      </div>

      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-white flex items-center gap-2">
          <GitFork size={20} />
          Repositories
        </h2>
        <div className="flex items-center gap-2">
          <button
            onClick={loadRepos}
            disabled={loading}
            className="p-2 text-gray-400 hover:text-white rounded-lg hover:bg-gray-700"
            title="Refresh"
          >
            <RefreshCw size={16} className={loading ? 'animate-spin' : ''} />
          </button>
          <button
            onClick={() => setShowAddForm(!showAddForm)}
            className="p-2 text-gray-400 hover:text-white rounded-lg hover:bg-gray-700"
            title="Add Repository"
          >
            <Plus size={16} />
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-900/50 border border-red-700 rounded-lg text-red-200 text-sm">
          {error}
          <button 
            onClick={() => setError(null)} 
            className="ml-2 text-red-300 hover:text-red-100"
          >
            Dismiss
          </button>
        </div>
      )}

      {showAddForm && (
        <div className="mb-4 p-4 bg-gray-700 rounded-lg">
          <h3 className="text-sm font-medium text-white mb-3">Add GitHub Repository</h3>
          <div className="space-y-3">
            <div>
              <label className="block text-xs text-gray-400 mb-1">Repository URL</label>
              <input
                type="text"
                value={newRepoUrl}
                onChange={(e) => setNewRepoUrl(e.target.value)}
                placeholder="https://github.com/owner/repo"
                className="w-full px-3 py-2 bg-gray-600 text-white rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">GitHub Token (optional, for private repos)</label>
              <input
                type="password"
                value={newRepoToken}
                onChange={(e) => setNewRepoToken(e.target.value)}
                placeholder="ghp_xxxxxxxxxxxx"
                className="w-full px-3 py-2 bg-gray-600 text-white rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Default Branch</label>
              <input
                type="text"
                value={newRepoBranch}
                onChange={(e) => setNewRepoBranch(e.target.value)}
                placeholder="main"
                className="w-full px-3 py-2 bg-gray-600 text-white rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div className="flex gap-2">
              <button
                onClick={handleAddRepo}
                disabled={adding || !newRepoUrl.trim()}
                className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded-lg text-sm flex items-center justify-center gap-2"
              >
                {adding ? (
                  <>
                    <Loader2 size={14} className="animate-spin" />
                    Adding...
                  </>
                ) : (
                  <>
                    <Plus size={14} />
                    Add Repository
                  </>
                )}
              </button>
              <button
                onClick={() => setShowAddForm(false)}
                className="px-4 py-2 bg-gray-600 hover:bg-gray-500 text-white rounded-lg text-sm"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="space-y-2">
        {loading && repos.length === 0 ? (
          <p className="text-gray-400 text-sm text-center py-4">Loading repositories...</p>
        ) : repos.length === 0 ? (
          <p className="text-gray-400 text-sm text-center py-4">No repositories connected. Click + to add one.</p>
        ) : (
          repos.map((repo) => (
            <div
              key={repo.repo_id}
              className="bg-gray-700 rounded-lg overflow-hidden"
            >
              <div
                onClick={() => setExpandedRepo(expandedRepo === repo.repo_id ? null : repo.repo_id)}
                className="p-3 cursor-pointer hover:bg-gray-600 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <GitFork size={16} className="text-gray-400" />
                    <span className="text-white font-medium">{repo.owner}/{repo.repo_name}</span>
                    <span className="flex items-center gap-1 px-2 py-0.5 bg-gray-600 text-xs text-gray-300 rounded">
                      {getStatusIcon(repo.status)}
                      {getStatusText(repo.status)}
                    </span>
                  </div>
                  <div className="flex items-center gap-1">
                    {repo.status === 'cloned' && sessionId && (
                      <button
                        onClick={(e) => { e.stopPropagation(); handleLinkToSession(repo.repo_id); }}
                        className="p-1.5 text-gray-400 hover:text-blue-400 rounded"
                        title="Link to current session"
                      >
                        <Link2 size={14} />
                      </button>
                    )}
                    {repo.status === 'cloned' && (
                      <button
                        onClick={(e) => { e.stopPropagation(); handlePull(repo.repo_id); }}
                        disabled={cloning === repo.repo_id}
                        className="p-1.5 text-gray-400 hover:text-green-400 rounded disabled:opacity-50"
                        title="Pull latest changes"
                      >
                        {cloning === repo.repo_id ? (
                          <Loader2 size={14} className="animate-spin" />
                        ) : (
                          <RefreshCw size={14} />
                        )}
                      </button>
                    )}
                    {repo.status === 'pending' && (
                      <button
                        onClick={(e) => { e.stopPropagation(); handleClone(repo.repo_id); }}
                        disabled={cloning === repo.repo_id}
                        className="p-1.5 text-gray-400 hover:text-blue-400 rounded disabled:opacity-50"
                        title="Clone repository"
                      >
                        {cloning === repo.repo_id ? (
                          <Loader2 size={14} className="animate-spin" />
                        ) : (
                          <GitBranch size={14} />
                        )}
                      </button>
                    )}
                    <a
                      href={repo.repo_url.replace('.git', '')}
                      target="_blank"
                      rel="noopener noreferrer"
                      onClick={(e) => e.stopPropagation()}
                      className="p-1.5 text-gray-400 hover:text-blue-400 rounded"
                      title="Open on GitHub"
                    >
                      <ExternalLink size={14} />
                    </a>
                    <button
                      onClick={(e) => { e.stopPropagation(); handleDelete(repo.repo_id); }}
                      className="p-1.5 text-gray-400 hover:text-red-400 rounded"
                      title="Delete"
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                </div>
                <p className="text-gray-400 text-xs mt-1">
                  Branch: {repo.default_branch} | Added: {new Date(repo.created_at).toLocaleDateString()}
                </p>
              </div>

              {expandedRepo === repo.repo_id && (
                <div className="px-3 pb-3 border-t border-gray-600">
                  <div className="mt-3 space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Repository ID:</span>
                      <span className="text-gray-300 font-mono">{repo.repo_id}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Local Path:</span>
                      <span className="text-gray-300 font-mono text-xs">{repo.local_path || 'Not cloned'}</span>
                    </div>
                    {repo.last_synced && (
                      <div className="flex justify-between">
                        <span className="text-gray-400">Last Synced:</span>
                        <span className="text-gray-300">{new Date(repo.last_synced).toLocaleString()}</span>
                      </div>
                    )}
                    {repo.has_token && (
                      <div className="flex justify-between">
                        <span className="text-gray-400">Authentication:</span>
                        <span className="text-green-400">Token configured</span>
                      </div>
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
