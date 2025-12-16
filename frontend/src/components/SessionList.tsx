import { useState, useEffect } from 'react';
import { Session } from '../types';
import { useApi } from '../hooks/useApi';
import { Plus, Trash2, RefreshCw } from 'lucide-react';

interface SessionListProps {
  onSelectSession: (session: Session) => void;
  selectedSessionId?: string;
}

export function SessionList({ onSelectSession, selectedSessionId }: SessionListProps) {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [workingDir, setWorkingDir] = useState('.');
  const [model, setModel] = useState('gpt-4o');
  const [maxIterations, setMaxIterations] = useState(50);
  
  const api = useApi();

  const loadSessions = async () => {
    try {
      const data = await api.listSessions();
      setSessions(data);
    } catch (e) {
      console.error('Failed to load sessions:', e);
    }
  };

  useEffect(() => {
    loadSessions();
    const interval = setInterval(loadSessions, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleCreateSession = async () => {
    try {
      const session = await api.createSession({
        working_directory: workingDir,
        model,
        max_iterations: maxIterations,
      });
      setSessions([...sessions, session]);
      setShowCreateForm(false);
      onSelectSession(session);
    } catch (e) {
      console.error('Failed to create session:', e);
    }
  };

  const handleDeleteSession = async (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm('Are you sure you want to delete this session?')) return;
    
    try {
      await api.deleteSession(sessionId);
      setSessions(sessions.filter(s => s.session_id !== sessionId));
    } catch (e) {
      console.error('Failed to delete session:', e);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'bg-green-500';
      case 'idle': return 'bg-gray-400';
      case 'error': return 'bg-red-500';
      default: return 'bg-gray-400';
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-white">Sessions</h2>
        <div className="flex gap-2">
          <button
            onClick={loadSessions}
            className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
            title="Refresh"
          >
            <RefreshCw size={16} />
          </button>
          <button
            onClick={() => setShowCreateForm(!showCreateForm)}
            className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
            title="New Session"
          >
            <Plus size={16} />
          </button>
        </div>
      </div>

      {showCreateForm && (
        <div className="mb-4 p-3 bg-gray-700 rounded-lg">
          <div className="space-y-3">
            <div>
              <label className="block text-sm text-gray-300 mb-1">Working Directory</label>
              <input
                type="text"
                value={workingDir}
                onChange={(e) => setWorkingDir(e.target.value)}
                className="w-full px-3 py-2 bg-gray-600 text-white rounded border border-gray-500 focus:border-blue-500 focus:outline-none"
                placeholder="/path/to/repo"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-300 mb-1">Model</label>
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="w-full px-3 py-2 bg-gray-600 text-white rounded border border-gray-500 focus:border-blue-500 focus:outline-none"
              >
                <option value="gpt-4o">GPT-4o</option>
                <option value="gpt-4o-mini">GPT-4o Mini</option>
                <option value="claude-3-5-sonnet-20241022">Claude 3.5 Sonnet</option>
              </select>
            </div>
            <div>
              <label className="block text-sm text-gray-300 mb-1">Max Iterations</label>
              <input
                type="number"
                value={maxIterations}
                onChange={(e) => setMaxIterations(parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-gray-600 text-white rounded border border-gray-500 focus:border-blue-500 focus:outline-none"
                min={1}
                max={100}
              />
            </div>
            <button
              onClick={handleCreateSession}
              disabled={api.loading}
              className="w-full py-2 bg-blue-600 hover:bg-blue-700 text-white rounded font-medium disabled:opacity-50"
            >
              {api.loading ? 'Creating...' : 'Create Session'}
            </button>
          </div>
        </div>
      )}

      <div className="space-y-2">
        {sessions.length === 0 ? (
          <p className="text-gray-400 text-sm text-center py-4">No sessions yet</p>
        ) : (
          sessions.map((session) => (
            <div
              key={session.session_id}
              onClick={() => onSelectSession(session)}
              className={`p-3 rounded-lg cursor-pointer transition-colors ${
                selectedSessionId === session.session_id
                  ? 'bg-blue-600'
                  : 'bg-gray-700 hover:bg-gray-600'
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${getStatusColor(session.status)}`} />
                  <span className="text-white font-medium">{session.session_id}</span>
                </div>
                <button
                  onClick={(e) => handleDeleteSession(session.session_id, e)}
                  className="p-1 text-gray-400 hover:text-red-400"
                >
                  <Trash2 size={14} />
                </button>
              </div>
              <div className="mt-1 text-sm text-gray-300 truncate">
                {session.working_directory}
              </div>
              <div className="mt-1 text-xs text-gray-400">
                Tasks: {session.total_tasks} | Iteration: {session.iteration}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
