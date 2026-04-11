import { useState, useEffect, useRef } from 'react';
import { Session, Task } from '../types';
import { useApi } from '../hooks/useApi';
import { Plus, Trash2, RefreshCw, Zap, Clock, CheckCircle2, AlertCircle, Search, X } from 'lucide-react';

interface SessionListProps {
  onSelectSession: (session: Session) => void;
  selectedSessionId?: string;
  onNewSession?: () => void;
  /** When this number changes, reload sessions from the API (e.g. after creating a session). */
  refreshTrigger?: number;
}

function formatTime(dateStr: string): string {
  const d = new Date(dateStr);
  const diff = Date.now() - d.getTime();
  if (diff < 60000) return 'just now';
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
  return `${Math.floor(diff / 86400000)}d ago`;
}

function StatusIcon({ status }: { status: string }) {
  if (status === 'running') return <Zap size={12} className="text-[#00ff99] animate-pulse" />;
  if (status === 'error') return <AlertCircle size={12} className="text-red-400" />;
  return <CheckCircle2 size={12} className="text-[#525252]" />;
}

export function SessionList({
  onSelectSession,
  selectedSessionId,
  onNewSession,
  refreshTrigger = 0,
}: SessionListProps) {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [sessionTitles, setSessionTitles] = useState<Record<string, string>>({});
  const [searchQuery, setSearchQuery] = useState('');
  const [showSearch, setShowSearch] = useState(false);
  const searchRef = useRef<HTMLInputElement>(null);
  const api = useApi();

  const loadSessions = async () => {
    try {
      const data = await api.listSessions();
      setSessions(data);
      // Use backend title if available, else fetch from tasks
      for (const s of data) {
        if (!sessionTitles[s.session_id]) {
          if (s.title) {
            setSessionTitles(prev => ({ ...prev, [s.session_id]: s.title! }));
          } else if (s.total_tasks > 0) {
            fetchSessionTitle(s.session_id);
          }
        }
      }
    } catch (e) {
      console.error('Failed to load sessions:', e);
    }
  };

  const fetchSessionTitle = async (sessionId: string) => {
    try {
      const tasks = await api.listTasks(sessionId);
      if (tasks && tasks.length > 0) {
        const firstTask = tasks[0] as Task;
        const desc = firstTask.description || '';
        // Truncate to a reasonable title length
        const title = desc.length > 60 ? desc.slice(0, 57) + '…' : desc;
        setSessionTitles(prev => ({ ...prev, [sessionId]: title }));
      }
    } catch {
      // Ignore errors — title just won't show
    }
  };

  useEffect(() => {
    loadSessions();
    const interval = setInterval(loadSessions, 5000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (refreshTrigger > 0) {
      loadSessions();
    }
  }, [refreshTrigger]);

  useEffect(() => {
    if (showSearch && searchRef.current) {
      searchRef.current.focus();
    }
  }, [showSearch]);

  const handleDeleteSession = async (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm('Delete this session?')) return;
    try {
      await api.deleteSession(sessionId);
      setSessions(sessions.filter(s => s.session_id !== sessionId));
      setSessionTitles(prev => {
        const next = { ...prev };
        delete next[sessionId];
        return next;
      });
    } catch (e) {
      console.error('Failed to delete session:', e);
    }
  };

  // Filter sessions by search query
  const filteredSessions = searchQuery.trim()
    ? sessions.filter(s => {
      const title = sessionTitles[s.session_id] || s.session_id;
      return title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        s.session_id.toLowerCase().includes(searchQuery.toLowerCase());
    })
    : sessions;

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-3 px-1">
        <span className="text-[10px] uppercase tracking-widest text-[#525252] font-bold">Sessions</span>
        <div className="flex gap-1">
          <button
            onClick={() => { setShowSearch(v => !v); if (showSearch) setSearchQuery(''); }}
            className={`p-1 rounded transition-colors ${showSearch ? 'text-[#00ff99]' : 'text-[#525252] hover:text-[#a3a3a3]'}`}
            title="Search sessions"
          >
            <Search size={12} />
          </button>
          <button
            onClick={loadSessions}
            className="p-1 text-[#525252] hover:text-[#a3a3a3] rounded transition-colors"
            title="Refresh"
          >
            <RefreshCw size={12} />
          </button>
          <button
            onClick={onNewSession}
            className="p-1 text-[#525252] hover:text-[#00ff99] rounded transition-colors"
            title="New Session"
          >
            <Plus size={12} />
          </button>
        </div>
      </div>

      {/* Search input */}
      {showSearch && (
        <div className="relative mb-2">
          <Search size={11} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-[#525252]" />
          <input
            ref={searchRef}
            type="text"
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            placeholder="Search sessions…"
            className="w-full pl-7 pr-7 py-1.5 bg-[#0a0a0a] text-white text-xs rounded-lg border border-[#262626] focus:border-[#00ff99]/30 focus:outline-none placeholder-[#525252]"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery('')}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-[#525252] hover:text-white"
            >
              <X size={10} />
            </button>
          )}
        </div>
      )}

      {/* Session list */}
      <div className="space-y-1">
        {filteredSessions.length === 0 ? (
          sessions.length === 0 ? (
            <div
              onClick={onNewSession}
              className="p-3 rounded-lg border border-dashed border-[#262626] text-[#525252] text-xs text-center cursor-pointer hover:border-[#00ff99]/30 hover:text-[#00ff99]/70 transition-colors"
            >
              + New session
            </div>
          ) : (
            <div className="p-3 text-[#525252] text-xs text-center">
              No sessions match "{searchQuery}"
            </div>
          )
        ) : (
          filteredSessions.map((session) => {
            const isSelected = selectedSessionId === session.session_id;
            const isRunning = session.status === 'running';
            const title = sessionTitles[session.session_id];
            return (
              <div
                key={session.session_id}
                onClick={() => onSelectSession(session)}
                className={`group relative p-3 rounded-lg cursor-pointer transition-all ${isSelected
                  ? 'bg-[#1a1a1a] border border-[#00ff99]/20 shadow-[inset_0_0_0_1px_rgba(0,255,153,0.1)]'
                  : 'border border-transparent hover:border-[#262626] hover:bg-[#1a1a1a]/50'
                  }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2 min-w-0">
                    <StatusIcon status={session.status} />
                    <div className="min-w-0">
                      {title ? (
                        <>
                          <div className={`text-xs font-semibold truncate leading-tight ${isSelected ? 'text-white' : 'text-[#d1d1d1]'}`}>
                            {title}
                          </div>
                          <div className="text-[9px] text-[#525252] font-mono mt-0.5">#{session.session_id}</div>
                        </>
                      ) : (
                        <span className={`text-xs font-semibold font-mono ${isSelected ? 'text-white' : 'text-[#d1d1d1]'}`}>
                          {session.session_id}
                        </span>
                      )}
                    </div>
                    {isRunning && (
                      <span className="flex-shrink-0 px-1.5 py-0.5 bg-[#00ff99]/10 text-[#00ff99] text-[9px] font-bold uppercase rounded-full border border-[#00ff99]/20">
                        live
                      </span>
                    )}
                  </div>
                  <button
                    onClick={(e) => handleDeleteSession(session.session_id, e)}
                    className="opacity-0 group-hover:opacity-100 p-0.5 text-[#525252] hover:text-red-400 transition-all flex-shrink-0"
                  >
                    <Trash2 size={11} />
                  </button>
                </div>

                <div className="mt-1.5 flex items-center gap-2 text-[10px] text-[#525252]">
                  <Clock size={9} />
                  <span>{formatTime(session.created_at)}</span>
                  <span>·</span>
                  <span>{session.total_tasks} task{session.total_tasks !== 1 ? 's' : ''}</span>
                  {session.iteration > 0 && (
                    <>
                      <span>·</span>
                      <span>iter {session.iteration}</span>
                    </>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
