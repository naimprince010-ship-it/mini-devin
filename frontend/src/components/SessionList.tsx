import { useState, useEffect } from 'react';
import { Session } from '../types';
import { useApi } from '../hooks/useApi';
import { Plus, Trash2, RefreshCw, Zap, Clock, CheckCircle2, AlertCircle } from 'lucide-react';

interface SessionListProps {
  onSelectSession: (session: Session) => void;
  selectedSessionId?: string;
  onNewSession?: () => void;
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
  if (status === 'running') return <Zap size={12} className="text-[#00ff99]" />;
  if (status === 'error') return <AlertCircle size={12} className="text-red-400" />;
  return <CheckCircle2 size={12} className="text-[#525252]" />;
}

export function SessionList({ onSelectSession, selectedSessionId, onNewSession }: SessionListProps) {
  const [sessions, setSessions] = useState<Session[]>([]);
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

  const handleDeleteSession = async (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm('Delete this session?')) return;
    try {
      await api.deleteSession(sessionId);
      setSessions(sessions.filter(s => s.session_id !== sessionId));
    } catch (e) {
      console.error('Failed to delete session:', e);
    }
  };

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-3 px-1">
        <span className="text-[10px] uppercase tracking-widest text-[#525252] font-bold">Sessions</span>
        <div className="flex gap-1">
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

      {/* Session list */}
      <div className="space-y-1">
        {sessions.length === 0 ? (
          <div
            onClick={onNewSession}
            className="p-3 rounded-lg border border-dashed border-[#262626] text-[#525252] text-xs text-center cursor-pointer hover:border-[#00ff99]/30 hover:text-[#00ff99]/70 transition-colors"
          >
            + New session
          </div>
        ) : (
          sessions.map((session) => {
            const isSelected = selectedSessionId === session.session_id;
            const isRunning = session.status === 'running';
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
                    <span className={`text-xs font-semibold truncate ${isSelected ? 'text-white' : 'text-[#d1d1d1]'}`}>
                      {session.session_id}
                    </span>
                    {isRunning && (
                      <span className="flex-shrink-0 px-1.5 py-0.5 bg-[#00ff99]/10 text-[#00ff99] text-[9px] font-bold uppercase rounded-full border border-[#00ff99]/20">
                        live
                      </span>
                    )}
                  </div>
                  <button
                    onClick={(e) => handleDeleteSession(session.session_id, e)}
                    className="opacity-0 group-hover:opacity-100 p-0.5 text-[#525252] hover:text-red-400 transition-all"
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
