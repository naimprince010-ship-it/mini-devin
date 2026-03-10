import React, { useState, useEffect, useCallback, useRef, Fragment } from 'react';
import { Session, Task, WebSocketMessage } from '../types';
import { useApi } from '../hooks/useApi';
import { useWebSocket } from '../hooks/useWebSocket';
import { Send, Bot, User, AlertCircle, Square, ChevronRight } from 'lucide-react';
import { StreamingOutput } from './StreamingOutput';
import { useSessionEvents } from '../contexts/SessionEventsContext';

interface TaskPanelProps {
  session: Session;
}

const PHASE_LABELS: Record<string, string> = {
  intake: 'Understanding',
  explore: 'Exploring',
  plan: 'Planning',
  execute: 'Executing',
  verify: 'Verifying',
  repair: 'Repairing',
  finalize: 'Finalizing',
  complete: 'Complete',
};

const PHASE_ORDER = ['intake', 'explore', 'plan', 'execute', 'verify', 'finalize'];

function formatRelativeTime(date: string | Date): string {
  const d = typeof date === 'string' ? new Date(date) : date;
  const diff = Date.now() - d.getTime();
  if (diff < 60000) return 'just now';
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
  return `${Math.floor(diff / 86400000)}d ago`;
}

export function TaskPanel({ session }: TaskPanelProps) {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [taskDescription, setTaskDescription] = useState('');
  const [streamingContent, setStreamingContent] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const currentTaskIdRef = useRef<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const toolCallIdMapRef = useRef<Map<string, string>>(new Map()); // ws tool -> context id

  const api = useApi();
  const events = useSessionEvents();

  const loadTasks = async () => {
    try {
      const data = await api.listTasks(session.session_id);
      setTasks(data);
    } catch (e) {
      console.error('Failed to load tasks:', e);
    }
  };

  const handleWebSocketMessage = useCallback((message: WebSocketMessage) => {
    const data = message.data || {};

    switch (message.type) {
      case 'token':
      case 'tokens_batch':
        setIsStreaming(true);
        setStreamingContent(prev => prev + (data.content as string || ''));
        break;

      case 'response':
        setIsStreaming(true);
        setStreamingContent(prev => prev + (message.content || data.content as string || ''));
        break;

      case 'task_started':
        setIsStreaming(true);
        setStreamingContent('');
        events.onTaskStarted();
        break;

      case 'phase_changed': {
        const phase = (data.phase as string || '').toLowerCase();
        events.onPhaseChange(phase as import('../contexts/SessionEventsContext').AgentPhase);
        break;
      }

      case 'iteration_started':
      case 'iteration': {
        const iter = (message.iteration ?? data.iteration ?? 0) as number;
        const max = (message.max ?? data.max ?? 50) as number;
        events.onIterationUpdate(iter, max);
        break;
      }

      case 'tool_started': {
        const tool = (data.tool as string) || 'unknown';
        const input = (data.input as Record<string, unknown>) || {};
        const ctxId = events.onToolStarted(tool, input);
        // store mapping: we'll need the ws-side id when completed
        const wsKey = `${tool}-${Date.now()}`;
        toolCallIdMapRef.current.set(wsKey, ctxId);
        // store latest for completion
        toolCallIdMapRef.current.set('__latest__', ctxId);
        break;
      }

      case 'tool_completed':
      case 'tool_result': {
        const latestId = toolCallIdMapRef.current.get('__latest__');
        if (latestId) {
          const output = (data.output as Record<string, unknown>) || {};
          const duration = (data.duration_ms as number) || 0;
          events.onToolCompleted(latestId, output, duration);
        }
        break;
      }

      case 'tool_failed': {
        const latestId = toolCallIdMapRef.current.get('__latest__');
        if (latestId) {
          events.onToolFailed(latestId, (data.error as string) || 'Unknown error');
        }
        break;
      }

      case 'task_completed':
        setIsStreaming(false);
        setStreamingContent(prev => {
          const finalContent = prev + '\n\n✅ Task completed successfully.';
          if (currentTaskIdRef.current) {
            setTasks(prevTasks => prevTasks.map(t =>
              t.task_id === currentTaskIdRef.current
                ? { ...t, status: 'completed', summary: finalContent }
                : t
            ));
          }
          return finalContent;
        });
        events.onTaskEnded();
        break;

      case 'task_failed':
        setIsStreaming(false);
        setStreamingContent(prev => {
          const finalContent = prev + `\n\n❌ Task failed: ${data.error || 'Unknown error'}`;
          if (currentTaskIdRef.current) {
            setTasks(prevTasks => prevTasks.map(t =>
              t.task_id === currentTaskIdRef.current
                ? { ...t, status: 'failed', summary: finalContent }
                : t
            ));
          }
          return finalContent;
        });
        events.onTaskEnded();
        break;

      default:
        break;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const { isConnected, sendMessage } = useWebSocket({
    sessionId: session.session_id,
    onMessage: handleWebSocketMessage,
  });

  useEffect(() => {
    loadTasks();
    setStreamingContent('');
    events.resetForSession();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session.session_id]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [streamingContent, tasks]);

  const handleSubmitTask = async () => {
    if (!taskDescription.trim() || isStreaming) return;

    const newTaskId = `task-${Date.now()}`;
    const mockTask: Task = {
      task_id: newTaskId,
      session_id: session.session_id,
      description: taskDescription,
      status: 'running',
      created_at: new Date().toISOString(),
      started_at: new Date().toISOString(),
      completed_at: null,
      iteration: 0,
      error_message: null,
    };

    currentTaskIdRef.current = newTaskId;
    setTasks(prev => [...prev, mockTask]);
    setTaskDescription('');
    setStreamingContent('');
    setIsStreaming(true);
    sendMessage(taskDescription);
  };

  const handleStop = () => {
    // Send stop signal via WebSocket (best-effort)
    sendMessage('__STOP__');
    setIsStreaming(false);
    events.onTaskEnded();
  };

  const currentPhase = events.phase;
  const phaseLabel = currentPhase ? (PHASE_LABELS[currentPhase] || currentPhase) : null;

  return (
    <div className="flex flex-col h-full bg-[#0f0f0f]">
      {/* Header */}
      <div className="px-6 py-4 border-b border-[#262626] bg-[#0f0f0f]/80 backdrop-blur-md sticky top-0 z-10">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-sm font-semibold text-white">Current Session</h2>
            <div className="flex items-center gap-2 mt-1">
              <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-[#00ff99] shadow-[0_0_6px_#00ff99]' : 'bg-red-500'}`} />
              <span className="text-[10px] uppercase tracking-wider text-[#a3a3a3] font-bold">
                {isConnected ? (isStreaming ? 'Agent Running' : 'Active Agent') : 'Agent Offline'}
              </span>
            </div>
          </div>

          {/* Phase + Iteration chips */}
          {isStreaming && (
            <div className="flex items-center gap-2">
              {events.iteration > 0 && (
                <div className="px-2 py-0.5 rounded-full bg-[#1a1a1a] border border-[#262626] text-[10px] text-[#a3a3a3]">
                  iter {events.iteration}/{events.maxIterations}
                </div>
              )}
              {phaseLabel && (
                <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-[#00ff99]/10 border border-[#00ff99]/25">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#00ff99] animate-pulse" />
                  <span className="text-[10px] font-bold uppercase tracking-widest text-[#00ff99]">
                    {phaseLabel}
                  </span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Phase progress bar */}
        {isStreaming && currentPhase && PHASE_ORDER.includes(currentPhase) && (
          <div className="mt-3 flex items-center gap-1">
            {PHASE_ORDER.map((p, i) => {
              const currentIdx = PHASE_ORDER.indexOf(currentPhase);
              const isPast = i < currentIdx;
              const isCurrent = i === currentIdx;
              return (
                <Fragment key={p}>
                  <div className={`flex items-center gap-1 text-[9px] uppercase tracking-wider font-bold transition-colors ${isCurrent ? 'text-[#00ff99]' : isPast ? 'text-[#404040]' : 'text-[#2a2a2a]'
                    }`}>
                    <div className={`w-1.5 h-1.5 rounded-full ${isCurrent ? 'bg-[#00ff99] animate-pulse' : isPast ? 'bg-[#404040]' : 'bg-[#2a2a2a]'
                      }`} />
                    <span className="hidden sm:inline">{PHASE_LABELS[p]}</span>
                  </div>
                  {i < PHASE_ORDER.length - 1 && (
                    <ChevronRight size={9} className={isPast || isCurrent ? 'text-[#404040]' : 'text-[#2a2a2a]'} />
                  )}
                </Fragment>
              );
            })}
          </div>
        )}
      </div>

      {/* Messages Area */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-8 space-y-10 custom-scrollbar">
        {tasks.map((task) => (
          <div key={task.task_id} className="space-y-8">
            {/* User message */}
            <div className="flex gap-4 max-w-2xl ml-auto flex-row-reverse">
              <div className="w-8 h-8 rounded-full bg-[#1a1a1a] flex items-center justify-center flex-shrink-0 border border-[#262626]">
                <User size={16} className="text-[#a3a3a3]" />
              </div>
              <div className="space-y-1">
                <div className="bg-[#1a1a1a] px-4 py-3 rounded-2xl rounded-tr-none border border-[#262626] text-sm leading-relaxed text-white">
                  {task.description}
                </div>
                <div className="text-right text-[10px] text-[#525252]">
                  {formatRelativeTime(task.created_at)}
                </div>
              </div>
            </div>

            {/* AI response */}
            {task.status !== 'pending' && (
              <div className="flex gap-4 max-w-3xl mr-auto">
                <div className="w-8 h-8 rounded-full bg-[#00ff99]/10 flex items-center justify-center flex-shrink-0 border border-[#00ff99]/20">
                  <Bot size={16} className="text-[#00ff99]" />
                </div>
                <div className="space-y-3 flex-1">
                  <div className="text-sm leading-relaxed text-[#d1d1d1] whitespace-pre-wrap">
                    {task.task_id === currentTaskIdRef.current && isStreaming ? (
                      <div className="space-y-3">
                        {/* Inline tool call card (last running tool) */}
                        {events.toolCalls.length > 0 && (() => {
                          const last = events.toolCalls[events.toolCalls.length - 1];
                          if (last.status === 'running') return (
                            <div className="flex items-center gap-2 px-3 py-2 bg-[#0d0d0d] border border-[#262626] rounded-lg text-xs text-[#a3a3a3] font-mono animate-in fade-in duration-200">
                              <span className="w-1.5 h-1.5 rounded-full bg-[#00ff99] animate-pulse flex-shrink-0" />
                              <span className="text-[#00ff99] font-semibold">{last.tool}</span>
                              <span className="truncate opacity-60">
                                {Object.values(last.input)[0] as string || ''}
                              </span>
                            </div>
                          );
                          return null;
                        })()}
                        <StreamingOutput content={streamingContent} isStreaming={isStreaming} />
                      </div>
                    ) : (
                      <div>
                        {task.summary ? (
                          <StreamingOutput content={task.summary} isStreaming={false} />
                        ) : (
                          <div className="p-4 bg-[#1a1a1a]/30 rounded-xl border border-[#262626] italic text-[#a3a3a3]">
                            Agent response is loading...
                          </div>
                        )}
                      </div>
                    )}
                  </div>

                  {task.error_message && (
                    <div className="flex items-center gap-3 p-3 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-sm">
                      <AlertCircle size={16} />
                      <p>{task.error_message}</p>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        ))}

        {tasks.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center text-center space-y-4 opacity-40">
            <Bot size={64} className="text-[#262626]" />
            <div className="space-y-1">
              <p className="text-sm font-medium">No messages yet</p>
              <p className="text-xs text-[#a3a3a3]">Send a message to start working</p>
            </div>
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="p-6 bg-[#0f0f0f] border-t border-[#262626]">
        <div className="max-w-3xl mx-auto relative">
          <textarea
            value={taskDescription}
            onChange={(e) => setTaskDescription(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSubmitTask();
              }
            }}
            placeholder="Type a task for the agent..."
            className="w-full bg-[#121212] border border-[#262626] rounded-[20px] px-5 py-4 pr-14 text-sm focus:outline-none focus:border-[#00ff99]/50 transition-colors resize-none placeholder-[#737373] min-h-[56px] max-h-40 custom-scrollbar"
            rows={1}
            disabled={isStreaming}
          />
          {isStreaming ? (
            <button
              onClick={handleStop}
              className="absolute right-3 bottom-3 p-2 rounded-full bg-red-500/20 border border-red-500/30 text-red-400 hover:bg-red-500/30 transition-all"
              title="Stop agent"
            >
              <Square size={18} />
            </button>
          ) : (
            <button
              onClick={handleSubmitTask}
              disabled={!taskDescription.trim()}
              className={`absolute right-3 bottom-3 p-2 rounded-full transition-all ${taskDescription.trim()
                ? 'bg-[#00ff99] text-[#0f0f0f] hover:scale-110'
                : 'bg-[#262626] text-[#737373] cursor-not-allowed'
                }`}
            >
              <Send size={18} />
            </button>
          )}
        </div>
        <p className="text-center mt-3 text-[10px] text-[#737373] font-medium uppercase tracking-widest">
          Shift + Enter for new line
        </p>
      </div>
    </div>
  );
}
