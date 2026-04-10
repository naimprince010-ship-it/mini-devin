import React, { useState, useEffect, useCallback, useRef, Fragment } from 'react';
import { Session, Task, WebSocketMessage } from '../types';
import { useApi } from '../hooks/useApi';
import { useWebSocket } from '../hooks/useWebSocket';
import { Send, Bot, User, AlertCircle, Square, ChevronRight, Target, Coins, HelpCircle, X, DollarSign, Cpu, Download } from 'lucide-react';
import { StreamingOutput } from './StreamingOutput';
import { useSessionEvents } from '../contexts/SessionEventsContext';
import { PlanStepsView } from './PlanStepsView';
import { useToast } from './Toast';
import { ExportButtons } from './ExportButtons';

interface TaskPanelProps {
  session: Session;
  onTitleUpdated?: (title: string) => void;
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

// GPT-4o pricing (per 1M tokens)
const COST_PER_1M_INPUT = 2.50;
const COST_PER_1M_OUTPUT = 10.00;

function calcEstimatedCost(promptTokens: number, completionTokens: number): string {
  const cost = (promptTokens / 1_000_000) * COST_PER_1M_INPUT
    + (completionTokens / 1_000_000) * COST_PER_1M_OUTPUT;
  if (cost < 0.001) return '<$0.001';
  return `$${cost.toFixed(3)}`;
}

export function TaskPanel({ session, onTitleUpdated }: TaskPanelProps) {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [taskDescription, setTaskDescription] = useState('');
  const [streamingContent, setStreamingContent] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [showExport, setShowExport] = useState(false);
  const currentTaskIdRef = useRef<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const toolCallIdMapRef = useRef<Map<string, string>>(new Map()); // ws tool -> context id

  // Clarification state (local to TaskPanel for the modal)
  const [clarificationQuestion, setClarificationQuestion] = useState<string | null>(null);
  const [clarificationAnswer, setClarificationAnswer] = useState('');
  const [isSubmittingAnswer, setIsSubmittingAnswer] = useState(false);

  const api = useApi();
  const events = useSessionEvents();
  const toast = useToast();

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

      case 'token_usage':
        if (data.total_tokens !== undefined) {
          events.onTokenUsage({
            total_tokens: data.total_tokens as number,
            prompt_tokens: data.prompt_tokens as number,
            completion_tokens: data.completion_tokens as number
          });
        }
        break;

      case 'clarification_needed':
        setClarificationQuestion(data.question as string || 'The agent needs clarification.');
        events.onClarificationNeeded(data.question as string || '');
        break;

      case 'response':
        setIsStreaming(true);
        setStreamingContent(prev => prev + (message.content || data.content as string || ''));
        break;

      case 'task_started':
        setIsStreaming(true);
        setStreamingContent('');
        setClarificationQuestion(null);
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
        const wsKey = `${tool}-${Date.now()}`;
        toolCallIdMapRef.current.set(wsKey, ctxId);
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

      case 'plan_created':
      case 'plan': {
        const steps = (data.steps as string[]) || [];
        if (steps.length > 0) events.onPlanCreated(steps);
        break;
      }

      case 'step_started': {
        const idx = (data.index as number) ?? (data.step_index as number) ?? 0;
        events.onStepStarted(idx);
        break;
      }

      case 'step_completed': {
        const idx = (data.index as number) ?? (data.step_index as number) ?? 0;
        events.onStepCompleted(idx);
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
        toast.success('Task completed', 'Agent finished the task successfully.');
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
        toast.error('Task failed', String(data.error || 'Unknown error'));
        break;

      case 'session_title_updated':
        if (onTitleUpdated && data.title) {
          onTitleUpdated(data.title as string);
        }
        break;

      case 'browser_event':
        events.onBrowserEventFromWS(
          (data.event_type as string) || 'other',
          data.url as string | undefined,
          data.query as string | undefined,
          data.screenshot_base64 as string | undefined,
        );
        break;

      case 'file_changed':
        events.onFileChangedFromWS(
          (data.path as string) || '',
          (data.content as string) || '',
        );
        break;

      case 'tool_output': {
        // Shell live streaming — add line to shell output in WorkspacePanel
        const line = data.line as string | undefined;
        if (line) {
          events.onShellLine?.(line);
        }
        break;
      }

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
    // Restore conversation history from backend on session load
    api.getSessionHistory(session.session_id).then(hist => {
      if (hist.total > 0) {
        // Reconstruct a readable chat history from the conversation
        const lines: string[] = [];
        for (const msg of hist.messages) {
          if (msg.role === 'user' && msg.content) {
            lines.push(`**[You]:** ${msg.content}`);
          } else if (msg.role === 'assistant' && msg.content) {
            lines.push(msg.content);
          }
        }
        if (lines.length > 0) {
          setStreamingContent(lines.join('\n\n---\n\n') + '\n\n_— Session restored from history —_');
        }
      }
    }).catch(() => {/* ignore */});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session.session_id]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [streamingContent, tasks]);

  const handleSubmitTask = async () => {
    if (!taskDescription.trim()) return;

    // If agent is running, send as follow-up via WebSocket
    if (isStreaming) {
      sendMessage(taskDescription);
      setStreamingContent(prev => prev + `\n\n**[You]:** ${taskDescription}\n`);
      setTaskDescription('');
      return;
    }

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

  const handleStop = async () => {
    // REST API call for proper cancellation
    await api.stopSession(session.session_id);
    setIsStreaming(false);
    events.onTaskEnded();
    toast.error('Agent stopped', 'The agent was interrupted.');
  };

  const handleAnswerClarification = async () => {
    if (!clarificationAnswer.trim() || isSubmittingAnswer) return;
    setIsSubmittingAnswer(true);
    try {
      const response = await fetch(`/api/sessions/${session.session_id}/answer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ answer: clarificationAnswer })
      });
      if (response.ok) {
        toast.success('Answer sent', 'The agent will now resume.');
        setClarificationQuestion(null);
        events.dismissClarification();
        setClarificationAnswer('');
      } else {
        toast.error('Failed to send answer', 'The API returned an error.');
      }
    } catch (e) {
      toast.error('Error', 'Failed to connect to API.');
    } finally {
      setIsSubmittingAnswer(false);
    }
  };

  const currentPhase = events.phase;
  const phaseLabel = currentPhase ? (PHASE_LABELS[currentPhase] || currentPhase) : null;

  return (
    <div className="flex flex-col h-full bg-[#0f0f0f]">
      {/* Header */}
      <div className="px-6 py-4 border-b border-[#262626] bg-[#0f0f0f]/80 backdrop-blur-md sticky top-0 z-10">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-sm font-semibold text-white truncate max-w-[200px]">
              {session.title || 'Current Session'}
            </h2>
            <div className="flex items-center gap-2 mt-1">
              <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-[#00ff99] shadow-[0_0_6px_#00ff99]' : 'bg-red-500'}`} />
              <span className="text-[10px] uppercase tracking-wider text-[#a3a3a3] font-bold">
                {isConnected ? (isStreaming ? 'Agent Running' : 'Active Agent') : 'Agent Offline'}
              </span>
              {/* Model badge */}
              {session.model && (
                <>
                  <span className="text-[#2a2a2a]">·</span>
                  <div className="flex items-center gap-1 text-[10px] text-[#525252]">
                    <Cpu size={9} />
                    <span>{session.model}</span>
                  </div>
                </>
              )}
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* Export button */}
            <div className="relative">
              <button
                onClick={() => setShowExport(p => !p)}
                className="p-1.5 rounded-lg text-[#525252] hover:text-[#a3a3a3] hover:bg-[#1a1a1a] transition-colors"
                title="Export conversation"
              >
                <Download size={14} />
              </button>
              {showExport && (
                <div className="absolute right-0 top-full mt-1 bg-[#111111] border border-[#262626] rounded-xl shadow-2xl p-3 z-50 min-w-[160px]">
                  <p className="text-[10px] uppercase tracking-wider text-[#525252] mb-2 font-bold">Export As</p>
                  <ExportButtons sessionId={session.session_id} />
                </div>
              )}
            </div>

            {/* Phase + Iteration chips (only show when streaming) */}
            {isStreaming && (
              <div className="flex items-center gap-2">
                {events.iteration > 0 && (
                  <div className="px-2 py-0.5 rounded-full bg-[#1a1a1a] border border-[#262626] text-[10px] text-[#a3a3a3]">
                    iter {events.iteration}/{events.maxIterations}
                  </div>
                )}
                {phaseLabel && (
                  <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-[#00ff99]/10 border border-[#00ff99]/25">
                    <span className={`w-1.5 h-1.5 rounded-full ${clarificationQuestion ? 'bg-yellow-400' : 'bg-[#00ff99] animate-pulse'}`} />
                    <span className={`text-[10px] font-bold uppercase tracking-widest ${clarificationQuestion ? 'text-yellow-400' : 'text-[#00ff99]'}`}>
                      {clarificationQuestion ? 'Waiting for User' : phaseLabel}
                    </span>
                  </div>
                )}
              </div>
            )}
          </div>
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

      {/* Token Usage Status Bar — shown whenever tokens have been used */}
      {events.tokenUsage.total_tokens > 0 && (
        <div className="px-6 py-2 border-b border-[#1a1a1a] bg-[#0a0a0a] flex items-center gap-4">
          {/* Token count */}
          <div
            className="group relative flex items-center gap-1.5 cursor-default"
            title="Prompt / Completion tokens"
          >
            <Coins size={11} className="text-yellow-500/70" />
            <span className="text-[11px] font-mono text-[#737373]">
              {events.tokenUsage.total_tokens.toLocaleString()}
              <span className="text-[#525252]"> tokens</span>
            </span>
            {/* Hover breakdown */}
            <div className="absolute bottom-full left-0 mb-1.5 hidden group-hover:flex flex-col gap-0.5 bg-[#1a1a1a] border border-[#262626] rounded-lg px-3 py-2 text-[10px] font-mono text-[#a3a3a3] shadow-xl whitespace-nowrap z-50">
              <span>↑ prompt: <span className="text-white">{events.tokenUsage.prompt_tokens.toLocaleString()}</span></span>
              <span>↓ completion: <span className="text-white">{events.tokenUsage.completion_tokens.toLocaleString()}</span></span>
              <span>∑ total: <span className="text-white">{events.tokenUsage.total_tokens.toLocaleString()}</span></span>
            </div>
          </div>

          <span className="text-[#2a2a2a]">·</span>

          {/* Estimated cost */}
          <div className="flex items-center gap-1.5">
            <DollarSign size={11} className="text-emerald-500/70" />
            <span className="text-[11px] font-mono text-[#737373]">
              {calcEstimatedCost(
                events.tokenUsage.prompt_tokens,
                events.tokenUsage.completion_tokens
              )}
              <span className="text-[#525252]"> est. cost</span>
            </span>
          </div>

          {/* Live pulse when running */}
          {isStreaming && (
            <>
              <span className="text-[#2a2a2a]">·</span>
              <div className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-[#00ff99] animate-pulse" />
                <span className="text-[10px] uppercase tracking-wider text-[#00ff99]/70 font-bold">Live</span>
              </div>
            </>
          )}
        </div>
      )}

      {/* Acceptance Criteria badge */}
      {events.acceptanceCriteria && (
        <div className="mx-6 mt-3 mb-1 flex items-start gap-2 px-3 py-2 bg-[#0d1a0d] border border-[#00ff99]/15 rounded-lg">
          <Target size={12} className="text-[#00ff99] flex-shrink-0 mt-0.5" />
          <span className="text-[11px] text-[#00ff99]/70 leading-relaxed">
            <span className="font-bold text-[#00ff99]/90">Success: </span>
            {events.acceptanceCriteria}
          </span>
        </div>
      )}

      {/* Plan Steps */}
      {events.planSteps.length > 0 && (
        <PlanStepsView
          steps={events.planSteps}
          currentIndex={events.currentStepIndex}
          isRunning={events.isRunning}
        />
      )}

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
            placeholder={isStreaming ? 'Send a follow-up message to the agent...' : 'Type a task for the agent...'}
            className={`w-full bg-[#121212] border rounded-[20px] px-5 py-4 pr-24 text-sm focus:outline-none transition-colors resize-none placeholder-[#737373] min-h-[56px] max-h-40 custom-scrollbar ${isStreaming
              ? 'border-[#00ff99]/20 focus:border-[#00ff99]/40'
              : 'border-[#262626] focus:border-[#00ff99]/50'
              }`}
            rows={1}
          />
          <div className="absolute right-3 bottom-3 flex items-center gap-1.5">
            {/* Send / follow-up button */}
            <button
              onClick={handleSubmitTask}
              disabled={!taskDescription.trim()}
              className={`p-2 rounded-full transition-all ${taskDescription.trim()
                ? isStreaming
                  ? 'bg-[#00ff99]/20 text-[#00ff99] border border-[#00ff99]/30 hover:bg-[#00ff99]/30'
                  : 'bg-[#00ff99] text-[#0f0f0f] hover:scale-110'
                : 'bg-[#262626] text-[#737373] cursor-not-allowed'
                }`}
              title={isStreaming ? 'Send follow-up' : 'Send task'}
            >
              <Send size={16} />
            </button>
            {/* Stop button — only when streaming */}
            {isStreaming && (
              <button
                onClick={handleStop}
                className="p-2 rounded-full bg-red-500/20 border border-red-500/30 text-red-400 hover:bg-red-500/30 transition-all"
                title="Stop agent"
              >
                <Square size={16} />
              </button>
            )}
          </div>
        </div>
        <p className="text-center mt-3 text-[10px] text-[#737373] font-medium uppercase tracking-widest">
          {isStreaming ? 'Agent is running — you can send follow-up messages' : 'Shift + Enter for new line'}
        </p>
      </div>

      {/* Clarification Modal Overlay */}
      {clarificationQuestion && (
        <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-6 animate-in fade-in duration-200">
          <div className="bg-[#121212] border border-yellow-500/30 shadow-2xl rounded-2xl w-full max-w-lg overflow-hidden flex flex-col">
            <div className="px-5 py-4 border-b border-[#262626] flex justify-between items-center bg-yellow-500/5">
              <div className="flex items-center gap-2 text-yellow-500">
                <HelpCircle size={18} />
                <h3 className="font-semibold text-sm tracking-wide">Agent Needs Clarification</h3>
              </div>
              <button
                onClick={() => { setClarificationQuestion(null); events.dismissClarification(); }}
                className="text-[#737373] hover:text-white transition-colors"
                title="Dismiss (Agent will guess)"
              >
                <X size={16} />
              </button>
            </div>

            <div className="p-6 flex-1 overflow-y-auto">
              <p className="text-[#d1d1d1] text-sm leading-relaxed whitespace-pre-wrap font-medium">
                {clarificationQuestion}
              </p>

              <div className="mt-6 space-y-2">
                <label className="text-[10px] uppercase tracking-wider text-[#737373] font-bold">Your Answer</label>
                <textarea
                  autoFocus
                  value={clarificationAnswer}
                  onChange={e => setClarificationAnswer(e.target.value)}
                  placeholder="Type your answer here..."
                  className="w-full bg-[#0f0f0f] border border-[#262626] focus:border-yellow-500/50 rounded-xl px-4 py-3 text-sm text-white resize-none h-24 custom-scrollbar outline-none transition-colors"
                  onKeyDown={e => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleAnswerClarification();
                    }
                  }}
                />
              </div>
            </div>

            <div className="p-4 border-t border-[#262626] bg-[#0f0f0f]/50 flex justify-end gap-3">
              <button
                onClick={() => { setClarificationQuestion(null); events.dismissClarification(); }}
                className="px-4 py-2 text-xs font-semibold text-[#a3a3a3] hover:text-white transition-colors"
              >
                Ignore
              </button>
              <button
                onClick={handleAnswerClarification}
                disabled={!clarificationAnswer.trim() || isSubmittingAnswer}
                className="px-5 py-2 text-xs font-bold rounded-lg bg-yellow-500 text-black hover:bg-yellow-400 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {isSubmittingAnswer ? 'Sending...' : 'Send Answer'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
