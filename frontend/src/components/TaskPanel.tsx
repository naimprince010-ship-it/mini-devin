import { useState, useEffect, useCallback, useRef } from 'react';
import { Session, Task, WebSocketMessage } from '../types';
import { useApi } from '../hooks/useApi';
import { useWebSocket } from '../hooks/useWebSocket';
import { Send, Bot, User, Loader2, AlertCircle } from 'lucide-react';
import { StreamingOutput } from './StreamingOutput';

interface TaskPanelProps {
  session: Session;
}

export function TaskPanel({ session }: TaskPanelProps) {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [taskDescription, setTaskDescription] = useState('');
  const [streamingContent, setStreamingContent] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [taskSummaries, setTaskSummaries] = useState<Record<string, string>>({}); // kept for now, unused
  const currentTaskIdRef = useRef<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const api = useApi();

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

      case 'thinking':
        setIsStreaming(true);
        // We can choose to show thinking or not. Devin shows it subtly.
        break;

      case 'response':
        setIsStreaming(true);
        setStreamingContent(prev => prev + (message.content || data.content || ''));
        break;

      case 'task_started':
        setIsStreaming(true);
        setStreamingContent('');
        break;

      case 'task_completed':
        setIsStreaming(false);
        setStreamingContent(prev => {
          const finalContent = prev + '\n\n✅ Task completed successfully.';
          if (currentTaskIdRef.current) {
            // Directly update the task in local state with the streamed content as summary
            // This avoids loadTasks() replacing our mock task with a DB task (different ID, no summary)
            setTasks(prevTasks => prevTasks.map(t =>
              t.task_id === currentTaskIdRef.current
                ? { ...t, status: 'completed', summary: finalContent }
                : t
            ));
          }
          return finalContent;
        });
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
        break;

      default:
        break;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const getAgentStatusText = () => {
    if (!isStreaming) return null;
    if (streamingContent.includes('--- Thinking ---') || streamingContent.toLowerCase().includes('thinking')) return 'Thinking';
    if (streamingContent.toLowerCase().includes('tool:')) return 'Executing Tool';
    return 'Working';
  };

  const agentStatus = getAgentStatusText();

  const { isConnected, sendMessage } = useWebSocket({
    sessionId: session.session_id,
    onMessage: handleWebSocketMessage,
  });

  useEffect(() => {
    loadTasks();
    setStreamingContent('');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session.session_id]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [streamingContent, tasks]);

  const handleSubmitTask = async () => {
    if (!taskDescription.trim() || isStreaming) return;

    try {
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
        error_message: null
      };

      currentTaskIdRef.current = newTaskId;
      setTasks(prev => [...prev, mockTask]);
      setTaskDescription('');
      setStreamingContent('');
      setIsStreaming(true);
      sendMessage(taskDescription);
    } catch (e) {
      console.error('Failed to create task:', e);
    }
  };

  return (
    <div className="flex flex-col h-full bg-[#0f0f0f]">
      {/* Chat Header */}
      <div className="px-6 py-4 border-b border-[#262626] bg-[#0f0f0f]/80 backdrop-blur-md flex items-center justify-between sticky top-0 z-10">
        <div>
          <h2 className="text-sm font-semibold text-white">Current Session</h2>
          <div className="flex items-center gap-2 mt-1">
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-[#00ff99]' : 'bg-red-500'}`} />
            <span className="text-[10px] uppercase tracking-wider text-[#a3a3a3] font-bold">
              {isConnected ? 'Active Agent' : 'Agent Offline'}
            </span>
          </div>
        </div>
      </div>

      {/* Messages Area */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-8 space-y-10 custom-scrollbar">
        {tasks.map((task) => (
          <div key={task.task_id} className="space-y-8">
            {/* User Message */}
            <div className="flex gap-4 max-w-2xl ml-auto flex-row-reverse">
              <div className="w-8 h-8 rounded-full bg-[#1a1a1a] flex items-center justify-center flex-shrink-0 border border-[#262626]">
                <User size={16} className="text-[#a3a3a3]" />
              </div>
              <div className="bg-[#1a1a1a] px-4 py-3 rounded-2xl rounded-tr-none border border-[#262626] text-sm leading-relaxed text-white">
                {task.description}
              </div>
            </div>

            {/* AI Message (Original Task) */}
            {task.status !== 'pending' && (
              <div className="flex gap-4 max-w-3xl mr-auto">
                <div className="w-8 h-8 rounded-full bg-[#00ff99]/10 flex items-center justify-center flex-shrink-0 border border-[#00ff99]/20">
                  <Bot size={16} className="text-[#00ff99]" />
                </div>
                <div className="space-y-4 flex-1">
                  <div className="text-sm leading-relaxed text-[#d1d1d1] whitespace-pre-wrap">
                    {task.task_id === currentTaskIdRef.current && isStreaming ? (
                      <div className="space-y-4">
                        {agentStatus && (
                          <div className="flex items-center gap-2 mb-2 animate-in fade-in slide-in-from-left-2 duration-300">
                            <div className="px-2 py-0.5 rounded-full bg-[#00ff99]/10 border border-[#00ff99]/20 flex items-center gap-1.5">
                              <span className="w-1.5 h-1.5 rounded-full bg-[#00ff99] animate-pulse" />
                              <span className="text-[10px] font-bold uppercase tracking-widest text-[#00ff99]">
                                {agentStatus}
                              </span>
                            </div>
                          </div>
                        )}
                        <StreamingOutput content={streamingContent} isStreaming={isStreaming} />
                      </div>
                    ) : (
                      <div className="space-y-4">
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
                    <div className="flex items-center gap-3 p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-sm">
                      <AlertCircle size={18} />
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
              <p className="text-xs text-[#a3a3a3]">Send a message to start breading</p>
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
            placeholder="Type a message..."
            className="w-full bg-[#121212] border border-[#262626] rounded-[20px] px-5 py-4 pr-14 text-sm focus:outline-none focus:border-[#00ff99]/50 transition-colors resize-none placeholder-[#737373] min-h-[56px] max-h-40 custom-scrollbar"
            rows={1}
          />
          <button
            onClick={handleSubmitTask}
            disabled={!taskDescription.trim() || isStreaming}
            className={`absolute right-3 bottom-3 p-2 rounded-full transition-all ${taskDescription.trim() && !isStreaming
              ? 'bg-[#00ff99] text-[#0f0f0f] hover:scale-110'
              : 'bg-[#262626] text-[#737373] cursor-not-allowed'
              }`}
          >
            <Send size={18} />
          </button>
        </div>
        <p className="text-center mt-3 text-[10px] text-[#737373] font-medium uppercase tracking-widest">
          Shift + Enter for new line
        </p>
      </div>
    </div>
  );
}
