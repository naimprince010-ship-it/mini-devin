import { useState, useEffect, useCallback } from 'react';
import { Session, Task, WebSocketMessage } from '../types';
import { useApi } from '../hooks/useApi';
import { useWebSocket } from '../hooks/useWebSocket';
import { Send, Clock, CheckCircle, XCircle, Loader, Layers, Terminal, ListTodo, AlertTriangle } from 'lucide-react';
import { StreamingOutput } from './StreamingOutput';
import { ToolExecutionLog, ToolExecution } from './ToolExecutionLog';
import { PlanProgress, Plan } from './PlanProgress';
// FileUpload, MemoryView, ExportButtons reserved for future use

interface TaskPanelProps {
  session: Session;
}

type ViewTab = 'output' | 'tools' | 'plan' | 'files';

export function TaskPanel({ session }: TaskPanelProps) {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [taskDescription, setTaskDescription] = useState('');
  const [selectedTask, setSelectedTask] = useState<Task | null>(null);
  const [activeTab, setActiveTab] = useState<ViewTab>('output');

  const [streamingContent, setStreamingContent] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [toolExecutions, setToolExecutions] = useState<ToolExecution[]>([]);
  const [currentPlan, setCurrentPlan] = useState<Plan | null>(null);

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
    console.log('[WebSocket]', message.type, message);

    switch (message.type) {
      case 'token':
        setIsStreaming(true);
        setStreamingContent(prev => {
          console.log('[WebSocket token] appending to prev:', prev.length, 'chars');
          return prev + (data.content as string || '');
        });
        break;

      case 'tokens_batch':
        setIsStreaming(true);
        setStreamingContent(prev => prev + (data.content as string || ''));
        break;

      case 'connected':
        console.log('WebSocket connected to session:', message.session_id);
        break;

      case 'thinking':
        setIsStreaming(true);
        setStreamingContent(prev => prev + `--- Thinking ---\n${message.content || data.content || ''}\n\n`);
        break;

      case 'response':
        setIsStreaming(true);
        setStreamingContent(prev => prev + `--- Agent Response ---\n${message.content || data.content || ''}\n\n`);
        break;

      case 'iteration':
        setStreamingContent(prev => prev + `\n--- Iteration ${message.iteration || data.iteration}/${message.max || data.max} ---\n`);
        break;

      case 'tool_result': {
        const result = message.result || (data.result as { tool: string; success: boolean; output: string; error?: string });
        if (result) {
          const toolOutput = `**Tool: ${result.tool}** (${result.success ? 'Success' : 'Failed'})\n\`\`\`\n${result.output || result.error || ''}\n\`\`\`\n\n`;
          setStreamingContent(prev => prev + toolOutput);

          const newExecution: ToolExecution = {
            id: `${Date.now()}-${result.tool}`,
            tool: result.tool,
            input: {},
            status: result.success ? 'completed' : 'failed',
            output: { stdout: result.output },
            error: result.error,
            startTime: new Date().toISOString(),
            endTime: new Date().toISOString(),
          };
          setToolExecutions(prev => [...prev, newExecution]);
        }
        break;
      }

      case 'tool_started': {
        const newExecution: ToolExecution = {
          id: `${Date.now()}-${data.tool}`,
          tool: data.tool as string,
          input: data.input as Record<string, unknown>,
          status: 'running',
          startTime: message.timestamp || new Date().toISOString(),
        };
        setToolExecutions(prev => [...prev, newExecution]);
        break;
      }

      case 'tool_completed': {
        setToolExecutions(prev => prev.map(exec => {
          if (exec.tool === data.tool && exec.status === 'running') {
            return {
              ...exec,
              status: 'completed',
              output: data.output as Record<string, unknown>,
              endTime: message.timestamp,
              durationMs: data.duration_ms as number,
            };
          }
          return exec;
        }));
        break;
      }

      case 'tool_failed': {
        setToolExecutions(prev => prev.map(exec => {
          if (exec.tool === data.tool && exec.status === 'running') {
            return {
              ...exec,
              status: 'failed',
              error: data.error as string,
              endTime: message.timestamp,
            };
          }
          return exec;
        }));
        break;
      }

      case 'tool_output': {
        setToolExecutions(prev => prev.map(exec => {
          if (exec.tool === data.tool && exec.status === 'running') {
            const currentOutput = exec.output || {};
            return {
              ...exec,
              output: {
                ...currentOutput,
                stdout: ((currentOutput.stdout as string) || '') + (data.output as string || ''),
              },
            };
          }
          return exec;
        }));
        break;
      }

      case 'plan_created': {
        const planData = data.plan as Record<string, unknown>;
        const steps = (planData.steps as Array<Record<string, unknown>> || []).map((step, index) => ({
          id: `step-${index}`,
          description: step.description as string || `Step ${index + 1}`,
          status: 'pending' as const,
          substeps: (step.substeps as Array<Record<string, unknown>> || []).map((sub, subIndex) => ({
            id: `step-${index}-${subIndex}`,
            description: sub.description as string || `Substep ${subIndex + 1}`,
            status: 'pending' as const,
          })),
        }));

        setCurrentPlan({
          id: planData.id as string || `plan-${Date.now()}`,
          title: planData.title as string || 'Execution Plan',
          strategy: planData.strategy as string || 'sequential',
          steps,
          currentStepIndex: 0,
          totalSteps: steps.length,
          completedSteps: 0,
        });
        break;
      }

      case 'plan_updated':
      case 'step_completed': {
        setCurrentPlan(prev => {
          if (!prev) return prev;

          const stepIndex = data.step_index as number;
          const updatedSteps = prev.steps.map((step, index) => {
            if (index === stepIndex) {
              return { ...step, status: 'completed' as const };
            }
            if (index === stepIndex + 1) {
              return { ...step, status: 'in_progress' as const };
            }
            return step;
          });

          return {
            ...prev,
            steps: updatedSteps,
            currentStepIndex: stepIndex + 1,
            completedSteps: prev.completedSteps + 1,
          };
        });
        break;
      }

      case 'phase_changed':
        setStreamingContent(prev => prev + `\n\n--- Phase: ${data.phase} ---\n\n`);
        break;

      case 'iteration_started':
        setStreamingContent(prev => prev + `\n--- Iteration ${data.iteration} ---\n`);
        break;

      case 'task_started':
        setIsStreaming(true);
        setStreamingContent('');
        setToolExecutions([]);
        setCurrentPlan(null);
        break;

      case 'task_completed':
        setIsStreaming(false);
        setStreamingContent(prev => prev + `\n\n--- Task Completed ---\nSummary: ${data.summary || 'Done'}`);
        loadTasks();
        break;

      case 'task_failed':
        setIsStreaming(false);
        setStreamingContent(prev => prev + `\n\n--- Task Failed ---\nError: ${data.error || 'Unknown error'}`);
        loadTasks();
        break;

      case 'verification_started':
        setStreamingContent(prev => prev + '\n\n--- Running Verification ---\n');
        break;

      case 'verification_completed': {
        const passed = data.passed as boolean;
        setStreamingContent(prev => prev + `\nVerification ${passed ? 'PASSED' : 'FAILED'}\n`);
        break;
      }

      case 'repair_started':
        setStreamingContent(prev => prev + '\n--- Starting Repair Loop ---\n');
        break;

      case 'repair_completed':
        setStreamingContent(prev => prev + '\n--- Repair Complete ---\n');
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
    setToolExecutions([]);
    setCurrentPlan(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session.session_id]);

  useEffect(() => {
    if (selectedTask) {
      if (selectedTask.task_id.startsWith('task-')) {
        // This is a local mock task, no need to fetch output from API
        setStreamingContent('Waiting for agent response...');
        setIsStreaming(true);
      } else {
        setStreamingContent('Loading task output...');
        api.getTaskOutput(session.session_id, selectedTask.task_id)
          .then(output => {
            if (output.outputs && output.outputs.length > 0) {
              const content = output.outputs
                .map(o => {
                  if (o.type === 'thinking') return `--- Thinking ---\n${o.content}\n`;
                  if (o.type === 'response') return `\n--- Agent Response ---\n${o.content}`;
                  if (o.type === 'error') return `\n--- Error ---\n${o.content}`;
                  return o.content;
                })
                .join('\n');
              setStreamingContent(content);
            } else if (output.result) {
              setStreamingContent(output.result);
            } else {
              setStreamingContent('No output available');
            }
            setIsStreaming(output.status === 'running' || output.status === 'queued');
          })
          .catch(() => {
            setStreamingContent('Failed to load task output');
          });
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedTask?.task_id, session.session_id]);

  const _pollTaskOutput = useCallback(async (sessionId: string, taskId: string) => {
    if (taskId.startsWith('task-')) {
      // Local mock tasks don't have an output API endpoint
      return;
    }

    const maxAttempts = 60;
    let attempts = 0;

    const poll = async () => {
      try {
        const output = await api.getTaskOutput(sessionId, taskId);

        if (output.outputs && output.outputs.length > 0) {
          const content = output.outputs
            .map(o => {
              if (o.type === 'thinking') return `--- Thinking ---\n${o.content}\n`;
              if (o.type === 'response') return `\n--- Agent Response ---\n${o.content}`;
              if (o.type === 'error') return `\n--- Error ---\n${o.content}`;
              return o.content;
            })
            .join('\n');
          setStreamingContent(content);
        }

        if (output.status === 'completed' || output.status === 'failed') {
          setIsStreaming(false);
          loadTasks();
          return;
        }

        attempts++;
        if (attempts < maxAttempts && (output.status === 'queued' || output.status === 'running')) {
          setTimeout(poll, 1000);
        }
      } catch (e) {
        console.error('Failed to poll task output:', e);
        attempts++;
        if (attempts < maxAttempts) {
          setTimeout(poll, 2000);
        }
      }
    };

    poll();
  }, [api]);

  const handleSubmitTask = async () => {
    if (!taskDescription.trim() || isStreaming) return;

    try {
      // Create a local mock task for the UI
      const mockTask: Task = {
        task_id: `task-${Date.now()}`,
        session_id: session.session_id,
        description: taskDescription,
        status: 'running',
        created_at: new Date().toISOString(),
        started_at: new Date().toISOString(),
        completed_at: null,
        iteration: 0,
        error_message: null
      };

      setTasks([...tasks, mockTask]);
      setTaskDescription('');
      setSelectedTask(mockTask);
      setStreamingContent('Waiting for agent response...');
      setIsStreaming(true);
      setToolExecutions([]);
      setCurrentPlan(null);
      setActiveTab('output');

      // Send the task description over the active WebSocket connection
      sendMessage(taskDescription);
    } catch (e) {
      console.error('Failed to create task:', e);
    }
  };

  const _handleCancelTask = async (taskId: string) => {
    try {
      await api.cancelTask(session.session_id, taskId);
      loadTasks();
    } catch (e) {
      console.error('Failed to cancel task:', e);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running': return <Loader className="animate-spin text-blue-400" size={16} />;
      case 'completed': return <CheckCircle className="text-green-400" size={16} />;
      case 'failed': return <XCircle className="text-red-400" size={16} />;
      case 'pending': return <Clock className="text-yellow-400" size={16} />;
      default: return <Clock className="text-gray-400" size={16} />;
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Task Input (OpenHands Style) */}
      <div className="p-4 border-b border-gray-800 bg-gray-900/50 backdrop-blur-sm z-10 shadow-sm relative shrink-0">
        <div className="max-w-3xl mx-auto w-full flex flex-col gap-2">
          {/* Status Indicator */}
          <div className="flex items-center gap-2 px-1">
            <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-gray-800/80 border border-gray-700/50 text-xs text-gray-400 w-fit">
              <div className={`w-1.5 h-1.5 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
              {isConnected ? 'Connected to Agent' : 'Disconnected'}
            </div>
          </div>

          {/* Input Area */}
          <div className="relative group flex items-end gap-2 bg-[#1E1E24] border border-gray-700/80 rounded-xl p-2 focus-within:ring-1 focus-within:ring-blue-500/50 focus-within:border-blue-500/50 transition-all shadow-inner">
            <textarea
              value={taskDescription}
              onChange={(e) => setTaskDescription(e.target.value)}
              placeholder={isStreaming ? "Agent is working..." : "What do you want to build?"}
              disabled={api.loading || isStreaming}
              className="flex-1 max-h-48 min-h-[44px] bg-transparent text-gray-100 placeholder-gray-500 px-3 py-2 focus:outline-none resize-none scrollbar-thin scrollbar-thumb-gray-600 w-full disabled:opacity-50"
              rows={taskDescription.split('\n').length > 1 ? Math.min(taskDescription.split('\n').length, 5) : 1}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  if (!isStreaming && !api.loading) {
                    handleSubmitTask();
                  }
                }
              }}
            />

            <button
              onClick={handleSubmitTask}
              disabled={api.loading || !taskDescription.trim() || isStreaming}
              className="shrink-0 p-2.5 rounded-lg bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-40 disabled:hover:bg-blue-600 transition-colors shadow-sm self-end"
              title="Send Message (Enter)"
            >
              <Send size={18} className={taskDescription.trim() && !isStreaming ? "translate-x-0.5" : ""} />
            </button>
          </div>
          <div className="flex justify-between items-center px-2">
            <p className="text-[11px] text-gray-500 font-medium">Press <kbd className="font-sans px-1 py-0.5 rounded bg-gray-800 border border-gray-700">Enter</kbd> to submit, <kbd className="font-sans px-1 py-0.5 rounded bg-gray-800 border border-gray-700">Shift</kbd> + <kbd className="font-sans px-1 py-0.5 rounded bg-gray-800 border border-gray-700">Enter</kbd> for newline</p>
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 overflow-hidden flex flex-col relative bg-[#1A1A1E]">
        <div className="flex-1 overflow-y-auto w-full max-w-4xl mx-auto p-4 custom-scrollbar">
          {selectedTask ? (
            <div className="flex flex-col gap-6 w-full pb-32">
              {/* Task Header Bubble */}
              <div className="self-end max-w-[85%] bg-blue-600 rounded-2xl p-4 shadow-sm">
                <div className="flex flex-col gap-2">
                  <div className="flex items-center gap-2 text-blue-100 mb-1">
                    {getStatusIcon(selectedTask.status)}
                    <span className="font-medium text-sm">Task: {selectedTask.task_id}</span>
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ml-auto ${selectedTask.status === 'completed' ? 'bg-green-500/20 text-green-100' :
                      selectedTask.status === 'failed' ? 'bg-red-500/20 text-red-100' :
                        selectedTask.status === 'running' ? 'bg-white/20 text-white' :
                          'bg-gray-800 border border-gray-700 text-gray-300'
                      }`}>
                      {selectedTask.status}
                    </span>
                  </div>
                  <p className="text-white text-base leading-relaxed">{selectedTask.description}</p>
                </div>
              </div>

              {/* Error Message */}
              {selectedTask.error_message && (
                <div className="self-start max-w-[85%] bg-red-950/40 border border-red-900/50 rounded-2xl p-4 shadow-sm">
                  <div className="flex items-center gap-2 text-red-400 text-sm mb-2">
                    <AlertTriangle size={16} />
                    <span className="font-medium">Execution Error</span>
                  </div>
                  <p className="text-red-200/80 text-sm whitespace-pre-wrap">{selectedTask.error_message}</p>
                </div>
              )}

              {/* Streaming Output / Agent Response Bubble */}
              <div className="self-start max-w-[95%] w-full">
                <StreamingOutput
                  content={streamingContent}
                  isStreaming={isStreaming}
                  title="Agent Response"
                />
              </div>

              {/* Tools & Plan Accordions */}
              {(toolExecutions.length > 0 || currentPlan) && (
                <div className="self-start max-w-[95%] w-full flex flex-col gap-3 mt-2">
                  <div className="flex items-center gap-4 text-sm text-gray-500 py-2 border-b border-gray-800/50">
                    <span className="font-medium">Execution Details</span>
                  </div>

                  {toolExecutions.length > 0 && (
                    <div className="bg-gray-800/30 border border-gray-700/50 rounded-xl overflow-hidden">
                      <div
                        className="flex items-center gap-2 px-4 py-3 bg-gray-800/50 cursor-pointer hover:bg-gray-800 transition-colors"
                        onClick={() => setActiveTab(activeTab === 'tools' ? 'output' : 'tools')}
                      >
                        <Layers size={16} className={activeTab === 'tools' ? 'text-blue-400' : 'text-gray-400'} />
                        <span className="font-medium text-gray-300">Tool Executions</span>
                        <span className="ml-2 px-2 py-0.5 text-xs bg-gray-700 text-gray-300 rounded-full">
                          {toolExecutions.length}
                        </span>
                      </div>
                      {activeTab === 'tools' && (
                        <div className="p-4 border-t border-gray-700/50 bg-[#1A1A1E]">
                          <ToolExecutionLog executions={toolExecutions} />
                        </div>
                      )}
                    </div>
                  )}

                  {currentPlan && (
                    <div className="bg-gray-800/30 border border-gray-700/50 rounded-xl overflow-hidden">
                      <div
                        className="flex items-center gap-2 px-4 py-3 bg-gray-800/50 cursor-pointer hover:bg-gray-800 transition-colors"
                        onClick={() => setActiveTab(activeTab === 'plan' ? 'output' : 'plan')}
                      >
                        <ListTodo size={16} className={activeTab === 'plan' ? 'text-blue-400' : 'text-gray-400'} />
                        <span className="font-medium text-gray-300">Execution Plan</span>
                        <span className="ml-2 px-2 py-0.5 text-xs bg-gray-700 text-gray-300 rounded-full">
                          {currentPlan.completedSteps}/{currentPlan.totalSteps}
                        </span>
                      </div>
                      {activeTab === 'plan' && (
                        <div className="p-4 border-t border-gray-700/50 bg-[#1A1A1E]">
                          <PlanProgress plan={currentPlan} />
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-gray-500 gap-4">
              <div className="w-16 h-16 rounded-full bg-gray-800/50 flex items-center justify-center border border-gray-700/50">
                <Terminal size={32} className="text-gray-400" />
              </div>
              <div className="text-center">
                <h3 className="text-lg font-medium text-gray-300 mb-1">Mini-Devin Agent</h3>
                <p className="text-sm">Enter a task to start building.</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
