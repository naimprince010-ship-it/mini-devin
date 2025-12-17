import { useState, useEffect, useCallback } from 'react';
import { Session, Task, WebSocketMessage } from '../types';
import { useApi } from '../hooks/useApi';
import { useWebSocket } from '../hooks/useWebSocket';
import { Send, Square, Clock, CheckCircle, XCircle, Loader, Layers, Terminal, ListTodo } from 'lucide-react';
import { StreamingOutput } from './StreamingOutput';
import { ToolExecutionLog, ToolExecution } from './ToolExecutionLog';
import { PlanProgress, Plan } from './PlanProgress';

interface TaskPanelProps {
  session: Session;
}

type ViewTab = 'output' | 'tools' | 'plan';

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
    
    switch (message.type) {
      case 'token':
        setIsStreaming(true);
        setStreamingContent(prev => prev + (data.content as string || ''));
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

  const { isConnected } = useWebSocket({
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
  }, [selectedTask?.task_id, session.session_id, api]);

  const pollTaskOutput = useCallback(async (sessionId: string, taskId: string) => {
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
    if (!taskDescription.trim()) return;
    
    try {
      const task = await api.createTask(session.session_id, {
        description: taskDescription,
        acceptance_criteria: [],
      });
      setTasks([...tasks, task]);
      setTaskDescription('');
      setSelectedTask(task);
      setStreamingContent('Waiting for agent response...');
      setIsStreaming(true);
      setToolExecutions([]);
      setCurrentPlan(null);
      setActiveTab('output');
      
      pollTaskOutput(session.session_id, task.task_id);
    } catch (e) {
      console.error('Failed to create task:', e);
    }
  };

  const handleCancelTask = async (taskId: string) => {
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
      {/* Task Input */}
      <div className="p-4 border-b border-gray-700">
        <div className="flex items-center gap-2 mb-2">
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
          <span className="text-sm text-gray-400">
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
        <div className="flex gap-2">
          <textarea
            value={taskDescription}
            onChange={(e) => setTaskDescription(e.target.value)}
            placeholder="Describe your task..."
            className="flex-1 px-3 py-2 bg-gray-700 text-white rounded border border-gray-600 focus:border-blue-500 focus:outline-none resize-none"
            rows={3}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && e.metaKey) {
                handleSubmitTask();
              }
            }}
          />
          <button
            onClick={handleSubmitTask}
            disabled={api.loading || !taskDescription.trim()}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded font-medium disabled:opacity-50 self-end"
          >
            <Send size={20} />
          </button>
        </div>
        <p className="text-xs text-gray-500 mt-1">Press Cmd+Enter to submit</p>
      </div>

      {/* Task List */}
      <div className="flex-1 overflow-hidden flex">
        {/* Tasks sidebar */}
        <div className="w-64 border-r border-gray-700 overflow-y-auto">
          <div className="p-2">
            <h3 className="text-sm font-medium text-gray-400 px-2 py-1">Tasks</h3>
            {tasks.length === 0 ? (
              <p className="text-gray-500 text-sm px-2 py-4">No tasks yet</p>
            ) : (
              <div className="space-y-1">
                {tasks.map((task) => (
                  <div
                    key={task.task_id}
                    onClick={() => setSelectedTask(task)}
                    className={`p-2 rounded cursor-pointer ${
                      selectedTask?.task_id === task.task_id
                        ? 'bg-blue-600'
                        : 'hover:bg-gray-700'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      {getStatusIcon(task.status)}
                      <span className="text-sm text-white truncate flex-1">
                        {task.description.slice(0, 30)}...
                      </span>
                    </div>
                    <div className="text-xs text-gray-400 mt-1">
                      Iteration: {task.iteration}
                    </div>
                    {task.status === 'running' && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleCancelTask(task.task_id);
                        }}
                        className="mt-1 text-xs text-red-400 hover:text-red-300 flex items-center gap-1"
                      >
                        <Square size={12} /> Cancel
                      </button>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Task Details / Streaming Output */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {selectedTask ? (
            <>
              {/* Task Header */}
              <div className="p-4 border-b border-gray-700">
                <div className="flex items-center gap-2 mb-2">
                  {getStatusIcon(selectedTask.status)}
                  <span className="font-medium text-white">{selectedTask.task_id}</span>
                  <span className={`px-2 py-0.5 rounded text-xs ${
                    selectedTask.status === 'completed' ? 'bg-green-900 text-green-300' :
                    selectedTask.status === 'failed' ? 'bg-red-900 text-red-300' :
                    selectedTask.status === 'running' ? 'bg-blue-900 text-blue-300' :
                    'bg-gray-700 text-gray-300'
                  }`}>
                    {selectedTask.status}
                  </span>
                </div>
                <p className="text-gray-300 text-sm">{selectedTask.description}</p>
                {selectedTask.error_message && (
                  <p className="text-red-400 text-sm mt-2">{selectedTask.error_message}</p>
                )}
              </div>

              {/* Tab Navigation */}
              <div className="flex border-b border-gray-700">
                <button
                  onClick={() => setActiveTab('output')}
                  className={`flex items-center gap-2 px-4 py-2 text-sm font-medium transition-colors ${
                    activeTab === 'output'
                      ? 'text-blue-400 border-b-2 border-blue-400 bg-gray-800'
                      : 'text-gray-400 hover:text-gray-200'
                  }`}
                >
                  <Terminal size={16} />
                  Output
                </button>
                <button
                  onClick={() => setActiveTab('tools')}
                  className={`flex items-center gap-2 px-4 py-2 text-sm font-medium transition-colors ${
                    activeTab === 'tools'
                      ? 'text-blue-400 border-b-2 border-blue-400 bg-gray-800'
                      : 'text-gray-400 hover:text-gray-200'
                  }`}
                >
                  <Layers size={16} />
                  Tools
                  {toolExecutions.length > 0 && (
                    <span className="ml-1 px-1.5 py-0.5 text-xs bg-gray-700 rounded">
                      {toolExecutions.length}
                    </span>
                  )}
                </button>
                <button
                  onClick={() => setActiveTab('plan')}
                  className={`flex items-center gap-2 px-4 py-2 text-sm font-medium transition-colors ${
                    activeTab === 'plan'
                      ? 'text-blue-400 border-b-2 border-blue-400 bg-gray-800'
                      : 'text-gray-400 hover:text-gray-200'
                  }`}
                >
                  <ListTodo size={16} />
                  Plan
                  {currentPlan && (
                    <span className="ml-1 px-1.5 py-0.5 text-xs bg-gray-700 rounded">
                      {currentPlan.completedSteps}/{currentPlan.totalSteps}
                    </span>
                  )}
                </button>
              </div>

              {/* Tab Content */}
              <div className="flex-1 overflow-hidden">
                {activeTab === 'output' && (
                  <StreamingOutput 
                    content={streamingContent} 
                    isStreaming={isStreaming}
                    title="Agent Output"
                  />
                )}
                {activeTab === 'tools' && (
                  <ToolExecutionLog executions={toolExecutions} />
                )}
                {activeTab === 'plan' && (
                  <PlanProgress plan={currentPlan} />
                )}
              </div>
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center text-gray-500">
              Select a task to view details
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
