import { useState, useEffect } from 'react';
import { Session, Task, WebSocketMessage } from '../types';
import { useApi } from '../hooks/useApi';
import { useWebSocket } from '../hooks/useWebSocket';
import { Send, Square, FileText, Clock, CheckCircle, XCircle, Loader } from 'lucide-react';

interface TaskPanelProps {
  session: Session;
}

export function TaskPanel({ session }: TaskPanelProps) {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [taskDescription, setTaskDescription] = useState('');
  const [selectedTask, setSelectedTask] = useState<Task | null>(null);
  const [streamingLogs, setStreamingLogs] = useState<string[]>([]);
  
  const api = useApi();

  const handleWebSocketMessage = (message: WebSocketMessage) => {
    const logEntry = `[${message.type}] ${JSON.stringify(message.data)}`;
    setStreamingLogs(prev => [...prev.slice(-100), logEntry]);
    
    // Update task status based on message
    if (message.task_id) {
      if (message.type === 'task_completed' || message.type === 'task_failed') {
        loadTasks();
      }
    }
  };

  const { isConnected, messages } = useWebSocket({
    sessionId: session.session_id,
    onMessage: handleWebSocketMessage,
  });

  const loadTasks = async () => {
    try {
      const data = await api.listTasks(session.session_id);
      setTasks(data);
    } catch (e) {
      console.error('Failed to load tasks:', e);
    }
  };

  useEffect(() => {
    loadTasks();
    setStreamingLogs([]);
  }, [session.session_id]);

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
      setStreamingLogs([]);
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

        {/* Task Details / Streaming Logs */}
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

              {/* Streaming Logs */}
              <div className="flex-1 overflow-y-auto p-4 bg-gray-900 font-mono text-sm">
                <h4 className="text-gray-400 mb-2">Live Logs</h4>
                {streamingLogs.length === 0 ? (
                  <p className="text-gray-500">Waiting for updates...</p>
                ) : (
                  <div className="space-y-1">
                    {streamingLogs.map((log, i) => (
                      <div key={i} className="text-gray-300 break-all">
                        {log}
                      </div>
                    ))}
                  </div>
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
