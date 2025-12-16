import { useState } from 'react';
import { 
  Terminal, 
  FileEdit, 
  Globe, 
  ChevronDown, 
  ChevronRight,
  Clock,
  CheckCircle,
  XCircle,
  Loader,
  Code
} from 'lucide-react';

export interface ToolExecution {
  id: string;
  tool: string;
  input: Record<string, unknown>;
  output?: Record<string, unknown>;
  status: 'running' | 'completed' | 'failed';
  startTime: string;
  endTime?: string;
  durationMs?: number;
  error?: string;
}

interface ToolExecutionLogProps {
  executions: ToolExecution[];
}

const getToolIcon = (tool: string) => {
  if (tool.includes('terminal') || tool.includes('shell') || tool.includes('command')) {
    return <Terminal size={14} className="text-green-400" />;
  }
  if (tool.includes('edit') || tool.includes('file') || tool.includes('write') || tool.includes('read')) {
    return <FileEdit size={14} className="text-blue-400" />;
  }
  if (tool.includes('browser') || tool.includes('search') || tool.includes('fetch')) {
    return <Globe size={14} className="text-purple-400" />;
  }
  return <Code size={14} className="text-gray-400" />;
};

const getStatusIcon = (status: string) => {
  switch (status) {
    case 'running':
      return <Loader size={14} className="text-blue-400 animate-spin" />;
    case 'completed':
      return <CheckCircle size={14} className="text-green-400" />;
    case 'failed':
      return <XCircle size={14} className="text-red-400" />;
    default:
      return <Clock size={14} className="text-gray-400" />;
  }
};

const formatDuration = (ms: number): string => {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}m`;
};


interface ToolExecutionItemProps {
  execution: ToolExecution;
}

function ToolExecutionItem({ execution }: ToolExecutionItemProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="border border-gray-700 rounded-lg overflow-hidden mb-2">
      <div
        className="flex items-center gap-2 px-3 py-2 bg-gray-800 cursor-pointer hover:bg-gray-750"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        {isExpanded ? (
          <ChevronDown size={14} className="text-gray-400" />
        ) : (
          <ChevronRight size={14} className="text-gray-400" />
        )}
        {getToolIcon(execution.tool)}
        <span className="text-sm font-medium text-gray-200 flex-1">
          {execution.tool}
        </span>
        {getStatusIcon(execution.status)}
        {execution.durationMs !== undefined && (
          <span className="text-xs text-gray-500">
            {formatDuration(execution.durationMs)}
          </span>
        )}
      </div>
      
      {isExpanded && (
        <div className="p-3 bg-gray-900 border-t border-gray-700">
          <div className="mb-3">
            <div className="text-xs text-gray-500 mb-1">Input</div>
            <pre className="text-xs text-gray-300 bg-gray-800 p-2 rounded overflow-x-auto">
              {JSON.stringify(execution.input, null, 2)}
            </pre>
          </div>
          
          {execution.output && (
            <div className="mb-3">
              <div className="text-xs text-gray-500 mb-1">Output</div>
              <pre className="text-xs text-gray-300 bg-gray-800 p-2 rounded overflow-x-auto max-h-48 overflow-y-auto">
                {typeof execution.output === 'string' 
                  ? execution.output 
                  : JSON.stringify(execution.output, null, 2)}
              </pre>
            </div>
          )}
          
          {execution.error && (
            <div>
              <div className="text-xs text-red-500 mb-1">Error</div>
              <pre className="text-xs text-red-300 bg-red-900/20 p-2 rounded overflow-x-auto">
                {execution.error}
              </pre>
            </div>
          )}
          
          <div className="flex items-center gap-4 mt-2 text-xs text-gray-500">
            <span>Started: {new Date(execution.startTime).toLocaleTimeString()}</span>
            {execution.endTime && (
              <span>Ended: {new Date(execution.endTime).toLocaleTimeString()}</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export function ToolExecutionLog({ executions }: ToolExecutionLogProps) {
  if (executions.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-gray-500 p-4">
        <Code size={32} className="mb-2 opacity-50" />
        <span className="text-sm">No tool executions yet</span>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center gap-2 px-4 py-2 bg-gray-800 border-b border-gray-700">
        <Code size={16} className="text-blue-400" />
        <span className="text-sm font-medium text-gray-200">Tool Executions</span>
        <span className="text-xs text-gray-500 ml-auto">{executions.length} total</span>
      </div>
      <div className="flex-1 overflow-y-auto p-3">
        {executions.map((execution) => (
          <ToolExecutionItem key={execution.id} execution={execution} />
        ))}
      </div>
    </div>
  );
}
