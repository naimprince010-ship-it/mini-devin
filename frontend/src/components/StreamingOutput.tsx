import { useEffect, useRef } from 'react';
import { Terminal, Sparkles } from 'lucide-react';

interface StreamingOutputProps {
  content: string;
  isStreaming: boolean;
  title?: string;
}

export function StreamingOutput({ content, isStreaming, title = 'Agent Output' }: StreamingOutputProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [content]);

  return (
    <div className="flex flex-col h-full bg-gray-900 rounded-lg overflow-hidden">
      <div className="flex items-center gap-2 px-4 py-2 bg-gray-800 border-b border-gray-700">
        <Terminal size={16} className="text-blue-400" />
        <span className="text-sm font-medium text-gray-200">{title}</span>
        {isStreaming && (
          <div className="flex items-center gap-1 ml-auto">
            <Sparkles size={14} className="text-yellow-400 animate-pulse" />
            <span className="text-xs text-yellow-400">Streaming...</span>
          </div>
        )}
      </div>
      <div
        ref={containerRef}
        className="flex-1 overflow-y-auto p-4 font-mono text-sm text-gray-200 whitespace-pre-wrap"
      >
        {content || (
          <span className="text-gray-500 italic">Waiting for agent response...</span>
        )}
        {isStreaming && (
          <span className="inline-block w-2 h-4 bg-blue-400 animate-pulse ml-0.5" />
        )}
      </div>
    </div>
  );
}
