import { useEffect, useRef } from 'react';
import { Terminal, Sparkles, Copy, Check } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { useState } from 'react';

interface StreamingOutputProps {
  content: string;
  isStreaming: boolean;
  title?: string;
}

function CodeBlock({ language, children }: { language: string; children: string }) {
  const [copied, setCopied] = useState(false);
  
  const handleCopy = () => {
    navigator.clipboard.writeText(children);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  
  return (
    <div className="relative group my-2">
      <button
        onClick={handleCopy}
        className="absolute right-2 top-2 p-1.5 rounded bg-gray-700 hover:bg-gray-600 opacity-0 group-hover:opacity-100 transition-opacity"
        title="Copy code"
      >
        {copied ? <Check size={14} className="text-green-400" /> : <Copy size={14} className="text-gray-300" />}
      </button>
      <SyntaxHighlighter
        style={oneDark}
        language={language || 'text'}
        PreTag="div"
        customStyle={{ margin: 0, borderRadius: '0.5rem', fontSize: '0.875rem' }}
      >
        {children}
      </SyntaxHighlighter>
    </div>
  );
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
        className="flex-1 overflow-y-auto p-4 text-sm text-gray-200 prose prose-invert prose-sm max-w-none"
      >
        {content ? (
          <ReactMarkdown
            components={{
              code({ className, children, ...props }) {
                const match = /language-(\w+)/.exec(className || '');
                const codeString = String(children).replace(/\n$/, '');
                return match ? (
                  <CodeBlock language={match[1]}>{codeString}</CodeBlock>
                ) : (
                  <code className="bg-gray-800 px-1.5 py-0.5 rounded text-pink-400" {...props}>
                    {children}
                  </code>
                );
              },
              p({ children }) {
                return <p className="mb-3 leading-relaxed">{children}</p>;
              },
              h1({ children }) {
                return <h1 className="text-xl font-bold text-white mb-3 mt-4">{children}</h1>;
              },
              h2({ children }) {
                return <h2 className="text-lg font-bold text-white mb-2 mt-3">{children}</h2>;
              },
              h3({ children }) {
                return <h3 className="text-base font-semibold text-white mb-2 mt-3">{children}</h3>;
              },
              ul({ children }) {
                return <ul className="list-disc list-inside mb-3 space-y-1">{children}</ul>;
              },
              ol({ children }) {
                return <ol className="list-decimal list-inside mb-3 space-y-1">{children}</ol>;
              },
              li({ children }) {
                return <li className="text-gray-300">{children}</li>;
              },
              strong({ children }) {
                return <strong className="font-bold text-white">{children}</strong>;
              },
            }}
          >
            {content}
          </ReactMarkdown>
        ) : (
          <span className="text-gray-500 italic">Waiting for agent response...</span>
        )}
        {isStreaming && (
          <span className="inline-block w-2 h-4 bg-blue-400 animate-pulse ml-0.5" />
        )}
      </div>
    </div>
  );
}
