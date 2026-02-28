import { useEffect, useRef } from 'react';
import { Terminal, Copy, Check } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { useState } from 'react';

interface StreamingOutputProps {
  content: string;
  isStreaming: boolean;
  title?: string;
  forceOpenHandsStyle?: boolean;
}

function CodeBlock({ language, children }: { language: string; children: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(children);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative group my-3 overflow-hidden rounded-xl border border-gray-700/50 shadow-sm">
      <div className="flex items-center justify-between px-4 py-2 bg-gray-800/80 border-b border-gray-700/50">
        <span className="text-xs font-medium text-gray-400 uppercase tracking-wider">{language || 'text'}</span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-transparent hover:bg-gray-700 text-gray-400 hover:text-gray-200 transition-colors"
          title="Copy code"
        >
          {copied ? <><Check size={14} className="text-green-400" /><span className="text-xs">Copied</span></> : <><Copy size={14} /><span className="text-xs">Copy</span></>}
        </button>
      </div>
      <SyntaxHighlighter
        style={oneDark}
        language={language || 'text'}
        PreTag="div"
        customStyle={{ margin: 0, padding: '1rem', background: '#1E1E24', fontSize: '0.875rem' }}
      >
        {children}
      </SyntaxHighlighter>
    </div>
  );
}

export function StreamingOutput({ content, isStreaming, title = 'Agent Output', forceOpenHandsStyle = false }: StreamingOutputProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [content]);

  // Use either the new OpenHands style or the legacy terminal style
  const containerClass = forceOpenHandsStyle
    ? "flex flex-col h-full bg-[#1E1E24] rounded-2xl border border-gray-700/50 shadow-sm overflow-hidden"
    : "flex flex-col h-full bg-gray-900 rounded-lg overflow-hidden";

  const headerClass = forceOpenHandsStyle
    ? "flex items-center gap-2 px-5 py-3 bg-[#1A1A1E] border-b border-gray-800/80"
    : "flex items-center gap-2 px-4 py-2 bg-gray-800 border-b border-gray-700";

  const contentClass = forceOpenHandsStyle
    ? "flex-1 overflow-y-auto p-5 text-[15px] leading-relaxed text-gray-200 prose prose-invert prose-sm max-w-none"
    : "flex-1 overflow-y-auto p-4 text-sm text-gray-200 prose prose-invert prose-sm max-w-none bg-gray-900";

  return (
    <div className={containerClass}>
      <div className={headerClass}>
        <Terminal size={16} className={forceOpenHandsStyle ? "text-blue-500" : "text-blue-400"} />
        <span className="text-sm font-medium text-gray-200">{title}</span>
        {isStreaming && (
          <div className="flex items-center gap-1.5 ml-auto">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500"></span>
            </span>
            <span className="text-[11px] uppercase tracking-wider font-semibold text-blue-400">Thinking</span>
          </div>
        )}
      </div>
      <div
        ref={containerRef}
        className={contentClass}
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
                  <code className="bg-gray-800/80 px-1.5 py-0.5 rounded text-blue-300 font-medium" {...props}>
                    {children}
                  </code>
                );
              },
              p({ children }) {
                return <p className="mb-4">{children}</p>;
              },
              h1({ children }) {
                return <h1 className="text-xl font-bold text-white mb-4 mt-6">{children}</h1>;
              },
              h2({ children }) {
                return <h2 className="text-lg font-semibold text-white mb-3 mt-5">{children}</h2>;
              },
              h3({ children }) {
                return <h3 className="text-base font-semibold text-white mb-2 mt-4">{children}</h3>;
              },
              ul({ children }) {
                return <ul className="list-disc list-outside ml-5 mb-4 space-y-1">{children}</ul>;
              },
              ol({ children }) {
                return <ol className="list-decimal list-outside ml-5 mb-4 space-y-1">{children}</ol>;
              },
              li({ children }) {
                return <li className="text-gray-300 pl-1">{children}</li>;
              },
              strong({ children }) {
                return <strong className="font-semibold text-white">{children}</strong>;
              },
            }}
          >
            {content}
          </ReactMarkdown>
        ) : (
          <div className="flex items-center justify-center h-full">
            <span className="text-gray-500 italic">Waiting for agent response...</span>
          </div>
        )}
        {isStreaming && (
          <span className="inline-block w-2 h-4 bg-blue-500 animate-pulse ml-1 align-middle" />
        )}
      </div>
    </div>
  );
}
