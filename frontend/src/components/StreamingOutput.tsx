import { useEffect, useRef, useState } from 'react';
import { Copy, Check, FileText } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface StreamingOutputProps {
  content: string;
  isStreaming: boolean;
  title?: string;
  sessionId?: string; // Reserved for potential backends
}

function CodeBlock({ language, children }: { language: string; children: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(children);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative group my-4 overflow-hidden rounded-xl border border-[#262626] shadow-md bg-[#000000]">
      <div className="flex items-center justify-between px-4 py-2 bg-[#121212] border-b border-[#262626]">
        <span className="text-[10px] font-bold text-[#a3a3a3] uppercase tracking-widest">{language || 'text'}</span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1.5 px-2 py-1 rounded-md text-[#a3a3a3] hover:text-[#00ff99] transition-colors"
          title="Copy code"
        >
          {copied ? <><Check size={12} /><span className="text-[10px] font-bold uppercase">Copied</span></> : <><Copy size={12} /><span className="text-[10px] font-bold uppercase">Copy</span></>}
        </button>
      </div>
      <SyntaxHighlighter
        style={oneDark}
        language={language || 'text'}
        PreTag="div"
        customStyle={{ margin: 0, padding: '1.25rem', background: 'transparent', fontSize: '0.85rem', lineHeight: '1.6' }}
      >
        {children}
      </SyntaxHighlighter>
    </div>
  );
}

export function StreamingOutput({ content, isStreaming, title = 'Output' }: StreamingOutputProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [content]);

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div ref={containerRef} className="flex-1 overflow-y-auto custom-scrollbar text-sm leading-relaxed text-[#d1d1d1] prose prose-invert prose-sm max-w-none">
        {content ? (
          <ReactMarkdown
            components={{
              code({ className, children, ...props }) {
                const match = /language-(\w+)/.exec(className || '');
                const codeString = String(children).replace(/\n$/, '');
                return match ? (
                  <CodeBlock language={match[1]}>{codeString}</CodeBlock>
                ) : (
                  <code className="bg-[#1a1a1a] px-1.5 py-0.5 rounded text-[#00ff99] font-medium" {...props}>
                    {children}
                  </code>
                );
              },
              p({ children }) {
                return <p className="mb-4 last:mb-0">{children}</p>;
              },
              h1({ children }) {
                return <h1 className="text-lg font-bold text-white mb-4 mt-6 border-b border-[#262626] pb-2">{children}</h1>;
              },
              h2({ children }) {
                return <h2 className="text-base font-semibold text-white mb-3 mt-5">{children}</h2>;
              },
              h3({ children }) {
                return <h3 className="text-sm font-semibold text-white mb-2 mt-4">{children}</h3>;
              },
              ul({ children }) {
                return <ul className="list-disc list-outside ml-5 mb-4 space-y-2">{children}</ul>;
              },
              ol({ children }) {
                return <ol className="list-decimal list-outside ml-5 mb-4 space-y-2">{children}</ol>;
              },
              li({ children }) {
                return <li className="text-[#a3a3a3] pl-1 font-normal leading-relaxed">{children}</li>;
              },
              strong({ children }) {
                return <strong className="font-bold text-white">{children}</strong>;
              },
              blockquote({ children }) {
                return <blockquote className="border-l-2 border-[#00ff99] pl-4 italic text-[#a3a3a3] my-4">{children}</blockquote>;
              }
            }}
          >
            {content}
          </ReactMarkdown>
        ) : (
          <div className="flex flex-col items-center justify-center h-full gap-3 opacity-20">
            <FileText size={40} />
            <span className="text-xs font-bold uppercase tracking-widest italic">Awaiting output...</span>
          </div>
        )}
        {isStreaming && (
          <div className="inline-flex items-center ml-1 align-baseline">
            <span className="w-1.5 h-3.5 bg-[#00ff99] animate-pulse" />
          </div>
        )}
      </div>
    </div>
  );
}
