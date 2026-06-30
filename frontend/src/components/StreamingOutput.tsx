import { useEffect, useRef, useState } from 'react';
import { Copy, Check, FileText, Terminal } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface StreamingOutputProps {
  content: string;
  isStreaming: boolean;
  title?: string;
  sessionId?: string; // Reserved for potential backends
}

// ── Terminal shell languages that should render as a terminal window ──────────
const TERMINAL_LANGS = new Set(['bash', 'sh', 'shell', 'zsh', 'terminal', 'console', 'cmd', 'powershell', 'ps1', 'fish']);

// ── Detect command-like patterns (lines starting with $ or >) ────────────────
function looksLikeShellOutput(text: string): boolean {
  const lines = text.split('\n').filter(l => l.trim());
  if (lines.length === 0) return false;
  const commandLines = lines.filter(l => /^(\$|>|#)\s/.test(l.trim()));
  return commandLines.length > 0;
}

// ── Colourize a single line for the terminal display ────────────────────────
function TerminalLine({ line }: { line: string }) {
  const trimmed = line.trimStart();
  // Command prompts
  if (/^(\$|>|#)\s/.test(trimmed)) {
    const [prompt, ...rest] = line.split(/\s+/);
    return (
      <div className="flex gap-2">
        <span className="text-[#00ff99] font-bold select-none flex-shrink-0">{prompt}</span>
        <span className="text-[#e2e8f0]">{rest.join(' ')}</span>
      </div>
    );
  }
  // Error / warning patterns
  if (/error|traceback|exception|fatal|fail/i.test(trimmed) && !/✓|success/i.test(trimmed)) {
    return <div className="text-red-400/90">{line}</div>;
  }
  // Warning patterns
  if (/warning|warn:/i.test(trimmed)) {
    return <div className="text-yellow-400/90">{line}</div>;
  }
  // Success patterns
  if (/✓|success|done|complete|built|compiled|passed/i.test(trimmed)) {
    return <div className="text-[#00ff99]/80">{line}</div>;
  }
  // Info / numbered lines (e.g. "added 123 packages")
  if (/^added \d+|^found \d+|^\d+ packages|^npm warn/i.test(trimmed)) {
    return <div className="text-[#94a3b8]">{line}</div>;
  }
  return <div className="text-[#cbd5e1]">{line}</div>;
}

// ── Terminal window component ────────────────────────────────────────────────
function TerminalBlock({ children, language }: { children: string; language?: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(children).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const lines = children.split('\n');
  // Remove trailing blank line that code blocks usually have
  if (lines[lines.length - 1] === '') lines.pop();

  const label = language && language !== 'terminal' && language !== 'console' ? language : 'Terminal';

  return (
    <div className="my-4 rounded-xl overflow-hidden border border-[#1e2a1e] shadow-[0_0_24px_rgba(0,255,153,0.04)] bg-[#050f05]">
      {/* Title bar */}
      <div className="flex items-center justify-between px-4 py-2.5 bg-[#0a140a] border-b border-[#1e2a1e]">
        <div className="flex items-center gap-3">
          {/* Traffic light dots */}
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-[#ff5f57] shadow-[0_0_4px_#ff5f57]" />
            <span className="w-2.5 h-2.5 rounded-full bg-[#febc2e] shadow-[0_0_4px_#febc2e]" />
            <span className="w-2.5 h-2.5 rounded-full bg-[#28c840] shadow-[0_0_4px_#28c840]" />
          </div>
          <div className="flex items-center gap-1.5 text-[#3a5a3a]">
            <Terminal size={11} />
            <span className="text-[10px] font-bold uppercase tracking-[0.15em] font-mono">{label}</span>
          </div>
        </div>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1.5 px-2 py-1 rounded-md text-[#3a5a3a] hover:text-[#00ff99] hover:bg-[#00ff99]/5 transition-all text-[10px] font-bold uppercase tracking-wide"
          title="Copy terminal output"
        >
          {copied
            ? <><Check size={11} /><span>Copied</span></>
            : <><Copy size={11} /><span>Copy</span></>
          }
        </button>
      </div>

      {/* Output body */}
      <div className="p-4 overflow-x-auto custom-scrollbar max-h-[480px] overflow-y-auto">
        <pre className="text-[12.5px] leading-[1.65] font-mono m-0">
          {lines.map((line, i) => (
            <TerminalLine key={i} line={line} />
          ))}
        </pre>
      </div>
    </div>
  );
}

function CodeBlock({ language, children }: { language: string; children: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(children);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Delegate terminal-like languages to TerminalBlock
  if (TERMINAL_LANGS.has(language.toLowerCase()) || looksLikeShellOutput(children)) {
    return <TerminalBlock language={language}>{children}</TerminalBlock>;
  }

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
  const [copiedAll, setCopiedAll] = useState(false);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [content]);

  const copyEntireReply = () => {
    if (!content?.trim()) return;
    navigator.clipboard.writeText(content).then(() => {
      setCopiedAll(true);
      setTimeout(() => setCopiedAll(false), 2000);
    });
  };

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {content.trim().length > 0 && (
        <div className="flex-shrink-0 flex items-center justify-between gap-2 px-2 py-1.5 border-b border-[#1a1a1a] bg-[#080808]">
          <span className="text-[10px] font-mono text-[#525252] truncate uppercase tracking-wide">{title}</span>
          <button
            type="button"
            onClick={copyEntireReply}
            className="flex items-center gap-1.5 px-2 py-1 rounded-md text-[#a3a3a3] hover:text-[#00ff99] hover:bg-[#141414] transition-colors text-[10px] font-bold uppercase shrink-0"
            title="Copy full agent reply (raw markdown / text)"
          >
            {copiedAll ? (
              <>
                <Check size={12} />
                Copied
              </>
            ) : (
              <>
                <Copy size={12} />
                Copy reply
              </>
            )}
          </button>
        </div>
      )}
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
