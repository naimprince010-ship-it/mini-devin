import React, { useState } from 'react';
import { CheckCircle2, XCircle, Loader2, Terminal, FileText, Globe, Search, Code2, Wrench, Copy, Check, ChevronDown, ChevronRight } from 'lucide-react';
import { ToolCallEntry } from '../contexts/SessionEventsContext';

interface ToolCallLogProps {
    toolCalls: ToolCallEntry[];
}

function getToolIcon(toolName: string) {
    const name = toolName.toLowerCase();
    if (name.includes('bash') || name.includes('shell') || name.includes('command') || name.includes('run')) {
        return <Terminal size={13} className="text-[#00ff99]" />;
    }
    if (name.includes('read') || name.includes('file') || name.includes('write') || name.includes('view')) {
        return <FileText size={13} className="text-blue-400" />;
    }
    if (name.includes('search') || name.includes('grep') || name.includes('find')) {
        return <Search size={13} className="text-yellow-400" />;
    }
    if (name.includes('browser') || name.includes('url') || name.includes('web')) {
        return <Globe size={13} className="text-purple-400" />;
    }
    if (name.includes('code') || name.includes('edit') || name.includes('replace')) {
        return <Code2 size={13} className="text-orange-400" />;
    }
    return <Wrench size={13} className="text-[#a3a3a3]" />;
}

function getInputSummary(tool: string, input: Record<string, unknown>): string {
    const name = tool.toLowerCase();
    if (name.includes('command') || name.includes('bash') || name.includes('run')) {
        return (input.command as string) || (input.cmd as string) || JSON.stringify(input).slice(0, 60);
    }
    if (name.includes('read') || name.includes('view') || name.includes('file')) {
        return (input.path as string) || (input.file as string) || JSON.stringify(input).slice(0, 60);
    }
    if (name.includes('write') || name.includes('edit') || name.includes('replace')) {
        return (input.path as string) || (input.file as string) || JSON.stringify(input).slice(0, 60);
    }
    if (name.includes('search') || name.includes('grep')) {
        return (input.query as string) || (input.pattern as string) || JSON.stringify(input).slice(0, 60);
    }
    if (name.includes('url') || name.includes('browser')) {
        return (input.url as string) || JSON.stringify(input).slice(0, 60);
    }
    const first = Object.values(input)[0];
    if (typeof first === 'string') return first.slice(0, 60);
    return JSON.stringify(input).slice(0, 60);
}

function getSuccessDetails(output: Record<string, unknown> | undefined): string[] {
    if (!output) return [];
    const details: string[] = [];
    const prUrl = output.pr_url;
    if (typeof prUrl === 'string' && prUrl) {
        details.push(`PR: ${prUrl}`);
    }
    const branchName = output.branch_name;
    if (typeof branchName === 'string' && branchName) {
        details.push(`Branch: ${branchName}`);
    }
    const commitSha = output.commit_sha;
    if (typeof commitSha === 'string' && commitSha) {
        details.push(`Commit: ${commitSha.slice(0, 12)}`);
    }
    const issueNumber = output.issue_number;
    if (typeof issueNumber === 'number') {
        details.push(`Issue: #${issueNumber}`);
    }
    const message = output.message;
    if (details.length === 0 && typeof message === 'string' && message) {
        details.push(message);
    }
    return details.slice(0, 3);
}

// ── Inline terminal output panel ─────────────────────────────────────────────
function TerminalOutputPanel({ output }: { output: Record<string, unknown> }) {
    const [expanded, setExpanded] = useState(false);
    const [copied, setCopied] = useState(false);

    const stdout = (output.stdout as string) || (output.output as string) || '';
    const stderr = (output.stderr as string) || '';
    const exitCode = output.exit_code != null ? Number(output.exit_code) : null;
    const combined = [stdout, stderr].filter(Boolean).join('\n').trim();

    if (!combined) return null;

    const lines = combined.split('\n');
    const preview = lines.slice(0, 2).join('\n');
    const hasMore = lines.length > 2;

    const handleCopy = () => {
        navigator.clipboard.writeText(combined).then(() => {
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        });
    };

    return (
        <div className="mt-2 rounded-lg overflow-hidden border border-[#1e2a1e] bg-[#050f05]">
            {/* Mini title bar */}
            <div className="flex items-center justify-between px-2.5 py-1.5 bg-[#0a140a] border-b border-[#1e2a1e]">
                <div className="flex items-center gap-2">
                    <div className="flex items-center gap-1">
                        <span className="w-2 h-2 rounded-full bg-[#ff5f57]" />
                        <span className="w-2 h-2 rounded-full bg-[#febc2e]" />
                        <span className="w-2 h-2 rounded-full bg-[#28c840]" />
                    </div>
                    <span className="text-[9px] font-bold uppercase tracking-[0.15em] font-mono text-[#3a5a3a]">
                        stdout
                        {exitCode !== null && (
                            <span className={`ml-1.5 ${exitCode === 0 ? 'text-[#00ff99]/50' : 'text-red-500/70'}`}>
                                exit {exitCode}
                            </span>
                        )}
                    </span>
                </div>
                <div className="flex items-center gap-1">
                    <button
                        onClick={handleCopy}
                        className="flex items-center gap-1 px-1.5 py-0.5 rounded text-[#3a5a3a] hover:text-[#00ff99] transition-colors text-[9px] font-bold uppercase"
                        title="Copy output"
                    >
                        {copied ? <><Check size={9} /><span>Copied</span></> : <><Copy size={9} /><span>Copy</span></>}
                    </button>
                    {hasMore && (
                        <button
                            onClick={() => setExpanded(e => !e)}
                            className="flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[#3a5a3a] hover:text-[#00ff99] transition-colors text-[9px]"
                        >
                            {expanded ? <ChevronDown size={9} /> : <ChevronRight size={9} />}
                        </button>
                    )}
                </div>
            </div>
            {/* Content */}
            <div className={`px-3 py-2 overflow-x-auto ${expanded ? 'max-h-64 overflow-y-auto' : 'max-h-14 overflow-hidden'} transition-all duration-200`}>
                <pre className="text-[11px] leading-relaxed font-mono text-[#7cbb7c] whitespace-pre-wrap">
                    {expanded ? combined : preview}
                    {!expanded && hasMore && (
                        <span className="text-[#3a5a3a] cursor-pointer" onClick={() => setExpanded(true)}>
                            {' '}…{lines.length - 2} more line{lines.length - 2 !== 1 ? 's' : ''}
                        </span>
                    )}
                </pre>
            </div>
        </div>
    );
}

function isTerminalTool(toolName: string): boolean {
    const n = toolName.toLowerCase();
    return n.includes('terminal') || n.includes('bash') || n.includes('shell') || n.includes('command') || n.includes('run');
}

export const ToolCallLog: React.FC<ToolCallLogProps> = ({ toolCalls }) => {
    if (toolCalls.length === 0) {
        return (
            <div className="flex flex-col items-center justify-center h-full text-[#a3a3a3] text-sm py-12 gap-3">
                <Wrench size={32} className="opacity-20" />
                <p className="text-xs text-center">No tool calls yet.<br />Tool activity will appear here when the agent runs.</p>
            </div>
        );
    }

    return (
        <div className="space-y-1 p-3">
            {toolCalls.map((entry) => (
                <div
                    key={entry.id}
                    className={`group flex items-start gap-2.5 px-3 py-2.5 rounded-lg border transition-colors ${entry.status === 'running'
                        ? 'bg-[#00ff99]/5 border-[#00ff99]/20'
                        : entry.status === 'failed'
                            ? 'bg-red-500/5 border-red-500/20'
                            : 'bg-[#1a1a1a] border-[#262626] hover:border-[#363636]'
                        }`}
                >
                    {/* Status icon */}
                    <div className="flex-shrink-0 mt-0.5">
                        {entry.status === 'running' ? (
                            <Loader2 size={13} className="animate-spin text-[#00ff99]" />
                        ) : entry.status === 'failed' ? (
                            <XCircle size={13} className="text-red-400" />
                        ) : (
                            <CheckCircle2 size={13} className="text-[#00ff99]" />
                        )}
                    </div>

                    {/* Tool icon + content */}
                    <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-1.5">
                            {getToolIcon(entry.tool)}
                            <span className="text-xs font-semibold text-white truncate">{entry.tool}</span>
                            {entry.durationMs !== undefined && (
                                <span className="ml-auto text-[10px] text-[#737373] flex-shrink-0">
                                    {entry.durationMs < 1000
                                        ? `${entry.durationMs}ms`
                                        : `${(entry.durationMs / 1000).toFixed(1)}s`}
                                </span>
                            )}
                        </div>
                        <div className="mt-0.5 text-[11px] text-[#737373] font-mono truncate">
                            {getInputSummary(entry.tool, entry.input)}
                        </div>
                        {entry.status === 'failed' && entry.output?.error !== undefined && (
                            <div className="mt-1 text-[11px] text-red-400 truncate">
                                {String(entry.output.error)}
                            </div>
                        )}
                        {entry.status === 'completed' && isTerminalTool(entry.tool) && entry.output && (
                            <TerminalOutputPanel output={entry.output} />
                        )}
                        {entry.status === 'completed' && !isTerminalTool(entry.tool) && getSuccessDetails(entry.output).length > 0 && (
                            <div className="mt-1 space-y-0.5">
                                {getSuccessDetails(entry.output).map((detail) => (
                                    <div key={detail} className="text-[11px] text-[#9ae6b4] truncate">
                                        {detail}
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            ))}
        </div>
    );
};
