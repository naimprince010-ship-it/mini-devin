import React, { useState, useRef, useEffect } from 'react';
import { Terminal, FileCode, History, Maximize2, Globe, ExternalLink, Search, X, Brain } from 'lucide-react';
import { MemoryView } from './MemoryView';
import { FileExplorer } from './FileExplorer';
import { ToolCallLog } from './ToolCallLog';
import { FileDiffView } from './FileDiffView';
import { MonacoEditorPanel } from './MonacoEditorPanel';
import { useSessionEvents } from '../contexts/SessionEventsContext';

interface WorkspacePanelProps {
    sessionId?: string;
}

type TabType = 'shell' | 'worklog' | 'editor' | 'browser' | 'memory';

export const WorkspacePanel: React.FC<WorkspacePanelProps> = ({ sessionId }) => {
    const [activeTab, setActiveTab] = useState<TabType>('shell');
    const shellRef = useRef<HTMLDivElement>(null);
    const events = useSessionEvents();
    // Monaco editor state: which file is open
    const [openFile, setOpenFile] = useState<string | null>(null);

    // Auto-scroll shell when new lines come in
    useEffect(() => {
        if (shellRef.current) {
            shellRef.current.scrollTop = shellRef.current.scrollHeight;
        }
    }, [events.shellLines]);

    // Switch to worklog when a tool starts
    useEffect(() => {
        if (events.toolCalls.length > 0 && events.isRunning) {
            if (activeTab === 'shell') setActiveTab('worklog');
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [events.toolCalls.length, events.isRunning]);

    // Auto-switch to editor tab when file edits appear
    useEffect(() => {
        if (events.fileEdits.length > 0 && events.isRunning && activeTab !== 'browser') {
            setActiveTab('editor');
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [events.fileEdits.length]);

    // Auto-switch to browser tab when browser events appear
    useEffect(() => {
        if (events.browserEvents.length > 0 && events.isRunning) {
            setActiveTab('browser');
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [events.browserEvents.length]);

    const tabs: { id: TabType; label: string; icon: React.ReactNode; badge?: number }[] = [
        { id: 'shell', label: 'Shell', icon: <Terminal size={13} /> },
        {
            id: 'worklog',
            label: 'Worklog',
            icon: <History size={13} />,
            badge: events.toolCalls.length > 0 ? events.toolCalls.length : undefined,
        },
        {
            id: 'editor',
            label: 'IDE',
            icon: <FileCode size={13} />,
            badge: events.fileEdits.length > 0 ? events.fileEdits.length : undefined,
        },
        {
            id: 'browser',
            label: 'Browser',
            icon: <Globe size={13} />,
            badge: events.browserEvents.length > 0 ? events.browserEvents.length : undefined,
        },
        {
            id: 'memory',
            label: 'Memory',
            icon: <Brain size={13} />,
        },
    ];

    const lastBrowserEvent = events.browserEvents[events.browserEvents.length - 1];
    const currentUrl = lastBrowserEvent?.url || lastBrowserEvent?.query;

    return (
        <div className="h-full w-full flex flex-col bg-[#0a0a0a] border-l border-[#262626]">
            {/* Tab Header */}
            <div className="flex items-center justify-between px-3 py-2 border-b border-[#262626] bg-[#111111]">
                <div className="flex gap-1">
                    {tabs.map(tab => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            className={`relative flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${activeTab === tab.id
                                ? 'bg-[#1e1e1e] text-white shadow-sm'
                                : 'text-[#737373] hover:text-[#a3a3a3] hover:bg-[#1a1a1a]'
                                }`}
                        >
                            <span className={activeTab === tab.id ? 'text-[#00ff99]' : ''}>{tab.icon}</span>
                            {tab.label}
                            {tab.badge !== undefined && (
                                <span className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-[#00ff99] text-[#0f0f0f] text-[9px] font-bold flex items-center justify-center">
                                    {tab.badge > 9 ? '9+' : tab.badge}
                                </span>
                            )}
                        </button>
                    ))}
                </div>
                <button className="p-1.5 text-[#525252] hover:text-[#a3a3a3] rounded-md hover:bg-[#1a1a1a] transition-colors">
                    <Maximize2 size={13} />
                </button>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-hidden relative">

                {/* SHELL */}
                {activeTab === 'shell' && (
                    <div
                        ref={shellRef}
                        className="absolute inset-0 p-4 font-mono text-xs overflow-y-auto bg-[#050505] custom-scrollbar"
                    >
                        <div className="text-[#00ff99] mb-3 text-[11px]">
                            ubuntu@mini-devin:~$ <span className="animate-pulse">_</span>
                        </div>
                        {events.shellLines.length === 0 ? (
                            <div className="text-[#3a3a3a] italic text-[11px]">
                                Shell output will appear here when the agent runs commands...
                            </div>
                        ) : (
                            events.shellLines.map((line, i) => (
                                <div
                                    key={i}
                                    className={`leading-5 ${line.startsWith('$')
                                        ? 'text-[#00ff99]'
                                        : line.toLowerCase().includes('error') || line.toLowerCase().includes('fail')
                                            ? 'text-red-400'
                                            : 'text-[#c0c0c0]'
                                        }`}
                                >
                                    {line}
                                </div>
                            ))
                        )}
                    </div>
                )}

                {/* WORKLOG */}
                {activeTab === 'worklog' && (
                    <div className="absolute inset-0 overflow-y-auto custom-scrollbar">
                        {events.toolCalls.length === 0 && !events.isRunning ? (
                            <div className="p-4">
                                <h3 className="text-[#525252] text-[10px] font-semibold mb-3 uppercase tracking-wider">
                                    Agent Worklog
                                </h3>
                                <p className="text-[#3a3a3a] text-xs italic">
                                    Tool calls and actions will appear here in real time...
                                </p>
                            </div>
                        ) : (
                            <>
                                <div className="px-3 pt-3 pb-1 flex items-center justify-between">
                                    <h3 className="text-[#525252] text-[10px] font-semibold uppercase tracking-wider">
                                        Agent Worklog
                                    </h3>
                                    <span className="text-[10px] text-[#525252]">
                                        {events.toolCalls.filter(t => t.status === 'completed').length}/
                                        {events.toolCalls.length} done
                                    </span>
                                </div>
                                <ToolCallLog toolCalls={events.toolCalls} />
                            </>
                        )}
                    </div>
                )}

                {/* IDE — Monaco or File Diff View */}
                {activeTab === 'editor' && (
                    <div className="absolute inset-0 flex flex-col">
                        {openFile && sessionId ? (
                            // Monaco editor for the selected file
                            <MonacoEditorPanel
                                sessionId={sessionId}
                                filePath={openFile}
                                initialContent={events.fileEdits.find(f => f.path === openFile)?.content}
                                onClose={() => setOpenFile(null)}
                                onSaved={() => { }}
                            />
                        ) : events.fileEdits.length > 0 ? (
                            // File diff view when agent has written files
                            <div className="flex flex-col h-full">
                                <div className="flex items-center gap-2 px-3 py-1.5 border-b border-[#1a1a1a] bg-[#111] flex-shrink-0">
                                    <span className="text-[10px] text-[#525252] uppercase tracking-wider">Agent file edits — click a file to edit</span>
                                </div>
                                <FileDiffView
                                    fileEdits={events.fileEdits}
                                    onFileSelect={(path) => { setActiveTab('editor'); setOpenFile(path); }}
                                />
                            </div>
                        ) : (
                            // File explorer when no edits yet
                            <FileExplorer
                                sessionId={sessionId}
                                onFileSelect={(path: string) => { setActiveTab('editor'); setOpenFile(path); }}
                            />
                        )}
                    </div>
                )}

                {/* BROWSER */}
                {activeTab === 'browser' && (
                    <div className="absolute inset-0 flex flex-col bg-[#050505]">
                        {events.browserEvents.length === 0 ? (
                            <div className="flex flex-col items-center justify-center h-full text-center p-8 gap-4">
                                <Globe size={48} className="text-[#2a2a2a]" />
                                <div>
                                    <p className="text-[#525252] text-sm font-medium">Browser</p>
                                    <p className="text-[#3a3a3a] text-xs mt-1">
                                        Web browser activity will be visible here when the agent browses the internet.
                                    </p>
                                </div>
                            </div>
                        ) : (
                            <>
                                {/* URL Bar */}
                                <div className="flex items-center gap-2 px-3 py-2 border-b border-[#1a1a1a] bg-[#0d0d0d]">
                                    <div className="w-2 h-2 rounded-full bg-[#00ff99] animate-pulse flex-shrink-0" />
                                    <div className="flex-1 flex items-center gap-2 px-3 py-1.5 bg-[#111111] border border-[#1e1e1e] rounded-lg">
                                        {lastBrowserEvent?.type === 'search' ? (
                                            <Search size={10} className="text-[#525252] flex-shrink-0" />
                                        ) : (
                                            <Globe size={10} className="text-[#525252] flex-shrink-0" />
                                        )}
                                        <span className="text-[11px] font-mono text-[#a3a3a3] truncate">
                                            {currentUrl || 'Agent browsing...'}
                                        </span>
                                    </div>
                                    {currentUrl && (
                                        <a href={currentUrl} target="_blank" rel="noreferrer"
                                            className="p-1 text-[#525252] hover:text-[#a3a3a3] transition-colors">
                                            <ExternalLink size={12} />
                                        </a>
                                    )}
                                </div>

                                {/* Screenshot or activity log */}
                                <div className="flex-1 overflow-y-auto custom-scrollbar">
                                    {lastBrowserEvent?.screenshotBase64 ? (
                                        <div className="flex items-center justify-center p-4">
                                            <img
                                                src={`data:image/png;base64,${lastBrowserEvent.screenshotBase64}`}
                                                alt="Browser screenshot"
                                                className="max-w-full rounded-lg border border-[#262626] shadow-2xl"
                                            />
                                        </div>
                                    ) : (
                                        <div className="p-4 space-y-2">
                                            <p className="text-[9px] uppercase tracking-wider text-[#3a3a3a] font-bold mb-3">Browser Activity</p>
                                            {[...events.browserEvents].reverse().map((ev, i) => (
                                                <div key={ev.id} className={`flex items-start gap-2.5 p-2.5 rounded-lg border ${i === 0 ? 'bg-[#0d1a0d] border-[#00ff99]/15' : 'bg-[#0d0d0d] border-[#1a1a1a]'}`}>
                                                    <div className="flex-shrink-0 mt-0.5">
                                                        {ev.type === 'search' ? <Search size={11} className="text-[#00ff99]" /> : <Globe size={11} className="text-[#00ff99]" />}
                                                    </div>
                                                    <div className="flex-1 min-w-0">
                                                        <p className="text-[10px] font-bold uppercase tracking-wider text-[#525252] mb-0.5">
                                                            {ev.type === 'navigate' ? 'Navigate' : ev.type === 'search' ? 'Search' : ev.type === 'screenshot' ? 'Screenshot' : ev.type}
                                                        </p>
                                                        <p className="text-[11px] text-[#a3a3a3] truncate font-mono">
                                                            {ev.url || ev.query || '—'}
                                                        </p>
                                                    </div>
                                                    <span className="text-[9px] text-[#3a3a3a] flex-shrink-0">
                                                        {ev.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                                    </span>
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            </>
                        )}
                    </div>
                )}

                {/* MEMORY */}
                {activeTab === 'memory' && sessionId && (
                    <div className="absolute inset-0 overflow-y-auto custom-scrollbar p-4">
                        <MemoryView sessionId={sessionId} />
                    </div>
                )}
                {activeTab === 'memory' && !sessionId && (
                    <div className="absolute inset-0 flex items-center justify-center text-[#3a3a3a] text-xs italic">
                        Select a session to view memory.
                    </div>
                )}
            </div>
        </div>
    );
};
